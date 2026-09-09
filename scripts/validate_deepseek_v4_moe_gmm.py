"""Gate shared-GMM MoE with captured real FFN inputs and checkpoint weights.

This isolates routed experts, not a full-layer/model or Engine benchmark.
Both backends consume identical captured activations, expert IDs and mixing
weights. Compilation, loading and CPU oracle work are excluded from timings.
"""

import argparse
import hashlib
import json
import re
import time
import traceback
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4.moe import grouped_fp4_experts
from sgl_jax.srt.kernels.deepseek_v4.moe_gmm import gmm_fp4_experts
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer
from sgl_jax.test.kernels.test_deepseek_v4_low_bit import (
    _bf16,
    _reference_activation,
    _reference_weight,
)


def captured_array(prefix, key):
    schema = json.loads(prefix.with_suffix(".json").read_text())[key]
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as data:
        value = data[key]
    if schema["dtype"] == "bfloat16":
        value = value.view(ml_dtypes.bfloat16)
    if list(value.shape) != schema["shape"] or str(value.dtype) != schema["dtype"]:
        raise ValueError(f"invalid capture field {key}")
    return value


def numpy_one_token(checkpoint, layer, x, ids, mixing, limit):
    """Independent byte/NumPy math; only selected CPU experts are expanded."""
    result = np.zeros(x.shape, np.float32)
    for expert in sorted(set(ids[0].tolist())):
        scale = np.sum(np.where(ids[0] == expert, mixing[0], 0), dtype=np.float32)
        if scale == 0:
            continue

        def project(value, projection, expert=expert):
            host = checkpoint.load_linear(f"layers.{layer}.ffn.experts.{expert}.w{projection}")
            weight = _reference_weight(host.data, host.scales, "fp4")
            return _bf16(_reference_activation(value) @ weight.T)

        gate = np.minimum(project(x, 1), limit)
        up = np.clip(project(x, 3), -limit, limit)
        hidden = _bf16(scale * (gate / (1 + np.exp(-gate))) * up)
        result += project(hidden, 2)
    return result


def expert_slice_lines(hlo, weights):
    # Look at actual optimized instructions, not names in source metadata.
    shapes = {f"u8[1,{w.shape[1]},{w.shape[2]}]" for w in weights}
    return [
        line.strip()
        for line in hlo.splitlines()
        if any(f"= {shape}" in line for shape in shapes)
        and re.search(r"\b(?:fusion|dynamic-slice)\(", line)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-prefix", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 128])
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 5 or any(t <= 0 for t in args.tokens):
        parser.error("requires positive batch sizes and at least five timing samples")
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("this gate requires the actual four-device TPU slice")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "scope": __doc__,
        "layer": args.layer,
        "checkpoint": str(args.checkpoint),
        "capture": str(args.capture_prefix),
        "framework_source_fingerprint": framework_fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "jax_version": jax.__version__,
        "devices": [d.device_kind for d in jax.devices()],
        "checks": [],
        "benchmarks": [],
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        capture_report = json.loads((args.capture_prefix.parent / "report.json").read_text())
        capture_checkpoint = capture_report.get("checkpoint")
        if capture_checkpoint is None:
            fixture_path = Path(capture_report["fixture"])
            fixture = json.loads(fixture_path.read_text())
            capture_checkpoint = fixture["checkpoint"]
            report["capture_checkpoint_binding"] = {
                "method": "historical capture omitted checkpoint; inferred from its fixture report",
                "fixture": str(fixture_path),
                "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            }
        if Path(capture_checkpoint).resolve() != args.checkpoint.resolve():
            raise ValueError("capture and weights must use the same checkpoint")
        if f"layer-{args.layer:02d}-" not in args.capture_prefix.name:
            raise ValueError("capture layer does not match requested weights")
        report["capture_source_fingerprint"] = capture_report.get("source_fingerprint")
        report["capture_schema_sha256"] = hashlib.sha256(
            args.capture_prefix.with_suffix(".json").read_bytes()
        ).hexdigest()
        x = captured_array(args.capture_prefix, "ffn.norm")
        ids = captured_array(args.capture_prefix, "expert_ids")
        mixing = captured_array(args.capture_prefix, "routing_weights")
        if max(args.tokens) > len(x):
            raise ValueError("not enough captured real rows")
        selected = np.linspace(0, len(x) - 1, max(args.tokens), dtype=np.int32)
        x, ids, mixing = x[selected], ids[selected], mixing[selected]
        report["capture_rows"] = selected.tolist()
        checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
        limit = checkpoint.config["swiglu_limit"]
        mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
        with jax.set_mesh(mesh):
            loaded = load_layer(checkpoint, args.layer, mesh)
            weights = tuple(
                loaded["experts." + key] for key in ("w1", "w3", "w2", "s1", "s3", "s2")
            )
            report["packed_expert_bytes_global"] = sum(w.nbytes for w in weights)
            report["packed_expert_bytes_per_chip"] = sum(
                w.addressable_shards[0].data.nbytes for w in weights
            )
            specs = (P(), *(P("tensor", None, None) for _ in weights), P(), P())
            functions = {
                "legacy": lambda *v: grouped_fp4_experts(*v, swiglu_limit=limit),
                "gmm": lambda *v: gmm_fp4_experts(*v, swiglu_limit=limit),
            }
            for tokens in args.tokens:
                inputs = (x[:tokens], *weights, ids[:tokens], mixing[:tokens])
                inputs = tuple(
                    jax.device_put(v, NamedSharding(mesh, spec))
                    for v, spec in zip(inputs, specs, strict=True)
                )
                compiled, results = {}, {}
                for name, fn in functions.items():
                    run = jax.jit(
                        jax.shard_map(fn, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False)
                    )
                    started = time.perf_counter()
                    executable = run.lower(*inputs).compile()
                    compiled[name] = executable
                    results[name] = np.asarray(jax.block_until_ready(executable(*inputs)))
                    hlo = executable.as_text()
                    (args.output / f"{name}-m{tokens}.hlo.txt").write_text(hlo)
                    slices = expert_slice_lines(hlo, weights[:3])
                    report.setdefault("hlo_evidence", []).append(
                        {
                            "backend": name,
                            "tokens": tokens,
                            "expert_slice_instructions": slices,
                            "shared_fp4_gmm_marker": "gmm_checkpoint_fp4" in hlo,
                            "compile_and_first_call_seconds": time.perf_counter() - started,
                        }
                    )
                    if name == "gmm" and (slices or "gmm_checkpoint_fp4" not in hlo):
                        raise AssertionError(
                            "candidate must use shared FP4 GMM without whole-expert slices"
                        )
                metrics = compare_arrays(results["legacy"], results["gmm"])
                report["checks"].append(
                    {"tokens": tokens, "comparison": "legacy_vs_gmm", **metrics}
                )
                save()
                print(json.dumps(report["checks"][-1]), flush=True)
                if not metrics["bitwise_equal"] or not np.isfinite(results["gmm"]).all():
                    raise AssertionError("real-weight GMM must match the retained path bitwise")
                if tokens == 1:
                    oracle = numpy_one_token(
                        checkpoint, args.layer, x[:1], ids[:1], mixing[:1], limit
                    )
                    metrics = compare_arrays(oracle, results["gmm"])
                    report["checks"].append(
                        {
                            "tokens": 1,
                            "comparison": "independent_numpy_vs_gmm",
                            "nrmse_limit": 0.005,
                            **metrics,
                        }
                    )
                    save()
                    if not np.isfinite(oracle).all() or metrics["nrmse"] > 0.005:
                        raise AssertionError("independent real-weight oracle failed")
                samples = {name: [] for name in compiled}
                for executable in compiled.values():
                    for _ in range(3):
                        jax.block_until_ready(executable(*inputs))
                for iteration in range(args.repeats):
                    order = ("legacy", "gmm") if iteration % 2 else ("gmm", "legacy")
                    for name in order:
                        started = time.perf_counter_ns()
                        jax.block_until_ready(compiled[name](*inputs))
                        samples[name].append((time.perf_counter_ns() - started) / 1e6)
                row = {
                    "tokens": tokens,
                    "timing_scope": "routed MoE only; block_until_ready wall time",
                    "samples_ms": samples,
                    **{
                        f"{name}_p50_ms": float(np.median(values))
                        for name, values in samples.items()
                    },
                    **{
                        f"{name}_p95_ms": float(np.percentile(values, 95))
                        for name, values in samples.items()
                    },
                }
                row["p50_speedup"] = row["legacy_p50_ms"] / row["gmm_p50_ms"]
                report["benchmarks"].append(row)
                save()
                print(json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True)
        report["complete"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
