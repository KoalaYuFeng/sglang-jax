"""Real single-layer head-TP experiment, without constructing a model/Engine.

Historical 8023 inputs are immutable operator fixtures, not a same-source
full-model replay. B4 clones B2 requests onto independent physical pages and
private slots. Every timed call starts from an independent copy of the same
frozen cache, prepared outside timing/profiling, and donates that copy. These
are not advancing-decode/model throughput measurements.
"""

import argparse
import gc
import hashlib
import json
import re
import time
import traceback
from collections import deque
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.kernels.deepseek_v4.attention import attention
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import (
    V4PagedBackend,
    V4PagedMetadata,
)
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer
from sgl_jax.test.kernels.csa_compressor_cases import read_arrays
from sgl_jax.test.test_deepseek_v4_paged import make_batch

from benchmark.kernels.csa.bench_v4_compressor import memory_bytes, profile_call

HEAD_WEIGHTS = {
    "attn.wq_b.weight",
    "attn.wq_b.scale",
    "attn.wo_a.weight",
    "attn.wo_a.scale",
    "attn.attn_sink",
}


def read_all(prefix):
    return read_arrays(prefix, json.loads(prefix.with_suffix(".json").read_text()))


def array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(
        str((a.shape, str(a.dtype))).encode() + a.tobytes()
    ).hexdigest()


def clone_b4(hidden, inputs, cache, metadata, config):
    """Duplicate two real requests without sharing writable pages or slots."""
    if hidden.shape[0] != 2 or not np.all(metadata.query_lens == 1):
        raise ValueError("B4 clone requires the two live captured decode requests")
    sources = [0, 1, 0, 1]
    count = config.max_context // 128
    pages = [list(range(1 + i * count, 1 + (i + 1) * count)) for i in range(4)]
    batch = make_batch(inputs["positions"][sources], [1] * 4, pages=pages, decode=True)
    md = V4PagedBackend(max_context=config.max_context).get_forward_metadata(batch)
    cloned = {
        key: np.full_like(value, -np.inf if key.endswith("score") else 0)
        for key, value in cache.items()
    }
    for destination, origin in enumerate(sources):
        used = (int(metadata.seq_lens[origin]) + 127) // 128
        old_pages = metadata.page_table[origin, :used]
        new_pages = np.asarray(pages[destination][:used])
        for key, value in cache.items():
            if key.endswith((".kv", ".score")):
                cloned[key][destination + 1] = value[metadata.req_slots[origin]]
            elif ".snapshot_" in key:
                cloned[key][new_pages] = value[old_pages]
            else:
                rows_per_page = 128 if key == "window" else 128 // config.ratio
                original_pages = value.reshape(-1, rows_per_page, value.shape[-1])
                target_pages = cloned[key].reshape(original_pages.shape)
                target_pages[new_pages] = original_pages[old_pages]
    return (
        hidden[sources],
        {"positions": batch.positions, "locations": batch.out_cache_loc},
        cloned,
        md,
    )


def weight_specs(weights, tp):
    return {
        key: P("tensor", *([None] * (value.ndim - 1)))
        if tp and key in HEAD_WEIGHTS
        else P()
        for key, value in weights.items()
    }


def build(mesh, config, weights, *, tp, trace):
    if tp and (config.heads % mesh.size or config.groups % mesh.size):
        raise ValueError("head TP must keep complete wo_a groups")
    local = (
        replace(
            config, heads=config.heads // mesh.size, groups=config.groups // mesh.size
        )
        if tp
        else config
    )
    specs = (P(), P(), weight_specs(weights, tp), P(), P(), P())
    trace_specs = {key: P() for key in ("q", "qr", "kv", "indices", "attention_value")}
    if config.ratio == 4:
        trace_specs.update(
            {key: P() for key in ("index_q", "index_score", "index_selected")}
        )
    if tp:
        trace_specs.update(
            q=P(None, "tensor", None), attention_value=P(None, "tensor", None)
        )

    def rank_local(x, positions, w, cache, metadata, locations):
        observed = {} if trace else None
        output, updated = attention(
            x,
            positions,
            w,
            cache,
            local,
            metadata,
            locations,
            hca_backend="pallas",
            csa_backend="pallas",
            trace=observed,
            output_tensor_axis="tensor" if tp else None,
        )
        return (output, updated, observed) if trace else (output, updated)

    run = jax.jit(
        jax.shard_map(
            rank_local,
            mesh=mesh,
            in_specs=specs,
            out_specs=(P(), P(), trace_specs) if trace else (P(), P()),
            check_vma=False,
        ),
        donate_argnums=() if trace else (3,),
    )
    return run, specs


def place(mesh, specs, operands):
    return jax.tree.map(
        lambda spec, value: jax.tree.map(
            lambda a: jax.device_put(a, NamedSharding(mesh, spec)), value
        ),
        specs,
        operands,
        is_leaf=lambda value: isinstance(value, P),
    )


def metrics(expected, actual, *, tolerance, exact):
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise AssertionError(
            f"shape/dtype mismatch: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
        )
    # Device layouts can expose non-contiguous host views (notably HCA cache
    # leaves during prefill). Compare logical C-order bytes, not host strides.
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    bitwise = np.array_equal(a.view(np.uint8), b.view(np.uint8))
    mask_a, mask_b = np.isfinite(a), np.isfinite(b)
    special_ok = (
        np.array_equal(mask_a, mask_b)
        and not np.any(np.isnan(a))
        and not np.any(np.isnan(b))
    )
    special_ok = special_ok and bool(np.all(a[~mask_a] == b[~mask_a]))
    error, maximum = 0.0, 0.0
    if not bitwise:
        af, bf = a[mask_a].astype(np.float64), b[mask_a].astype(np.float64)
        error = float(np.linalg.norm(bf - af) / max(np.linalg.norm(af), 1e-12))
        maximum = float(np.max(np.abs(bf - af), initial=0))
    return {
        "bitwise_equal": bool(bitwise),
        "nrmse": error,
        "max_abs": maximum,
        "nonfinite_contract_passed": bool(special_ok),
        "tolerance": tolerance,
        "exact_required": exact,
        "passed": bool(special_ok and error <= tolerance and (bitwise or not exact)),
    }


class IndependentSteps:
    def __init__(self, compiled, operands, host_cache, mesh):
        self.compiled, self.operands = compiled, tuple(operands)
        self.host_cache, self.mesh = host_cache, mesh
        self.pending = deque()

    def prepare(self, count):
        """Allocate independent donated inputs OUTSIDE timed/profiled calls."""
        if self.pending:
            raise ValueError("unconsumed prepared steps")
        with jax.set_mesh(self.mesh):
            sharding = NamedSharding(self.mesh, P())
            for _ in range(count):
                cache = jax.tree.map(
                    lambda a: jax.device_put(np.array(a, copy=True), sharding),
                    self.host_cache,
                )
                jax.block_until_ready(cache)
                self.pending.append(self.operands[:3] + (cache,) + self.operands[4:])

    def __call__(self):
        if not self.pending:
            raise ValueError("prepare frozen inputs before timing")
        return self.compiled(*self.pending.popleft())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    parser.add_argument(
        "--frames", nargs="+", default=["B1-case1", "B2", "B4-cloned-B2"]
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        raise RuntimeError("requires exactly four physical TPU chips")
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.capture / "report.json").read_text())
    replay = json.loads((args.replay / "report.json").read_text())
    if (
        not replay["faithful"]
        or replay["framework_source_fingerprint"]
        != source["framework_source_fingerprint"]
    ):
        raise ValueError("requires a faithful matching historical capture/replay")
    checkpoint = DeepSeekV4Checkpoint(source["checkpoint"])
    from analyze_deepseek_v4_native_profile import fingerprint

    report = {
        "complete": False,
        "scope": __doc__,
        "framework_source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": source["checkpoint"],
        "capture": str(args.capture),
        "replay": str(args.replay),
        "capture_sha256": hashlib.sha256(
            (args.capture / "report.json").read_bytes()
        ).hexdigest(),
        "replay_sha256": hashlib.sha256(
            (args.replay / "report.json").read_bytes()
        ).hexdigest(),
        "checks": [],
        "cases": [],
    }

    def flush():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def check(label, expected, actual, *, tolerance=0, exact=True):
        row = {
            "label": label,
            **metrics(expected, actual, tolerance=tolerance, exact=exact),
        }
        report["checks"].append(row)
        if not row["passed"]:
            flush()
            raise AssertionError(row)

    def compare_result(label, expected, actual):
        check(label + "/output", expected[0], actual[0], tolerance=0.005, exact=False)
        for key, value in expected[1].items():
            check(label + "/cache/" + key, value, actual[1][key])
        if len(expected) == 3 and len(actual) == 3:
            for key, value in expected[2].items():
                tol = 2e-4 if key == "attention_value" else 0
                check(
                    label + "/trace/" + key,
                    value,
                    actual[2][key],
                    tolerance=tol,
                    exact=tol == 0,
                )

    flush()
    try:
        single_mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",))
        four_mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
        for layer in args.layers:
            config = config_for_layer(checkpoint.config, layer, 8192)
            if config.ratio not in (4, 128):
                raise ValueError("this gate covers CSA/HCA layers")
            with jax.set_mesh(single_mesh):
                staged = load_layer(
                    checkpoint, layer, single_mesh, include_experts=False
                )
                weights = {
                    key: np.asarray(value)
                    for key, value in staged.items()
                    if key.startswith("attn.")
                }
            del staged
            for frame in args.frames:
                label = f"layer{layer}/{frame}"
                folder = args.capture / ("B2" if frame == "B4-cloned-B2" else frame)
                inputs, cache = (
                    read_all(folder / "inputs"),
                    read_all(folder / "before" / f"layer-{layer:02d}"),
                )
                metadata = V4PagedMetadata(**read_all(folder / "metadata"))
                trace_prefix = args.replay / f"{folder.name}-layer-{layer:02d}-trace"
                hidden = read_arrays(trace_prefix, ["attn.norm"])["attn.norm"]
                if frame == "B4-cloned-B2":
                    hidden, inputs, cache, metadata = clone_b4(
                        hidden, inputs, cache, metadata, config
                    )
                operands = (
                    hidden,
                    inputs["positions"],
                    weights,
                    cache,
                    metadata,
                    inputs["locations"],
                )
                case = {
                    "label": label,
                    "ratio": config.ratio,
                    "input_digest": array_digest(hidden),
                    "operand_digests": {
                        "positions": array_digest(inputs["positions"]),
                        "locations": array_digest(inputs["locations"]),
                        "weights": {
                            key: array_digest(value) for key, value in weights.items()
                        },
                        "cache": {
                            key: array_digest(value) for key, value in cache.items()
                        },
                        "metadata": {
                            key: array_digest(value)
                            for key, value in vars(metadata).items()
                        },
                    },
                    "variants": {},
                }
                report["cases"].append(case)
                runtimes, golden = {}, None
                for name, mesh, tp in (
                    ("single", single_mesh, False),
                    ("replicated4", four_mesh, False),
                    ("head_tp4", four_mesh, True),
                ):
                    print(
                        json.dumps(
                            {"event": "compile", "case": label, "variant": name}
                        ),
                        flush=True,
                    )
                    with jax.set_mesh(mesh):
                        traced, specs = build(mesh, config, weights, tp=tp, trace=True)
                        traced_args = place(mesh, specs, operands)
                        observed = traced(*traced_args)
                        actual = jax.tree.map(np.asarray, observed)
                        if golden is None:
                            golden = actual
                        else:
                            compare_result(label + "/" + name, golden, actual)
                        local_heads = config.heads // mesh.size if tp else config.heads
                        shapes = [
                            list(s.data.shape)
                            for s in observed[2]["q"].addressable_shards
                        ]
                        assert len(shapes) == mesh.size and all(
                            s == [hidden.shape[0], local_heads, config.head_dim]
                            for s in shapes
                        )
                        for key, value in observed[1].items():
                            for i, shard in enumerate(value.addressable_shards):
                                check(
                                    f"{label}/{name}/replica{i}/{key}",
                                    golden[1][key],
                                    shard.data,
                                )
                        run, specs = build(mesh, config, weights, tp=tp, trace=False)
                        call_args = place(mesh, specs, operands)
                        compiled = run.lower(*call_args).compile()
                        hlo = compiled.as_text()
                        stem = f"layer{layer}-{frame}-{name}"
                        (args.output / f"{stem}.hlo.txt").write_text(hlo)
                        family = (
                            "csa-joint-attention"
                            if config.ratio == 4
                            else "hca-paged-stream"
                        )
                        calls = [
                            line
                            for line in hlo.splitlines()
                            if "custom-call(" in line and family in line
                        ]
                        if (
                            len(calls) != 1
                            or f"-h{local_heads}-d512-v4" not in calls[0]
                        ):
                            raise AssertionError(
                                "missing actual local-head Pallas dispatch"
                            )
                        collectives = [
                            line.split("metadata=")[0]
                            for line in hlo.splitlines()
                            if re.search(
                                r"\b(?:all-gather|all-reduce|reduce-scatter)\(", line
                            )
                        ]
                        if tp and (
                            not any("all-gather(" in line for line in collectives)
                            or any("all-reduce(" in line for line in collectives)
                        ):
                            raise AssertionError(
                                "TP must gather groups without a floating sum collective"
                            )
                        runtime = IndependentSteps(compiled, call_args, cache, mesh)
                        runtime.prepare(2)
                        first = jax.tree.map(np.asarray, runtime())
                        compare_result(label + "/" + name + "/untraced", golden, first)
                        second = jax.tree.map(np.asarray, runtime())
                        check(
                            label + "/" + name + "/fresh_snapshot_output",
                            first[0],
                            second[0],
                        )
                        for key in first[1]:
                            check(
                                label + "/" + name + "/fresh_snapshot_cache/" + key,
                                first[1][key],
                                second[1][key],
                            )
                        runtimes[name] = runtime
                        case["variants"][name] = {
                            "query_shard_shapes": shapes,
                            "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
                            "memory_analysis": memory_bytes(compiled),
                            "collectives": collectives,
                        }
                        del traced, traced_args, observed, actual, first, second
                    flush()
                samples = {name: [] for name in runtimes}
                for step in range(args.iterations + 4):
                    names = (
                        list(runtimes) if step % 2 == 0 else list(reversed(runtimes))
                    )
                    for name in names:
                        runtimes[name].prepare(1)
                        start = time.perf_counter_ns()
                        jax.block_until_ready(runtimes[name]())
                        if step >= 4:
                            samples[name].append((time.perf_counter_ns() - start) / 1e6)
                for name, values in samples.items():
                    if values:
                        case["variants"][name]["host_ms"] = {
                            "samples": values,
                            "median": float(np.median(values)),
                            "p90": float(np.percentile(values, 90)),
                        }
                    if args.profile:
                        runtimes[name].prepare(8)
                        case["variants"][name]["profile"] = profile_call(
                            runtimes[name],
                            ((),),
                            args.output / "traces" / f"layer{layer}-{frame}-{name}",
                            8,
                        )
                    runtimes[name].prepare(1)
                    compare_result(
                        label + "/" + name + "/after_timing", golden, runtimes[name]()
                    )
                print(
                    json.dumps(
                        {
                            "event": "case_passed",
                            "case": label,
                            "host_ms": {
                                n: v.get("host_ms", {}).get("median")
                                for n, v in case["variants"].items()
                            },
                        }
                    ),
                    flush=True,
                )
                flush()
                del runtimes, golden, operands, cache, hidden, runtime, call_args
                gc.collect()
            del weights
            jax.clear_caches()
            gc.collect()
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        flush()


if __name__ == "__main__":
    main()
