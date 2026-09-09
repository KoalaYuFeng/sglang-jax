"""Real-weight EP4 FP4 adapter A/B, isolated from the serving scheduler.

Original checkpoint bytes and captured layer-3 routes/activations are retained.
No BF16 weight expansion, sum-tree change, global defaults or production loader
changes. M1/M4/M128 refer to routed tokens, not end-to-end serving throughput.
Compile, setup, validation, warmup and tracing are excluded from rotated timings.
Failed candidates remain in the report and are never timed as accepted results.
"""

import argparse
import hashlib
import json
import re
import time
import traceback
from pathlib import Path

import jax
import numpy as np
import profile_deepseek_v4_fp4_moe as diagnostic_module
import sgl_jax.srt.kernels.low_bit.fp4_tuning as candidate_module
from jax.experimental.layout import Format, Layout
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from profile_deepseek_v4_fp4_moe import diagnostic_moe, route_geometry
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4.moe_gmm import gmm_fp4_experts
from sgl_jax.srt.kernels.low_bit.fp4_tuning import (
    CandidateCheckpointFP4Rhs,
    transpose_compact_scales,
)
from sgl_jax.srt.model_executor.deepseek_v4_reference import source_fingerprint
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer
from validate_deepseek_v4_moe_gmm import captured_array, numpy_one_token


def scale_copies(hlo):
    """Concrete scale-shape copies; include SSA witnesses rather than source labels."""
    return [
        line.strip()
        for line in hlo.splitlines()
        if re.search(
            r"= u8\[64,(?:4096,64|64,4096|2048,128|128,2048)\].*? copy\(", line
        )
    ]


def tile_geometry(group_sizes, tile_m):
    """Logical group tile counts, not measured MXU utilization or DMA traffic."""
    sizes = np.asarray(group_sizes, np.int64).copy()
    sizes[-1] += (-int(sizes.sum())) % tile_m
    ends = np.cumsum(sizes)
    starts = ends - sizes
    counts = np.where(sizes > 0, (ends + tile_m - 1) // tile_m - starts // tile_m, 0)
    return [int(counts[chip * 64 : (chip + 1) * 64].sum()) for chip in range(4)]


def stress_routes(x, ids, mixing):
    """Synthetic routing adversaries over unchanged REAL checkpoint weights."""
    yield "reordered_rows", x[::-1].copy(), ids[::-1].copy(), mixing[::-1].copy()
    yield "all_inactive", x, ids, np.zeros_like(mixing)
    first = np.broadcast_to(np.arange(ids.shape[1], dtype=ids.dtype), ids.shape).copy()
    yield "only_chip0_active", x, first, mixing
    yield "only_chip3_active", x, first + 250, mixing
    duplicate_mix = np.zeros_like(mixing)
    duplicate_mix[:, :3] = (0.25, -0.25, 0.5)
    yield "duplicate_cancel_and_coalesce", x, np.zeros_like(ids), duplicate_mix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument(
        "--capture-prefix",
        type=Path,
        help="restored capture path; bytes must match the accepted fixture hashes",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 128])
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "production",
            "pinned_scale_layout",
            "transpose_m8_n128",
            "transpose_m8_n256",
            "transpose_m16_n128",
            "transpose_m32_n128",
        ],
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--stress-routes", action="store_true")
    args = parser.parse_args()
    if args.repeats < 10 or not args.tokens or min(args.tokens) <= 0:
        raise ValueError("positive token counts and at least ten repeats required")
    fixture = json.loads(args.fixture.read_text())
    if not fixture.get("complete") or fixture["layer"] != 3:
        raise ValueError("requires accepted layer-3 real-weight fixture")
    if args.variants[0] != "production":
        raise ValueError("fresh unchanged production control must be first")
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("requires an idle four-chip TPU slice")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "scope": __doc__,
        "framework_source_fingerprint": framework_fingerprint(),
        "source_fingerprint": source_fingerprint(),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "fixture_source_fingerprint": fixture["framework_source_fingerprint"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_kernel_sha256": hashlib.sha256(
            Path(candidate_module.__file__).read_bytes()
        ).hexdigest(),
        "diagnostic_glue_sha256": hashlib.sha256(
            Path(diagnostic_module.__file__).read_bytes()
        ).hexdigest(),
        "checkpoint": fixture["checkpoint"],
        "capture": fixture["capture"],
        "capture_rows": fixture["capture_rows"],
        "jax_version": jax.__version__,
        "devices": [str(d) + ": " + d.device_kind for d in jax.devices()],
        "checks": [],
        "captures": [],
        "runs": {},
        "failed_candidates": [],
    }

    def save(event):
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    def check(label, actual, expected):
        metrics = compare_arrays(expected, actual)
        metrics["all_finite"] = bool(
            np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
        )
        report["checks"].append({"label": label, **metrics})
        if not metrics["bitwise_equal"] or not metrics["all_finite"]:
            raise AssertionError(f"bitwise numerical gate failed: {label}: {metrics}")

    try:
        capture = args.capture_prefix or Path(fixture["capture"])
        report["actual_capture"] = str(capture)
        report["capture_sha256"] = {
            suffix: hashlib.sha256(capture.with_suffix(suffix).read_bytes()).hexdigest()
            for suffix in (".json", ".npz")
        }
        if report["capture_sha256"][".json"] != fixture["capture_schema_sha256"]:
            raise ValueError("restored capture schema differs from accepted fixture")
        if args.capture_manifest is not None:
            manifest = json.loads(args.capture_manifest.read_text())
            if (
                manifest["subset_npz_sha256"] != report["capture_sha256"][".npz"]
                or manifest["schema_sha256"] != report["capture_sha256"][".json"]
            ):
                raise ValueError(
                    "restored capture subset differs from original-member manifest"
                )
            report["capture_subset_manifest"] = manifest
        selected = np.asarray(fixture["capture_rows"], np.int32)
        x, ids, mixing = (
            captured_array(capture, key)[selected]
            for key in ("ffn.norm", "expert_ids", "routing_weights")
        )
        if max(args.tokens) > len(x):
            raise ValueError("not enough accepted captured rows")
        checkpoint = DeepSeekV4Checkpoint(Path(fixture["checkpoint"]))
        limit = checkpoint.config["swiglu_limit"]
        mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
        spec = P("tensor", None, None)
        specs = (P(), *(spec for _ in range(6)), P(), P())
        with jax.set_mesh(mesh):
            loaded = load_layer(checkpoint, 3, mesh)
            raw = tuple(
                loaded["experts." + key] for key in ("w1", "w3", "w2", "s1", "s3", "s2")
            )
            transpose = jax.jit(
                jax.shard_map(
                    transpose_compact_scales,
                    mesh=mesh,
                    in_specs=spec,
                    out_specs=spec,
                    check_vma=False,
                )
            )
            transposed = (
                *raw[:3],
                *(jax.block_until_ready(transpose(s)) for s in raw[3:]),
            )
            for index in range(3):
                # Scale-byte reversibility is independent of the model output gate.
                np.testing.assert_array_equal(
                    np.asarray(transposed[index + 3]).swapaxes(1, 2),
                    np.asarray(raw[index + 3]),
                )
            report["logical_raw_bytes_per_chip"] = sum(
                value.addressable_shards[0].data.nbytes for value in raw
            )
            report["logical_candidate_bytes_per_chip"] = sum(
                value.addressable_shards[0].data.nbytes for value in transposed
            )
            save({"event": "loaded_and_scale_bytes_checked"})
            for tokens in args.tokens:
                row = {
                    "geometry": route_geometry(ids[:tokens], mixing[:tokens]),
                    "variants": {},
                }
                report["runs"][str(tokens)] = row
                calls, inputs, expected = {}, {}, None
                for name in args.variants:
                    start = time.perf_counter()
                    try:
                        weights = (
                            transposed
                            if name.startswith("transpose_") or name == "tuned_entry"
                            else raw
                        )
                        values = (x[:tokens], *weights, ids[:tokens], mixing[:tokens])
                        values = tuple(
                            jax.device_put(v, NamedSharding(mesh, s))
                            for v, s in zip(values, specs, strict=True)
                        )
                        jit_kwargs = {}
                        if name == "pinned_scale_layout":
                            # Pin both placement AND the JIT ABI; merely device_put-ing
                            # a different layout does not change a default-layout JIT.
                            target = Format(
                                Layout((0, 1, 2), ((32, 128), (4, 1))),
                                NamedSharding(mesh, spec),
                            )
                            values = (
                                *values[:6],
                                jax.device_put(values[6], target),
                                *values[7:],
                            )
                            jit_kwargs["in_shardings"] = tuple(v.format for v in values)
                        if name == "tuned_entry":
                            from sgl_jax.srt.kernels.low_bit.fp4_tuning import (
                                tuned_fp4_tile_m,
                            )

                            tile_m, tile_n = tuned_fp4_tile_m(tokens), 256
                            fn = lambda *v: gmm_fp4_experts(
                                *v, swiglu_limit=limit, tuned=True
                            )
                        elif name in ("production", "pinned_scale_layout"):
                            tile_m, tile_n = 8, 128
                            fn = lambda *v: gmm_fp4_experts(*v, swiglu_limit=limit)
                        else:
                            match = re.fullmatch(
                                r"transpose_m(8|16|32)_n(128|256|512)(_packed)?", name
                            )
                            if match is None:
                                raise ValueError(f"unknown variant {name}")
                            adapter = CandidateCheckpointFP4Rhs(
                                transpose_scales=True,
                                tile_m=int(match[1]),
                                tile_n=int(match[2]),
                                packed_scale=bool(match[3]),
                            )
                            tile_m, tile_n = adapter.tile_m, adapter.tile_n
                            fn = lambda *v, adapter=adapter: diagnostic_moe(
                                *v, adapter=adapter, limit=limit
                            )
                        run = jax.jit(
                            jax.shard_map(
                                fn,
                                mesh=mesh,
                                in_specs=specs,
                                out_specs=P(),
                                check_vma=False,
                            ),
                            **jit_kwargs,
                        )
                        compiled = run.lower(*values).compile()
                        actual = np.asarray(compiled(*values))
                        if expected is None:
                            expected = actual
                            if tokens == 1:
                                oracle = numpy_one_token(
                                    checkpoint,
                                    3,
                                    x[:1].astype(np.float32),
                                    ids[:1],
                                    mixing[:1],
                                    limit,
                                )
                                check("production_M1_independent_numpy", actual, oracle)
                        check(f"M{tokens}/{name}", actual, expected)
                        hlo = compiled.as_text()
                        (args.output / f"M{tokens}-{name}.hlo.txt").write_text(hlo)
                        row["variants"][name] = {
                            "compile_setup_check_seconds": time.perf_counter() - start,
                            "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
                            "scale_copy_witnesses": scale_copies(hlo),
                            "input_formats": [str(v.format) for v in values],
                            "input_physical_bytes_per_chip": [
                                v.addressable_shards[0].data.on_device_size_in_bytes()
                                for v in values
                            ],
                            "compiler_temp_hbm_bytes_per_chip": compiled.memory_analysis().temp_size_in_bytes,
                            "samples_ms": [],
                            "tile_m": tile_m,
                            "tile_n": tile_n,
                            "logical_group_m_tiles_by_chip": tile_geometry(
                                row["geometry"]["group_sizes"], tile_m
                            ),
                        }
                        calls[name], inputs[name] = compiled, values
                        for _ in range(5):
                            jax.block_until_ready(compiled(*values))
                        save(
                            {
                                "event": "candidate_bitwise_checked",
                                "tokens": tokens,
                                "variant": name,
                                "scale_copies": len(scale_copies(hlo)),
                            }
                        )
                    except Exception:
                        if name == "production":
                            raise
                        report["failed_candidates"].append(
                            {
                                "tokens": tokens,
                                "variant": name,
                                "error": traceback.format_exc(),
                            }
                        )
                        save(
                            {
                                "event": "candidate_rejected",
                                "tokens": tokens,
                                "variant": name,
                                "error": report["failed_candidates"][-1]["error"],
                            }
                        )
                names = list(calls)
                for index in range(args.repeats):
                    order = names[index % len(names) :] + names[: index % len(names)]
                    for name in order:
                        start = time.perf_counter()
                        jax.block_until_ready(calls[name](*inputs[name]))
                        row["variants"][name]["samples_ms"].append(
                            (time.perf_counter() - start) * 1000
                        )
                for name, stats in row["variants"].items():
                    stats.update(
                        p50_ms=float(np.median(stats["samples_ms"])),
                        p95_ms=float(np.percentile(stats["samples_ms"], 95)),
                    )
                    check(
                        f"M{tokens}/{name}/after_timing",
                        np.asarray(calls[name](*inputs[name])),
                        expected,
                    )
                save(
                    {
                        "event": "timed",
                        "tokens": tokens,
                        "p50_ms": {n: s["p50_ms"] for n, s in row["variants"].items()},
                    }
                )
                if args.stress_routes:
                    for label, sx, si, sm in stress_routes(
                        x[:tokens], ids[:tokens], mixing[:tokens]
                    ):
                        stress_expected = None
                        for name in names:
                            values = (
                                jax.device_put(sx, inputs[name][0].sharding),
                                *inputs[name][1:7],
                                jax.device_put(si, inputs[name][7].sharding),
                                jax.device_put(sm, inputs[name][8].sharding),
                            )
                            actual = np.asarray(calls[name](*values))
                            if stress_expected is None:
                                stress_expected = actual
                                if label == "all_inactive":
                                    np.testing.assert_array_equal(
                                        actual, np.zeros_like(actual)
                                    )
                            check(f"M{tokens}/{name}/{label}", actual, stress_expected)
                    save({"event": "stress_routes_passed", "tokens": tokens})
                if args.profile:
                    for name in names:
                        options = jax.profiler.ProfileOptions()
                        options.host_tracer_level = 2
                        options.python_tracer_level = 0
                        options.enable_hlo_proto = True
                        options.raise_error_on_start_failure = True
                        options.advanced_configuration = {
                            "tpu_trace_mode": "TRACE_ONLY_XLA",
                            "tpu_num_chips_to_profile_per_task": 4,
                            "tpu_num_sparse_cores_to_trace": 0,
                        }
                        jax.profiler.start_trace(
                            str(args.output / "traces" / f"M{tokens}_{name}"),
                            profiler_options=options,
                        )
                        seconds = []
                        try:
                            for step in range(8):
                                start = time.perf_counter()
                                with jax.profiler.StepTraceAnnotation(
                                    "V4_PREFILL" if tokens == 128 else "V4_DECODE",
                                    step_num=step,
                                ):
                                    result = jax.block_until_ready(
                                        calls[name](*inputs[name])
                                    )
                                seconds.append(time.perf_counter() - start)
                        finally:
                            jax.profiler.stop_trace()
                        check(
                            f"M{tokens}/{name}/trace_replay",
                            np.asarray(result),
                            expected,
                        )
                        report["captures"].append(
                            {
                                "label": f"M{tokens}_{name}",
                                "seconds": seconds,
                                "chips": 4,
                            }
                        )
                        save({"event": "profiled", "tokens": tokens, "variant": name})
            report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        save({"event": "finished", "complete": report["complete"]})


if __name__ == "__main__":
    main()
