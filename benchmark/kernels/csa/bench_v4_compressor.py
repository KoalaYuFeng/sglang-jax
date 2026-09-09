"""Standalone V4 raw-window compressor correctness/freeze and A/B timing.

No model, scheduler, checkpoint allocation, or Engine imports. Host-ready
latency includes Python dispatch and synchronization; it is not device time.
Historical inputs and expected outputs are content-addressed in each receipt.
"""

import argparse
import functools
import hashlib
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.csa.compressor import csa_emit_selected_pallas
from sgl_jax.test.kernels.csa_compressor_cases import (
    make_case,
    oracle_check,
    real_cases,
)


def baseline(values, scores, norm, phase, valid):
    """The actual V4 two-gather + mask + selected-emitter boundary."""
    groups, _, width = values.shape
    dim = width // 2
    columns = jnp.arange(dim)[None] + (jnp.arange(8) >= 4)[:, None] * dim
    columns = jnp.broadcast_to(columns, (groups, 8, dim)).reshape(-1, dim)
    values = jnp.take_along_axis(values.reshape(-1, width), columns, axis=1).reshape(groups, 8, dim)
    scores = jnp.take_along_axis(scores.reshape(-1, width), columns, axis=1).reshape(groups, 8, dim)
    scores = jnp.where(valid[:, None, None], scores, 0)
    return csa_emit_selected_pallas(values, scores, norm, phase, valid)


def source_hashes():
    root = Path(__file__).resolve().parents[3]
    names = [
        "python/sgl_jax/srt/kernels/csa/compressor.py",
        "python/sgl_jax/srt/kernels/csa/tune.py",
        "python/sgl_jax/srt/kernels/low_bit/formats.py",
        "python/sgl_jax/test/kernels/csa_compressor_cases.py",
        "benchmark/kernels/csa/bench_v4_compressor.py",
    ]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def memory_bytes(compiled):
    memory = compiled.memory_analysis()
    return {
        key: getattr(memory, key, None)
        for key in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
        )
    }


def paired_timings(calls, ring, iterations):
    samples = {name: [] for name in calls}
    for operands in ring:
        for call in calls.values():
            jax.block_until_ready(call(*operands))
    for i in range(iterations):
        order = list(calls) if i % 2 == 0 else list(reversed(calls))
        for name in order:
            begin = time.perf_counter_ns()
            jax.block_until_ready(calls[name](*ring[i % len(ring)]))
            samples[name].append((time.perf_counter_ns() - begin) / 1e6)
    return {
        name: {
            "samples": values,
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }
        for name, values in samples.items()
    }


def profile_call(call, ring, directory, iterations):
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    with jax.profiler.trace(str(directory), profiler_options=options):
        for i in range(iterations):
            jax.block_until_ready(call(*ring[i % len(ring)]))
    paths = list(directory.rglob("*.xplane.pb"))
    if len(paths) != 1:
        raise ValueError(f"expected one XPlane: {directory}")
    data = jax.profiler.ProfileData.from_file(str(paths[0]))
    lanes = []
    for lane in range(2 * jax.device_count()):
        name = f"/device:TPU:{lane}"
        plane = data.find_plane_with_name(name)
        if plane is None:
            continue
        modules = []
        for line in plane.lines:
            if line.name == "XLA Modules":
                modules.extend(line.events)
        if not modules:
            continue  # inactive TensorCores are not extra executing lanes
        modules.sort(key=lambda event: event.start_ns)
        if len(modules) != iterations:
            raise ValueError(
                f"incomplete/ambiguous module coverage: {name}: {len(modules)} != {iterations}"
            )
        lanes.append(
            {
                "name": name,
                "modules": [
                    {
                        "name": event.name,
                        "start_ns": int(event.start_ns),
                        "end_ns": int(event.end_ns),
                    }
                    for event in modules
                ],
            }
        )
    if not lanes:
        raise ValueError("no active TPU module timeline")
    envelope = [
        (
            max(lane["modules"][i]["end_ns"] for lane in lanes)
            - min(lane["modules"][i]["start_ns"] for lane in lanes)
        )
        / 1e6
        for i in range(iterations)
    ]
    return {
        "xplane": str(paths[0]),
        "iterations": iterations,
        "active_lanes": lanes,
        "device_module_envelope_ms": {"samples": envelope, "median": float(np.median(envelope))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path)
    parser.add_argument("--candidate", choices=("selected", "overlap"), default="selected")
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 5, 9, 65])
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--tile-groups", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-groups", type=int, nargs="+", default=[5, 1985])
    parser.add_argument("--profile-iterations", type=int, default=10)
    args = parser.parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError("requires a real TPU, not Pallas interpretation")
    if any(g < 1 for g in args.groups) or args.iterations < 0:
        raise ValueError("groups must be positive and iterations nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    baseline_report = None
    frozen = None
    if args.baseline_receipt:
        baseline_report = json.loads((args.baseline_receipt / "report.json").read_text())
        if not baseline_report["complete"] or baseline_report["candidate"] != "selected":
            raise ValueError("requires an accepted frozen selected-emitter baseline")
        if (
            hashlib.sha256((args.baseline_receipt / "expected.npz").read_bytes()).hexdigest()
            != baseline_report["expected_sha256"]
        ):
            raise ValueError("frozen expected-output checksum mismatch")
        frozen = np.load(args.baseline_receipt / "expected.npz", allow_pickle=False)
    if args.candidate == "overlap" and frozen is None:
        raise ValueError("candidate must be checked against a previously frozen baseline")
    fn = baseline
    if args.candidate == "overlap":
        from sgl_jax.srt.kernels.csa.compressor import csa_emit_overlap_pallas

        fn = functools.partial(csa_emit_overlap_pallas, tile_groups=args.tile_groups)
    run = jax.jit(fn)
    report = {
        "complete": False,
        "scope": __doc__,
        "candidate": args.candidate,
        "tile_groups": args.tile_groups if args.candidate == "overlap" else 4,
        "timing_contract": "alternating A/B order, four input buffers; host-ready includes dispatch; device envelopes reported separately",
        "sources": source_hashes(),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "execution_device": str(jax.devices()[0]),
        "baseline_receipt": (str(args.baseline_receipt) if args.baseline_receipt else None),
        "checks": [],
        "cases": {},
    }
    expected = {}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        cases = [make_case(dim, groups) for dim in (128, 512) for groups in args.groups]
        cases.extend(real_cases(args.fixture_root))
        with jax.default_device(jax.devices()[0]):
            for case in cases:
                operands = tuple(jnp.asarray(a) for a in case.inputs())
                compiled = run.lower(*operands).compile()
                value = np.asarray(compiled(*operands))
                check = {
                    "case": case.name,
                    "input_sha256": case.digest(),
                    **oracle_check(case, value),
                }
                if frozen is not None:
                    if baseline_report["cases"][case.name]["input_sha256"] != case.digest():
                        raise ValueError(f"fixture changed: {case.name}")
                    check["frozen_bitwise_equal"] = bool(
                        np.array_equal(value.view(np.uint16), frozen[case.name])
                    )
                report["checks"].append(check)
                save()
                if not (check["passed"] and check.get("frozen_bitwise_equal", True)):
                    raise AssertionError(check)
                expected[case.name] = value.view(np.uint16)
                hlo = compiled.as_text()
                (args.output / f"{case.name}.hlo.txt").write_text(hlo)
                stats = {
                    "input_sha256": case.digest(),
                    "shape": list(case.values.shape),
                    "valid_groups": int(case.valid.sum()),
                    "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
                    "compiled_memory_bytes": memory_bytes(compiled),
                    "logical_selected_window_bytes": int(
                        2 * case.values.shape[0] * 8 * case.dim * 4
                    ),
                    "raw_window_vmem_tile_bytes": int(2 * args.tile_groups * 8 * 2 * case.dim * 4),
                }
                if args.iterations or args.profile:
                    calls = {args.candidate: compiled}
                    if args.candidate == "overlap":
                        selected = jax.jit(baseline).lower(*operands).compile()
                        selected_value = np.asarray(selected(*operands))
                        if not np.array_equal(selected_value.view(np.uint16), frozen[case.name]):
                            raise ValueError(f"paired baseline changed: {case.name}")
                        calls = {"selected": selected, **calls}
                        stats["paired_baseline_memory_bytes"] = memory_bytes(selected)
                    ring = [operands] + [
                        tuple(jnp.asarray(a.copy()) for a in case.inputs()) for _ in range(3)
                    ]
                    if args.iterations:
                        stats["paired_host_ready_ms"] = paired_timings(calls, ring, args.iterations)
                    if (
                        args.profile
                        and case.values.shape[0] in args.profile_groups
                        and case.name.startswith("random")
                    ):
                        stats["profiles"] = {
                            name: profile_call(
                                call,
                                ring,
                                args.output / "traces" / case.name / name,
                                args.profile_iterations,
                            )
                            for name, call in calls.items()
                        }
                report["cases"][case.name] = stats
                # Keep raw samples/timelines in the immutable receipt, not
                # thousands of duplicated lines in an interactive SSH log.
                summary = {
                    "check": check,
                    "compiled_memory_bytes": stats["compiled_memory_bytes"],
                    "host_ready_median_ms": {
                        name: timing["median"]
                        for name, timing in stats.get("paired_host_ready_ms", {}).items()
                    },
                    "device_median_ms": {
                        name: profile["device_module_envelope_ms"]["median"]
                        for name, profile in stats.get("profiles", {}).items()
                    },
                }
                print(json.dumps(summary), flush=True)
                save()
        np.savez(args.output / "expected.npz", **expected)
        report["expected_sha256"] = hashlib.sha256(
            (args.output / "expected.npz").read_bytes()
        ).hexdigest()
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        if frozen is not None:
            frozen.close()
        save()


if __name__ == "__main__":
    main()
