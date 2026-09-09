"""Profile the already-gated V4 reference without changing numerical source.

Raw XPlane traces contain measured execution intervals. Compiler cost/memory
analysis and LLO slot utilization are not physical memory-traffic counters.
Only diagnostic names are monkey-patched, with bitwise replay checks against
the unannotated execution. No weights are preconverted or offloaded.
"""

import argparse
import collections
import functools
import hashlib
import importlib
import json
import os
import resource
import time
from pathlib import Path

import jax
import numpy as np
import psutil

import sgl_jax.srt.model_executor.deepseek_v4_reference as reference


def array_inventory(model):
    categories = collections.defaultdict(lambda: collections.defaultdict(int))

    def add(category, array):
        for shard in array.addressable_shards:
            categories[category][str(shard.device.id)] += shard.data.nbytes

    for layer in model.layers:
        for name, value in layer.items():
            if name.startswith("experts."):
                category = "routed_fp4_scales" if name.startswith("experts.s") else "routed_fp4"
            elif name.endswith(".scale"):
                category = "replicated_fp8_scales"
            elif value.dtype == np.uint8:
                category = "replicated_fp8"
            else:
                category = "replicated_" + str(value.dtype)
            add(category, value)
    for name, value in model.shared.items():
        add("embedding" if name == "embed.weight" else "head_and_final_norm", value)
    for cache in model.cache:
        for name, value in cache.items():
            category = (
                "kv_window"
                if name == "window"
                else "kv_compressed" if name.endswith(".compressed") else "compressor_scratch"
            )
            add(category, value)
    return {key: dict(value) for key, value in categories.items()}


def memory_snapshot():
    return {
        "devices": [dict(id=d.id, stats=d.memory_stats()) for d in jax.devices()],
        "host_rss_bytes": psutil.Process().memory_info().rss,
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def install_names():
    """Metadata-only wrappers, restored by the caller after profiling."""
    originals = []

    def wrap(owner, name, label):
        original = getattr(owner, name)
        originals.append((owner, name, original))

        @functools.wraps(original)
        def call(*args, **kwargs):
            scope = label(*args, **kwargs) if callable(label) else label
            with jax.named_scope(scope):
                return original(*args, **kwargs)

        setattr(owner, name, call)

    for name, label in (
        ("attention", "V4_ATTENTION"),
        ("moe", "V4_MOE"),
        ("route", "V4_ROUTER"),
        ("routed_fp4_experts", "V4_ROUTED_EXPERTS"),
        ("sparse_attention", "V4_SPARSE_ATTENTION"),
        ("mhc_pre_fused", "V4_MHC_PRE"),
        ("mhc_post_fused", "V4_MHC_POST"),
        ("rms_norm", "V4_RMS_NORM"),
        ("rope", "V4_ROPE"),
        ("hadamard_rotate", "V4_HADAMARD"),
        ("official_head_collapse", "V4_HEAD_MHC"),
        ("activation_fp4_roundtrip", "V4_INDEX_FP4_QAT"),
        ("activation_fp8_roundtrip", "V4_KV_FP8_QAT"),
    ):
        wrap(reference, name, label)
    wrap(reference, "linear", lambda x, weights, prefix, **kw: "V4_LINEAR_" + prefix)
    wrap(
        reference,
        "compress",
        lambda *a, index=False, **kw: "V4_INDEX_COMPRESSOR" if index else "V4_MAIN_COMPRESSOR",
    )
    matmul = importlib.import_module("sgl_jax.srt.kernels.low_bit.matmul")
    for name, label in (
        ("_unpack_fp4_vmem", "V4_WEIGHT_FP4_UNPACK"),
        ("decode_fp8", "V4_WEIGHT_FP8_DECODE"),
        ("decode_e8m0", "V4_WEIGHT_SCALE_DECODE"),
        ("_expand_columns", "V4_WEIGHT_SCALE_EXPAND"),
        ("activation_fp8_roundtrip", "V4_ACTIVATION_FP8_QAT"),
    ):
        wrap(matmul, name, label)
    matmul.low_bit_matmul.clear_cache()
    reference.compiled_layer.cache_clear()
    reference.compiled_head.cache_clear()
    return originals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--correctness-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compute-trace", action="store_true")
    parser.add_argument("--unannotated-only", action="store_true")
    captures = parser.add_mutually_exclusive_group()
    captures.add_argument("--decode-only", action="store_true")
    captures.add_argument("--prefill-only", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.correctness_report.read_text())
    if (
        not fixture.get("complete")
        or fixture["source_fingerprint"] != reference.source_fingerprint()
        or Path(fixture["checkpoint"]).resolve() != args.checkpoint.resolve()
        or fixture["loaded_main_layers"] != 43
        or args.repeats < 1
    ):
        raise RuntimeError("requires a same-source complete official-model correctness report")
    if args.output.exists():
        raise FileExistsError("use a new output directory to preserve previous profiles")
    args.output.mkdir(parents=True)
    report = {
        "complete": False,
        "source_fingerprint": reference.source_fingerprint(),
        "instrumentation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint),
        "scope": "batch one, 132-token prefill and positions 132..135 decode, all 43 layers, EP4",
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "internal_names_enabled": not args.unannotated_only,
        "measurements": {},
        "compiler": {},
        "captures": [],
        "events": [],
    }
    phase = "load"
    layer_events = []
    originals = []
    saved_factories = {}

    def save(event):
        print(json.dumps(event), flush=True)
        report["events"].append(event)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def progress(event):
        if event["event"] == "layer_loaded":
            if event["layer"] % 8 == 0 or event["layer"] == 42:
                save({key: value for key, value in event.items() if key != "hbm"})
        else:
            layer_events.append(dict(phase=phase, **event))

    def compiler_factory(name):
        factory = getattr(reference, name)
        saved_factories[name] = factory

        @functools.lru_cache(maxsize=16)
        def wrapped(config, mesh):
            compiled_fn = factory(config, mesh)

            def call(*inputs):
                key = f"{name}-ratio{config.ratio}-hash{int(config.hash_routing)}-m{inputs[0].shape[0]}"
                if key not in report["compiler"]:
                    compiled = compiled_fn.lower(*inputs).compile()
                    stats = compiled.memory_analysis()
                    info = {
                        "scope": "compiler static accounting, not measured traffic",
                        "hbm_per_device": {
                            field: getattr(stats, field)
                            for field in (
                                "argument_size_in_bytes",
                                "output_size_in_bytes",
                                "temp_size_in_bytes",
                                "alias_size_in_bytes",
                            )
                        },
                        "cost_analysis": compiled.cost_analysis(),
                    }
                    report["compiler"][key] = info
                    destination = args.output / "compiler"
                    destination.mkdir(exist_ok=True)
                    (destination / f"{key}.hlo.txt").write_text(compiled.as_text())
                return compiled_fn(*inputs)

            return call

        setattr(reference, name, wrapped)

    try:
        report["before_load"] = memory_snapshot()
        model = reference.DeepSeekV4Reference(args.checkpoint, max_context=256, progress=progress)
        start = time.perf_counter()
        model.load()
        report["load_seconds"] = time.perf_counter() - start
        report["after_load"] = memory_snapshot()
        report["array_inventory"] = array_inventory(model)
        save({"event": "weights_loaded", "seconds": report["load_seconds"]})
        ids = fixture["input_ids"]
        expected_ids = fixture["generated_ids"]

        def prefill():
            model.reset()
            start = time.perf_counter()
            with jax.profiler.StepTraceAnnotation("V4_PREFILL", step_num=0):
                logits, _ = model.step(ids)
            return logits, time.perf_counter() - start

        def decode(logits, count=4):
            generated, seconds, checks = [], [], []
            for step in range(count):
                start = time.perf_counter()
                with jax.profiler.StepTraceAnnotation("V4_DECODE", step_num=step):
                    token = int(np.argmax(np.asarray(logits, np.float32)[0]))
                    generated.append(token)
                    logits, _ = model.step([token])
                seconds.append(time.perf_counter() - start)
                checks.append(np.asarray(logits, np.float32))
            if generated != expected_ids[:count] or not all(np.all(np.isfinite(x)) for x in checks):
                raise AssertionError("profile workload changed the known correctness fixture")
            return checks, seconds

        def rounds(label, repeats, expected=None):
            nonlocal phase
            phase = label
            prefill_seconds, decode_seconds = [], []
            for _ in range(repeats):
                logits, elapsed = prefill()
                values, seconds = decode(logits)
                if expected is not None:
                    for actual, oracle in zip(values, expected):
                        np.testing.assert_array_equal(actual, oracle)
                prefill_seconds.append(elapsed)
                decode_seconds.extend(seconds)
            report["measurements"][label] = {
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "decode_p50_ms": float(np.median(decode_seconds)) * 1000,
            }
            save({"event": label, **report["measurements"][label]})
            return values

        baseline = rounds("unannotated_compile_warmup", 1)
        rounds("unannotated_control", args.repeats, baseline)
        if not args.unannotated_only:
            originals = install_names()
        compiler_factory("compiled_layer")
        compiler_factory("compiled_head")
        label = "compiler_export" if args.unannotated_only else "annotated"
        rounds(label + "_compile_warmup", 1, baseline)
        rounds(label + "_control", args.repeats, baseline)
        report["after_warmup"] = memory_snapshot()
        try:
            jax.profiler.save_device_memory_profile(str(args.output / "device-memory.pprof"))
        except Exception as error:
            report["device_memory_pprof_error"] = str(error)

        def capture(label, *, mode, chips, is_prefill=False, count=4):
            nonlocal phase
            phase = label + "_prepare"
            logits, _ = prefill()
            if is_prefill:
                model.reset()
            options = jax.profiler.ProfileOptions()
            options.host_tracer_level = 2
            options.python_tracer_level = 0
            options.enable_hlo_proto = True
            options.raise_error_on_start_failure = True
            options.advanced_configuration = {
                "tpu_trace_mode": mode,
                "tpu_num_chips_to_profile_per_task": chips,
                "tpu_num_sparse_cores_to_trace": 0,
                "tpu_perf_counters": True,
            }
            directory = args.output / "traces" / label
            phase = label
            save(
                {
                    "event": "capture_start",
                    "label": label,
                    "options": options.advanced_configuration,
                }
            )
            jax.profiler.start_trace(str(directory), profiler_options=options)
            try:
                if is_prefill:
                    start = time.perf_counter()
                    with jax.profiler.StepTraceAnnotation("V4_PREFILL", step_num=0):
                        captured, _ = model.step(ids)
                    elapsed = [time.perf_counter() - start]
                    np.testing.assert_array_equal(np.asarray(captured), np.asarray(logits))
                else:
                    actual, elapsed = decode(logits, count)
                    for value, expected in zip(actual, baseline):
                        np.testing.assert_array_equal(value, expected)
            finally:
                jax.profiler.stop_trace()
            files = [
                {"path": str(p.relative_to(args.output)), "bytes": p.stat().st_size}
                for p in directory.rglob("*")
                if p.is_file()
            ]
            report["captures"].append(
                dict(label=label, mode=mode, chips=chips, seconds=elapsed, files=files)
            )
            save({"event": "capture_complete", "label": label, "seconds": elapsed, "files": files})

        if not args.prefill_only:
            capture("decode_xla", mode="TRACE_ONLY_XLA", chips=4)
        if not args.decode_only:
            capture("prefill_xla", mode="TRACE_ONLY_XLA", chips=4, is_prefill=True)
        if args.compute_trace:
            capture("decode_compute", mode="TRACE_COMPUTE_AND_SYNC", chips=1, count=1)
        rounds("post_profile_control", args.repeats, baseline)
        report["final_memory"] = memory_snapshot()
        report["complete"] = True
    finally:
        report["layer_host_timings"] = layer_events
        for name, factory in saved_factories.items():
            setattr(reference, name, factory)
        for owner, name, function in reversed(originals):
            setattr(owner, name, function)
        save({"event": "profile_finished", "complete": report["complete"]})


if __name__ == "__main__":
    main()
