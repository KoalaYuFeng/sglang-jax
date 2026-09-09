"""43-layer 8K golden runs, paged multi-request decode and warm TPU profiles.

Only this process owns the TPU. B2/B4 restore real B1 prefill pages (outside
timing); independent cold concurrent prefill is covered by the Engine gate.
With --cold-prefill, B2/B4 instead compute all packed prefill chunks from empty
cache and compare every 128-token boundary against the independently packed B1.
The late independent oracle receives the native prefix cache, so it checks a
full long-context decode, not an independently recomputed 8K prefill.
With --profile-reused-b1, also measure B1 from an accepted reconstructed prefix
without retaining a diagnostic reference model in the timing process.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from deepseek_v4_execution_options import (
    HELPER_SHA256,
    add_execution_arguments,
    assert_runner_options,
    model_overrides,
    options_from_namespace,
    options_from_receipt,
)
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from run_deepseek_v4_framework import (
    compare_arrays,
    framework_fingerprint,
    memory_snapshot,
)
from run_deepseek_v4_paged import PagedWorkerSession, reference_layers, server_args
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    DeepSeekV4Reference,
    source_fingerprint,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from transformers import AutoTokenizer

CONTEXT = 8192
PROMPT = 7936
DECODE = CONTEXT - PROMPT
CAPACITY = CONTEXT * 4 + 512  # admission/page-rounding headroom, not extra context


def prompts_for(tokenizer):
    texts = (
        "Geography notebook A: The capital of France is Paris. Rivers, mountains and cities. ",
        "Arithmetic notebook B: One plus one equals two. Count integers and compare numbers. ",
    )
    result = []
    for text in texts:
        seed = tokenizer.encode(text, add_special_tokens=False)
        if not seed:
            raise ValueError("empty tokenizer fixture")
        result.append([int(x) for x in (seed * (PROMPT // len(seed) + 1))[:PROMPT]])
    if result[0][:128] == result[1][:128]:
        raise AssertionError("pressure fixtures must not share a complete prefix page")
    return result


def summary(records):
    warm = [r for r in records if not r["cache_misses"] and not r.get("profiled")]
    ms = np.asarray([r["seconds"] * 1000 for r in warm])
    return {
        "calls": len(records),
        "cold_or_cache_miss_seconds": sum(
            r["seconds"] for r in records if r["cache_misses"]
        ),
        "warm_calls": len(warm),
        "warm_p50_ms": float(np.median(ms)) if len(ms) else None,
        "warm_p95_ms": float(np.percentile(ms, 95)) if len(ms) else None,
        "warm_tokens_per_second": (
            sum(r["tokens"] for r in warm) / sum(r["seconds"] for r in warm)
            if warm
            else None
        ),
    }


def measured_decode_batches(*, profile, reuse_goldens, cold_prefill, profile_reused_b1):
    if profile_reused_b1:
        if not profile or not reuse_goldens or cold_prefill:
            raise ValueError(
                "--profile-reused-b1 requires --profile and --reuse-goldens, "
                "without --cold-prefill"
            )
        return (1, 2, 4)
    return (2, 4)


def profile_schedule(index, label, *, enabled, boundary=False):
    if not enabled:
        return None
    if boundary and index == 127:
        return {
            "label": label + "_boundary8063",
            "start": True,
            "stop": True,
            "kind": "r4_r128_boundary",
        }
    if index in (128, 129):
        return {
            "label": label,
            "start": index == 128,
            "stop": index == 129,
            "kind": "interior_decode",
        }
    return None


def restore_prefix(session, key, saved):
    session.new(key)
    request = session.requests[key]
    locs = session.runner.token_to_kv_pool_allocator.alloc_extend(
        np.asarray([0], np.int32), np.asarray([PROMPT], np.int32), [-1], PROMPT
    )
    if locs is None:
        raise RuntimeError("fixture prefix allocation exhausted the pool")
    request.locations = [int(x) for x in locs]
    session.runner.req_to_token_pool.write(
        (request.req_pool_idx, slice(0, PROMPT)), locs
    )
    session.runner.token_to_kv_pool.load_cpu_copy(saved, locs)


def late_reference(session, key, checkpoint, token):
    """Read-only physical-to-logical prefix conversion for the independent oracle."""
    runner, request = session.runner, session.requests[key]
    ref = DeepSeekV4Reference(checkpoint, max_context=CONTEXT, progress=lambda _: None)
    ref.mesh = runner.mesh
    ref.layers = reference_layers(runner)
    ref.shared = {
        **runner.model.shared.get_value(),
        "embed.weight": runner.model.embed_tokens.embedding.get_value(),
        "head.weight": jax.reshard(
            runner.model.lm_head.embedding.get_value(), NamedSharding(runner.mesh, P())
        ),
    }
    ref.position = len(request.locations)
    physical = jnp.asarray(
        request.locations + [0] * (CONTEXT - ref.position), jnp.int32
    )
    for config, layer in zip(ref.configs, runner.token_to_kv_pool.layers, strict=True):
        logical = {"window": layer["window"][physical]}
        if config.ratio:
            terminal = jnp.arange(config.ratio - 1, CONTEXT, config.ratio)
            for prefix in ("main", "index"):
                if prefix + ".compressed" not in layer:
                    continue
                values = layer[prefix + ".compressed"][
                    physical[terminal] // config.ratio
                ]
                logical[prefix + ".compressed"] = jnp.where(
                    (terminal < ref.position)[:, None], values, 0
                )
                for suffix in ("kv", "score"):
                    logical[prefix + "." + suffix] = layer[prefix + "." + suffix][
                        request.req_pool_idx + 1
                    ]
        ref.cache.append(logical)
    logits, _ = ref.step([token])
    result = np.asarray(logits, np.float32)
    del ref
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-boundary",
        action="store_true",
        help="with --profile, separately capture position 8063; do not average it into interior traces",
    )
    add_execution_arguments(parser)
    parser.add_argument("--check-kernel-dispatch", action="store_true")
    parser.add_argument("--reuse-goldens", type=Path)
    parser.add_argument(
        "--profile-reused-b1",
        action="store_true",
        help="also profile B1 from the accepted prefix without loading a diagnostic oracle",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="also require compatibility with an immutable earlier B1 full-logit baseline",
    )
    parser.add_argument("--cold-prefill", action="store_true")
    parser.add_argument("--continue-on-numerical-failure", action="store_true")
    parser.add_argument("--capture-failure-state", action="store_true")
    options = parser.parse_args()
    selected = options_from_namespace(options)
    if options.profile_boundary and not options.profile:
        parser.error("--profile-boundary requires --profile")
    try:
        decode_batches = measured_decode_batches(
            profile=options.profile,
            reuse_goldens=options.reuse_goldens,
            cold_prefill=options.cold_prefill,
            profile_reused_b1=options.profile_reused_b1,
        )
    except ValueError as error:
        parser.error(str(error))
    options.output.mkdir(parents=True, exist_ok=False)
    sa = server_args(options.checkpoint, CONTEXT)
    sa.json_model_override_args = json.dumps(model_overrides(selected))
    sa.max_total_tokens = CAPACITY
    report = {
        "complete": False,
        "checkpoint": str(options.checkpoint),
        "source_fingerprint": source_fingerprint(),
        "framework_source_fingerprint": framework_fingerprint(),
        **selected,
        "execution_helper_sha256": HELPER_SHA256,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "server_args": dataclasses.asdict(sa),
        "prompt_tokens": PROMPT,
        "generation_tokens": DECODE,
        "cold_concurrent_prefill": options.cold_prefill,
        "profile_boundary": options.profile_boundary,
        "profile_reused_b1": options.profile_reused_b1,
        "scope": __doc__,
        "events": [],
        "checks": [],
        "runs": {},
        "captures": [],
        "numerical_failures": [],
        "finished": False,
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (options.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    def check(label, expected, actual):
        metrics = compare_arrays(expected, actual)
        metrics["all_finite"] = bool(
            np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))
        )
        metrics["top1_equal"] = bool(
            np.array_equal(np.argmax(expected, axis=-1), np.argmax(actual, axis=-1))
        )
        report["checks"].append({"label": label, **metrics})
        if (
            not metrics["all_finite"]
            or not metrics["top1_equal"]
            or metrics["nrmse"] > 0.005
        ):
            failure = {"label": label, **metrics}
            report["numerical_failures"].append(failure)
            number = len(report["numerical_failures"])
            if number <= 4:
                np.savez_compressed(
                    options.output / f"failure-{number:04d}.npz",
                    expected=expected,
                    actual=actual,
                )
            if number <= 4 or number % 64 == 0:
                emit({"event": "numerical_failure", **failure, "failure_count": number})
            if options.capture_failure_state and number == 1:
                # Copy only AFTER the unchanged production call and numerical
                # comparison. No trace outputs/callbacks alter ModelRunner JIT.
                from debug_deepseek_v4_8023 import save_arrays

                target = options.output / "failure-state"
                target.mkdir()
                batch = session.last_batch
                metadata = worker.model_runner.attn_backend.get_forward_metadata(batch)
                save_arrays(
                    target / "metadata", dataclasses.asdict(metadata), compressed=False
                )
                save_arrays(
                    target / "inputs",
                    {
                        "ids": batch.input_ids,
                        "positions": batch.positions,
                        "locations": batch.out_cache_loc,
                    },
                    compressed=False,
                )
                for layer, cache in enumerate(
                    worker.model_runner.token_to_kv_pool.layers
                ):
                    save_arrays(target / f"layer-{layer:02d}", cache, compressed=False)
                emit(
                    {
                        "event": "production_failure_state_saved",
                        "directory": str(target),
                        "layers": len(worker.model_runner.token_to_kv_pool.layers),
                    }
                )
            if not options.continue_on_numerical_failure:
                raise AssertionError(f"8K full-logit comparison failed: {label}")

    def step(session, items, *, bucket, index, label):
        capture_plan = profile_schedule(
            index, label, enabled=options.profile, boundary=options.profile_boundary
        )
        profile = capture_plan is not None
        if profile and capture_plan["start"]:
            jax.profiler.start_trace(
                str(options.output / "traces" / capture_plan["label"])
            )
        try:
            with jax.profiler.TraceAnnotation(
                "V4_8K_DECODE", batch=bucket, position=PROMPT + index
            ):
                logits, record = session.step(items, decode=True, bucket=bucket)
        finally:
            if profile and capture_plan["stop"]:
                jax.profiler.stop_trace()
        record.update(position=PROMPT + index, profiled=profile)
        if profile:
            label = capture_plan["label"]
            capture = next((c for c in report["captures"] if c["label"] == label), None)
            if capture is None:
                capture = {"label": label, "kind": capture_plan["kind"], "seconds": []}
                report["captures"].append(capture)
            capture["seconds"].append(record["seconds"])
        return logits, record

    try:
        mesh = create_device_mesh([1, 4], [1, 1])
        emit({"event": "loading"})
        worker = ModelWorker(sa, mesh)
        report["actual_execution"] = assert_runner_options(
            worker.model_runner, selected
        )
        session = PagedWorkerSession(worker)
        emit({"event": "loaded", "memory": memory_snapshot()})
        tokenizer = AutoTokenizer.from_pretrained(
            options.checkpoint, local_files_only=True
        )
        prompts = prompts_for(tokenizer)
        report["prompts"] = prompts
        baseline = None
        if options.baseline_report:
            original = json.loads(options.baseline_report.read_text())
            if (
                original["prompts"] != prompts
                or Path(original["checkpoint"]).resolve()
                != options.checkpoint.resolve()
            ):
                raise ValueError(
                    "compatibility baseline must use the same checkpoint and prompts"
                )
            with np.load(
                options.baseline_report.parent / "golden_logits.npz"
            ) as arrays:
                baseline = [arrays[f"case{i}"] for i in range(2)]
            if any(
                b.shape != (DECODE + 1, worker.model_runner.model_config.vocab_size)
                for b in baseline
            ):
                raise ValueError("invalid compatibility baseline dimensions")
            report["baseline_report"] = str(options.baseline_report)
            report["baseline_framework_source_fingerprint"] = original[
                "framework_source_fingerprint"
            ]
        golden, saved, prefill_goldens = [], [], []
        reused_prefill = False
        if options.reuse_goldens:
            previous = json.loads(options.reuse_goldens.read_text())
            oracle_checks = [
                c
                for c in previous["checks"]
                if "independent_decode_at_8191" in c["label"]
            ]
            if (
                previous["framework_source_fingerprint"] != framework_fingerprint()
                or options_from_receipt(previous) != selected
                or previous["prompts"] != prompts
                or len(oracle_checks) != 2
                or any(not c["top1_equal"] or c["nrmse"] > 0.005 for c in oracle_checks)
            ):
                raise ValueError(
                    "only same-source, independently checked B1 goldens can be reused"
                )
            with np.load(options.reuse_goldens.parent / "golden_logits.npz") as arrays:
                golden = [arrays[f"case{i}"] for i in range(2)]
            if any(
                g.shape != (DECODE + 1, worker.model_runner.model_config.vocab_size)
                for g in golden
            ):
                raise ValueError("invalid full-logit golden dimensions")
            report["reused_golden_report"] = str(options.reuse_goldens)
            report["reused_oracle_checks"] = oracle_checks
            if options.cold_prefill:
                prefill_path = options.reuse_goldens.parent / "prefill_logits.npz"
                with np.load(prefill_path, allow_pickle=False) as arrays:
                    prefill_goldens = [arrays[f"case{i}"] for i in range(2)]
                if any(
                    g.shape
                    != (PROMPT // 128, worker.model_runner.model_config.vocab_size)
                    or not np.all(np.isfinite(g))
                    for g in prefill_goldens
                ):
                    raise ValueError(
                        "invalid same-source prefill full-logit dimensions/values"
                    )
                reused_prefill = True
                report["reused_prefill_logits"] = str(prefill_path)
        for case, prompt in enumerate(prompts):
            if reused_prefill:
                emit({"event": "same_source_prefill_golden_reused", "case": case})
                continue
            session.new(case)
            prefill_records, decode_records, rows, prefill_rows = [], [], [], []
            for begin in range(0, PROMPT, 128):
                logits, record = session.step(
                    [(case, prompt[begin : begin + 128])], bucket=1
                )
                prefill_rows.append(logits[0])
                prefill_records.append({"position": begin, **record})
                if begin % 1024 == 0:
                    emit(
                        {
                            "event": "golden_prefill",
                            "case": case,
                            "position": begin,
                            **record,
                        }
                    )
            rows.append(logits[0])
            prefill_goldens.append(np.stack(prefill_rows))
            if baseline is not None:
                check(f"B1/{case}/original_prefill", baseline[case][0], logits[0])
            saved.append(
                worker.model_runner.token_to_kv_pool.get_cpu_copy(
                    session.requests[case].locations
                )
            )
            emit(
                {"event": "prefill_complete", "case": case, "memory": memory_snapshot()}
            )
            if options.reuse_goldens:
                check(f"reconstructed_prefill/{case}", golden[case][0], logits[0])
                report["runs"][f"reconstructed_prefix{case}"] = {
                    "prefill": prefill_records,
                    "prefill_summary": summary(prefill_records),
                }
                session.free(case)
                emit({"event": "golden_prefix_reconstructed", "case": case})
                continue
            for index in range(DECODE):
                token = int(np.argmax(rows[-1]))
                if index == DECODE - 1:
                    with jax.set_mesh(mesh):
                        expected = late_reference(
                            session, case, options.checkpoint, token
                        )
                    emit(
                        {
                            "event": "late_oracle_ready",
                            "case": case,
                            "position": CONTEXT - 1,
                        }
                    )
                logits, record = step(
                    session,
                    [(case, [token])],
                    bucket=1,
                    index=index,
                    label=f"B1_case{case}",
                )
                if index == DECODE - 1:
                    check(f"B1/{case}/independent_decode_at_8191", expected, logits)
                if baseline is not None:
                    check(
                        f"B1/{case}/original_decode{index}",
                        baseline[case][index + 1],
                        logits[0],
                    )
                rows.append(logits[0])
                decode_records.append(record)
                if index % 64 == 0:
                    emit(
                        {
                            "event": "golden_decode",
                            "case": case,
                            "index": index,
                            **record,
                        }
                    )
            golden.append(np.stack(rows))
            report["runs"][f"B1_case{case}"] = {
                "prefill": prefill_records,
                "decode": decode_records,
                "prefill_summary": summary(prefill_records),
                "decode_summary": summary(decode_records),
            }
            session.free(case)
            emit({"event": "golden_complete", "case": case})
        report["expected_output_ids"] = [
            np.argmax(g, axis=-1).tolist()[:DECODE] for g in golden
        ]
        np.savez_compressed(
            options.output / "golden_logits.npz", case0=golden[0], case1=golden[1]
        )
        np.savez_compressed(
            options.output / "prefill_logits.npz",
            case0=prefill_goldens[0],
            case1=prefill_goldens[1],
        )
        for batch in decode_batches:
            with jax.set_mesh(mesh):
                for key in range(batch):
                    if options.cold_prefill:
                        session.new(key)
                    else:
                        restore_prefix(session, key, saved[key % 2])
            cold_records = []
            if options.cold_prefill:
                chunk = (
                    128 // batch
                )  # ordinary packed-token capacity, not a larger bucket
                for begin in range(0, PROMPT, chunk):
                    end = begin + chunk
                    order = list(range(batch))[:: -1 if (begin // chunk) % 2 else 1]
                    logits, record = session.step(
                        [(key, prompts[key % 2][begin:end]) for key in order],
                        bucket=batch,
                    )
                    cold_records.append(record)
                    if end % 128 == 0:
                        for row, key in enumerate(order):
                            check(
                                f"B{batch}/{key}/cold_prefill{end}",
                                prefill_goldens[key % 2][end // 128 - 1],
                                logits[row],
                            )
                    if end % 1024 == 0 or end == PROMPT:
                        emit(
                            {
                                "event": "cold_batched_prefill",
                                "batch": batch,
                                "length": end,
                            }
                        )
            records = []
            for index in range(DECODE):
                order = list(range(batch))[:: -1 if index % 2 else 1]
                logits, record = step(
                    session,
                    [(key, [int(np.argmax(golden[key % 2][index]))]) for key in order],
                    bucket=batch,
                    index=index,
                    label=f"B{batch}",
                )
                for row, key in enumerate(order):
                    check(
                        f"B{batch}/{key}/decode{index}",
                        golden[key % 2][index + 1],
                        logits[row],
                    )
                records.append(record)
                if index % 64 == 0:
                    emit(
                        {
                            "event": "batched_decode",
                            "batch": batch,
                            "index": index,
                            **record,
                        }
                    )
            report["runs"][f"B{batch}"] = {
                "decode": records,
                "decode_summary": summary(records),
            }
            if options.cold_prefill:
                report["runs"][f"B{batch}"]["prefill"] = cold_records
            for key in range(batch):
                session.free(key)
            emit(
                {"event": "batch_complete", "batch": batch, "memory": memory_snapshot()}
            )
        if options.check_kernel_dispatch:
            from deepseek_v4_kernel_evidence import export_mhc_evidence

            report["kernel_evidence"] = export_mhc_evidence(session, options.output)
            emit({"event": "kernel_dispatch_verified", **report["kernel_evidence"]})
        report["finished"] = True
        report["complete"] = not report["numerical_failures"]
        emit(
            {
                "event": "native_8k_measurement_finished",
                "acceptance_passed": report["complete"],
                "numerical_failures": len(report["numerical_failures"]),
            }
        )
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
