"""Same-source B1 prefill chunk experiment, not a scheduler/default change.

Each chunk independently passes two full 7936-token prompts and 256 decode
steps before profiling. Compare every available 128-token golden boundary,
all page-owned KV/snapshots and live compressor state at the prompt end.
The 128-token run is the fresh state baseline. Children are separate complete
receipts; a later failed chunk never relabels an earlier receipt or the parent.
"""

import argparse
import dataclasses
import hashlib
import json
import time
import traceback
from pathlib import Path

import jax
import numpy as np
from deepseek_v4_execution_options import (
    DEFAULTS,
    HELPER_SHA256,
    assert_runner_options,
    model_overrides,
    options_from_receipt,
)
from run_deepseek_v4_8k_native import (
    CAPACITY,
    CONTEXT,
    DECODE,
    PROMPT,
    prompts_for,
    summary,
)
from run_deepseek_v4_framework import (
    compare_arrays,
    framework_fingerprint,
    memory_snapshot,
)
from run_deepseek_v4_paged import PagedWorkerSession, server_args
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.deepseek_v4_reference import source_fingerprint
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from transformers import AutoTokenizer

CHUNKS = (128, 256, 512)
TRACE_PREFIXES = (1024, 7168)


def chunk_plan(chunk, length=PROMPT):
    if chunk not in CHUNKS or length <= 0 or length % 128:
        raise ValueError("requires a supported chunk and positive page-aligned prompt")
    return [(begin, min(begin + chunk, length)) for begin in range(0, length, chunk)]


def validate_fixture(native, oracle, *, checkpoint, native_path, source, reference):
    if not native.get("complete") or not native.get("finished"):
        raise ValueError("requires a complete native fixture")
    if not oracle.get("complete") or not oracle.get("finished"):
        raise ValueError("requires a complete independent oracle fixture")
    if any(r["framework_source_fingerprint"] != source for r in (native, oracle)):
        raise ValueError("fixture production source changed")
    if (
        native["source_fingerprint"] != reference
        or oracle["reference_source_fingerprint"] != reference
    ):
        raise ValueError("fixture independent reference changed")
    if any(
        Path(r["checkpoint"]).resolve() != checkpoint.resolve()
        for r in (native, oracle)
    ):
        raise ValueError("fixture checkpoint changed")
    if Path(oracle["native_report"]).resolve() != native_path.resolve():
        raise ValueError("independent oracle checked a different native fixture")
    if (native["prompt_tokens"], native["generation_tokens"]) != (PROMPT, DECODE):
        raise ValueError("fixture must cover 7936 + 256 tokens")
    return options_from_receipt(native)


def logical_state(session, key):
    """CPU-only observations, outside all timing; exclude unowned/padded pages."""
    request = session.requests[key]
    pool = session.runner.token_to_kv_pool
    saved = pool.get_cpu_copy(request.locations)
    result = {}
    for layer_id, (pages, layer) in enumerate(
        zip(saved["layers"], pool.layers, strict=True)
    ):
        for name, value in pages.items():
            result[f"layer{layer_id:02d}/{name}"] = np.ascontiguousarray(value)
        for name, value in layer.items():
            if name.endswith((".kv", ".score")):
                result[f"layer{layer_id:02d}/live/{name}"] = np.ascontiguousarray(
                    jax.device_get(value[request.req_pool_idx + 1])
                )
    return result


def state_record(label, value):
    return {
        "label": label,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(
            memoryview(np.ascontiguousarray(value).view(np.uint8)).cast("B")
        ).hexdigest(),
    }


def same_state(expected, actual):
    a, b = state_record("", expected), state_record("", actual)
    return a == b


def check_logits(expected, actual):
    metrics = compare_arrays(expected, actual)
    metrics["all_finite"] = bool(
        np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))
    )
    metrics["top1_equal"] = bool(
        np.array_equal(np.argmax(expected, axis=-1), np.argmax(actual, axis=-1))
    )
    metrics["passed"] = (
        metrics["all_finite"] and metrics["top1_equal"] and metrics["nrmse"] <= 0.005
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--chunks", nargs="+", type=int, default=list(CHUNKS), choices=CHUNKS
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        args.chunks[0] != 128
        or len(set(args.chunks)) != len(args.chunks)
        or args.repeats < 3
    ):
        parser.error(
            "start with chunk 128, no duplicates, and at least three warm rounds"
        )
    native = json.loads(args.native_report.read_text())
    oracle = json.loads(args.oracle_report.read_text())
    source, reference = framework_fingerprint(), source_fingerprint()
    selected = validate_fixture(
        native,
        oracle,
        checkpoint=args.checkpoint,
        native_path=args.native_report,
        source=source,
        reference=reference,
    )
    if selected != {
        **DEFAULTS,
        "mhc_backend": "pallas",
        "hca_backend": "pallas",
        "csa_backend": "pallas",
        "moe_backend": "gmm",
        "attention_tp": True,
        "csa_decode_batch": True,
    }:
        parser.error("requires the accepted combined GMM/TP4/batched-CSA path")
    args.output.mkdir(parents=True, exist_ok=False)
    sa = server_args(args.checkpoint, CONTEXT)
    sa.max_total_tokens = CAPACITY
    sa.max_prefill_tokens = sa.chunked_prefill_size = max(args.chunks)
    sa.precompile_token_paddings = sorted(args.chunks)
    sa.json_model_override_args = json.dumps(model_overrides(selected))
    common = {
        "scope": __doc__,
        "checkpoint": str(args.checkpoint.resolve()),
        "framework_source_fingerprint": source,
        "source_fingerprint": reference,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "session_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("run_deepseek_v4_paged.py").read_bytes()
        ).hexdigest(),
        "execution_helper_sha256": HELPER_SHA256,
        "server_args": dataclasses.asdict(sa),
        "native_report": str(args.native_report.resolve()),
        "native_report_sha256": hashlib.sha256(
            args.native_report.read_bytes()
        ).hexdigest(),
        "oracle_report": str(args.oracle_report.resolve()),
        "oracle_report_sha256": hashlib.sha256(
            args.oracle_report.read_bytes()
        ).hexdigest(),
        "prompt_tokens": PROMPT,
        "generation_tokens": DECODE,
        **selected,
    }
    parent = {
        **common,
        "complete": False,
        "finished": False,
        "children": [],
        "events": [],
    }

    def emit(report, folder, event):
        report["events"].append({"time": time.time(), **event})
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"folder": folder.name, **event}), flush=True)

    current, folder = parent, args.output
    try:
        emit(parent, args.output, {"event": "loading"})
        mesh = create_device_mesh([1, 4], [1, 1])
        worker = ModelWorker(sa, mesh)
        common["actual_execution"] = assert_runner_options(
            worker.model_runner, selected
        )
        session = PagedWorkerSession(worker)
        emit(parent, args.output, {"event": "loaded", "memory": memory_snapshot()})
        prompts = prompts_for(
            AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        )
        if prompts != native["prompts"]:
            raise ValueError("tokenized prompts differ from accepted fixtures")
        with np.load(
            args.native_report.parent / "prefill_logits.npz", allow_pickle=False
        ) as data:
            prefill_goldens = [data[f"case{i}"] for i in range(2)]
        with np.load(
            args.native_report.parent / "golden_logits.npz", allow_pickle=False
        ) as data:
            decode_goldens = [data[f"case{i}"] for i in range(2)]
        vocab = worker.model_runner.model_config.vocab_size
        if any(
            x.shape != (PROMPT // 128, vocab) or not np.all(np.isfinite(x))
            for x in prefill_goldens
        ):
            raise ValueError("invalid full-vocabulary prefill goldens")
        if any(
            x.shape != (DECODE + 1, vocab) or not np.all(np.isfinite(x))
            for x in decode_goldens
        ):
            raise ValueError("invalid full-vocabulary decode goldens")
        state_baselines = []
        for chunk in args.chunks:
            folder = args.output / f"chunk{chunk}"
            folder.mkdir()
            current = {
                **common,
                "chunk": chunk,
                "complete": False,
                "finished": False,
                "correctness_complete": False,
                "checks": [],
                "state_checks": [],
                "runs": {},
                "captures": [],
                "events": [],
                "memory_scope": "allocator high-water includes diagnostic KV/state readback, not serving-only peak",
            }
            emit(current, folder, {"event": "chunk_start", "chunk": chunk})
            run_chunk(
                chunk,
                args.repeats,
                session,
                prompts,
                prefill_goldens,
                decode_goldens,
                state_baselines,
                current,
                folder,
                emit,
            )
            if framework_fingerprint() != source or source_fingerprint() != reference:
                raise ValueError(
                    "production/reference source changed during experiment"
                )
            current["complete"] = current["finished"] = True
            emit(
                current,
                folder,
                {
                    "event": "chunk_complete",
                    "chunk": chunk,
                    "prefill_summary": current["prefill_summary"],
                },
            )
            parent["children"].append(
                {
                    "chunk": chunk,
                    "report": str(folder / "report.json"),
                    "report_sha256": hashlib.sha256(
                        (folder / "report.json").read_bytes()
                    ).hexdigest(),
                }
            )
            emit(parent, args.output, {"event": "accepted_chunk", "chunk": chunk})
        parent["complete"] = parent["finished"] = True
        emit(parent, args.output, {"event": "experiment_complete"})
    except BaseException:
        current["error"] = traceback.format_exc()
        emit(current, folder, {"event": "failed", "error": current["error"]})
        if current is not parent:
            parent["error"] = current["error"]
            emit(
                parent,
                args.output,
                {"event": "failed_chunk", "chunk": current["chunk"]},
            )
        raise


def export_prefill_hlo(session, folder):
    from deepseek_v4_moe_evidence import compiled_moe_evidence
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

    runner, batch = session.runner, session.last_batch
    if batch.forward_mode.is_decode():
        raise ValueError("prefill evidence requires an executed EXTEND batch")
    forward = ForwardBatch.init_new(batch, runner)
    runner.attn_backend.forward_metadata = runner.attn_backend.get_forward_metadata(
        batch
    )
    with jax.set_mesh(runner.mesh):
        compiled = runner._jitted_run_model.lower(
            runner._model_def,
            runner._model_state_def,
            runner.model_state_leaves,
            forward,
            runner.memory_pools,
            LogitsMetadata.from_model_worker_batch(batch, runner.mesh),
        ).compile()
    hlo = compiled.as_text()
    (folder / "prefill_optimized_hlo.txt").write_text(hlo)
    return {
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "tokens": len(batch.input_ids),
        "real_tokens": int(batch.real_input_ids_len),
        "moe_evidence": compiled_moe_evidence(hlo, layers=len(runner.model.configs)),
        "scope": "static GMM ownership for the executed EXTEND shape; dynamic timing is in XPlane",
    }


def run_chunk(
    chunk,
    repeats,
    session,
    prompts,
    prefill_goldens,
    decode_goldens,
    state_baselines,
    report,
    folder,
    emit,
):
    def check(label, expected, actual):
        metrics = check_logits(expected, actual)
        report["checks"].append({"label": label, **metrics})
        if not metrics["passed"]:
            np.savez_compressed(
                folder / "first-logit-failure.npz", expected=expected, actual=actual
            )
            raise AssertionError(f"full-logit comparison failed: {label}: {metrics}")

    def prefill(case, label, *, tracing=False):
        session.new(0)
        records = []
        for begin, end in chunk_plan(chunk):
            capture = tracing and begin in TRACE_PREFIXES
            capture_label = f"prefill{chunk}_prefix{begin}"
            if capture:
                jax.profiler.start_trace(str(folder / "traces" / capture_label))
            try:
                with jax.profiler.TraceAnnotation(
                    "V4_8K_PREFILL_CHUNK", chunk=chunk, prefix=begin
                ):
                    logits, record = session.step(
                        [(0, prompts[case][begin:end])],
                        bucket=1,
                        token_bucket=chunk,
                    )
            finally:
                if capture:
                    jax.profiler.stop_trace()
            records.append(
                {
                    **record,
                    "begin": begin,
                    "end": end,
                    "token_bucket": chunk,
                    "profiled": capture,
                }
            )
            check(
                f"{label}/prefill{end}",
                prefill_goldens[case][end // 128 - 1],
                logits[0],
            )
            if capture:
                report["captures"].append(
                    {
                        "label": capture_label,
                        "kind": "prefill",
                        "prefix": begin,
                        "tokens": end - begin,
                        "seconds": [record["seconds"]],
                    }
                )
            if begin % 1024 == 0:
                emit(
                    report,
                    folder,
                    {
                        "event": "prefill_progress",
                        "run": label,
                        "end": end,
                        **records[-1],
                    },
                )
        report["runs"][label] = {
            "prefill": records,
            "prefill_summary": summary(records),
        }
        return records

    for case in range(2):
        label = f"correctness_case{case}"
        prefill(case, label)
        observed = logical_state(session, 0)
        if chunk == 128:
            state_baselines.append(observed)
        baseline = state_baselines[case]
        if set(baseline) != set(observed):
            raise AssertionError("cache observation fields changed")
        for key, value in observed.items():
            actual = state_record(key, value)
            expected = state_record(key, baseline[key])
            passed = actual == expected
            report["state_checks"].append(
                {
                    "case": case,
                    **actual,
                    "expected_sha256": expected["sha256"],
                    "bitwise_equal": passed,
                }
            )
            if not passed:
                np.savez_compressed(
                    folder / "first-state-failure.npz",
                    expected=baseline[key].astype(np.float32),
                    actual=value.astype(np.float32),
                    expected_bytes=baseline[key].view(np.uint8),
                    actual_bytes=value.view(np.uint8),
                )
                raise AssertionError(
                    f"chunk={chunk}, case={case}: state differs at {key}"
                )
        emit(
            report,
            folder,
            {
                "event": "prefill_state_passed",
                "case": case,
                "arrays": len(observed),
                "baseline_is_self": chunk == 128,
            },
        )
        decode_records = []
        for index in range(DECODE):
            token = int(np.argmax(decode_goldens[case][index]))
            logits, record = session.step([(0, [token])], decode=True, bucket=1)
            check(
                f"{label}/decode{PROMPT + index}",
                decode_goldens[case][index + 1],
                logits[0],
            )
            decode_records.append({**record, "position": PROMPT + index})
            if index % 64 == 0:
                emit(
                    report,
                    folder,
                    {
                        "event": "decode_progress",
                        "case": case,
                        "position": PROMPT + index,
                        **record,
                    },
                )
        report["runs"][label]["decode"] = decode_records
        session.free(0)
        emit(report, folder, {"event": "correctness_case_passed", "case": case})
    report["correctness_complete"] = True
    emit(report, folder, {"event": "correctness_passed", "chunk": chunk})
    warm_records = []
    for repeat in range(repeats):
        label = f"warm_round{repeat}_case{repeat % 2}"
        records = prefill(repeat % 2, label, tracing=repeat == repeats - 1)
        if any(r["cache_misses"] for r in records):
            raise AssertionError(
                "measurement phase unexpectedly compiled a new variant"
            )
        warm_records.extend(records)
        if repeat == repeats - 1:
            report["prefill_kernel_evidence"] = export_prefill_hlo(session, folder)
        session.free(0)
        emit(
            report,
            folder,
            {
                "event": "warm_round_complete",
                "repeat": repeat,
                "prefill_summary": summary(records),
            },
        )
    report["prefill_summary"] = summary(warm_records)
    report["final_memory"] = memory_snapshot()


if __name__ == "__main__":
    main()
