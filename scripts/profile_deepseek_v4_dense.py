"""Same-machine opt-in dense-kernel profile through the real V4 ModelWorker.

Run control and candidate in separate, otherwise identical TPU processes.
Historical, independently checked logits are compatibility fixtures, NOT
same-source acceptance receipts. Every 128-token prefill boundary and every
teacher-forced decode row is checked; no diagnostic reference model is loaded.
This does not replace cold concurrent-prefill/state or Engine/HTTP acceptance.
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
    add_execution_arguments,
    assert_runner_options,
    model_overrides,
    options_from_namespace,
    options_from_receipt,
)
from profile_deepseek_v4_prefill import check_logits
from run_deepseek_v4_8k_native import (
    CAPACITY,
    CONTEXT,
    DECODE,
    PROMPT,
    prompts_for,
    restore_prefix,
    summary,
)
from run_deepseek_v4_framework import framework_fingerprint, memory_snapshot
from run_deepseek_v4_paged import PagedWorkerSession, server_args
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.deepseek_v4_reference import source_fingerprint
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from transformers import AutoTokenizer

DENSE_OPTIONS = {"fp8_backend", "fused_norm", "merged_projections", "fused_wo_a"}
BASE_OPTIONS = {
    **DEFAULTS,
    "moe_backend": "gmm",
    "attention_tp": True,
    "csa_decode_batch": True,
}


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_compatibility(
    native, oracle, *, checkpoint, native_path, reference, base_options=BASE_OPTIONS
):
    """Allow a different production source explicitly; never relabel the fixture."""
    if any(not r.get("complete") or not r.get("finished") for r in (native, oracle)):
        raise ValueError("requires complete native and independent oracle fixtures")
    if (
        native["source_fingerprint"] != reference
        or oracle["reference_source_fingerprint"] != reference
        or native["framework_source_fingerprint"]
        != oracle["framework_source_fingerprint"]
        or Path(oracle["native_report"]).resolve() != native_path.resolve()
        or any(
            Path(r["checkpoint"]).resolve() != checkpoint.resolve()
            for r in (native, oracle)
        )
        or (native["prompt_tokens"], native["generation_tokens"]) != (PROMPT, DECODE)
        or options_from_receipt(native) != base_options
        or native.get("numerical_failures")
    ):
        raise ValueError("inconsistent historical compatibility fixtures")
    checks = oracle["checks"]
    expected = {
        (case, position) for case in range(2) for position in range(PROMPT - 1, CONTEXT)
    }
    if (
        len(checks) != len(expected)
        or {(c["case"], c["position"]) for c in checks} != expected
    ):
        raise ValueError(
            "independent oracle must cover both complete 8K decode fixtures"
        )
    if any(
        not c["all_finite"] or not c["top1_equal"] or not c["nrmse"] <= 0.005
        for c in checks
    ):
        raise ValueError("independent oracle numerical check failed")


def validate_control(control, report, *, base_options=BASE_OPTIONS):
    if not control.get("complete") or not control.get("finished"):
        raise ValueError("A/B control must have completed")
    for key in (
        "framework_source_fingerprint",
        "source_fingerprint",
        "script_sha256",
        "execution_helper_sha256",
        "fixture_sha256",
        "checkpoint",
        "rounds",
        "devices",
        "jax_version",
        "prompt_tokens",
        "generation_tokens",
    ):
        if control[key] != report[key]:
            raise ValueError(f"A/B control differs in {key}")
    a, b = dict(control["server_args"]), dict(report["server_args"])
    a.pop("json_model_override_args")
    b.pop("json_model_override_args")
    if a != b or options_from_receipt(control) != base_options:
        raise ValueError("A/B control has a different serving configuration")


def capture_plan(*, decode, round_id, case=0, batch=1, index):
    if round_id != 1:
        return None
    if not decode and case == 0 and index in (56, 57):
        return (
            "prefill_case0_prefix7168",
            index == 56,
            index == 57,
            "prefill_128_tokens",
        )
    if decode and batch == 4 and index == 127:
        return ("B4_boundary8063", True, True, "r4_r128_boundary")
    if decode and index in (128, 129):
        return (f"B{batch}", index == 128, index == 129, "interior_decode")
    return None


def warm_summary(records):
    # Keep every measurement; warmup exclusion is explicit, not outlier trimming.
    result = summary([r for r in records if not r.get("warmup")])
    result["all_calls"] = len(records)
    result["warmup_calls"] = sum(bool(r.get("warmup")) for r in records)
    result["profiled_calls"] = sum(bool(r.get("profiled")) for r in records)
    result["cache_miss_calls"] = sum(bool(r["cache_misses"]) for r in records)
    return result


def main(
    *,
    base_options=BASE_OPTIONS,
    candidate_options=DENSE_OPTIONS,
    scope=__doc__,
    entry_source=None,
    require_control_bitwise=False,
    fixture_options=None,
):
    if fixture_options is None:
        fixture_options = base_options
    parser = argparse.ArgumentParser(description=scope)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--control-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    add_execution_arguments(parser)
    args = parser.parse_args()
    selected = options_from_namespace(args)
    if args.rounds < 2 or any(
        selected[k] != v for k, v in base_options.items() if k not in candidate_options
    ):
        parser.error(
            "requires >=2 rounds and the accepted FP4-GMM/head-TP4/batched-CSA base"
        )
    if selected != base_options and not args.control_report:
        parser.error("candidate must name a fresh same-source control report")
    native = json.loads(args.native_report.read_text())
    oracle = json.loads(args.oracle_report.read_text())
    source, reference = framework_fingerprint(), source_fingerprint()
    validate_compatibility(
        native,
        oracle,
        checkpoint=args.checkpoint,
        native_path=args.native_report,
        reference=reference,
        base_options=fixture_options,
    )
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("requires exclusive use of all four physical v5p chips")
    sa = server_args(args.checkpoint, CONTEXT)
    sa.max_total_tokens = CAPACITY
    sa.json_model_override_args = json.dumps(model_overrides(selected))
    fixture_files = {
        "native": args.native_report,
        "oracle": args.oracle_report,
        "prefill_logits": args.native_report.parent / "prefill_logits.npz",
        "decode_logits": args.native_report.parent / "golden_logits.npz",
    }
    report = {
        "complete": False,
        "finished": False,
        "scope": scope,
        "checkpoint": str(args.checkpoint.resolve()),
        "framework_source_fingerprint": source,
        "source_fingerprint": reference,
        "historical_fixture_production_fingerprint": native[
            "framework_source_fingerprint"
        ],
        "fixture_sha256": {k: sha256_file(p) for k, p in fixture_files.items()},
        "fixture_paths": {k: str(p.resolve()) for k, p in fixture_files.items()},
        "fixture_execution": options_from_receipt(native),
        "script_sha256": sha256_file(Path(__file__)),
        "entry_source_sha256": sha256_file(Path(entry_source or __file__)),
        "execution_helper_sha256": HELPER_SHA256,
        "server_args": dataclasses.asdict(sa),
        "rounds": args.rounds,
        "control_bitwise_required": require_control_bitwise,
        **selected,
        "jax_version": jax.__version__,
        "devices": [{"id": d.id, "kind": d.device_kind} for d in jax.devices()],
        "prompt_tokens": PROMPT,
        "generation_tokens": DECODE,
        "memory_scope": "allocator snapshots; high-water includes CPU prefix copy/restore and compiled shapes, not serving-only peak",
        "timing_scope": "worker generation, wait, full-logit host transfer, finite/argmax checks; excludes page preparation, oracle comparisons, prefix restore, compilation, three warmup calls and profiled calls",
        "runs": {},
        "checks": [],
        "captures": [],
        "events": [],
    }
    control = None
    control_checks = {}
    if args.control_report:
        control = json.loads(args.control_report.read_text())
        validate_control(control, report, base_options=base_options)
        if control.get("entry_source_sha256") != report["entry_source_sha256"]:
            raise ValueError("A/B entry source differs")
        control_checks = {c["label"]: c for c in control["checks"]}
        if len(control_checks) != len(control["checks"]):
            raise ValueError("duplicate A/B control checks")
        report["control_report"] = str(args.control_report.resolve())
        report["control_report_sha256"] = sha256_file(args.control_report)
    args.output.mkdir(parents=True, exist_ok=False)

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    def check(label, expected, actual):
        metrics = check_logits(expected, actual)
        metrics["actual_sha256"] = hashlib.sha256(
            np.ascontiguousarray(actual, dtype=np.float32).tobytes()
        ).hexdigest()
        if control is not None:
            metrics["bitwise_equal_control"] = (
                metrics["actual_sha256"] == control_checks[label]["actual_sha256"]
            )
        report["checks"].append({"label": label, **metrics})
        if not metrics["passed"] or (
            require_control_bitwise
            and control is not None
            and not metrics["bitwise_equal_control"]
        ):
            np.savez_compressed(
                args.output / "numerical_failure.npz", expected=expected, actual=actual
            )
            raise AssertionError(
                f"full-vocabulary compatibility failed: {label}: {metrics}"
            )

    active_trace = None

    def step(items, *, decode, round_id, index, case=0, batch=1):
        nonlocal active_trace
        plan = capture_plan(
            decode=decode, round_id=round_id, case=case, batch=batch, index=index
        )
        if plan and plan[1]:
            jax.profiler.start_trace(str(args.output / "traces" / plan[0]))
            active_trace = plan[0]
        with jax.profiler.TraceAnnotation(
            "V4_8K_DECODE" if decode else "V4_8K_PREFILL_CHUNK",
            batch=batch,
            position=PROMPT + index if decode else index * 128,
        ):
            logits, record = session.step(items, decode=decode, bucket=batch)
        if plan:
            capture = next(
                (c for c in report["captures"] if c["label"] == plan[0]), None
            )
            if capture is None:
                capture = {"label": plan[0], "kind": plan[3], "seconds": []}
                report["captures"].append(capture)
            capture["seconds"].append(record["seconds"])
            if plan[2]:
                jax.profiler.stop_trace()
                active_trace = None
        record.update(
            profiled=plan is not None,
            warmup=index < 3,
            round=round_id,
            position=PROMPT + index if decode else index * 128,
        )
        return logits, record

    try:
        emit({"event": "loading", "selected": selected})
        mesh = create_device_mesh([1, 4], [1, 1])
        worker = ModelWorker(sa, mesh)
        session = PagedWorkerSession(worker)
        report["actual_execution"] = assert_runner_options(
            worker.model_runner, selected
        )
        emit(
            {
                "event": "loaded",
                "memory": memory_snapshot(),
                "actual_execution": report["actual_execution"],
            }
        )
        prompts = prompts_for(
            AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        )
        if prompts != native["prompts"]:
            raise ValueError("tokenized prompts differ from compatibility fixture")
        vocab = worker.model_runner.model_config.vocab_size
        with np.load(fixture_files["prefill_logits"], allow_pickle=False) as data:
            prefill_gold = [data[f"case{i}"] for i in range(2)]
        with np.load(fixture_files["decode_logits"], allow_pickle=False) as data:
            decode_gold = [data[f"case{i}"] for i in range(2)]
        for arrays, shape in (
            (prefill_gold, (PROMPT // 128, vocab)),
            (decode_gold, (DECODE + 1, vocab)),
        ):
            if any(a.shape != shape or not np.all(np.isfinite(a)) for a in arrays):
                raise ValueError("invalid compatibility logits")
        saved = []
        for case, prompt in enumerate(prompts):
            records = []
            for round_id in range(args.rounds):
                session.new(case)
                for index, begin in enumerate(range(0, PROMPT, 128)):
                    logits, record = step(
                        [(case, prompt[begin : begin + 128])],
                        decode=False,
                        round_id=round_id,
                        index=index,
                        case=case,
                    )
                    records.append(record)
                    check(
                        f"prefill/case{case}/round{round_id}/end{begin + 128}",
                        prefill_gold[case][index],
                        logits[0],
                    )
                    if index % 16 == 0:
                        emit({"event": "prefill", "case": case, **record})
                if round_id == args.rounds - 1:
                    saved.append(
                        worker.model_runner.token_to_kv_pool.get_cpu_copy(
                            session.requests[case].locations
                        )
                    )
                session.free(case)
            report["runs"][f"prefill_case{case}"] = {
                "prefill": records,
                "prefill_summary": warm_summary(records),
            }
            emit(
                {
                    "event": "prefill_complete",
                    "case": case,
                    "summary": warm_summary(records),
                    "memory": memory_snapshot(),
                }
            )
        for batch in (1, 2, 4):
            records = []
            for round_id in range(args.rounds):
                with jax.set_mesh(mesh):
                    for key in range(batch):
                        restore_prefix(session, key, saved[key % 2])
                for index in range(DECODE):
                    order = list(range(batch))[:: -1 if index % 2 else 1]
                    logits, record = step(
                        [
                            (key, [int(np.argmax(decode_gold[key % 2][index]))])
                            for key in order
                        ],
                        decode=True,
                        round_id=round_id,
                        index=index,
                        batch=batch,
                    )
                    records.append(record)
                    for row, key in enumerate(order):
                        check(
                            f"decode/B{batch}/round{round_id}/key{key}/index{index}",
                            decode_gold[key % 2][index + 1],
                            logits[row],
                        )
                    if index % 64 == 0:
                        emit({"event": "decode", "batch": batch, **record})
                for key in range(batch):
                    session.free(key)
            report["runs"][f"B{batch}"] = {
                "decode": records,
                "decode_summary": warm_summary(records),
            }
            emit(
                {
                    "event": "decode_complete",
                    "batch": batch,
                    "summary": warm_summary(records),
                    "memory": memory_snapshot(),
                }
            )
        from deepseek_v4_kernel_evidence import export_mhc_evidence

        report["kernel_evidence"] = export_mhc_evidence(session, args.output)
        if control is not None and {c["label"] for c in report["checks"]} != set(
            control_checks
        ):
            raise ValueError("A/B checks did not cover identical workloads")
        if framework_fingerprint() != source or source_fingerprint() != reference:
            raise ValueError("source changed during profile")
        report["complete"] = report["finished"] = True
        report["all_rows_bitwise_equal_control"] = (
            all(c["bitwise_equal_control"] for c in report["checks"])
            if control is not None
            else None
        )
        emit(
            {
                "event": "profile_complete",
                "checks": len(report["checks"]),
                "all_rows_bitwise_equal_control": report[
                    "all_rows_bitwise_equal_control"
                ],
            }
        )
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise
    finally:
        if active_trace is not None:
            jax.profiler.stop_trace()


if __name__ == "__main__":
    main()
