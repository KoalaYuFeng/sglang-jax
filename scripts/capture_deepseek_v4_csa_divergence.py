"""Uninstrumented whole-ModelRunner A/B capture around a CSA numerical failure.

One read-only checkpoint allocation is reused. Each explicit backend starts
from a fresh request/cache history, using the immutable baseline's teacher
tokens. Only host-side captures surround ordinary production invocations.
"""

import argparse
import dataclasses
import hashlib
import json
import re
import time
import traceback
from pathlib import Path

import jax
import numpy as np

from debug_deepseek_v4_8023 import save_arrays
from deepseek_v4_kernel_evidence import export_mhc_evidence
from run_deepseek_v4_8k_native import CAPACITY, CONTEXT, PROMPT
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from run_deepseek_v4_paged import ModelWorker, PagedWorkerSession, server_args
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-begin", type=int)
    args = parser.parse_args()
    candidate = json.loads(args.candidate_report.read_text())
    baseline = json.loads(args.baseline_report.read_text())
    fingerprint = framework_fingerprint()
    if candidate["framework_source_fingerprint"] != fingerprint:
        raise ValueError("reproduction requires the unchanged failing source")
    if (
        candidate["checkpoint"] != baseline["checkpoint"]
        or candidate["prompts"] != baseline["prompts"]
    ):
        raise ValueError("baseline checkpoint and prompts must match the failed run")
    failure = candidate["numerical_failures"][0]
    match = re.fullmatch(r"B1/(\d+)/original_decode(\d+)", failure["label"])
    if match is None:
        raise ValueError("this capture requires a B1 compatibility failure")
    case, index = map(int, match.groups())
    position = PROMPT + index
    capture_begin = position - 1 if args.capture_begin is None else args.capture_begin
    if not PROMPT <= capture_begin < position:
        raise ValueError("capture begin must precede the failure within its decode history")
    with np.load(args.baseline_report.parent / "golden_logits.npz", allow_pickle=False) as arrays:
        golden = arrays[f"case{case}"]
    with np.load(args.candidate_report.parent / "failure-0001.npz", allow_pickle=False) as arrays:
        failed_logits = arrays["actual"]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "scope": __doc__,
        "checkpoint": candidate["checkpoint"],
        "framework_source_fingerprint": fingerprint,
        "baseline_framework_source_fingerprint": baseline["framework_source_fingerprint"],
        "candidate_report": str(args.candidate_report),
        "baseline_report": str(args.baseline_report),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case": case,
        "position": position,
        "capture_begin": capture_begin,
        "events": [],
        "checks": [],
        "frames": [],
        "kernel_evidence": {},
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    def compare(label, expected, actual, *, required=False):
        a, b = np.asarray(expected), np.asarray(actual)
        metrics = {
            "label": label,
            **compare_arrays(a, b),
            "all_finite": bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b))),
            "top1_equal": bool(np.array_equal(np.argmax(a, -1), np.argmax(b, -1))),
            "faithfulness_required": required,
        }
        report["checks"].append(metrics)
        if required and (
            not metrics["all_finite"] or not metrics["top1_equal"] or metrics["nrmse"] >= 1e-5
        ):
            emit({"event": "unfaithful_reproduction", **metrics})
            raise AssertionError(label)
        if metrics["nrmse"] > 1e-6 or label.endswith(f"/{position}"):
            emit({"event": "comparison", **metrics})

    try:
        sa = server_args(candidate["checkpoint"], CONTEXT)
        sa.max_total_tokens = CAPACITY
        sa.json_model_override_args = json.dumps(
            {"v4_mhc_backend": "pallas", "v4_hca_backend": "pallas", "v4_csa_backend": "reference"}
        )
        emit({"event": "loading"})
        worker = ModelWorker(sa, create_device_mesh([1, 4], [1, 1]))
        runner = worker.model_runner
        original_forward = worker.forward_batch_generation
        selected = None

        def forward(batch):
            if selected is None:
                return original_forward(batch)
            mode, absolute = selected
            folder = args.output / mode / f"step-{absolute}"
            folder.mkdir()
            metadata = runner.attn_backend.get_forward_metadata(batch)
            save_arrays(folder / "metadata", dataclasses.asdict(metadata), compressed=False)
            save_arrays(
                folder / "inputs",
                {
                    "ids": batch.input_ids,
                    "positions": batch.positions,
                    "locations": batch.out_cache_loc,
                },
                compressed=False,
            )
            if absolute == capture_begin:
                before = folder / "before"
                before.mkdir()
                for layer, cache in enumerate(runner.token_to_kv_pool.layers):
                    save_arrays(before / f"layer-{layer:02d}", cache, compressed=False)
                emit({"event": "before_state_saved", "backend": mode, "position": absolute})
            output, sampled, misses = original_forward(batch)
            jax.block_until_ready((output, sampled))
            save_arrays(
                folder / "production", {"logits": output.next_token_logits}, compressed=False
            )
            report["frames"].append(f"{mode}/step-{absolute}")
            emit({"event": "frame_saved", "backend": mode, "position": absolute})
            return output, sampled, misses

        worker.forward_batch_generation = forward
        for mode in ("reference", "pallas"):
            # Rebuild the standard JIT entry so the explicit static backend is
            # actually part of its GraphDef, not only a changed Python label.
            # The exported executed HLO below verifies which backend ran.
            runner.model.csa_backend = mode
            runner.initialize_jit()
            runner.req_to_token_pool.clear()
            runner.token_to_kv_pool_allocator.clear()
            folder = args.output / mode
            folder.mkdir()
            session = PagedWorkerSession(worker)
            session.new(0)
            prompt = baseline["prompts"][case]
            for begin in range(0, PROMPT, 128):
                logits, _ = session.step([(0, prompt[begin : begin + 128])], bucket=1)
                if begin % 1024 == 0:
                    emit({"event": "prefill", "backend": mode, "position": begin})
            compare(f"{mode}/prefill", golden[0], logits[0], required=True)
            rows = [logits[0]]
            for step in range(index + 1):
                absolute = PROMPT + step
                selected = (mode, absolute) if absolute >= capture_begin else None
                logits, _ = session.step(
                    [(0, [int(np.argmax(golden[step]))])], decode=True, bucket=1
                )
                rows.append(logits[0])
                compare(
                    f"{mode}/{absolute}", golden[step + 1], logits[0], required=mode == "reference"
                )
                if step % 16 == 0:
                    emit({"event": "decode", "backend": mode, "position": absolute})
            selected = None
            if mode == "pallas":
                compare(f"faithful_failure/{position}", failed_logits, logits[0], required=True)
            np.savez_compressed(folder / "logits.npz", logits=np.stack(rows))
            report["kernel_evidence"][mode] = export_mhc_evidence(session, folder)
            if mode == "reference" and any(
                v
                for k, v in report["kernel_evidence"][mode]["compiled_custom_call_counts"].items()
                if k.startswith("csa_")
            ):
                raise AssertionError(
                    "reference capture unexpectedly executed original CSA programs"
                )
            session.free(0)
            emit({"event": "backend_complete", "backend": mode})
        report["complete"] = True
        emit({"event": "dual_capture_complete"})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
