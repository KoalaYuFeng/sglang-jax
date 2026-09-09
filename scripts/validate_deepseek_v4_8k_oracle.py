"""Independent 43-layer, from-empty-cache 8K numerical gate.

Unlike late_reference in the native batch test, this run shares no hidden
states, prefix KV, or compressor scratch with ModelWorker. The independent
reference evaluates the entire 7936-token prompt in one prefill and then all
256 teacher-forced decode positions. It compares complete vocabulary logits,
not just generated tokens. This is a correctness test, not a performance run.
"""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np

from run_deepseek_v4_8k_native import CONTEXT, DECODE, PROMPT
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    DeepSeekV4Reference,
    source_fingerprint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native_report.read_text())
    if not native["complete"] or native["numerical_failures"]:
        raise ValueError("first pass the complete native B1/B2/B4 numerical gate")
    if native["framework_source_fingerprint"] != framework_fingerprint():
        raise ValueError("native goldens must come from the current production source")
    prompts = native["prompts"]
    if len(prompts) != 2 or any(len(prompt) != PROMPT for prompt in prompts):
        raise ValueError("requires the complete two-prompt 8K fixture")
    with np.load(args.native_report.parent / "golden_logits.npz", allow_pickle=False) as arrays:
        goldens = [arrays[f"case{i}"] for i in range(2)]
    if any(g.ndim != 2 or g.shape[0] != DECODE + 1 for g in goldens):
        raise ValueError("requires all prefill/decode full-logit rows")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "finished": False,
        "scope": __doc__,
        "native_report": str(args.native_report),
        "checkpoint": native["checkpoint"],
        "framework_source_fingerprint": framework_fingerprint(),
        "reference_source_fingerprint": source_fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": [],
        "events": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    def check(case, index, expected):
        expected = np.asarray(expected, np.float32)[0]
        actual = goldens[case][index]
        metrics = compare_arrays(expected, actual)
        metrics["all_finite"] = bool(np.all(np.isfinite(expected)) and np.all(np.isfinite(actual)))
        metrics["top1_equal"] = bool(np.argmax(expected) == np.argmax(actual))
        row = {"case": case, "position": PROMPT - 1 + index, **metrics}
        report["checks"].append(row)
        if not metrics["all_finite"] or not metrics["top1_equal"] or metrics["nrmse"] > 0.005:
            np.savez_compressed(args.output / "failure.npz", expected=expected, actual=actual)
            emit({"event": "numerical_failure", **row})
            raise AssertionError(f"independent 8K comparison failed: {row}")
        if index % 32 == 0 or index == DECODE:
            emit({"event": "comparison", **row})
        return expected

    try:
        emit({"event": "loading_independent_reference"})

        def progress(event):
            # Make long full-prefill/compile progress visible without emitting
            # every decode layer (all acceptance checks are persisted above).
            if event["event"] == "layer_loaded" or event.get("tokens", 0) > 1:
                print(json.dumps(event), flush=True)

        ref = DeepSeekV4Reference(native["checkpoint"], max_context=CONTEXT, progress=progress)
        ref.load()
        for case, prompt in enumerate(prompts):
            ref.reset()
            emit({"event": "independent_prefill", "case": case, "tokens": len(prompt)})
            logits, _ = ref.step(prompt)
            rows = [check(case, 0, logits)]
            for index in range(DECODE):
                # Fixed teacher forcing isolates numerical errors from an
                # autoregressive change of input after a token discrepancy.
                token = int(np.argmax(goldens[case][index]))
                logits, _ = ref.step([token])
                rows.append(check(case, index + 1, logits))
            np.savez_compressed(args.output / f"case{case}-logits.npz", logits=np.stack(rows))
            if ref.position != CONTEXT:
                raise AssertionError("independent reference did not reach the full 8192 boundary")
            emit({"event": "case_complete", "case": case, "cache_length": ref.position})
        report["complete"] = report["finished"] = True
        emit({"event": "independent_8k_gate_passed", "full_logit_rows": len(report["checks"])})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
