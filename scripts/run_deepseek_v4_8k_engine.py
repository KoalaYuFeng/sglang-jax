"""Normal Engine/scheduler gate for the real-checkpoint V4 8K worker result."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
import traceback
from pathlib import Path

from run_deepseek_v4_8k_worker import server_args
from run_deepseek_v4_framework import framework_fingerprint

from sgl_jax.srt.entrypoints.engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    gate = json.loads(options.worker_report.read_text())
    if not gate.get("complete") or gate.get(
        "framework_source_fingerprint"
    ) != framework_fingerprint():
        raise ValueError("requires a same-source complete V4 8K ModelWorker gate")
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite {options.output}")
    options.output.mkdir(parents=True)

    checkpoint = Path(gate["checkpoint"])
    args = server_args(checkpoint)
    args.skip_tokenizer_init = False
    args.skip_server_warmup = True
    args.log_level = "info"
    report = {
        "complete": False,
        "worker_report": str(options.worker_report.resolve()),
        "framework_source_fingerprint": gate["framework_source_fingerprint"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "server_args": dataclasses.asdict(args),
        "scope": (
            "Engine -> tokenizer manager -> standard SGLang scheduler -> 128-token "
            "chunked prefill -> native ModelWorker; B1 TP4/EP4"
        ),
        "events": [],
        "cases": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (options.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(json.dumps(event, default=str), flush=True)

    engine = None
    try:
        emit({"event": "engine_start"})
        start = time.perf_counter()
        engine = Engine(server_args=args)
        emit({"event": "engine_ready", "seconds": time.perf_counter() - start})

        cases = (
            (
                "near_8k_chunked_prefill",
                gate["long_input_ids"],
                gate["long_expected_output_ids"],
            ),
            (
                "request_slot_reset_after_8k",
                gate["short_reset_case"]["input_ids"],
                gate["short_reset_case"]["expected_output_ids"],
            ),
        )
        for label, input_ids, expected in cases:
            emit({"event": "request_start", "label": label, "prompt_tokens": len(input_ids)})
            start = time.perf_counter()
            output = engine.generate(
                input_ids=input_ids,
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": len(expected),
                    "ignore_eos": True,
                    "skip_special_tokens": True,
                },
            )
            seconds = time.perf_counter() - start
            finish = output["meta_info"]["finish_reason"]
            passed = (
                output["output_ids"] == expected
                and output["meta_info"]["prompt_tokens"] == len(input_ids)
                and output["meta_info"]["completion_tokens"] == len(expected)
                and finish == {"type": "length", "length": len(expected)}
            )
            row = {
                "label": label,
                "prompt_tokens": len(input_ids),
                "expected_output_ids": expected,
                "seconds": seconds,
                "output": output,
                "passed": passed,
            }
            report["cases"].append(row)
            emit({"event": "request_checked", **row})
            if not passed:
                raise AssertionError(f"Engine 8K scheduler gate failed: {label}")

        report["complete"] = True
        emit({"event": "v4_8k_engine_scheduler_gate_passed"})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()

