"""Bounded end-to-end Engine/scheduler gate, after the ModelWorker oracle.

Run in a fresh process after the numerical/profile process exits. The Engine
owns and shuts down its scheduler subprocess; never allocate a second model
while a separate TPU experiment is running.
"""

import argparse
import dataclasses
import hashlib
import json
import time
import traceback
from pathlib import Path

from run_deepseek_v4_framework import framework_fingerprint, server_args

from sgl_jax.srt.entrypoints.engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.framework_report.read_text())
    if (
        not fixture.get("complete")
        or fixture["framework_source_fingerprint"] != framework_fingerprint()
    ):
        raise ValueError(
            "first pass the same-source complete native ModelWorker correctness gate"
        )
    sa = server_args(Path(fixture["checkpoint"]))
    sa.skip_server_warmup = True
    sa.skip_tokenizer_init = False
    sa.log_level = "info"
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "framework_report": str(args.framework_report),
        "framework_source_fingerprint": fixture["framework_source_fingerprint"],
        "validation_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "server_args": dataclasses.asdict(sa),
        "scope": "Engine/tokenizer/scheduler/ModelWorker/greedy sampler, sequential B1 requests",
        "events": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(json.dumps(event, default=str), flush=True)

    engine = None
    try:
        emit({"event": "engine_start"})
        start = time.perf_counter()
        engine = Engine(server_args=sa)
        emit({"event": "engine_ready", "seconds": time.perf_counter() - start})
        cases = [
            (
                "chat",
                fixture["chat_smoke"]["prompt_ids"],
                fixture["chat_smoke"]["completion_ids"],
            ),
            (
                "132_token_generation",
                fixture["input_ids"],
                fixture["teacher_forced_ids"],
            ),
            (
                "chat_request_slot_reuse",
                fixture["chat_smoke"]["prompt_ids"],
                fixture["chat_smoke"]["completion_ids"],
            ),
        ]
        report["cases"] = []
        for label, inputs, expected in cases:
            stops_on_eos = label != "132_token_generation" and expected[-1] == 1
            expected_api_ids = expected[:-1] if stops_on_eos else expected
            emit(
                {"event": "request_start", "label": label, "input_tokens": len(inputs)}
            )
            start = time.perf_counter()
            output = engine.generate(
                input_ids=inputs,
                sampling_params={
                    "temperature": 0,
                    # Give EOS one spare slot so stop detection wins over the
                    # scheduler's max-length check. The public detokenized API
                    # intentionally omits the matched special token from
                    # output_ids, while completion_tokens still counts it.
                    "max_new_tokens": len(expected) + int(stops_on_eos),
                    # The long synthetic fixture is a fixed-length greedy
                    # regression. Chat exercises the official EOS token too.
                    "ignore_eos": label == "132_token_generation",
                    "stop_token_ids": [1] if label != "132_token_generation" else [],
                    "skip_special_tokens": True,
                },
            )
            finish = output["meta_info"]["finish_reason"]
            finish_matches = (
                finish == {"type": "stop", "matched": 1}
                if stops_on_eos
                else finish == {"type": "length", "length": len(expected)}
            )
            check = {
                "label": label,
                "seconds_including_cold_compilation": time.perf_counter() - start,
                "expected_internal_ids": expected,
                "expected_api_output_ids": expected_api_ids,
                "output": output,
                "matches_reference": (
                    output["output_ids"] == expected_api_ids
                    and output["meta_info"]["completion_tokens"] == len(expected)
                    and finish_matches
                ),
            }
            report["cases"].append(check)
            emit({"event": "request_checked", **check})
            if not check["matches_reference"]:
                raise AssertionError(
                    f"Engine/scheduler output differs from reference: {label}"
                )
        report["complete"] = True
        emit({"event": "engine_scheduler_smoke_passed"})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
