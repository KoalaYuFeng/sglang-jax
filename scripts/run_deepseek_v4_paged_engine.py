"""End-to-end native V4 scheduler/radix/streaming gate after ModelWorker passes."""

import argparse
import dataclasses
import hashlib
import json
import time
import traceback
from pathlib import Path

from deepseek_v4_execution_options import (
    HELPER_SHA256,
    assert_server_options,
    model_overrides,
    options_from_receipt,
)
from run_deepseek_v4_framework import framework_fingerprint
from run_deepseek_v4_paged import server_args
from sgl_jax.srt.entrypoints.engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.worker_report.read_text())
    if (
        not fixture.get("complete")
        or fixture["source_fingerprint"] != framework_fingerprint()
    ):
        raise ValueError("pass the complete, same-source ModelWorker gate first")
    args.output.mkdir(parents=True, exist_ok=False)
    selected = options_from_receipt(fixture)
    sa = server_args(fixture["checkpoint"], fixture["server_args"]["context_length"])
    sa.json_model_override_args = json.dumps(model_overrides(selected))
    sa.skip_tokenizer_init, sa.skip_server_warmup = False, True
    sa.disable_overlap_schedule = not args.overlap
    report = {
        "complete": False,
        "gate_version": 2,
        "gate_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_fingerprint": fixture["source_fingerprint"],
        "server_args": dataclasses.asdict(sa),
        "worker_report": str(args.worker_report),
        **selected,
        "execution_helper_sha256": HELPER_SHA256,
        "events": [],
        "cases": [],
    }
    engine = None

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(json.dumps(event, default=str), flush=True)

    def check(label, result, expected, *, prefix_hit=False, logs=False):
        if result["output_ids"] != expected:
            raise AssertionError(f"{label}: {result['output_ids']} != {expected}")
        if prefix_hit and result["meta_info"]["cached_tokens"] < 128:
            raise AssertionError(f"{label}: expected an actual shared-page radix hit")
        if logs and not result["meta_info"].get("output_token_logprobs"):
            raise AssertionError(f"{label}: output logprobs missing")
        report["cases"].append({"label": label, "output": result})
        emit(
            {
                "event": "case_passed",
                "label": label,
                "cached_tokens": result["meta_info"]["cached_tokens"],
            }
        )

    try:
        emit({"event": "engine_start"})
        engine = Engine(server_args=sa)
        report["reported_execution"] = assert_server_options(
            engine.get_server_info(), selected
        )
        emit({"event": "engine_ready"})
        prompts, expected = fixture["prompts"], fixture["expected_output_ids"]
        sampling = {"temperature": 0, "max_new_tokens": 2, "ignore_eos": True}
        # Cold distinct requests, followed by B4 radix forks using those pages.
        outputs = engine.generate(input_ids=prompts, sampling_params=sampling)
        for i, output in enumerate(outputs):
            check(f"cold_B2_{i}", output, expected[i])
        outputs = engine.generate(
            input_ids=[prompts[i % 2] for i in range(4)], sampling_params=sampling
        )
        for i, output in enumerate(outputs):
            check(f"prefix_B4_{i}", output, expected[i % 2], prefix_hit=True)
        # Streaming and sampled logprobs use the ordinary public interface.
        last = None
        for event in engine.generate(
            input_ids=prompts[0],
            sampling_params=sampling,
            stream=True,
            return_logprob=True,
            logprob_start_len=-1,
            top_logprobs_num=3,
        ):
            last = event
        if last is None:
            raise AssertionError("stream produced no events")
        check("stream_logprobs", last, expected[0], prefix_hit=True, logs=True)
        # Abort an owned request after its first streamed token, then reuse its
        # request/page capacity. No requests from another process are touched.
        stream = engine.generate(
            input_ids=prompts[1],
            sampling_params={**sampling, "max_new_tokens": 32},
            stream=True,
        )
        first = next(stream)
        if first["meta_info"].get("finish_reason") is not None:
            raise AssertionError("abort fixture finished before cancellation")
        engine.abort_request(first["meta_info"]["id"])
        aborted = first
        for event in stream:
            aborted = event
        if (aborted["meta_info"].get("finish_reason") or {}).get("type") != "abort":
            raise AssertionError(
                "cancelled request did not finish with an abort acknowledgement"
            )
        emit({"event": "request_aborted", "output": aborted})
        output = engine.generate(input_ids=prompts[1], sampling_params=sampling)
        check("after_abort_slot_reuse", output, expected[1], prefix_hit=True)

        # Exercise the framework's actual retract/resume path, not just a
        # manually restored cache. Every active request belongs to this gate.
        longer = {**sampling, "max_new_tokens": 32}
        baseline = engine.generate(input_ids=prompts[0], sampling_params=longer)
        if baseline["output_ids"][:2] != expected[0]:
            raise AssertionError("long-decode baseline disagrees with the reference")
        stream = engine.generate(
            input_ids=prompts[0], sampling_params=longer, stream=True
        )
        first = next(stream)
        if first["meta_info"].get("finish_reason") is not None:
            raise AssertionError("retract fixture finished before pause")
        engine.pause_generation(mode="retract")
        states = engine.get_server_info()["internal_states"]
        if not all(
            state["engine_paused"]
            and state["running_batch_size"] == 0
            and state["waiting_queue_size"] > 0
            for state in states
        ):
            raise AssertionError(f"request was not actually retracted: {states}")
        emit({"event": "request_retracted", "states": states})
        engine.continue_generation()
        resumed = first
        for event in stream:
            resumed = event
        check("after_retract_resume", resumed, baseline["output_ids"])

        if not engine.flush_cache():
            raise AssertionError("idle radix flush failed")
        output = engine.generate(input_ids=prompts[0], sampling_params=sampling)
        check("after_eviction", output, expected[0])
        if output["meta_info"]["cached_tokens"] != 0:
            raise AssertionError("flush did not evict prefix pages")
        report["complete"] = True
        emit({"event": "engine_paged_serving_passed"})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
