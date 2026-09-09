"""8K Engine concurrency, prefix reuse, queue pressure and genuine KV exhaustion.

Run normal first, then pressure in a fresh process with
SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=64. This deliberately under-reserves
generation length, but retraction must be caused by real page exhaustion, not
SGLANG_TEST_RETRACT or a manual pause. All requests belong to this test process.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np
from deepseek_v4_execution_options import (
    HELPER_SHA256,
    assert_server_options,
    model_overrides,
    options_from_receipt,
)
from deepseek_v4_test_observer import drain_observer
from run_deepseek_v4_8k_native import CAPACITY, CONTEXT, DECODE, PROMPT
from run_deepseek_v4_framework import framework_fingerprint
from run_deepseek_v4_paged import server_args
from sgl_jax.srt.entrypoints.engine import Engine

PUBLIC_DECODE = DECODE - 2  # standard max_req_len / scheduler safety headroom


def common_decode_window(rows):
    """Client-observed interval after all first tokens and before any last token.

    Queued groups need not have a common active interval; never label B8 as
    eight simultaneously decoding requests on a four-slot server.
    """
    start = max(row["token_events"][0]["elapsed"] for row in rows)
    stop = min(row["token_events"][-1]["elapsed"] for row in rows)
    if stop <= start:
        return None
    tokens = 0
    for row in rows:
        events = row["token_events"]
        before = max((e["tokens"] for e in events if e["elapsed"] <= start), default=0)
        after = max(
            (e["tokens"] for e in events if e["elapsed"] <= stop), default=before
        )
        tokens += after - before
    return {
        "seconds": stop - start,
        "tokens": tokens,
        "tokens_per_second": tokens / (stop - start),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pressure", action="store_true")
    parser.add_argument("--serving-report", type=Path)
    parser.add_argument("--run-log", type=Path)
    parser.add_argument(
        "--allow-failed-gates",
        action="store_true",
        help="diagnostic continuation only; failed acceptance stays failed",
    )
    args = parser.parse_args()
    fixture = json.loads(args.worker_report.read_text())
    fingerprint = framework_fingerprint()
    if fixture["framework_source_fingerprint"] != fingerprint or not (
        fixture["complete"] or (args.allow_failed_gates and fixture.get("finished"))
    ):
        raise ValueError("requires complete same-source 8K native worker gate")
    if args.pressure and (
        os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION") != "64"
        or args.serving_report is None
        or args.run_log is None
    ):
        raise ValueError(
            "pressure needs admission clip=64, prior serving report and own run log"
        )
    if os.environ.get("SGLANG_TEST_RETRACT"):
        raise ValueError("forced test-retract is not a memory-pressure acceptance gate")
    args.output.mkdir(parents=True, exist_ok=False)
    selected = options_from_receipt(fixture)
    sa = server_args(fixture["checkpoint"], CONTEXT)
    sa.json_model_override_args = json.dumps(model_overrides(selected))
    sa.max_total_tokens = 2 * PROMPT + 384 if args.pressure else CAPACITY
    sa.skip_tokenizer_init = False
    sa.skip_server_warmup = True
    sa.disable_overlap_schedule = False
    report = {
        "complete": False,
        "framework_source_fingerprint": fingerprint,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "server_args": dataclasses.asdict(sa),
        "pressure": args.pressure,
        "admission_estimate_clip": os.environ.get(
            "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096"
        ),
        "scope": __doc__,
        "events": [],
        "cases": [],
        "finished": False,
        "worker_numerical_gate_passed": fixture["complete"],
        **selected,
        "execution_helper_sha256": HELPER_SHA256,
        "public_generation_tokens": PUBLIC_DECODE,
    }
    engine = None

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(json.dumps(event, default=str), flush=True)

    async def run(
        label,
        indices,
        prompts,
        expected,
        *,
        count=PUBLIC_DECODE,
        cold=False,
        prefix=False,
    ):
        start = time.perf_counter()
        observations, rows, tasks = [], [None] * len(indices), []

        async def request(index, prompt_id):
            first, previous, seen = None, None, 0
            first_cached_tokens = None
            intervals, deltas = [], []
            token_events = []
            final = None
            stream = await engine.async_generate(
                input_ids=prompts[prompt_id],
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": count,
                    "ignore_eos": True,
                },
                stream=True,
            )
            async for output in stream:
                now = time.perf_counter()
                tokens = len(output["output_ids"])
                if tokens > seen:
                    token_events.append({"elapsed": now - start, "tokens": tokens})
                    if first is None:
                        first_cached_tokens = output["meta_info"]["cached_tokens"]
                    first = now if first is None else first
                    if previous is not None:
                        intervals.append((now - previous) / (tokens - seen))
                        deltas.append(tokens - seen)
                    previous, seen = now, tokens
                final = output
            if final is None or seen != count:
                raise AssertionError(f"{label}/{index}: incomplete generation: {final}")
            if final["meta_info"]["finish_reason"] != {
                "type": "length",
                "length": count,
            }:
                raise AssertionError(
                    f"{label}/{index}: unexpected finish: {final['meta_info']}"
                )
            numerical_equal = (
                expected[prompt_id] is None
                or final["output_ids"] == expected[prompt_id][:count]
            )
            first_mismatch = (
                next(
                    (
                        i
                        for i, (a, b) in enumerate(
                            zip(
                                final["output_ids"],
                                expected[prompt_id][:count],
                                strict=True,
                            )
                        )
                        if a != b
                    ),
                    None,
                )
                if expected[prompt_id] is not None
                else None
            )
            if not numerical_equal and not args.allow_failed_gates:
                raise AssertionError(
                    f"{label}/{index}: generated token IDs differ from serial baseline"
                )
            cached = final["meta_info"]["cached_tokens"]
            if cold and (
                first_cached_tokens != 0 or (cached != 0 and not args.pressure)
            ):
                raise AssertionError(
                    f"{label}/{index}: cold fixture reused {cached} tokens"
                )
            if prefix and cached < (PROMPT - 1) // 128 * 128:
                raise AssertionError(
                    f"{label}/{index}: missing long-prefix hit ({cached})"
                )
            rows[index] = {
                "prompt_id": prompt_id,
                "output": final,
                "numerically_equal": numerical_equal,
                "first_mismatch": first_mismatch,
                "first_cached_tokens": first_cached_tokens,
                "ttft_seconds": first - start,
                "e2e_seconds": time.perf_counter() - start,
                "decode_tpot_seconds": (previous - first) / (count - 1)
                if count > 1
                else None,
                "stream_interval_p50_ms_per_token": float(np.median(intervals) * 1000),
                "stream_interval_p95_ms_per_token": float(
                    np.percentile(intervals, 95) * 1000
                ),
                "max_coalesced_token_delta": max(deltas, default=0),
                "max_stream_gap_ms": max(
                    (a * b * 1000 for a, b in zip(intervals, deltas, strict=True)),
                    default=0,
                ),
                "token_events": token_events,
            }

        async def observe():
            while not all(task.done() for task in tasks):
                states = (await engine.async_get_server_info())["internal_states"]
                observations.append(
                    {"elapsed": time.perf_counter() - start, "states": states}
                )
                await asyncio.sleep(0.5)

        tasks.extend(
            asyncio.create_task(request(i, key)) for i, key in enumerate(indices)
        )
        monitor = asyncio.create_task(observe())
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=1800)
            elapsed = time.perf_counter() - start
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await drain_observer(monitor)
        states = (await asyncio.wait_for(engine.async_get_server_info(), timeout=60))[
            "internal_states"
        ]
        for state in states:
            if state["req_to_token_pool_used"] or state["waiting_queue_size"]:
                raise AssertionError(f"{label}: request slots did not return: {state}")
            if (
                state["available_kv_tokens"] + state["tree_cache_size"]
                != sa.max_total_tokens
            ):
                raise AssertionError(f"{label}: KV capacity accounting leaked: {state}")
        all_states = [s for o in observations for s in o["states"]]
        result = {
            "label": label,
            "indices": indices,
            "seconds": elapsed,
            "output_tokens": len(indices) * count,
            "e2e_output_tokens_per_second": len(indices) * count / elapsed,
            "common_decode_window": common_decode_window(rows),
            "max_running": max(
                (s["running_batch_size"] for s in all_states), default=0
            ),
            "max_waiting": max(
                (s["waiting_queue_size"] for s in all_states), default=0
            ),
            "min_free_kv_tokens": min(
                (s["available_kv_tokens"] for s in all_states), default=None
            ),
            "requests": rows,
            "observations": observations,
            "idle_state": states,
            "passed": all(r["numerically_equal"] for r in rows),
        }
        report["cases"].append(result)
        emit(
            {
                "event": "case_finished",
                **{
                    k: v
                    for k, v in result.items()
                    if k not in {"requests", "observations", "idle_state"}
                },
            }
        )
        return result

    def case(label, indices, *, cold=False, prefix=False, count=PUBLIC_DECODE):
        if cold and not engine.flush_cache():
            raise AssertionError("idle cache flush failed")
        emit({"event": "case_start", "label": label, "indices": indices, "cold": cold})
        return engine.loop.run_until_complete(
            run(
                label, indices, prompts, expected, count=count, cold=cold, prefix=prefix
            )
        )

    try:
        prompts = fixture["prompts"]
        expected = fixture["expected_output_ids"]
        if args.pressure:
            prior = json.loads(args.serving_report.read_text())
            if prior["framework_source_fingerprint"] != fingerprint or not (
                prior["complete"] or (args.allow_failed_gates and prior.get("finished"))
            ):
                raise ValueError("requires complete same-source normal serving gate")
            if options_from_receipt(prior) != selected:
                raise ValueError(
                    "pressure gate must preserve the accepted execution choices"
                )
            prompts, expected = prior["prompts"], prior["expected_output_ids"]
        else:
            prompts = prompts + [p[1:] + p[:1] for p in prompts]
            expected = expected + [None, None]
        if len({tuple(p[:128]) for p in prompts}) != 4:
            raise ValueError("cold B4 inputs must have four different first pages")
        report["prompts"], report["expected_output_ids"] = prompts, expected
        emit({"event": "engine_start"})
        engine = Engine(server_args=sa)
        report["reported_execution"] = assert_server_options(
            engine.get_server_info(), selected
        )
        emit({"event": "engine_ready"})
        if not args.pressure:
            case("cold_B1", [0], cold=True)
            for key in (2, 3):
                row = case(f"serial_variant_{key}", [key], cold=True)
                expected[key] = row["requests"][0]["output"]["output_ids"]
            case("cold_B2", [0, 1], cold=True)
            for repeat, order in enumerate(([3, 1, 2, 0], [0, 2, 1, 3])):
                row = case(f"cold_B4_round{repeat}", order, cold=True)
                if row["max_running"] != 4:
                    raise AssertionError("B4 test never observed four running requests")
            case("long_prefix_B4", [1, 3, 0, 2], prefix=True)
            row = case("queued_B8", [0, 1, 2, 3, 3, 2, 1, 0], prefix=True)
            if row["max_waiting"] < 1 or row["max_running"] != 4:
                raise AssertionError("queue pressure was not observed")
        else:
            case("kv_exhaustion_B2", [0, 1], cold=True)
            log = args.run_log.read_text()
            retractions = re.findall(
                r"KV cache pool is full\. Retract requests\. #retracted_reqs: (\d+), #aborted_reqs: (\d+)",
                log,
            )
            if not retractions or sum(int(r) for r, _ in retractions) < 1:
                raise AssertionError("no actual KV-pool exhaustion/retraction observed")
            if any(int(a) for _, a in retractions):
                raise AssertionError("memory pressure aborted an owned request")
            report["natural_retractions"] = [
                {"retracted": int(r), "aborted": int(a)} for r, a in retractions
            ]
            emit(
                {
                    "event": "real_kv_exhaustion_recovery_passed",
                    "retractions": report["natural_retractions"],
                }
            )
            case("evict_with_unique_2", [2])
            case("evict_with_unique_3", [3])
            row = case("recompute_evicted_0", [0])
            if row["requests"][0]["output"]["meta_info"]["cached_tokens"] != 0:
                raise AssertionError(
                    "old prefix was not fully evicted by unique 8K requests"
                )
        report["finished"] = True
        report["complete"] = fixture["complete"] and all(
            c["passed"] for c in report["cases"]
        )
        emit(
            {
                "event": "serving_8k_measurements_finished",
                "acceptance_passed": report["complete"],
            }
        )
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
