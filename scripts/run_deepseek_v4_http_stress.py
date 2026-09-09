"""Owned, loopback-only V4 HTTP concurrency and bounded-soak acceptance gate.

No JAX is imported by this client. One owned server process uses the TPU.
Token IDs are checked against a complete same-source Engine/8K worker gate.
Client streaming intervals include transport/coalescing; they are not isolated
device latency. Expected invalid-input/cancellation cases are reported separately.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import httpx
from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_execution_options import (
    DEFAULTS,
    HELPER_SHA256,
    assert_server_options,
    model_overrides,
    options_from_receipt,
)
from deepseek_v4_test_observer import drain_observer


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def compilation_observations(log):
    misses = [int(value) for value in re.findall(r"#cache_miss:\s*(\d+)", log)]
    return {"logged_misses": misses, "any_cache_miss": any(misses)}


def streaming_metrics(events, elapsed):
    if not events or any(
        b["tokens"] <= a["tokens"] or b["elapsed"] < a["elapsed"]
        for a, b in zip(events, events[1:])
    ):
        raise ValueError("stream token observations must be nonempty and monotonic")
    intervals = [
        (b["elapsed"] - a["elapsed"]) / (b["tokens"] - a["tokens"]) * 1000
        for a, b in zip(events, events[1:])
    ]
    decode_tokens = events[-1]["tokens"] - events[0]["tokens"]
    return {
        "ttft_seconds": events[0]["elapsed"],
        "e2e_seconds": elapsed,
        "decode_tpot_ms": (
            (events[-1]["elapsed"] - events[0]["elapsed"]) / decode_tokens * 1000
            if decode_tokens
            else None
        ),
        "stream_interval_p50_ms_per_token": percentile(intervals, 0.5),
        "stream_interval_p95_ms_per_token": percentile(intervals, 0.95),
        "max_coalesced_token_delta": max(
            [events[0]["tokens"]]
            + [b["tokens"] - a["tokens"] for a, b in zip(events, events[1:])]
        ),
    }


def validate_output(output, expected, count, prompt_tokens):
    meta = output["meta_info"]
    if len(expected) < count or output["output_ids"] != expected[:count]:
        raise AssertionError(
            "HTTP output token IDs differ from the same-source Engine baseline"
        )
    if meta["completion_tokens"] != count or meta["prompt_tokens"] != prompt_tokens:
        raise AssertionError("HTTP token accounting is incorrect")
    if meta["finish_reason"] != {"type": "length", "length": count}:
        raise AssertionError(f"unexpected HTTP finish reason: {meta['finish_reason']}")


def validate_idle(states, capacity):
    if not states:
        raise AssertionError("missing scheduler state")
    for state in states:
        if any(
            state[key]
            for key in (
                "req_to_token_pool_used",
                "waiting_queue_size",
                "running_batch_size",
            )
        ):
            return False
        if state["available_kv_tokens"] + state["tree_cache_size"] != capacity:
            raise AssertionError("HTTP request/page capacity leaked after idle")
    return True


def server_command(checkpoint, port, model_name, *, execution=None):
    return [
        sys.executable,
        "-m",
        "sgl_jax.launch_server",
        "--model-path",
        checkpoint,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        model_name,
        "--context-length",
        "8192",
        "--tp-size",
        "4",
        "--ep-size",
        "4",
        "--dp-size",
        "1",
        "--device",
        "tpu",
        "--dtype",
        "bfloat16",
        "--moe-backend",
        "epmoe",
        "--attention-backend",
        "deepseek_v4",
        "--page-size",
        "128",
        "--max-running-requests",
        "4",
        "--max-total-tokens",
        "33280",
        "--max-prefill-tokens",
        "128",
        "--chunked-prefill-size",
        "128",
        "--disable-hybrid-swa-memory",
        "--random-seed",
        "42",
        "--mem-fraction-static",
        "0.85",
        "--disable-precompile",
        "--skip-server-warmup",
        "--precompile-token-paddings",
        "128",
        "--precompile-bs-paddings",
        "1",
        "2",
        "4",
        "--watchdog-timeout",
        "1800",
        "--log-level-http",
        "warning",
        "--json-model-override-args",
        json.dumps(model_overrides(DEFAULTS if execution is None else execution)),
    ]


class HttpGate:
    def __init__(self, client, fixture, report, emit):
        self.client, self.fixture, self.report, self.emit = (
            client,
            fixture,
            report,
            emit,
        )
        self.prompts, self.expected = fixture["prompts"], fixture["expected_output_ids"]
        self.capacity = 33280

    async def states(self, *, timeout=60):
        response = await self.client.get("/get_server_info", timeout=timeout)
        response.raise_for_status()
        return response.json()["internal_states"]

    async def idle(self):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            states = await self.states()
            if validate_idle(states, self.capacity):
                return states
            await asyncio.sleep(0.25)
        raise AssertionError(
            "HTTP server did not release owned requests after completion"
        )

    async def flush(self):
        await self.idle()
        response = await self.client.post("/flush_cache", timeout=60)
        response.raise_for_status()
        states = await self.idle()
        if any(
            s["tree_cache_size"] or s["available_kv_tokens"] != self.capacity
            for s in states
        ):
            raise AssertionError("idle HTTP cache flush did not release every page")

    async def request(
        self, key, count=254, *, stream=True, abort=False, disconnect=False, logs=False
    ):
        start = time.perf_counter()
        rid = "v4-http-" + uuid.uuid4().hex
        payload = {
            "rid": rid,
            "input_ids": self.prompts[key],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": count,
                "ignore_eos": True,
            },
            "stream": stream,
        }
        if logs:
            payload.update(
                return_logprob=True, logprob_start_len=-1, top_logprobs_num=3
            )
        events, previous, first_cached, final, done = [], [], None, None, False
        if not stream:
            response = await self.client.post("/generate", json=payload)
            response.raise_for_status()
            final = response.json()
        else:
            async with self.client.stream(
                "POST", "/generate", json=payload
            ) as response:
                response.raise_for_status()
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise AssertionError("stream response has no SSE content type")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        done = True
                        break
                    final = json.loads(data)
                    if "error" in final:
                        raise AssertionError(f"HTTP SSE error: {final}")
                    ids = final["output_ids"]
                    if ids[: len(previous)] != previous:
                        raise AssertionError(
                            "stream changed previously delivered token IDs"
                        )
                    if len(ids) > len(previous):
                        if first_cached is None:
                            first_cached = final["meta_info"]["cached_tokens"]
                        events.append(
                            {"elapsed": time.perf_counter() - start, "tokens": len(ids)}
                        )
                        previous = ids.copy()
                        if (abort or disconnect) and len(events) == 1:
                            if final["meta_info"].get("finish_reason") is not None:
                                raise AssertionError(
                                    "cancellation fixture already finished"
                                )
                            if disconnect:
                                break
                            ack = await self.client.post(
                                "/abort_request", json={"rid": rid}, timeout=60
                            )
                            ack.raise_for_status()
                if not disconnect and not done:
                    raise AssertionError("HTTP SSE stream ended without [DONE]")
        if final is None:
            raise AssertionError("empty HTTP response")
        if abort or disconnect:
            if final["output_ids"] != self.expected[key][: len(final["output_ids"])]:
                raise AssertionError("cancelled stream prefix differs from baseline")
            if (
                abort
                and (final["meta_info"].get("finish_reason") or {}).get("type")
                != "abort"
            ):
                raise AssertionError(
                    "HTTP abort request was not acknowledged by the stream"
                )
        else:
            try:
                validate_output(
                    final, self.expected[key], count, len(self.prompts[key])
                )
            except AssertionError:
                self.report.setdefault("failed_responses", []).append(
                    {
                        "rid": rid,
                        "prompt_id": key,
                        "expected_output_ids": self.expected[key][:count],
                        "actual_response": final,
                    }
                )
                self.emit(
                    {"event": "http_numerical_failure", "rid": rid, "prompt_id": key}
                )
                raise
        if logs:
            values = final["meta_info"].get("output_token_logprobs", [])
            if not values or not all(math.isfinite(row[0]) for row in values):
                raise AssertionError("HTTP logprobs missing or nonfinite")
        elapsed = time.perf_counter() - start
        return {
            "prompt_id": key,
            "rid": rid,
            "stream": stream,
            "abort": abort,
            "disconnect": disconnect,
            "output": final,
            "first_cached_tokens": first_cached,
            "token_events": events,
            "passed": True,
            "metrics": streaming_metrics(events, elapsed)
            if stream
            else {"e2e_seconds": elapsed},
        }

    async def case(
        self, label, indices, *, cold=False, prefix=False, **request_options
    ):
        if cold:
            await self.flush()
        self.emit(
            {"event": "http_case_start", "label": label, "requests": len(indices)}
        )
        log_path = Path(self.report["server_log"])
        log_start = log_path.stat().st_size
        start, observations = time.perf_counter(), []
        tasks = [
            asyncio.create_task(self.request(key, **request_options)) for key in indices
        ]

        async def observe():
            while not all(task.done() for task in tasks):
                query_start = time.perf_counter()
                # First compilation can block the scheduler control plane.
                # Keep the same bounded budget as generation, but retain the
                # actual delay; the final idle check still has a 60s budget.
                states = await self.states(timeout=1800)
                observations.append(
                    {
                        "elapsed": time.perf_counter() - start,
                        "query_seconds": time.perf_counter() - query_start,
                        "states": states,
                    }
                )
                await asyncio.sleep(0.25)

        monitor = asyncio.create_task(observe())
        try:
            rows = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1800)
            elapsed = time.perf_counter() - start
            self.report["pending_case"] = {
                "label": label,
                "requests": rows,
                "seconds": elapsed,
            }
            self.emit({"event": "http_generation_complete", "label": label})
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await drain_observer(monitor)
        states = await self.idle()
        with log_path.open("rb") as stream:
            stream.seek(log_start)
            segment = stream.read()
        compilation = {
            "log_start_byte": log_start,
            "log_end_byte": log_start + len(segment),
            **compilation_observations(segment.decode(errors="replace")),
        }
        for row in rows:
            cached = row["output"]["meta_info"]["cached_tokens"]
            if cold and (cached != 0 or row["first_cached_tokens"] not in (0, None)):
                raise AssertionError("cold HTTP fixture reused prefix tokens")
            if (
                prefix
                and cached < (len(self.prompts[row["prompt_id"]]) - 1) // 128 * 128
            ):
                raise AssertionError("HTTP long-prefix reuse was not observed")
        all_states = [s for observation in observations for s in observation["states"]]
        total = sum(len(row["output"]["output_ids"]) for row in rows)
        result = {
            "label": label,
            "passed": True,
            "cold": cold,
            "prefix": prefix,
            "seconds": elapsed,
            "compilation_observations": compilation,
            "output_tokens": total,
            "e2e_output_tokens_per_second": total / elapsed,
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
            "max_status_query_seconds": max(
                (o["query_seconds"] for o in observations), default=None
            ),
            "idle_state": states,
        }
        self.report["cases"].append(result)
        self.report.pop("pending_case", None)
        self.emit(
            {
                "event": "http_case_passed",
                **{
                    k: v
                    for k, v in result.items()
                    if k not in ("requests", "observations", "idle_state")
                },
            }
        )
        return result

    async def openai_completion(self, model, expected_text, *, stream):
        payload = {
            "model": model,
            "prompt": self.prompts[0],
            "temperature": 0,
            "max_tokens": 64,
            "ignore_eos": True,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        text, finish, usage, done = "", None, None, False
        if not stream:
            response = await self.client.post("/v1/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            text, finish, usage = (
                data["choices"][0]["text"],
                data["choices"][0]["finish_reason"],
                data["usage"],
            )
        else:
            async with self.client.stream(
                "POST", "/v1/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line[6:] == "[DONE]":
                        done = True
                        break
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise AssertionError(f"OpenAI-compatible SSE error: {data}")
                    for choice in data["choices"]:
                        text += choice["text"]
                        finish = choice["finish_reason"] or finish
                    usage = data.get("usage") or usage
            if not done:
                raise AssertionError("OpenAI-compatible stream omitted [DONE]")
        if text != expected_text or finish != "length" or not usage:
            raise AssertionError(
                "OpenAI-compatible text/finish/usage disagrees with native HTTP"
            )
        if usage["completion_tokens"] != 64 or usage["prompt_tokens"] != len(
            self.prompts[0]
        ):
            raise AssertionError("OpenAI-compatible token accounting mismatch")
        await self.idle()
        self.report["protocol_checks"].append(
            {
                "label": "openai_completion",
                "stream": stream,
                "passed": True,
                "usage": usage,
            }
        )
        self.emit({"event": "openai_completion_passed", "stream": stream})


async def run_http(options, fixture, report, emit, process, model_name):
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{options.port}",
        timeout=httpx.Timeout(1800, connect=10),
        limits=httpx.Limits(max_connections=24, max_keepalive_connections=16),
        trust_env=False,
    ) as client:
        start, next_progress = time.monotonic(), 0
        while time.monotonic() - start < 1800:
            if process.poll() is not None:
                raise RuntimeError(
                    f"owned HTTP server exited with {process.returncode}; see server.log"
                )
            try:
                response = await client.get("/health", timeout=5)
                if response.status_code == 200:
                    break
            except httpx.RequestError:
                pass
            if time.monotonic() - start >= next_progress:
                emit(
                    {
                        "event": "http_server_loading",
                        "seconds": time.monotonic() - start,
                    }
                )
                next_progress += 30
            await asyncio.sleep(2)
        else:
            raise TimeoutError("owned HTTP server did not become ready")
        response = await client.get("/get_server_info", timeout=60)
        response.raise_for_status()
        info = response.json()
        report["reported_execution"] = assert_server_options(
            info, options_from_receipt(fixture)
        )
        expected_config = {
            "served_model_name": model_name,
            "context_length": 8192,
            "tp_size": 4,
            "ep_size": 4,
            "max_running_requests": 4,
            "max_total_tokens": 33280,
            "chunked_prefill_size": 128,
            "disable_overlap_schedule": False,
            "disable_radix_cache": False,
        }
        if any(info.get(key) != value for key, value in expected_config.items()):
            raise AssertionError(
                "HTTP server is not the requested owned Flash/EP4 configuration"
            )
        report["server_config"] = {key: info[key] for key in expected_config}
        response = await client.get("/v1/models", timeout=60)
        response.raise_for_status()
        if model_name not in [row["id"] for row in response.json()["data"]]:
            raise AssertionError("OpenAI-compatible model listing is incorrect")
        emit({"event": "http_server_ready", "seconds": time.monotonic() - start})
        gate = HttpGate(client, fixture, report, emit)
        await gate.idle()
        # Rejection is intentional and must not allocate lasting request state.
        response = await client.post(
            "/generate",
            json={
                "input_ids": fixture["prompts"][0] + [1] * 300,
                "sampling_params": {"max_new_tokens": 1},
            },
        )
        if response.status_code not in (400, 422):
            raise AssertionError(
                f"over-context request was not rejected: {response.status_code}"
            )
        await gate.idle()
        report["protocol_checks"].append(
            {
                "label": "over_context_rejected",
                "status": response.status_code,
                "passed": True,
            }
        )
        if options.lifecycle_only:
            # A fresh server must reconstruct both prefixes. This is explicitly
            # a subset receipt, not a new B4/B8/soak acceptance run.
            await gate.case("lifecycle_prime_B2", [0, 1], cold=True, count=64)
        else:
            native = await gate.case(
                "cold_nonstream_B1", [0], cold=True, count=64, stream=False
            )
            expected_text = native["requests"][0]["output"]["text"]
            await gate.openai_completion(model_name, expected_text, stream=False)
            await gate.openai_completion(model_name, expected_text, stream=True)
            cold = await gate.case("cold_stream_B4", [3, 1, 2, 0], cold=True)
            if cold["max_running"] != 4:
                raise AssertionError("HTTP cold B4 never reached four running requests")
            await gate.case("warm_prefix_B4", [1, 3, 0, 2], prefix=True, count=64)
            soak_start, rounds, warm_rounds = time.monotonic(), 0, 0
            while (
                warm_rounds < options.min_rounds
                or time.monotonic() - soak_start < options.soak_seconds
            ):
                if rounds >= 30:
                    raise AssertionError(
                        "bounded soak did not reach the requested duration in 30 rounds"
                    )
                indices = [0, 1, 2, 3, 3, 2, 1, 0][:: -1 if rounds % 2 else 1]
                result = await gate.case(
                    f"queued_B8_round{rounds}", indices, prefix=True
                )
                if result["max_waiting"] < 1 or result["max_running"] != 4:
                    raise AssertionError(
                        "HTTP queued burst did not exercise a four-slot queue"
                    )
                warm_rounds += not result["compilation_observations"]["any_cache_miss"]
                rounds += 1
            report["soak"] = {
                "seconds": time.monotonic() - soak_start,
                "rounds": rounds,
                "warm_rounds": warm_rounds,
                "submitted_requests": rounds * 8,
            }
        await gate.case("http_abort_acknowledged", [0], abort=True)
        await gate.case("after_abort_slot_reuse", [0], prefix=True, count=64, logs=True)
        await gate.case("http_client_disconnect", [1], disconnect=True)
        await gate.case("after_disconnect_slot_reuse", [1], prefix=True, count=64)
        await gate.case("flush_and_recompute", [0], cold=True, count=64)
        await gate.flush()
        report["final_idle_state"] = await gate.idle()
        report["finished"] = True
        emit(
            {
                "event": "http_selected_gates_passed",
                "mode": report["mode"],
                "cases": len(report["cases"]),
                "soak": report.get("soak"),
            }
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30124)
    parser.add_argument("--soak-seconds", type=float, default=300)
    parser.add_argument("--min-rounds", type=int, default=3)
    parser.add_argument(
        "--lifecycle-only",
        action="store_true",
        help="fresh-server cancellation/disconnect/reuse subset; does not certify B4/B8 soak",
    )
    options = parser.parse_args()
    if options.soak_seconds < 0 or options.min_rounds < 1 or options.min_rounds > 30:
        raise ValueError("invalid bounded soak duration/round count")
    fixture = json.loads(options.serving_report.read_text())
    source = fingerprint()
    if not fixture["complete"] or fixture["framework_source_fingerprint"] != source:
        raise ValueError("requires complete same-source normal Engine/8K serving gate")
    if len(fixture["prompts"]) != 4 or any(len(p) != 7936 for p in fixture["prompts"]):
        raise ValueError("requires the four distinct long-prompt Engine fixture")
    if any(
        expected is None or len(expected) < 254
        for expected in fixture["expected_output_ids"]
    ):
        raise ValueError("every HTTP prompt needs a complete accepted Engine baseline")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", options.port))
    options.output.mkdir(parents=True, exist_ok=False)
    model_name = "v4-csa-http-" + uuid.uuid4().hex[:12]
    selected = options_from_receipt(fixture)
    command = server_command(
        fixture["server_args"]["model_path"],
        options.port,
        model_name,
        execution=selected,
    )
    report = {
        "complete": False,
        "finished": False,
        "scope": __doc__,
        "mode": "lifecycle_only" if options.lifecycle_only else "full",
        "framework_source_fingerprint": source,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "serving_report": str(options.serving_report.resolve()),
        "command": command,
        "server_log": str((options.output / "server.log").resolve()),
        "requested_soak_seconds": options.soak_seconds,
        **selected,
        "execution_helper_sha256": HELPER_SHA256,
        "events": [],
        "cases": [],
        "protocol_checks": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (options.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    process = None
    try:
        environment = os.environ.copy()
        environment.update(JAX_PLATFORMS="tpu", USE_DEVICE_TYPE="tpu")
        environment.pop("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", None)
        if environment.get("SGLANG_TEST_RETRACT"):
            raise ValueError("HTTP acceptance must not force test retraction")
        with (options.output / "server.log").open("x") as server_log:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            report["owned_server_pid"] = process.pid
            emit(
                {
                    "event": "http_server_start",
                    "pid": process.pid,
                    "host": "127.0.0.1",
                    "port": options.port,
                }
            )
            asyncio.run(run_http(options, fixture, report, emit, process, model_name))
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise
    finally:
        if process is not None:
            # start_new_session gives this owned server a separate process
            # group. Never terminate unrelated Python/TPU/user processes.
            forced = False
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                forced = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            report["server_cleanup"] = {
                "reaped": True,
                "returncode": process.returncode,
                "forced_kill": forced,
            }
        report["complete"] = report["finished"] and "error" not in report
        emit({"event": "http_test_finished", "acceptance_passed": report["complete"]})


if __name__ == "__main__":
    main()
