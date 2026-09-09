"""HTTP acceptance must validate content, resource return and honest timing."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
with patch.object(sys, "path", [SCRIPTS, *sys.path]):
    from deepseek_v4_execution_options import DEFAULTS, model_overrides
    from run_deepseek_v4_http_stress import (
        HttpGate,
        compilation_observations,
        server_command,
        streaming_metrics,
        validate_idle,
        validate_output,
    )


def output(ids=(11, 12)):
    return {
        "output_ids": list(ids),
        "text": "ok",
        "meta_info": {
            "completion_tokens": len(ids),
            "prompt_tokens": 3,
            "cached_tokens": 0,
            "finish_reason": {"type": "length", "length": len(ids)},
        },
    }


def test_stream_metrics_do_not_invent_per_token_observations():
    events = [{"elapsed": 1, "tokens": 3}, {"elapsed": 2, "tokens": 5}, {"elapsed": 3, "tokens": 6}]
    metrics = streaming_metrics(events, 4)
    assert metrics["ttft_seconds"] == 1
    assert metrics["decode_tpot_ms"] == pytest.approx(2000 / 3)
    assert metrics["max_coalesced_token_delta"] == 3
    assert metrics["stream_interval_p50_ms_per_token"] == 750
    assert streaming_metrics(events[:1], 4)["decode_tpot_ms"] is None


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"elapsed": 2, "tokens": 2}, {"elapsed": 1, "tokens": 3}],
        [{"elapsed": 1, "tokens": 2}, {"elapsed": 2, "tokens": 2}],
    ],
)
def test_invalid_stream_observations_fail(events):
    with pytest.raises(ValueError):
        streaming_metrics(events, 3)


def test_token_comparison_requires_ids_count_and_finish():
    validate_output(output(), [11, 12], 2, 3)
    for ids in ([11], [11, 13]):
        with pytest.raises(AssertionError):
            validate_output(output(ids), [11, 12], 2, 3)
    bad = output()
    bad["meta_info"]["finish_reason"] = {"type": "abort"}
    with pytest.raises(AssertionError):
        validate_output(bad, [11, 12], 2, 3)


def test_idle_requires_capacity_and_all_request_slots_returned():
    state = {
        "req_to_token_pool_used": 0,
        "waiting_queue_size": 0,
        "running_batch_size": 0,
        "available_kv_tokens": 128,
        "tree_cache_size": 256,
    }
    assert validate_idle([state], 384)
    assert not validate_idle([{**state, "waiting_queue_size": 1}], 384)
    with pytest.raises(AssertionError):
        validate_idle([{**state, "available_kv_tokens": 0}], 384)


def test_launch_is_loopback_pallas_and_has_no_overlap_disable():
    command = server_command("/pinned/checkpoint", 30124, "owned-model")
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--disable-overlap-schedule" not in command
    assert "--disable-radix-cache" not in command
    assert json.loads(command[-1]) == model_overrides(DEFAULTS)


def test_http_launch_preserves_accepted_gmm_tp_and_decode_batch_options():
    selected = {**DEFAULTS, "moe_backend": "gmm", "attention_tp": True, "csa_decode_batch": True}
    command = server_command("/pinned/checkpoint", 30124, "owned-model", execution=selected)
    assert command[command.index("--moe-backend") + 1] == "epmoe"
    assert json.loads(command[command.index("--json-model-override-args") + 1]) == model_overrides(
        selected
    )


def test_compilation_uses_entire_case_log_not_final_response_counter():
    assert compilation_observations("prefill #cache_miss: 2\ndecode #cache_miss: 0") == {
        "logged_misses": [2, 0],
        "any_cache_miss": True,
    }
    assert compilation_observations("Decode batch. #running-req: 4") == {
        "logged_misses": [],
        "any_cache_miss": False,
    }


def test_cold_observer_budget_does_not_change_final_idle_budget():
    async def run():
        observed = []

        def handle(request):
            observed.append(request.extensions["timeout"]["read"])
            return httpx.Response(200, json={"internal_states": []})

        async with httpx.AsyncClient(
            base_url="http://127.0.0.1", transport=httpx.MockTransport(handle)
        ) as client:
            gate = HttpGate(client, {"prompts": [], "expected_output_ids": []}, {}, lambda _: None)
            await gate.states(timeout=1800)
            await gate.states()
        assert observed == [1800, 60]

    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, "missing_done", "error", "bad_ids"])
def test_http_sse_requires_done_and_correct_outputs(failure):
    final = output([11, 13] if failure == "bad_ids" else [11, 12])
    body = "data: " + json.dumps(final) + "\n\n"
    if failure == "error":
        body = 'data: {"error": {"message": "failed"}}\n\n'
    if failure != "missing_done":
        body += "data: [DONE]\n\n"

    async def execute():
        transport = httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
        )
        async with httpx.AsyncClient(base_url="http://127.0.0.1", transport=transport) as client:
            gate = HttpGate(
                client,
                {"prompts": [[1, 2, 3]], "expected_output_ids": [[11, 12]]},
                {},
                lambda _: None,
            )
            return await gate.request(0, count=2)

    if failure:
        with pytest.raises(AssertionError):
            asyncio.run(execute())
    else:
        assert asyncio.run(execute())["passed"]
