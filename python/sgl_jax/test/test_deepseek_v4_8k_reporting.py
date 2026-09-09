"""Acceptance-report timing and attribution must not overclaim measurements."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
with patch.object(sys, "path", [SCRIPTS, *sys.path]):
    import analyze_deepseek_v4_native_profile as profile
    from deepseek_v4_test_observer import drain_observer
    from run_deepseek_v4_8k_native import (
        PROMPT,
        measured_decode_batches,
        profile_schedule,
        prompts_for,
        summary,
    )
    from run_deepseek_v4_8k_serving import common_decode_window
    from run_deepseek_v4_framework import framework_fingerprint


@pytest.mark.parametrize("drain_timeout", [60, 0.001])
def test_observer_drain_preserves_engine_status_rpc(drain_timeout):
    from sgl_jax.srt.managers.tokenizer_manager import _Communicator

    class Sender:
        def __init__(self):
            self.sent = []

        def send_pyobj(self, value):
            self.sent.append(value)

    async def run():
        sender = Sender()
        channel = _Communicator(sender, 1)
        monitor = asyncio.create_task(channel("observed status"))
        await asyncio.sleep(0)
        draining = asyncio.create_task(drain_observer(monitor, timeout=drain_timeout))
        if drain_timeout < 1:
            with pytest.raises(TimeoutError):
                await draining
            assert not monitor.cancelled()
        else:
            await asyncio.sleep(0)
            assert not draining.done()
        channel.handle_recv("first reply")
        assert await monitor == ["first reply"]
        if drain_timeout >= 1:
            await draining
        next_query = asyncio.create_task(channel("idle status"))
        await asyncio.sleep(0)
        assert sender.sent == ["observed status", "idle status"]
        channel.handle_recv("idle reply")
        assert await next_query == ["idle reply"]
        assert channel._result_event is None
        assert not channel._ready_queue

    asyncio.run(run())


def source(module, function):
    line = next(start for start, _, name in profile.NATIVE_RANGES[module] if name == function)
    return f"/repo/kernels/deepseek_v4/{module}.py:{line}:0"


def test_warm_statistics_exclude_cold_and_profiled_calls():
    records = [
        {"seconds": 100, "cache_misses": 1, "tokens": 4},
        {"seconds": 20, "cache_misses": 0, "tokens": 4, "profiled": True},
        {"seconds": 0.01, "cache_misses": 0, "tokens": 4},
        {"seconds": 0.02, "cache_misses": 0, "tokens": 4},
    ]
    result = summary(records)
    assert result["warm_calls"] == 2
    assert result["warm_p50_ms"] == 15
    assert result["warm_p95_ms"] == pytest.approx(19.5)
    assert result["warm_tokens_per_second"] == pytest.approx(8 / 0.03)
    assert result["cold_or_cache_miss_seconds"] == 100


def test_no_warm_sample_does_not_invent_a_latency():
    result = summary([])
    assert result["warm_calls"] == 0
    assert result["warm_p50_ms"] is None
    assert result["warm_p95_ms"] is None
    assert result["warm_tokens_per_second"] is None


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("reuse_goldens", [False, True])
@pytest.mark.parametrize("cold_prefill", [False, True])
def test_reused_b1_profile_does_not_change_existing_gate_batches(
    profile, reuse_goldens, cold_prefill
):
    assert measured_decode_batches(
        profile=profile,
        reuse_goldens=reuse_goldens,
        cold_prefill=cold_prefill,
        profile_reused_b1=False,
    ) == (2, 4)
    if profile and reuse_goldens and not cold_prefill:
        assert measured_decode_batches(
            profile=profile,
            reuse_goldens=reuse_goldens,
            cold_prefill=cold_prefill,
            profile_reused_b1=True,
        ) == (1, 2, 4)
    else:
        with pytest.raises(ValueError, match="requires --profile and --reuse-goldens"):
            measured_decode_batches(
                profile=profile,
                reuse_goldens=reuse_goldens,
                cold_prefill=cold_prefill,
                profile_reused_b1=True,
            )


def test_boundary_trace_is_separate_and_leaves_a_warm_boundary_sample():
    plans = {i: profile_schedule(i, "B4", enabled=True, boundary=True) for i in range(256)}
    assert [i for i, value in plans.items() if value] == [127, 128, 129]
    assert plans[127] == {
        "label": "B4_boundary8063",
        "start": True,
        "stop": True,
        "kind": "r4_r128_boundary",
    }
    assert plans[128]["label"] == plans[129]["label"] == "B4"
    assert plans[128]["start"] and plans[129]["stop"]
    assert plans[255] is None
    assert profile_schedule(127, "B4", enabled=True) is None
    assert profile_schedule(128, "B4", enabled=False, boundary=True) is None


def test_common_decode_window_excludes_prefill_stalls():
    rows = [
        {
            "token_events": [
                {"elapsed": 1, "tokens": 1},
                {"elapsed": 3, "tokens": 2},
                {"elapsed": 4, "tokens": 3},
            ]
        },
        {
            "token_events": [
                {"elapsed": 2, "tokens": 1},
                {"elapsed": 3, "tokens": 2},
                {"elapsed": 4, "tokens": 3},
            ]
        },
    ]
    assert common_decode_window(rows) == {"seconds": 2, "tokens": 4, "tokens_per_second": 2}


def test_queued_groups_do_not_imply_simultaneous_decode():
    rows = [
        {"token_events": [{"elapsed": 1, "tokens": 1}, {"elapsed": 2, "tokens": 3}]},
        {"token_events": [{"elapsed": 3, "tokens": 1}, {"elapsed": 4, "tokens": 3}]},
    ]
    assert common_decode_window(rows) is None


def test_long_fixtures_are_distinct_and_exact_length():
    class Tokenizer:
        def encode(self, text, **_):
            return [1, 2, 3] if text.startswith("Geography") else [4, 5, 6]

    a, b = prompts_for(Tokenizer())
    assert len(a) == len(b) == PROMPT
    assert a[:128] != b[:128]
    variants = [a, b, a[1:] + a[:1], b[1:] + b[:1]]
    assert len({tuple(p[:128]) for p in variants}) == 4


def test_profile_fingerprint_matches_production_gate():
    assert profile.fingerprint() == framework_fingerprint()


def test_compressor_wins_over_outer_attention_projection():
    stack = source("attention", "attention") + "\n" + source("compressor", "compress")
    assert (
        profile.native_stage({"args": {"source_stack": stack}}, []) == "compressor_and_paged_state"
    )


def test_sparse_attention_wins_over_outer_projection():
    stack = source("attention", "attention") + "\n" + source("numerics", "_single_query_attention")
    assert profile.native_stage({"args": {"source_stack": stack}}, []) == "sparse_attention"


@pytest.mark.parametrize(
    "path,category,expected",
    [
        ("csa/compressor.py", "custom-call", "compressor_and_paged_state"),
        ("csa/joint_attention.py", "custom-call", "sparse_attention"),
        ("csa/indexer.py", "custom-call", "attention_index_scores"),
        ("dsa/streamindex_topk.py", "custom-call", "attention_index_scores"),
        ("dsa/streamindex_topk.py", "sort", "attention_topk"),
    ],
)
def test_original_csa_attribution_precedes_outer_projection(path, category, expected):
    stack = source("attention", "attention") + f"\n/repo/kernels/{path}:123:0"
    event = {"args": {"source_stack": stack, "hlo_category": category}}
    assert profile.native_stage(event, []) == expected


def test_fp4_kernel_attribution_includes_fused_dequant():
    event = {"args": {"source_stack": source("moe", "grouped_fp4_experts")}}
    assert profile.native_stage(event, []) == "routed_fp4_experts_including_online_dequant"


def test_collective_time_is_not_labeled_compute_utilization():
    event = {
        "args": {"source_stack": source("moe", "grouped_fp4_experts"), "hlo_category": "all-reduce"}
    }
    assert profile.native_stage(event, []) == "moe_all_reduce_including_wait"


def test_sampler_collectives_do_not_invalidate_layer_coverage():
    columns = ["category", "source_info", "occurrences"]
    table = {
        "cols": [{"id": name} for name in columns],
        "rows": [
            {"c": [{"v": value} for value in values]}
            for values in (
                ("all-reduce", source("moe", "grouped_fp4_experts"), 688),
                ("all-reduce", "/repo/sampler.py:31", 48),
            )
        ],
    }
    with patch.object(
        profile,
        "ORIGINAL_HLO_SUMMARY",
        return_value={"all_reduce_occurrences": 736, "expected_all_reduce_occurrences": 688},
    ):
        result = profile.native_hlo_summary(table, cores=8, steps=2, ranges=[])
    assert result["complete_collective_coverage"]
    assert result["moe_all_reduce_occurrences"] == 688
    assert result["additional_collective_occurrences"] == 48
    event = {"args": {"source_stack": "/repo/sampler.py:31", "hlo_category": "all-reduce"}}
    assert profile.native_stage(event, []) == "non_moe_collectives_including_wait"
