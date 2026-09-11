"""No inference: compilation evidence must follow the pinned logging schema."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from benchmark_deepseek_v4_release import compilation_evidence
from audit_deepseek_v4_release_benchmark import validate_metrics


def test_zero_misses_need_prefill_coverage():
    assert compilation_evidence("Prefill batch. #new-seq: 1, #new-token: 128") == []


@pytest.mark.parametrize("text", ["", "Cache flushed successfully!", "Decode batch."])
def test_empty_or_wrong_log_cannot_be_reported_compile_free(text):
    with pytest.raises(AssertionError, match="coverage"):
        compilation_evidence(text)


def test_positive_prefill_and_decode_misses_are_not_lost():
    text = "Prefill batch. #new-seq: 1\nPrefill batch. #bid: 0, #cache_miss: 3\nDecode batch. #cache_miss: 2"
    assert compilation_evidence(text) == [3, 2]


def synthetic_metrics():
    return {
        "completed": 1,
        "errors": [""],
        "input_lens": [128],
        "output_lens": [32],
        "cached_tokens": [0],
        "total_cached_tokens": 0,
        "total_input_tokens": 128,
        "total_output_tokens": 32,
        "ttfts": [0.5],
        "duration": 2.0,
        "mean_ttft_ms": 500.0,
        "mean_e2e_latency_ms": 1500.0,
        "mean_tpot_ms": 1000 / 31,
        "input_throughput": 64.0,
        "output_throughput": 16.0,
    }


def test_recompute_metrics():
    validate_metrics(synthetic_metrics(), 128, 1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("mean_ttft_ms", 400.0),
        ("mean_tpot_ms", 30.0),
        ("input_throughput", 128.0),
        ("total_cached_tokens", 128),
        ("output_lens", [31]),
    ],
)
def test_bad_metrics_rejected(key, value):
    result = synthetic_metrics()
    result[key] = value
    with pytest.raises(AssertionError):
        validate_metrics(result, 128, 1)
