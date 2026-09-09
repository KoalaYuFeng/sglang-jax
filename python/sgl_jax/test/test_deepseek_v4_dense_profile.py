"""CPU-only contracts for the opt-in full-model dense-kernel A/B experiment."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from profile_deepseek_v4_dense import (
    BASE_OPTIONS,
    capture_plan,
    validate_compatibility,
    validate_control,
    warm_summary,
)
from profile_deepseek_v4_fp4_model import CONTROL_OPTIONS, FIXTURE_OPTIONS


def fixtures(tmp_path):
    checkpoint, native_path = tmp_path / "checkpoint", tmp_path / "native.json"
    native = {
        "complete": True,
        "finished": True,
        "checkpoint": str(checkpoint),
        "framework_source_fingerprint": "historical",
        "source_fingerprint": "reference",
        "prompt_tokens": 7936,
        "generation_tokens": 256,
        "server_args": {
            "json_model_override_args": json.dumps({"v4_" + k: v for k, v in BASE_OPTIONS.items()})
        },
    }
    oracle = {
        "complete": True,
        "finished": True,
        "checkpoint": str(checkpoint),
        "framework_source_fingerprint": "historical",
        "reference_source_fingerprint": "reference",
        "native_report": str(native_path),
        "checks": [
            {"case": case, "position": p, "all_finite": True, "top1_equal": True, "nrmse": 0}
            for case in range(2)
            for p in range(7935, 8192)
        ],
    }
    return (
        native,
        oracle,
        dict(checkpoint=checkpoint, native_path=native_path, reference="reference"),
    )


def test_historical_fixtures_are_allowed_without_claiming_same_source(tmp_path):
    native, oracle, args = fixtures(tmp_path)
    validate_compatibility(native, oracle, **args)


@pytest.mark.parametrize(
    "owner,field,value",
    [
        ("native", "complete", False),
        ("oracle", "finished", False),
        ("native", "source_fingerprint", "changed"),
        ("oracle", "framework_source_fingerprint", "unrelated"),
        ("oracle", "reference_source_fingerprint", "changed"),
        ("oracle", "native_report", "/other"),
        ("native", "generation_tokens", 128),
        ("native", "numerical_failures", [1]),
        ("oracle", "checks", []),
    ],
)
def test_compatibility_rejects_invalid_receipts(tmp_path, owner, field, value):
    native, oracle, args = fixtures(tmp_path)
    (native if owner == "native" else oracle)[field] = value
    with pytest.raises(ValueError):
        validate_compatibility(native, oracle, **args)


@pytest.mark.parametrize(
    "field,value",
    [("top1_equal", False), ("all_finite", False), ("nrmse", 0.006), ("nrmse", float("nan"))],
)
def test_oracle_failure_cannot_be_hidden_by_complete_flag(tmp_path, field, value):
    native, oracle, args = fixtures(tmp_path)
    oracle["checks"][0][field] = value
    with pytest.raises(ValueError):
        validate_compatibility(native, oracle, **args)


def test_exact_capture_windows_and_unprofiled_first_round():
    assert capture_plan(decode=True, round_id=0, batch=4, index=128) is None
    assert capture_plan(decode=False, round_id=1, case=0, index=56)[1:3] == (True, False)
    assert capture_plan(decode=False, round_id=1, case=0, index=57)[1:3] == (False, True)
    assert capture_plan(decode=True, round_id=1, batch=4, index=127)[1:3] == (True, True)
    assert capture_plan(decode=True, round_id=1, batch=2, index=127) is None
    assert capture_plan(decode=True, round_id=1, batch=4, index=128)[0] == "B4"
    assert capture_plan(decode=False, round_id=1, case=1, index=56) is None


def test_warm_stats_exclude_compile_profiler_and_warmup_but_retain_outliers():
    common = {"cache_misses": 0, "tokens": 4, "seconds": 0.1}
    records = [
        common,
        {**common, "seconds": 1.0},
        {**common, "seconds": 10, "cache_misses": 1},
        {**common, "seconds": 10, "profiled": True},
        {**common, "seconds": 10, "warmup": True},
    ]
    result = warm_summary(records)
    assert result["all_calls"] == 5 and result["warm_calls"] == 2
    assert result["warm_tokens_per_second"] == pytest.approx(8 / 1.1)
    assert result["warmup_calls"] == result["profiled_calls"] == result["cache_miss_calls"] == 1


def test_control_only_allows_dense_option_changes(tmp_path):
    control, _, _ = fixtures(tmp_path)
    control.update(
        {
            k: "same"
            for k in (
                "script_sha256",
                "execution_helper_sha256",
                "fixture_sha256",
                "rounds",
                "devices",
                "jax_version",
            )
        }
    )
    report = copy.deepcopy(control)
    report["server_args"]["json_model_override_args"] = "{}"
    validate_control(control, report)
    for key in ("framework_source_fingerprint", "devices", "rounds", "fixture_sha256"):
        bad = copy.deepcopy(report)
        bad[key] = "different"
        with pytest.raises(ValueError):
            validate_control(control, bad)
    bad = copy.deepcopy(report)
    bad["server_args"]["context_length"] = 4096
    with pytest.raises(ValueError):
        validate_control(control, bad)


def test_fp4_profile_uses_the_dense_enabled_control_without_changing_dense_defaults(tmp_path):
    native, oracle, args = fixtures(tmp_path)
    assert CONTROL_OPTIONS == {
        **BASE_OPTIONS,
        "fp8_backend": "gmm",
        "fused_norm": True,
        "merged_projections": True,
        "fused_wo_a": True,
    }
    assert BASE_OPTIONS["fp8_backend"] == "legacy"
    assert FIXTURE_OPTIONS == {**CONTROL_OPTIONS, "moe_backend": "gmm_tuned"}
    with pytest.raises(ValueError):
        validate_compatibility(native, oracle, **args, base_options=CONTROL_OPTIONS)
    native["server_args"]["json_model_override_args"] = json.dumps(
        {"v4_" + k: v for k, v in CONTROL_OPTIONS.items()}
    )
    validate_compatibility(native, oracle, **args, base_options=CONTROL_OPTIONS)
    with pytest.raises(ValueError):
        validate_compatibility(native, oracle, **args)
    with pytest.raises(ValueError):
        validate_compatibility(native, oracle, **args, base_options=FIXTURE_OPTIONS)
    native["server_args"]["json_model_override_args"] = json.dumps(
        {"v4_" + k: v for k, v in FIXTURE_OPTIONS.items()}
    )
    validate_compatibility(native, oracle, **args, base_options=FIXTURE_OPTIONS)
    with pytest.raises(ValueError):
        validate_compatibility(native, oracle, **args, base_options=CONTROL_OPTIONS)
