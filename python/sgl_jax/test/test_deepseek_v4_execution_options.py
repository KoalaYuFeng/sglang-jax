"""Acceptance receipts must not silently fall back to a different MoE/TP path."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from deepseek_v4_execution_options import (
    DEFAULTS,
    add_execution_arguments,
    assert_runner_options,
    assert_server_options,
    model_overrides,
    options_from_namespace,
    options_from_receipt,
    validate_options,
)


def test_legacy_missing_overrides_do_not_imply_optimized_execution():
    assert options_from_receipt({"server_args": {"moe_backend": "epmoe"}}) == DEFAULTS


def test_native_parser_and_reported_server_use_the_same_explicit_options():
    parser = argparse.ArgumentParser()
    add_execution_arguments(parser)
    selected = options_from_namespace(
        parser.parse_args(["--moe-backend", "gmm", "--attention-tp", "--csa-decode-batch"])
    )
    assert selected == {
        **DEFAULTS,
        "moe_backend": "gmm",
        "attention_tp": True,
        "csa_decode_batch": True,
    }
    info = {"json_model_override_args": json.dumps(model_overrides(selected))}
    assert assert_server_options(info, selected) == selected
    with pytest.raises(AssertionError, match="reported_server_execution"):
        assert_server_options({}, selected)


@pytest.mark.parametrize("backend", ["gmm", "gmm_tuned"])
def test_full_selection_roundtrips_through_launched_server_args(backend):
    selected = {**DEFAULTS, "moe_backend": backend, "attention_tp": True, "csa_decode_batch": True}
    receipt = {
        **selected,
        "server_args": {"json_model_override_args": json.dumps(model_overrides(selected))},
    }
    assert options_from_receipt(receipt) == selected
    runner = SimpleNamespace(model=SimpleNamespace(**selected))
    assert assert_runner_options(runner, selected) == selected


def test_tuned_fp4_selection_does_not_change_defaults_or_fp8_choices():
    parser = argparse.ArgumentParser()
    add_execution_arguments(parser)
    assert options_from_namespace(parser.parse_args([])) == DEFAULTS
    assert DEFAULTS["moe_backend"] == "legacy"
    selected = options_from_namespace(parser.parse_args(["--moe-backend", "gmm_tuned"]))
    assert selected == {**DEFAULTS, "moe_backend": "gmm_tuned"}
    with pytest.raises(ValueError):
        validate_options({**DEFAULTS, "fp8_backend": "gmm_tuned"})


@pytest.mark.parametrize(
    "name,value", [("moe_backend", "gmm"), ("attention_tp", True), ("csa_decode_batch", True)]
)
def test_receipt_rejects_optimized_claim_without_launched_selection(name, value):
    with pytest.raises(ValueError, match="contradicts"):
        options_from_receipt({name: value, "server_args": {}})


@pytest.mark.parametrize("value", ["true", 1, None])
def test_boolean_flags_do_not_accept_truthy_strings_or_integers(value):
    with pytest.raises(ValueError, match="boolean"):
        validate_options({**DEFAULTS, "attention_tp": value})


def test_batched_projection_requires_pallas_and_worker_must_match():
    with pytest.raises(ValueError, match="Pallas"):
        validate_options({**DEFAULTS, "csa_backend": "reference", "csa_decode_batch": True})
    with pytest.raises(AssertionError, match="actual_execution"):
        assert_runner_options(
            SimpleNamespace(model=SimpleNamespace(**DEFAULTS)), {**DEFAULTS, "moe_backend": "gmm"}
        )


def test_dense_kernel_options_roundtrip_and_fail_closed():
    parser = argparse.ArgumentParser()
    add_execution_arguments(parser)
    selected = options_from_namespace(
        parser.parse_args(
            [
                "--fp8-backend",
                "gmm",
                "--fused-norm",
                "--merged-projections",
                "--fused-wo-a",
            ]
        )
    )
    receipt = {"server_args": {"json_model_override_args": json.dumps(model_overrides(selected))}}
    assert options_from_receipt(receipt) == selected
    assert_runner_options(SimpleNamespace(model=SimpleNamespace(**selected)), selected)
    with pytest.raises(ValueError, match="FP8 GMM"):
        validate_options({**DEFAULTS, "merged_projections": True})
    for name in ("fused_norm", "merged_projections", "fused_wo_a"):
        with pytest.raises(ValueError, match="boolean"):
            validate_options({**DEFAULTS, name: "true"})
        with pytest.raises(ValueError, match="contradicts"):
            options_from_receipt({"server_args": {}, name: True})
