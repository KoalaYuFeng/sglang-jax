"""Bind V4 numerical, Engine and HTTP receipts to the same execution choices.

This is test/acceptance plumbing, not a serving scheduler or model default.
"""

import hashlib
import json
from pathlib import Path

from sgl_jax.srt.configs import deepseek_v4_execution as execution
from sgl_jax.srt.configs.deepseek_v4_execution import (
    DEFAULTS,
    backend_choices,
    validate_options,
)

# Bind receipt plumbing to its shared policy as well as this CLI adapter.
HELPER_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes() + b"\0" + Path(execution.__file__).read_bytes()
).hexdigest()


def add_execution_arguments(parser):
    for name in (
        "mhc_backend",
        "hca_backend",
        "csa_backend",
        "moe_backend",
        "fp8_backend",
    ):
        choices = backend_choices(name)
        parser.add_argument(
            "--" + name.replace("_", "-"), choices=choices, default=DEFAULTS[name]
        )
    parser.add_argument(
        "--attention-tp",
        action="store_true",
        help="head-TP4 for CSA/HCA; SWA remains replicated",
    )
    parser.add_argument(
        "--csa-decode-batch",
        action="store_true",
        help="batched original GEMV for pure decode only",
    )
    for name in ("fused_norm", "merged_projections", "fused_wo_a"):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            action="store_true",
            help="experimental V4 kernel; requires separate numerical acceptance",
        )


def options_from_namespace(options):
    return validate_options(
        {name: getattr(options, name, default) for name, default in DEFAULTS.items()}
    )


def model_overrides(options):
    return {"v4_" + name: value for name, value in validate_options(options).items()}


def options_from_receipt(receipt):
    """An absent old option means its old default, never an inferred speedup.

    ServerArgs.moe_backend='epmoe' selects framework dispatch, not the V4
    expert compute implementation. Only json_model_override_args selects GMM.
    Reject a top-level claim inconsistent with what its worker was launched
    with instead of silently creating a different Engine/HTTP configuration.
    """
    raw = receipt["server_args"].get("json_model_override_args", "{}") or "{}"
    overrides = json.loads(raw)
    if not isinstance(overrides, dict):
        raise TypeError("model overrides must be a JSON object")
    selected = {
        name: overrides.get("v4_" + name, default) for name, default in DEFAULTS.items()
    }
    for name, value in selected.items():
        if name in receipt and (
            type(receipt[name]) is not type(value) or receipt[name] != value
        ):
            raise ValueError(f"receipt contradicts launched {name}")
    return validate_options(selected)


def assert_runner_options(runner, selected):
    actual = {
        name: getattr(runner.model, name, default) for name, default in DEFAULTS.items()
    }
    actual = validate_options(actual)
    if actual != validate_options(selected):
        raise AssertionError(
            {"requested_execution": selected, "actual_execution": actual}
        )
    return actual


def assert_server_options(info, selected):
    actual = options_from_receipt({"server_args": info})
    if actual != validate_options(selected):
        raise AssertionError(
            {"requested_execution": selected, "reported_server_execution": actual}
        )
    return actual
