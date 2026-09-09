"""Bind V4 numerical, Engine and HTTP receipts to the same execution choices.

This is test/acceptance plumbing, not a serving scheduler or model default.
"""

import hashlib
import json
from pathlib import Path

HELPER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

DEFAULTS = {
    "mhc_backend": "pallas",
    "hca_backend": "pallas",
    "csa_backend": "pallas",
    "moe_backend": "legacy",
    "attention_tp": False,
    "csa_decode_batch": False,
    "fp8_backend": "legacy",
    "fused_norm": False,
    "merged_projections": False,
    "fused_wo_a": False,
}


def backend_choices(name):
    if name == "moe_backend":
        return ("legacy", "gmm", "gmm_tuned")
    if name == "fp8_backend":
        return ("legacy", "gmm")
    return ("pallas", "reference")


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


def validate_options(options):
    if set(options) != set(DEFAULTS):
        raise ValueError("V4 execution options require all explicit selections")
    for name, value in options.items():
        if name in (
            "attention_tp",
            "csa_decode_batch",
            "fused_norm",
            "merged_projections",
            "fused_wo_a",
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        elif value not in backend_choices(name):
            raise ValueError(f"unsupported {name}: {value!r}")
    if options["merged_projections"] and options["fp8_backend"] != "gmm":
        raise ValueError("merged V4 projections require the checkpoint FP8 GMM backend")
    if options["csa_decode_batch"] and options["csa_backend"] != "pallas":
        raise ValueError("batched CSA decode requires the original Pallas projection")
    if options["attention_tp"] and any(
        options[name] != "pallas" for name in ("csa_backend", "hca_backend")
    ):
        raise ValueError(
            "attention TP requires the independently validated Pallas paths"
        )
    return dict(options)


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
