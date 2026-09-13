"""Shared V4 execution choices; standard library only, no serving imports."""

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


def options_from_config(config):
    """Resolve checkpoint overrides using the same defaults as acceptance tools."""
    return validate_options(
        {
            name: getattr(config, "v4_" + name, default)
            for name, default in DEFAULTS.items()
        }
    )
