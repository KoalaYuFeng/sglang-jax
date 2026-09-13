"""Shared execution policy, source identity and diagnostic-index contracts."""

import ast
import hashlib
import importlib
import itertools
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from sgl_jax.srt.configs.deepseek_v4_execution import (
    DEFAULTS,
    options_from_config,
    validate_options,
)

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def adapters(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    return importlib.import_module("deepseek_v4_execution_options")


def test_runtime_and_cli_share_policy_but_preserve_legacy_defaults(adapters):
    assert adapters.DEFAULTS is DEFAULTS
    assert adapters.validate_options is validate_options
    assert options_from_config(SimpleNamespace()) == {
        "mhc_backend": "pallas",
        "hca_backend": "pallas",
        "csa_backend": "pallas",
        "moe_backend": "legacy",
        "fp8_backend": "legacy",
        "attention_tp": False,
        "csa_decode_batch": False,
        "fused_norm": False,
        "merged_projections": False,
        "fused_wo_a": False,
    }
    from sgl_jax.srt.layers.deepseek_v4.linear import DenseKernels

    assert asdict(DenseKernels()) == {
        name: DEFAULTS[name]
        for name in ("fp8_backend", "fused_norm", "merged_projections", "fused_wo_a")
    }


def test_all_backend_and_flag_combinations_keep_the_existing_contract(adapters):
    names = tuple(DEFAULTS)
    choices = [
        (
            (False, True)
            if type(DEFAULTS[name]) is bool
            else adapters.backend_choices(name)
        )
        for name in names
    ]
    for values in itertools.product(*choices):
        options = dict(zip(names, values, strict=True))
        allowed = (
            (not options["merged_projections"] or options["fp8_backend"] == "gmm")
            and (not options["csa_decode_batch"] or options["csa_backend"] == "pallas")
            and (
                not options["attention_tp"]
                or (
                    options["csa_backend"] == "pallas"
                    and options["hca_backend"] == "pallas"
                )
            )
        )
        config = SimpleNamespace(**{"v4_" + k: v for k, v in options.items()})
        if allowed:
            assert options_from_config(config) == options
            assert adapters.validate_options(options) == options
        else:
            with pytest.raises(ValueError):
                options_from_config(config)
            with pytest.raises(ValueError):
                adapters.validate_options(options)


@pytest.mark.parametrize(
    "name", [name for name, value in DEFAULTS.items() if type(value) is bool]
)
@pytest.mark.parametrize("value", ["true", "false", 0, 1, None])
def test_runtime_rejects_non_boolean_flags(name, value):
    with pytest.raises(ValueError, match="boolean"):
        options_from_config(SimpleNamespace(**{"v4_" + name: value}))


def test_model_initialization_consumes_shared_options(monkeypatch):
    from sgl_jax.srt.models import deepseek_v4 as model

    selected = {
        **DEFAULTS,
        "moe_backend": "gmm_tuned",
        "fp8_backend": "gmm",
        "attention_tp": True,
        "csa_decode_batch": True,
        "fused_norm": True,
        "merged_projections": True,
        "fused_wo_a": True,
    }
    config = SimpleNamespace(
        **{"v4_" + k: v for k, v in selected.items()},
        num_hidden_layers=1,
        v4_max_context=128,
        to_dict=lambda: {},
    )

    class ReachedArchitectureCheck(Exception):
        pass

    def stop_before_allocation(*args):
        raise ReachedArchitectureCheck

    monkeypatch.setattr(model, "config_for_layer", stop_before_allocation)
    holder = SimpleNamespace()
    with pytest.raises(ReachedArchitectureCheck):
        model.DeepseekV4ForCausalLM.__init__(holder, config, None)
    assert {name: getattr(holder, name) for name in DEFAULTS} == selected
    assert asdict(holder.dense_kernels) == {
        name: selected[name] for name in asdict(holder.dense_kernels)
    }


def test_helper_receipt_binds_shared_config(adapters):
    from sgl_jax.srt.configs import deepseek_v4_execution as policy

    expected = hashlib.sha256(
        Path(adapters.__file__).read_bytes()
        + b"\0"
        + Path(policy.__file__).read_bytes()
    ).hexdigest()
    assert expected == adapters.HELPER_SHA256


@pytest.mark.parametrize(
    "target",
    [
        "python/sgl_jax/srt/configs/deepseek_v4_execution.py",
        "python/sgl_jax/srt/kernels/low_bit/fp4.py",
        "python/sgl_jax/srt/layers/deepseek_v4/linear.py",
        "scripts/deepseek_v4_source.py",
    ],
)
def test_shared_fingerprint_detects_source_changes(monkeypatch, target):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    provenance = importlib.import_module("deepseek_v4_source")
    baseline = provenance.framework_fingerprint()
    read_bytes = Path.read_bytes

    def changed(path):
        data = read_bytes(path)
        return (
            data + b"\n# simulated change\n"
            if path.resolve() == REPO / target
            else data
        )

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert provenance.framework_fingerprint() != baseline


def test_offline_helpers_do_not_require_accelerator_or_serving_packages():
    program = f"""
import sys
sys.path[:0] = [{str(REPO / 'python')!r}, {str(REPO / 'scripts')!r}]
import deepseek_v4_execution_options as options
import analyze_deepseek_v4_native_profile as profile
import deepseek_v4_source as source
assert profile.fingerprint is source.framework_fingerprint
assert len(profile.fingerprint()) == 64
assert not {{'jax', 'flax', 'transformers', 'torch'}}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-S", "-c", program], check=True, timeout=30)


def test_diagnostic_archive_index_covers_existing_entry_points():
    path = REPO / "scripts/diagnostics/deepseek_v4/README.md"
    links = re.findall(r"\]\((\.\./\.\./[^)]+\.py)\)", path.read_text())
    indexed = {(path.parent / link).resolve() for link in links}
    expected = {
        source.resolve()
        for prefix in ("debug", "probe", "replay", "isolate")
        for source in (REPO / "scripts").glob(f"{prefix}_deepseek_v4*.py")
    }
    assert indexed == expected
    assert len(links) == len(indexed)
    for source in indexed:
        ast.parse(source.read_text())
