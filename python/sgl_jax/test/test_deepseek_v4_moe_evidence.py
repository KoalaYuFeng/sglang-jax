"""CPU-only guards against accepting marker strings or misattributing GMM."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
with patch.object(sys, "path", [SCRIPTS, *sys.path]):
    import analyze_deepseek_v4_native_profile as profile
    from deepseek_v4_moe_evidence import compiled_moe_evidence


def fixture():
    lines = ["ENTRY %main () {"]
    for layer in range(2):
        for projection, k, n in ((1, 4096, 2048), (3, 4096, 2048), (2, 2048, 4096)):
            name = f"gmm_checkpoint_fp4-g_64-m_24-k_{k}-n_{n}-tm_8.{layer}.{projection}"
            lines.append(f"  %{name} = bf16[24,{n}] custom-call()")
            lines.append(
                f"  %consumer.{layer}.{projection} = bf16[24,{n}] copy(%{name}), "
                f'metadata={{op_name="v4_layer_{layer}/gate"}}'
            )
    return "\n".join([*lines, "}"])


def test_every_projection_needs_actual_ssa_layer_ownership():
    result = compiled_moe_evidence(fixture(), layers=2)
    assert result["custom_call_instructions"] == 6
    assert result["per_layer_projection_counts"][1] == {"gate_up": 2, "down": 1}
    assert all(w["method"] == "scoped_consumer" for w in result["witnesses"])


def test_marker_in_metadata_is_not_a_kernel_call():
    text = fixture().replace("custom-call()", "copy()")
    with pytest.raises(AssertionError, match="every layer"):
        compiled_moe_evidence(text, layers=2)


@pytest.mark.parametrize(
    "shape", ["2048,2048", "4096,1024", "2048,128", "4096,64", "128,2048", "64,4096"]
)
def test_full_expert_slices_are_rejected(shape):
    text = fixture().replace("\n}", f"\n  %slice = u8[1,{shape}] dynamic-slice(%weights)\n}}")
    with pytest.raises(AssertionError, match="weight/scale slices remain"):
        compiled_moe_evidence(text, layers=2)


def test_same_total_call_count_cannot_hide_a_missing_layer():
    text = fixture().replace("v4_layer_1/", "v4_layer_0/")
    with pytest.raises(AssertionError, match="every layer"):
        compiled_moe_evidence(text, layers=2)


def test_tuned_backend_requires_its_actual_adapter_on_every_projection():
    text = fixture().replace(
        "gmm_checkpoint_fp4-", "gmm_checkpoint_fp4_candidate_scale_kn_packed_scale-"
    )
    result = compiled_moe_evidence(text, layers=2, backend="gmm_tuned")
    assert result["backend"] == "gmm_tuned"
    assert result["custom_call_instructions"] == 6
    assert result["full_scale_layout_copies"] == []
    with pytest.raises(AssertionError):
        compiled_moe_evidence(text, layers=2, backend="gmm")
    with pytest.raises(AssertionError):
        compiled_moe_evidence(fixture(), layers=2, backend="gmm_tuned")
    mixed = text.replace(
        "gmm_checkpoint_fp4_candidate_scale_kn_packed_scale-", "gmm_checkpoint_fp4-", 1
    )
    with pytest.raises(AssertionError):
        compiled_moe_evidence(mixed, layers=2, backend="gmm_tuned")


@pytest.mark.parametrize("shape", ["4096,64", "64,4096", "2048,128", "128,2048"])
def test_tuned_backend_cannot_hide_full_scale_layout_copies(shape):
    text = fixture().replace(
        "gmm_checkpoint_fp4-", "gmm_checkpoint_fp4_candidate_scale_kn_packed_scale-"
    )
    text = text.replace("\n}", f"\n  %copy = u8[64,{shape}] copy(%scales)\n}}")
    with pytest.raises(AssertionError, match="scale.*copies"):
        compiled_moe_evidence(text, layers=2, backend="gmm_tuned")


@pytest.mark.parametrize("source", ["/deepseek_v4/moe_gmm.py:83", "/kernels/low_bit/gmm.py:81"])
def test_shared_gmm_attribution_precedes_the_outer_moe_wrapper(source):
    event = {
        "args": {
            "source_stack": source + "\n/deepseek_v4/moe.py:112",
            "hlo_category": "custom-call",
        }
    }
    assert profile.native_stage(event, []) == "routed_fp4_experts_including_online_dequant"


def test_gmm_collective_remains_moe_not_generic_communication():
    event = {"args": {"source_stack": "/deepseek_v4/moe_gmm.py:93", "hlo_category": "all-reduce"}}
    assert profile.native_stage(event, []) == "moe_all_reduce_including_wait"


def test_profile_and_runtime_fingerprints_cover_the_same_sources():
    with patch.object(sys, "path", [SCRIPTS, *sys.path]):
        from run_deepseek_v4_framework import framework_fingerprint

    assert profile.fingerprint() == framework_fingerprint()
