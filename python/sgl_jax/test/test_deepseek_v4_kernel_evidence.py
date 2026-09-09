"""Compiled kernel coverage requires SSA evidence, not ordinal call counts."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
with patch.object(sys, "path", [SCRIPTS, *sys.path]):
    from deepseek_v4_kernel_evidence import _hlo_graph, _owner, compiled_kernel_evidence


def fixture():
    lines = ["ENTRY %main () {"]
    for name in (
        "mhc-collapse-pre",
        "mhc-sinkhorn-gates",
        "csa-compressor-project-v4",
        "StreamIdxTC-v4",
    ):
        lines.append(
            f'  %{name} = f32[8,128] custom-call(), metadata={{op_name="v4_layer_2/compute"}}'
        )
    for name in (
        "csa-compressor-snapshot-d512-v4",
        "csa-compressor-snapshot-d128-v4",
        "csa-joint-attention-v4",
    ):
        lines.extend(
            [
                f'  %{name} = f32[8,128] custom-call(), metadata={{op_name="{name}/pallas_call"}}',
                f"  %{name}.copy = f32[8,128] copy(%{name})",
                f'  %{name}.use = f32[8,128] negate(%{name}.copy), metadata={{op_name="v4_layer_2/convert"}}',
            ]
        )
    return "\n".join([*lines, "}"])


def evidence(text):
    return compiled_kernel_evidence(
        text,
        mhc_backend="reference",
        hca_backend="reference",
        csa_backend="pallas",
        ratios=[0, 0, 4],
    )


def test_lost_layer_metadata_requires_scoped_consumer_witness():
    result = evidence(fixture())
    assert all(layers == [2] for layers in result["csa_layer_coverage"].values())
    witness = result["csa_layer_witnesses"]["csa_main_emission"][0]
    assert witness["method"] == "scoped_consumer"
    assert len(witness["path"]) == 3
    assert witness["consumer_op_name"] == "v4_layer_2/convert"


def test_dense_selection_requires_real_compiled_calls():
    options = dict(
        mhc_backend="reference",
        hca_backend="reference",
        csa_backend="pallas",
        ratios=[0, 0, 4],
        fp8_backend="gmm",
        fused_norm=True,
        fused_wo_a=True,
    )
    with pytest.raises(AssertionError, match="dense kernels"):
        compiled_kernel_evidence(fixture(), **options)
    extra = "\n".join(
        f'  %{name}.{i} = bf16[8,128] custom-call(), metadata={{op_name="v4_layer_{i}/compute"}}'
        for name in (
            "gmm_checkpoint_fp8",
            "v4_exact_rms_norm",
            "v4_exact_qnorm_rope",
            "v4_inverse_rope_checkpoint_fp8_wo_a",
        )
        for i in range(3)
    )
    result = compiled_kernel_evidence(fixture().replace("\n}", "\n" + extra + "\n}"), **options)
    assert result["compiled_custom_call_counts"]["qnorm_rope"] == 3


def test_kernel_marker_in_metadata_does_not_count_as_an_instruction():
    text = fixture().replace("%csa-compressor-project-v4 =", "%unrelated =")
    text = text.replace("v4_layer_2/compute", "v4_layer_2/csa-compressor-project-v4/compute")
    with pytest.raises(AssertionError, match="missing layers"):
        evidence(text)


def test_ambiguous_nearest_consumers_fail_closed():
    text = fixture().replace(
        "\n}",
        '\n  %other = f32[8,128] negate(%csa-compressor-snapshot-d512-v4.copy), metadata={op_name="v4_layer_4/convert"}\n}',
    )
    with pytest.raises(AssertionError, match="ambiguous"):
        evidence(text)


def test_tuple_aggregation_cannot_supply_layer_ownership():
    text = """ENTRY %main () {
  %kernel = f32[8,128] custom-call()
  %all = (f32[8,128]) tuple(%kernel)
  %value = f32[8,128] get-tuple-element(%all), metadata={op_name="v4_layer_2/compute"}
}
"""
    nodes, consumers = _hlo_graph(text)
    with pytest.raises(AssertionError, match="cannot establish"):
        _owner(("main", "kernel"), nodes, consumers)


def test_reused_ssa_names_do_not_cross_computation_boundaries():
    text = """%first () {
  %kernel = f32[8,128] custom-call()
}
ENTRY %second () {
  %kernel = f32[8,128] custom-call()
  %value = f32[8,128] copy(%kernel), metadata={op_name="v4_layer_2/compute"}
}
"""
    nodes, consumers = _hlo_graph(text)
    assert _owner(("second", "kernel"), nodes, consumers)["layer"] == 2
    with pytest.raises(AssertionError, match="cannot establish"):
        _owner(("first", "kernel"), nodes, consumers)


def test_parallel_claim_requires_local_head_and_batched_projection_instructions():
    def run(text, **kwargs):
        return compiled_kernel_evidence(
            text,
            mhc_backend="reference",
            hca_backend="reference",
            csa_backend="pallas",
            ratios=[0, 0, 4],
            **kwargs,
        )

    with pytest.raises(AssertionError, match="actual h16"):
        run(fixture(), attention_tp=True)
    with pytest.raises(AssertionError, match="batched main/index"):
        run(fixture(), csa_decode_batch=True)
    text = fixture().replace("csa-joint-attention-v4", "csa-joint-attention-h16-d512-v4")
    text = text.replace(
        "csa-compressor-project-v4", "csa-compressor-project-decode-batched-main-v4"
    )
    text = text.replace(
        "\n}",
        '\n  %csa-compressor-project-decode-batched-index-v4 = f32[4,512] custom-call(), metadata={op_name="v4_layer_2/index"}\n}',
    )
    result = run(text, attention_tp=True, csa_decode_batch=True)
    assert result["attention_tp"] and result["csa_decode_batch"]
    assert result["compiled_custom_call_counts"]["csa_projection"] == 2
