"""CPU-only accounting checks; these do not claim TPU performance coverage."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analyze_deepseek_v4_fp4_moe import stage, summarize, tensorcore_timelines  # noqa: E402
from analyze_deepseek_v4_native_profile import native_hlo_summary, native_stage  # noqa: E402


def row(name="other", category="loop fusion", source="", operation="", time=1000, count=1):
    return dict(
        hlo_op_name=name,
        category=category,
        source_info=source,
        tf_op_name=operation,
        total_self_time=time,
        occurrences=count,
    )


class AccountingTest(unittest.TestCase):
    def test_dense_gmm_is_not_misclassified_as_routed_moe(self):
        value = row("gmm_checkpoint_fp8-g_1", source="/kernels/gmm/megablox_gmm_kernel/gmm.py:600")
        self.assertEqual(stage(value, isolated=False), "fp8_projection_including_online_dequant")
        self.assertEqual(
            stage(row("v4_exact_qnorm_rope.1"), isolated=False), "v4_fused_normalization_and_rope"
        )

    def test_dense_gmm_metadata_uses_its_actual_scope(self):
        value = row(
            "pad.1",
            source="/kernels/gmm/megablox_gmm_kernel/gmm.py:600",
            operation="jit(gmm)/gmm_checkpoint_fp8-g_1/pallas_call/pad",
        )
        self.assertEqual(stage(value, isolated=False), "fp8_gmm_metadata_padding_and_glue")

    def test_empty_sparsecore_planes_are_not_tensorcores(self):
        value = {"cores": [{"name": "/device:TPU:0"}, {"name": "/device:TPU:0 SparseCore 0"}]}
        self.assertEqual(tensorcore_timelines(value), [{"name": "/device:TPU:0"}])

    def test_call_has_precedence_over_outer_source(self):
        value = row(
            "gmm_checkpoint_fp4-g_64-m_8-k_2048-n_4096-tm_8", source="/deepseek_v4/moe_gmm.py:81"
        )
        self.assertEqual(stage(value, isolated=False), "moe_down_gmm_including_conversion")

    def test_named_projection(self):
        value = row("gmm_checkpoint_fp4-g_64", operation="jit/DIAG_W1_GATE/pallas")
        self.assertEqual(stage(value, isolated=True), "moe_gate_gmm_including_conversion")

    def test_collective_does_not_hide_wait(self):
        value = row(category="all-reduce", source="/deepseek_v4/moe_gmm.py:90")
        self.assertEqual(stage(value, isolated=False), "moe_all_reduce_including_wait")

    def test_attention_gather_is_communication_not_projection_or_moe(self):
        value = row(category="all-gather", source="/deepseek_v4/attention.py:100")
        self.assertEqual(stage(value, isolated=False), "attention_tp_all_gather_including_wait")
        value["source_info"] = "/models/deepseek_v4.py:100"
        self.assertEqual(stage(value, isolated=False), "non_moe_collectives_including_wait")

    def test_new_gathers_are_counted_without_claiming_moe_coverage(self):
        records = [
            row(category="all-gather", source="/deepseek_v4/attention.py:100", count=41),
            row(category="all-gather", source="/models/deepseek_v4.py:100", count=1),
        ]
        columns = list(records[0])
        table = {
            "cols": [{"id": k} for k in columns],
            "rows": [{"c": [{"v": r[k]} for k in columns]} for r in records],
        }
        with patch("analyze_deepseek_v4_profile.event_stage", native_stage):
            result = native_hlo_summary(table, cores=1, steps=1, ranges=[])
        self.assertEqual(result["attention_tp_all_gather_occurrences"], 41)
        self.assertEqual(result["other_all_gather_occurrences"], 1)
        self.assertEqual(result["additional_collective_occurrences"], 42)
        self.assertFalse(result["complete_collective_coverage"])
        detailed = summarize(table, cores=1, steps=1, isolated=False)
        self.assertEqual(detailed["total_hlo_self_ms"], 2)
        self.assertEqual(detailed["stage_ms"]["attention_tp_all_gather_including_wait"], 1)
        self.assertEqual(detailed["stage_ms"]["non_moe_collectives_including_wait"], 1)
        self.assertEqual(sum(detailed["stage_ms"].values()), detailed["total_hlo_self_ms"])

    def test_shared_route_vs_metadata(self):
        for source, expected in (
            ("/kernels/gmm/routing.py:12", "moe_route_pack_and_gather"),
            ("/kernels/gmm/megablox_gmm_kernel/gmm.py:200", "moe_shared_gmm_metadata_and_zeroing"),
        ):
            self.assertEqual(stage(row(source=source), isolated=True), expected)

    def test_exclusive_time_and_occurrence_coverage(self):
        records = [
            row("gmm_checkpoint_fp4-g_64", time=32000, count=192),
            row(category="all-reduce", time=32000, count=64),
        ]
        columns = list(records[0])
        table = {
            "cols": [{"id": k} for k in columns],
            "rows": [{"c": [{"v": r[k]} for k in columns]} for r in records],
        }
        result = summarize(table, cores=8, steps=8, isolated=True)
        self.assertEqual(result["total_hlo_self_ms"], 1.0)
        self.assertTrue(result["complete_gmm_coverage"])
        self.assertTrue(result["complete_moe_collective_coverage"])
        self.assertEqual(sum(result["stage_ms"].values()), result["total_hlo_self_ms"])
        self.assertFalse(
            summarize(table, cores=8, steps=8, isolated=False)["complete_gmm_coverage"]
        )

    def test_empty_capture_is_not_a_measurement(self):
        with self.assertRaises(ValueError):
            summarize({}, cores=0, steps=8, isolated=True)

    def test_layout_copy_is_attributed_by_ssa_not_display_category(self):
        records = [row("gmm_checkpoint_fp4-test"), row("copy.s", category="data formatting")]
        records[0]["hlo_op_expression"] = (
            "%gmm_checkpoint_fp4-test = bf16[8,128] custom-call(%a,%b,%c,%d,%e,%f,%g,%copy.s)"
        )
        records[1]["hlo_op_expression"] = "%copy.s = u8[64,4096,64] copy(%original)"
        columns = list(records[0])
        table = {
            "cols": [{"id": k} for k in columns],
            "rows": [{"c": [{"v": r[k]} for k in columns]} for r in records],
        }
        result = summarize(table, cores=1, steps=1, isolated=False)
        self.assertEqual(result["stage_ms"]["moe_gmm_input_layout_copy_scale"], 1.0)
        self.assertEqual(len(result["gmm_input_copy_witnesses"]), 1)

    def test_fp8_copy_and_shared_metadata_are_not_charged_entirely_to_moe(self):
        records = [
            row("gmm_checkpoint_fp4-test"),
            row("gmm_checkpoint_fp8-test"),
            row("copy.s"),
            row("shared.meta"),
        ]
        records[0]["hlo_op_expression"] = (
            "%gmm_checkpoint_fp4-test = bf16[8,128] custom-call(%shared.meta,%b,%c,%d,%e,%f,%g,%fp4.scale)"
        )
        records[1]["hlo_op_expression"] = (
            "%gmm_checkpoint_fp8-test = bf16[8,128] custom-call(%shared.meta,%b8,%c8,%d8,%e8,%f8,%g8,%copy.s)"
        )
        records[2]["hlo_op_expression"] = "%copy.s = u8[1,32,1,32] copy(%original)"
        records[3]["hlo_op_expression"] = "%shared.meta = s32[] constant(0)"
        columns = list(records[0])
        table = {
            "cols": [{"id": k} for k in columns],
            "rows": [{"c": [{"v": r[k]} for k in columns]} for r in records],
        }
        result = summarize(table, cores=1, steps=1, isolated=False)
        self.assertEqual(result["stage_ms"]["fp8_gmm_input_layout_copy_scale"], 1.0)
        self.assertEqual(
            result["stage_ms"]["shared_fp8_and_moe_shared_gmm_metadata_and_zeroing"], 1.0
        )
        self.assertEqual(sum(result["stage_ms"].values()), 4.0)


if __name__ == "__main__":
    unittest.main()
