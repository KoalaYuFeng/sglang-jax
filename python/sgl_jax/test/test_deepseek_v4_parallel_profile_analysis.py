"""Dependency-free instruction classification guards for raw TP profiles."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analyze_deepseek_v4_parallel_profiles import event_family


class ParallelProfileAnalysisTest(unittest.TestCase):
    def test_collective_opcode_not_just_its_ssa_spelling(self):
        self.assertEqual(
            event_family("%all_gather.3 = bf16[4,8192] all-gather(%x)"), "all_gather_including_wait"
        )

    def test_real_attention_custom_call(self):
        self.assertEqual(
            event_family(
                "%csa-joint-attention-b1-k64-h16-d512-v4.1 = bf16[4,16,512] custom-call(%x)"
            ),
            "csa_attention",
        )

    def test_copy_with_kernel_name_is_not_kernel_execution(self):
        self.assertEqual(
            event_family("%csa-joint-attention.buffer = bf16[4,16,512] copy(%x)"), "copies"
        )

    def test_control_flow_envelopes_are_not_additive_work(self):
        self.assertIsNone(event_family("%while.1 = f32[1] while(%x)"))

    def test_batched_projection_is_distinct_from_sequential(self):
        self.assertEqual(
            event_family(
                "%csa-compressor-project-decode-batched-k4096-n128-v4.1 = f32[4,1,2048] custom-call(%x)"
            ),
            "csa_batched_decode_projection",
        )

    def test_marker_in_metadata_is_not_dispatch(self):
        self.assertIsNone(
            event_family('%fusion.1 = f32[1] fusion(%x), metadata={op_name="csa-joint-attention"}')
        )


if __name__ == "__main__":
    unittest.main()
