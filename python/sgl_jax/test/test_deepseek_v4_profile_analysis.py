"""Interval accounting must not count nested or overlapping events twice."""

import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/analyze_deepseek_v4_profile.py"
_SPEC = importlib.util.spec_from_file_location("v4_profile_analysis", _SCRIPT)
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


class ProfileAnalysisTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(analysis.interval_union([]), 0)

    def test_disjoint_and_adjacent(self):
        self.assertEqual(analysis.interval_union([(3, 5), (0, 2), (5, 7)]), 6)

    def test_nested_and_overlapping(self):
        self.assertEqual(analysis.interval_union([(0, 10), (2, 4), (8, 12)]), 12)

    def test_trace_accounting_and_units(self):
        trace = {
            "traceEvents": [
                {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "TPU:0"}},
                {"ph": "M", "pid": 1, "tid": 2, "name": "thread_name", "args": {"name": "Ops"}},
                {"ph": "X", "pid": 1, "tid": 2, "name": "outer", "ts": 0, "dur": 2000},
                {"ph": "X", "pid": 1, "tid": 2, "name": "inner", "ts": 100, "dur": 1000},
            ]
        }
        plane = analysis.summarize_trace(trace)["planes"][0]
        self.assertEqual(plane["name"], "TPU:0")
        self.assertEqual(plane["lines"][0]["name"], "Ops")
        self.assertEqual(plane["lines"][0]["union_ms"], 2)
        self.assertEqual(plane["lines"][0]["summed_ms"], 3)

    def test_partition_clips_windows_and_preserves_gaps(self):
        result = analysis.partition_intervals([(0, 10), (20, 25)], [(3, 30, "module", 0)])
        self.assertEqual(result, {"outside_device_modules": 3, "module": 12})

    def test_partition_priority_and_nested_children(self):
        result = analysis.partition_intervals(
            [(0, 10)],
            [(0, 10, "parent", 0), (2, 7, "child", 0), (4, 8, "collective", 4)],
        )
        self.assertEqual(result, {"parent": 4, "child": 2, "collective": 4})

    def test_incomplete_device_trace_is_flagged(self):
        trace = {
            "traceEvents": [
                {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
                {
                    "ph": "M",
                    "pid": 1,
                    "tid": 2,
                    "name": "thread_name",
                    "args": {"name": "XLA Modules"},
                },
                {"ph": "X", "pid": 0, "tid": 0, "name": "V4_DECODE", "ts": 0, "dur": 1000},
                {"ph": "X", "pid": 1, "tid": 2, "name": "jit_run(1)", "ts": 100, "dur": 200},
            ]
        }
        result = analysis.workload_breakdown(trace)
        self.assertEqual(result["host_steps"], 1)
        self.assertFalse(result["cores"][0]["complete_layer_coverage"])
        self.assertAlmostEqual(sum(result["cores"][0]["partition_ms"].values()), 1)

    def test_all_gather_wait_is_not_hidden_by_an_overlapping_device_region(self):
        trace = {
            "traceEvents": [
                {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
                {
                    "ph": "M",
                    "pid": 1,
                    "tid": 2,
                    "name": "thread_name",
                    "args": {"name": "XLA Ops"},
                },
                {"ph": "X", "pid": 0, "tid": 0, "name": "V4_8K_DECODE", "ts": 0, "dur": 1000},
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 2,
                    "name": "all-gather.1",
                    "ts": 100,
                    "dur": 600,
                    "args": {"hlo_category": "all-gather"},
                },
                {"ph": "X", "pid": 1, "tid": 2, "name": "region.1", "ts": 200, "dur": 200},
            ]
        }
        partition = analysis.workload_breakdown(trace)["cores"][0]["partition_ms"]
        self.assertAlmostEqual(partition["all_gather_including_wait"], 0.6)
        self.assertAlmostEqual(sum(partition.values()), 1.0)

    def test_source_stage_uses_outer_attention_not_shared_helper(self):
        ranges = [(1, 5, "rms_norm"), (10, 15, "linear"), (20, 30, "attention")]
        event = {
            "args": {"source_stack": "deepseek_v4_reference.py:3:0\ndeepseek_v4_reference.py:22:0"}
        }
        self.assertEqual(analysis.event_stage(event, ranges), "attention_projection_and_index")

    def test_attention_children_and_topk_are_separate(self):
        ranges = [
            (20, 30, "compress"),
            (40, 50, "_single_query_attention"),
            (60, 70, "attention"),
        ]
        self.assertEqual(
            analysis.event_stage(
                {"args": {"source_stack": "deepseek_v4_reference.py:22:0"}}, ranges
            ),
            "compressor",
        )
        self.assertEqual(
            analysis.event_stage(
                {
                    "args": {
                        "source_stack": "deepseek_v4_reference.py:44:0\n"
                        "deepseek_v4_reference.py:65:0"
                    }
                },
                ranges,
            ),
            "sparse_attention",
        )
        self.assertEqual(
            analysis.event_stage(
                {
                    "args": {
                        "source_stack": "deepseek_v4_reference.py:65:0",
                        "hlo_category": "sort",
                    }
                },
                ranges,
            ),
            "attention_topk",
        )

    def test_framework_dispatch_does_not_require_43_separate_layer_modules(self):
        trace = {
            "traceEvents": [
                {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
                {
                    "ph": "M",
                    "pid": 1,
                    "tid": 2,
                    "name": "thread_name",
                    "args": {"name": "XLA Modules"},
                },
                {
                    "ph": "X",
                    "pid": 0,
                    "tid": 0,
                    "name": "V4_FRAMEWORK_DECODE",
                    "ts": 0,
                    "dur": 1000,
                },
                {
                    "ph": "X",
                    "pid": 1,
                    "tid": 2,
                    "name": "jit_jitted_run_model(1)",
                    "ts": 100,
                    "dur": 800,
                },
            ]
        }
        result = analysis.workload_breakdown(trace)
        self.assertEqual(result["execution_path"], "framework")
        self.assertTrue(result["cores"][0]["complete_model_dispatch_coverage"])
        self.assertIsNone(result["cores"][0]["complete_layer_coverage"])
        self.assertAlmostEqual(result["cores"][0]["partition_ms"]["outside_device_modules"], 0.2)

    def test_8k_worker_marker_is_a_framework_dispatch(self):
        trace = {
            "traceEvents": [
                {
                    "ph": "X",
                    "pid": 0,
                    "tid": 0,
                    "name": "V4_8K_PREFILL_CHUNK",
                    "ts": 0,
                    "dur": 1000,
                }
            ]
        }

        result = analysis.workload_breakdown(trace)

        self.assertEqual(result["host_steps"], 1)
        self.assertEqual(result["execution_path"], "framework")

    def test_no_markers_does_not_claim_complete_coverage(self):
        result = analysis.workload_breakdown({"traceEvents": []})
        self.assertEqual(result["host_steps"], 0)
        self.assertEqual(result["cores"], [])

    def test_native_output_head_scope_is_attributed(self):
        self.assertEqual(analysis.event_stage({"name": "v4_output_head/dot"}, []), "lm_head")

    def test_collective_wait_is_separate_from_expert_compute(self):
        event = {"args": {"source": "/kernels/low_bit/moe.py:48", "hlo_category": "all-reduce"}}
        self.assertEqual(analysis.event_stage(event, []), "moe_all_reduce_including_wait")

    def test_missing_source_is_not_invented(self):
        self.assertIsNone(analysis.event_stage({"name": "region.104"}, []))

    def test_invalid_json_is_reexported_without_overwriting_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "host.trace.json.gz"
            with gzip.open(original, "wt") as stream:
                stream.write('{"traceEvents": [')
            converter = Mock()
            converter.xspace_to_tool_data.return_value = (
                json.dumps({"traceEvents": []}),
                "application/json",
            )
            summary, _ = analysis.raw_summary(root / "host.xplane.pb", converter)
            self.assertTrue(summary["json_export_issue"])
            self.assertTrue((root / "xprof-recovered.trace.json.gz").exists())
            with gzip.open(original, "rt") as stream:
                self.assertEqual(stream.read(), '{"traceEvents": [')

    def test_hlo_uses_self_time_not_nested_total_time(self):
        columns = ["category", "hlo_op_name", "total_self_time", "total_time", "occurrences"]
        table = {
            "cols": [{"id": column} for column in columns],
            "rows": [{"c": [{"v": v} for v in ["all-reduce", "psum.1", 8000, 40000, 344]]}],
        }
        result = analysis.summarize_hlo_table(table, cores=8, steps=1, ranges=[])
        self.assertEqual(result["total_hlo_self_ms"], 1)
        self.assertTrue(result["complete_collective_coverage"])


if __name__ == "__main__":
    unittest.main()
