"""CPU-only profile accounting; not device performance validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from analyze_deepseek_v4_ep_profile import modelworker_timeline, stage, summarize  # noqa: E402
from sgl_jax.test.test_deepseek_v4_fp4_profile_analysis import row  # noqa: E402


def test_ep_gather_and_local_add_are_separate_from_attention():
    for category, source, expected in (
        ("all-gather", "collectives.py:26", "moe_ep_all_gather_including_wait"),
        ("loop fusion", "collectives.py:29", "moe_ep_ordered_local_sum_and_layout"),
        ("all-gather", "attention.py:100", "attention_tp_all_gather_including_wait"),
    ):
        assert stage(row(category=category, source=f"/deepseek_v4/{source}")) == expected


def test_ordered_ep_coverage_does_not_count_other_gathers():
    records = [
        row(category="all-gather", source="/deepseek_v4/collectives.py:26", count=43 * 8),
        row(category="all-gather", source="/deepseek_v4/attention.py:100", count=41 * 8),
    ]
    columns = list(records[0])
    table = dict(
        cols=[dict(id=k) for k in columns],
        rows=[dict(c=[dict(v=r[k]) for k in columns]) for r in records],
    )
    summary = summarize(table, cores=8, steps=1)
    assert summary["complete_moe_collective_coverage"]
    assert summary["expected_moe_ep_gather_occurrences"] == 344
    records[0]["occurrences"] = 343
    table["rows"][0]["c"] = [dict(v=records[0][k]) for k in columns]
    assert not summarize(table, cores=8, steps=1)["complete_moe_collective_coverage"]


def test_generic_marker_requires_actual_modelworker_dispatch():
    timeline = dict(
        host_steps=1,
        cores=[dict(name=f"/device:TPU:{i}", module_counts={"jit_jitted_run_model": 1})
               for i in range(8)],
    )
    assert modelworker_timeline(timeline, 1)["execution_path"] == "framework"
    timeline["cores"][0]["module_counts"] = {"jit_run": 43}
    with pytest.raises(ValueError, match="ModelWorker"):
        modelworker_timeline(timeline, 1)
