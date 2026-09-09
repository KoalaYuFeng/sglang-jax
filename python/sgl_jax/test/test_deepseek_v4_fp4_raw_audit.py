"""Raw trace accounting must distinguish one layer from the full model."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import audit_deepseek_v4_fp4_profiles as module


@pytest.mark.parametrize("layers,steps", [(1, 8), (43, 1), (43, 2)])
def test_raw_audit_counts_each_layer_and_ignores_duplicate_async_lines(
    tmp_path, monkeypatch, layers, steps
):
    report = {"complete": True, "captures": [{"label": "B4", "seconds": [1] * steps}]}
    (tmp_path / "report.json").write_text(json.dumps(report))
    analysis = tmp_path / "analysis/B4"
    analysis.mkdir(parents=True)
    summary = {
        "stage_ms": {
            "moe_gate_up_gmm_including_conversion": layers * 3,
            "moe_all_reduce_including_wait": layers,
        }
    }
    (analysis / "fp4_moe_breakdown.json").write_text(json.dumps(summary))
    trace = tmp_path / "traces/B4/host.xplane.pb"
    trace.parent.mkdir(parents=True)
    trace.write_bytes(b"mock trace; no TPU performance claim")
    event_stats = [("device_duration_ps", 1e9), ("Time Scale Multiplier", 1)]
    events = [
        SimpleNamespace(
            name="%gmm_checkpoint_fp4_candidate_scale_kn_packed_scale-1 = custom-call()",
            stats=event_stats,
        )
        for _ in range(3 * layers * steps)
    ] + [
        SimpleNamespace(name="%sum = f32[] all-reduce(%x)", stats=event_stats)
        for _ in range(layers * steps)
    ]
    if layers == 43:
        sampler = SimpleNamespace(
            name="%sampler_sum = f32[1] all-reduce(%logits)", stats=event_stats
        )
        events += [sampler] * (3 * steps)
        columns = ("category", "hlo_op_expression", "source_info", "tf_op_name")
        rows = [
            (
                "all-reduce",
                "%sum = f32[] all-reduce(%x)",
                "/deepseek_v4/moe_gmm.py:1",
                "v4_layer_0/psum",
            ),
            ("all-reduce", sampler.name, "", "jit(jitted_sampler)/Sampler/cond:"),
        ]
        (analysis / "hlo_stats.json").write_text(
            json.dumps(
                {
                    "cols": [{"id": column} for column in columns],
                    "rows": [{"c": [{"v": cell} for cell in row]} for row in rows],
                }
            )
        )
    plane = SimpleNamespace(
        lines=[
            SimpleNamespace(name="XLA Ops", events=events),
            SimpleNamespace(name="Async Ops", events=events),
        ]
    )
    data = SimpleNamespace(find_plane_with_name=lambda name: plane)
    monkeypatch.setattr(
        module.jax.profiler, "ProfileData", SimpleNamespace(from_file=lambda _: data)
    )
    result = module.audit(tmp_path, layers=layers)
    assert result["B4"]["gmm"] == layers * 3
    assert result["B4"]["all_reduce_including_wait"] == layers
    assert json.loads((tmp_path / "raw_timing_audit.json").read_text())["layers"] == layers
    if layers == 43:
        receipt = json.loads((tmp_path / "raw_timing_audit.json").read_text())
        assert (
            receipt["captures"]["B4"]["collective_identity_witness"]["sampler_raw_ms_per_call"] == 3
        )
        events.append(SimpleNamespace(name="%unknown = f32[] all-reduce(%x)", stats=event_stats))
        with pytest.raises(ValueError, match="no exact source witness"):
            module.audit(tmp_path, layers=layers)
        events.pop()
    with pytest.raises(AssertionError, match="incomplete raw coverage"):
        events.pop(0)
        module.audit(tmp_path, layers=layers)


@pytest.mark.parametrize(
    "source,operation",
    [("", ""), ("/deepseek_v4/moe_gmm.py:1", "jit(jitted_sampler)/Sampler/cond:")],
)
def test_raw_audit_rejects_unknown_or_ambiguous_collective_ownership(tmp_path, source, operation):
    table = tmp_path / "hlo_stats.json"
    columns = ("category", "hlo_op_expression", "source_info", "tf_op_name")
    row = ("all-reduce", "%sum = f32[] all-reduce(%x)", source, operation)
    table.write_text(
        json.dumps(
            {
                "cols": [{"id": column} for column in columns],
                "rows": [{"c": [{"v": cell} for cell in row]}],
            }
        )
    )
    with pytest.raises(ValueError, match="unknown or ambiguous"):
        module.collective_owners(table)
