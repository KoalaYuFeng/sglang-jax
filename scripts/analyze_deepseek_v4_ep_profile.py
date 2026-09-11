"""Account for V4 ordered EP gather and local sum in real ModelWorker traces.

Reuse the established XProf exporter and exclusive HLO self-time analysis.
Collective duration includes waiting; HLO times are not hardware-bandwidth
measurements or separately measured online-dequantization costs.
"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import analyze_deepseek_v4_fp4_moe as moe
import analyze_deepseek_v4_native_profile as native
import analyze_deepseek_v4_profile as base

ORIGINAL_STAGE = moe.stage


def stage(row, *, isolated=False):
    source = row.get("source_info") or ""
    if "/deepseek_v4/collectives.py:" in source:
        if row["category"] == "all-gather":
            return "moe_ep_all_gather_including_wait"
        return "moe_ep_ordered_local_sum_and_layout"
    return ORIGINAL_STAGE(row, isolated=isolated)


def summarize(table, *, cores, steps):
    with patch.object(moe, "stage", stage):
        summary = moe.summarize(table, cores=cores, steps=steps, isolated=False)
    occurrences = summary["stage_occurrences"].get("moe_ep_all_gather_including_wait", 0)
    summary.pop("expected_moe_all_reduce_occurrences")
    summary.update(
        expected_moe_ep_gather_occurrences=43 * cores * steps,
        complete_moe_collective_coverage=occurrences == 43 * cores * steps,
        collective_coverage_scope="43 ordered EP gathers per full-model call; excludes attention/head collectives",
    )
    return summary


def modelworker_timeline(timeline, steps):
    """The harness uses generic V4 markers; verify the actual module dispatch."""
    cores = moe.tensorcore_timelines(timeline)
    if len(cores) != 8 or timeline["host_steps"] != steps:
        raise ValueError("incomplete TensorCore/host-step coverage")
    for core in cores:
        if core["module_counts"].get("jit_jitted_run_model") != steps:
            raise ValueError("trace is not the expected full-model ModelWorker dispatch")
        core["complete_layer_coverage"] = None
        core["complete_model_dispatch_coverage"] = True
    timeline["execution_path"] = "framework"
    timeline["execution_path_evidence"] = (
        "generic V4_PREFILL/V4_DECODE markers; verified jit_jitted_run_model count on all 8 TensorCores"
    )
    return timeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--capture", action="append")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    report_path = args.profile / "report.json"
    report = json.loads(report_path.read_text())
    if not report["complete"] or report["source_fingerprint"] != native.fingerprint():
        raise ValueError("requires complete same-source numerical/performance gate")
    if args.export:
        from xprof.convert import raw_to_tool_data
    for capture in report["captures"]:
        label = capture["label"]
        if args.capture and label not in args.capture:
            continue
        folder = args.profile / "analysis" / label
        folder.mkdir(parents=True, exist_ok=True)
        paths = sorted((args.profile / "traces" / label).rglob("*.xplane.pb"))
        if len(paths) != 1:
            raise ValueError(f"expected one-host XPlane: {paths}")
        if args.export:
            # Full native fingerprint above binds the source attribution.
            with patch.object(base, "event_stage", native.native_stage):
                raw, timeline = base.raw_summary(paths[0], raw_to_tool_data, ())
            timeline = modelworker_timeline(timeline, len(capture["seconds"]))
            for name, value in (("raw_summary", raw), ("workload_breakdown", timeline)):
                (folder / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
            available = raw_to_tool_data.xspace_to_tool_names([str(paths[0])])
            (folder / "available_tools.json").write_text(json.dumps(available, indent=2) + "\n")
            for tool in ("hlo_stats", "perf_counters"):
                if tool not in {name.rstrip("^") for name in available}:
                    continue
                data, _ = raw_to_tool_data.xspace_to_tool_data(
                    [str(paths[0])], tool, {"use_saved_result": False}
                )
                if isinstance(data, bytes):
                    data = data.decode()
                (folder / f"{tool}.json").write_text(json.dumps(json.loads(data), indent=2) + "\n")
        timeline = json.loads((folder / "workload_breakdown.json").read_text())
        modelworker_timeline(timeline, len(capture["seconds"]))
        cores = moe.tensorcore_timelines(timeline)
        steps = len(capture["seconds"])
        summary = summarize(json.loads((folder / "hlo_stats.json").read_text()), cores=len(cores), steps=steps)
        summary.update(
            capture=label,
            position=capture["position"],
            batch=report["batch"],
            input_length=report["input_length"],
            profiled_host_ms_per_call=sum(capture["seconds"]) * 1000 / steps,
            mean_module_union_ms_per_call=sum(c["module_union_ms"] for c in cores) / len(cores) / steps,
            excluded_device_planes=[c["name"] for c in timeline["cores"] if c not in cores],
            analysis_source_sha256={Path(m).name: hashlib.sha256(Path(m).read_bytes()).hexdigest()
                for m in (__file__, moe.__file__, native.__file__, base.__file__)},
            capture_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            xplane_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        )
        (folder / "ep_breakdown.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items()
            if k not in ("largest_witnesses_by_stage", "gmm_input_copy_witnesses", "scope")}), flush=True)
        if len(cores) != 8 or not summary["complete_gmm_coverage"] or not summary["complete_moe_collective_coverage"]:
            raise AssertionError("incomplete full-model TensorCore/GMM/EP coverage; inspect raw evidence")


if __name__ == "__main__":
    main()
