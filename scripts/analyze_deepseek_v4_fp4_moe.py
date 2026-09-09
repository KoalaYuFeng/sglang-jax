"""Small, auditable FP4 MoE XProf export and exclusive HLO self-time grouping.

Times are per TensorCore per profiled call, averaged across profiled cores.
They are not isolated dequant or hardware-bandwidth measurements. Keep raw
XPlane, HLO tables, per-core timelines and grouping witnesses beside this file.
Counterfactual warm timings live in the capture report, not profiler timings.
"""

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import analyze_deepseek_v4_native_profile as native
import analyze_deepseek_v4_profile as base


def tensorcore_timelines(timeline):
    # XProf can emit empty SparseCore planes even when profiling SC is disabled.
    # They are not extra TensorCores and must not dilute all measured times.
    return [c for c in timeline["cores"] if re.fullmatch(r"/device:TPU:\d+", c["name"])]


def stage(row, *, isolated):
    source = row.get("source_info") or ""
    name = row["hlo_op_name"]
    operation = row.get("tf_op_name") or ""
    category = row["category"]
    # A storage-adapter GMM is no longer synonymous with routed-expert work.
    if name.startswith(("gmm_checkpoint_fp8", "checkpoint_fp8_online_dequant_matmul")):
        return "fp8_projection_including_online_dequant"
    if name.startswith("v4_inverse_rope_checkpoint_fp8_wo_a"):
        return "inverse_rope_fp8_wo_a"
    if name.startswith(("v4_exact_rms_norm", "v4_exact_qnorm_rope")):
        return "v4_fused_normalization_and_rope"
    if "gmm_checkpoint_fp8" in operation or "/deepseek_v4/fp8.py:" in source:
        return "fp8_gmm_metadata_padding_and_glue"
    if "/deepseek_v4/normalization.py:" in source:
        return "v4_normalization_layout_and_helpers"
    if "/deepseek_v4/projections.py:" in source:
        return "fp8_projection_layout_and_glue"
    moe = (
        isolated
        or any(
            marker in source
            for marker in (
                "/deepseek_v4/moe_gmm.py:",
                "/kernels/low_bit/gmm.py:",
                "/kernels/gmm/",
            )
        )
        or "/jit(gmm)/" in operation
    )
    if category == "all-reduce" and moe:
        return "moe_all_reduce_including_wait"
    if name.startswith(("gmm_checkpoint_fp4", "gmm_diagnostic_predecoded_bf16")):
        if "DIAG_W1_GATE" in operation:
            return "moe_gate_gmm_including_conversion"
        if "DIAG_W3_UP" in operation:
            return "moe_up_gmm_including_conversion"
        if "-k_2048-n_4096-" in name or "DIAG_W2_DOWN" in operation:
            return "moe_down_gmm_including_conversion"
        return "moe_gate_up_gmm_including_conversion"
    if "DIAG_GATE_UP_ACTIVATION_QAT_ONCE" in operation:
        return "diagnostic_gate_up_activation_qat_once"
    if "DIAG_DOWN_ACTIVATION_QAT_ONCE" in operation:
        return "diagnostic_down_activation_qat_once"
    if "/kernels/gmm/routing.py:" in source and moe:
        return "moe_route_pack_and_gather"
    if "/kernels/gmm/" in source and moe:
        return "moe_shared_gmm_metadata_and_zeroing"
    if moe:
        if "DIAG_ROUTE_PACK_AND_GATHER" in operation or any(
            15 <= int(line) <= 41
            for line in re.findall(r"/deepseek_v4/moe_gmm.py:(\d+)", source)
        ):
            return "moe_route_pack_and_gather"
        if "DIAG_SWIGLU_AND_ROUTE_SCALE" in operation:
            return "moe_swiglu_and_route_scale"
        if "DIAG_UNPERMUTE_AND_LOCAL_SUM" in operation:
            return "moe_unpermute_and_local_sum"
        return "moe_other_glue_including_swiglu_and_combine"
    return (
        native.native_stage(
            {"args": {"source_stack": source, "hlo_category": category}}, ()
        )
        or "unattributed_hlo"
    )


def summarize(table, *, cores, steps, isolated):
    if cores <= 0 or steps <= 0:
        raise ValueError("cannot normalize an empty/incomplete capture")
    columns = [c["id"] for c in table["cols"]]
    rows = [
        dict(zip(columns, [c.get("v") for c in record["c"]], strict=True))
        for record in table["rows"]
    ]
    input_roles = collections.defaultdict(set)
    for row in rows:
        if row["hlo_op_name"].startswith(
            (
                "gmm_checkpoint_fp4",
                "gmm_diagnostic_predecoded_bf16",
                "gmm_checkpoint_fp8",
            )
        ):
            # HLO Stats keeps typed SSA operands. Use actual consumers, not
            # '.rhs_scale' spelling or a broad outer model stack, for copies.
            refs = re.findall(r"%([-\w.]+)", row.get("hlo_op_expression") or "")
            if len(refs) == 9 and refs[0] == row["hlo_op_name"]:
                owner = (
                    "fp8"
                    if row["hlo_op_name"].startswith("gmm_checkpoint_fp8")
                    else "moe"
                )
                for name, role in zip(
                    refs[1:],
                    (*("metadata" for _ in range(5)), "lhs", "rhs", "scale"),
                    strict=True,
                ):
                    input_roles[(row.get("program_id"), name)].add((owner, role))
    totals, counts = collections.defaultdict(float), collections.defaultdict(int)
    witnesses = collections.defaultdict(list)
    input_copy_witnesses = []
    gmm_calls = 0
    for row in rows:
        label = stage(row, isolated=isolated)
        consumers = input_roles.get((row.get("program_id"), row["hlo_op_name"]), set())
        owners, roles = (
            {owner for owner, _ in consumers},
            {role for _, role in consumers},
        )
        owner = "shared_fp8_and_moe" if len(owners) > 1 else next(iter(owners), "moe")
        if re.search(r"\bcopy\(", row.get("hlo_op_expression") or "") and roles & {
            "scale",
            "rhs",
            "lhs",
        }:
            label = owner + "_gmm_input_layout_copy_" + "_".join(sorted(roles))
            input_copy_witnesses.append(
                {
                    k: row.get(k)
                    for k in (
                        "hlo_op_name",
                        "hlo_op_expression",
                        "program_id",
                        "occurrences",
                        "total_self_time",
                    )
                }
            )
        elif "metadata" in roles and (owner != "moe" or not label.startswith("moe_")):
            label = owner + "_shared_gmm_metadata_and_zeroing"
        elapsed = row["total_self_time"] / (cores * steps * 1000)
        totals[label] += elapsed
        counts[label] += int(row["occurrences"])
        witnesses[label].append(
            {
                "hlo_op_name": row["hlo_op_name"],
                "tf_op_name": row.get("tf_op_name"),
                "category": row["category"],
                "occurrences": row["occurrences"],
                "self_ms_per_core_per_call": elapsed,
                "source_info": row.get("source_info"),
            }
        )
        if row["hlo_op_name"].startswith(
            ("gmm_checkpoint_fp4", "gmm_diagnostic_predecoded_bf16")
        ):
            gmm_calls += int(row["occurrences"])
    layers = 1 if isolated else 43
    return {
        "scope": __doc__,
        "tensorcores": cores,
        "steps": steps,
        "total_hlo_self_ms": sum(totals.values()),
        "stage_ms": dict(sorted(totals.items(), key=lambda v: -v[1])),
        "stage_occurrences": dict(counts),
        "gmm_occurrences": gmm_calls,
        "expected_gmm_occurrences": 3 * layers * cores * steps,
        "complete_gmm_coverage": gmm_calls == 3 * layers * cores * steps,
        "expected_moe_all_reduce_occurrences": layers * cores * steps,
        "complete_moe_collective_coverage": counts["moe_all_reduce_including_wait"]
        == layers * cores * steps,
        "gmm_input_copy_witnesses": input_copy_witnesses,
        "largest_witnesses_by_stage": {
            k: sorted(v, key=lambda r: -r["self_ms_per_core_per_call"])[:8]
            for k, v in witnesses.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--capture", action="append")
    parser.add_argument(
        "--export", action="store_true", help="requires the CPU XProf venv"
    )
    args = parser.parse_args()
    report = json.loads((args.profile / "report.json").read_text())
    if (
        not report.get("complete")
        or report["framework_source_fingerprint"] != native.fingerprint()
    ):
        raise ValueError("requires a complete, same-source capture")
    captures = {c["label"]: c for c in report["captures"]}
    ranges = base.source_ranges(args.profile)
    base.event_stage = native.native_stage
    if args.export:
        from xprof.convert import raw_to_tool_data
    for label, capture in captures.items():
        if args.capture and label not in args.capture:
            continue
        folder = args.profile / "analysis" / label
        folder.mkdir(parents=True, exist_ok=True)
        paths = sorted((args.profile / "traces" / label).rglob("*.xplane.pb"))
        if args.export:
            if len(paths) != 1:
                raise ValueError(f"expected one-host XPlane: {paths}")
            raw, timeline = base.raw_summary(paths[0], raw_to_tool_data, ranges)
            for name, data in (("raw_summary", raw), ("workload_breakdown", timeline)):
                (folder / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
            available = raw_to_tool_data.xspace_to_tool_names([str(p) for p in paths])
            (folder / "available_tools.json").write_text(
                json.dumps(available, indent=2) + "\n"
            )
            # Do not infer actual memory bandwidth from HLO/roofline estimates.
            tools = ["hlo_stats"]
            if "perf_counters" in {name.rstrip("^") for name in available}:
                tools.append("perf_counters")
            for tool in tools:
                data, _ = raw_to_tool_data.xspace_to_tool_data(
                    [str(p) for p in paths], tool, {"use_saved_result": False}
                )
                if isinstance(data, bytes):
                    data = data.decode()
                (folder / f"{tool}.json").write_text(
                    json.dumps(json.loads(data), indent=2) + "\n"
                )
        timeline = json.loads((folder / "workload_breakdown.json").read_text())
        tensorcores = tensorcore_timelines(timeline)
        summary = summarize(
            json.loads((folder / "hlo_stats.json").read_text()),
            cores=len(tensorcores),
            steps=len(capture["seconds"]),
            isolated=args.isolated,
        )
        summary["analysis_source_sha256"] = {
            Path(module).name: hashlib.sha256(Path(module).read_bytes()).hexdigest()
            for module in (__file__, native.__file__, base.__file__)
        }
        summary["capture_report_sha256"] = hashlib.sha256(
            (args.profile / "report.json").read_bytes()
        ).hexdigest()
        summary["profiled_host_ms_per_call"] = (
            sum(capture["seconds"]) * 1000 / len(capture["seconds"])
        )
        summary["mean_module_union_ms_per_call"] = (
            sum(c["module_union_ms"] for c in tensorcores)
            / len(tensorcores)
            / len(capture["seconds"])
        )
        summary["excluded_device_planes"] = [
            c["name"] for c in timeline["cores"] if c not in tensorcores
        ]
        (folder / "fp4_moe_breakdown.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "capture": label,
                    **{
                        k: v
                        for k, v in summary.items()
                        if k
                        not in (
                            "scope",
                            "largest_witnesses_by_stage",
                            "gmm_input_copy_witnesses",
                        )
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
