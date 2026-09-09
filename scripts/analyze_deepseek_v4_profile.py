"""Export XProf data and summarize non-overlapping raw execution intervals.

Run in the separate profile-tools venv. Raw tool exports are retained so the
report can be audited without a running web server. Hardware duration and
compiler-estimated FLOPs/bytes must never be conflated.
"""

import argparse
import ast
import collections
import gzip
import hashlib
import heapq
import json
import re
from pathlib import Path


def interval_union(intervals):
    total = 0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
        else:
            total += max(0, stop - end)
        end = stop if end is None else max(end, stop)
    return total


def partition_intervals(windows, regions):
    """Disjoint accounting: higher priority wins, then the shortest child.

    Regions are (start, end, label, priority). Nothing outside windows counts.
    This is timeline attribution, not a claim about overlapping hardware units.
    """
    bounds = []
    for index, (start, stop) in enumerate(windows):
        bounds.extend(((start, 1, "window", index), (stop, -1, "window", index)))
    for index, (start, stop, _label, _priority) in enumerate(regions):
        bounds.extend(((start, 1, "region", index), (stop, -1, "region", index)))
    active, heap, totals = set(), [], collections.defaultdict(float)
    window_count, previous = 0, None
    for time, sign, kind, index in sorted(bounds):
        while heap and heap[0][2] not in active:
            heapq.heappop(heap)
        if previous is not None and window_count and time > previous:
            label = heap[0][3] if heap else "outside_device_modules"
            totals[label] += time - previous
        if kind == "window":
            window_count += sign
        elif sign == -1:
            active.discard(index)
        else:
            active.add(index)
            start, stop, label, priority = regions[index]
            heapq.heappush(heap, (-priority, stop - start, index, label))
        previous = time
    return dict(totals)


def source_ranges(profile):
    """Only use source-line attribution when the numerical fingerprint matches."""
    root = Path(__file__).resolve().parents[1] / "python/sgl_jax/srt"
    source = root / "model_executor/deepseek_v4_reference.py"
    paths = [source, root / "model_loader/deepseek_v4_checkpoint.py"]
    for directory in ("kernels/low_bit", "kernels/mhc"):
        paths.extend((root / directory).glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    if (
        digest.hexdigest()
        != json.loads((profile / "report.json").read_text())["source_fingerprint"]
    ):
        raise RuntimeError("source fingerprint mismatch; cannot safely attribute old stack lines")
    return [
        (node.lineno, node.end_lineno, node.name)
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
    ]


def event_stage(event, ranges):
    info = event.get("args", {})
    stack = info.get("source_stack", info.get("source", ""))
    if info.get("hlo_category") == "all-reduce":
        return "moe_all_reduce_including_wait"
    if "v4_output_head" in event.get("name", "") or any(
        "v4_output_head" in value for value in info.values() if isinstance(value, str)
    ):
        return "lm_head"
    if "/kernels/mhc/" in stack:
        return "mhc"
    if "/low_bit/moe.py:" in stack:
        return "routed_experts"
    functions = [
        name
        for line in re.findall(r"deepseek_v4_reference\.py:(\d+)", stack)
        for start, end, name in ranges
        if start <= int(line) <= end
    ]
    if info.get("hlo_category") == "sort" and any(
        name in ("attention", "sparse_attention", "_single_query_attention")
        for name in functions
    ):
        return "attention_topk"
    for name in functions:
        if name == "compress":
            return "compressor"
        if name in ("sparse_attention", "_single_query_attention"):
            return "sparse_attention"
        if name == "attention":
            return "attention_projection_and_index"
        if name == "route":
            return "router"
        if name == "moe":
            return "shared_experts_and_moe_combine"
        if name in ("compiled_head", "official_head_collapse"):
            return "lm_head"
    if "rms_norm" in functions:
        return "layer_norm"
    return None


def summarize_hlo_table(table, *, cores, steps, ranges):
    """Full XPlane HLO self-times bypass the Chrome viewer's event cap."""
    columns = [column["id"] for column in table["cols"]]
    stages, categories, kernels = (collections.defaultdict(float) for _ in range(3))
    collectives = 0
    for record in table["rows"]:
        row = dict(zip(columns, [cell.get("v") for cell in record["c"]]))
        elapsed = row["total_self_time"] / (1000 * cores * steps)
        stage = (
            event_stage(
                {
                    "args": {
                        "source_stack": row.get("source_info", ""),
                        "hlo_category": row["category"],
                    }
                },
                ranges,
            )
            or "unattributed_hlo"
        )
        stages[stage] += elapsed
        categories[row["category"]] += elapsed
        if row["hlo_op_name"].startswith(("checkpoint_", "mhc-")):
            kernels[stage + "/" + row["hlo_op_name"].split(".")[0]] += elapsed
        if row["category"] == "all-reduce":
            collectives += int(row["occurrences"])
    return {
        "scope": "full XPlane HLO self-times; ms per TensorCore per workload step; not bandwidth counters",
        "tensorcores": cores,
        "steps": steps,
        "all_reduce_occurrences": collectives,
        "expected_all_reduce_occurrences": 43 * cores * steps,
        "complete_collective_coverage": collectives == 43 * cores * steps,
        "total_hlo_self_ms": sum(stages.values()),
        "stage_ms": dict(stages),
        "category_ms": dict(categories),
        "kernel_ms": dict(kernels),
    }


def workload_breakdown(trace, ranges=()):
    processes, threads, groups = {}, {}, collections.defaultdict(list)
    markers = []
    for event in trace["traceEvents"]:
        key = (event.get("pid"), event.get("tid"))
        if event.get("ph") == "M":
            if event["name"] == "process_name":
                processes[key[0]] = event["args"]["name"]
            elif event["name"] == "thread_name":
                threads[key] = event["args"]["name"]
        elif event.get("ph") == "X":
            groups[key].append(event)
            if event["name"] in (
                "V4_DECODE",
                "V4_PREFILL",
                "V4_FRAMEWORK_DECODE",
                "V4_FRAMEWORK_PREFILL",
                "V4_8K_DECODE",
                "V4_8K_PREFILL_CHUNK",
            ):
                markers.append(event)
    framework = any(
        e["name"].startswith(("V4_FRAMEWORK_", "V4_8K_")) for e in markers
    )
    windows = [(e["ts"], e["ts"] + e["dur"]) for e in markers]
    result = {
        "scope": "measured per-TensorCore timeline, not MXU utilization or traffic counters",
        "host_steps": len(markers),
        "host_wall_ms": interval_union(windows) / 1000,
        "execution_path": "framework" if framework else "reference",
        "cores": [],
    }
    for pid, name in processes.items():
        if not name.startswith("/device:TPU:"):
            continue
        lines = {threads.get(key): events for key, events in groups.items() if key[0] == pid}
        modules, ops = lines.get("XLA Modules", []), lines.get("XLA Ops", [])
        regions = [(e["ts"], e["ts"] + e["dur"], "inside_module_unattributed", 0) for e in modules]
        stages = [
            (
                e["ts"],
                e["ts"] + e["dur"],
                "lm_head" if e["name"].startswith("jit_head(") else "inside_module_unattributed",
                0,
            )
            for e in modules
        ]
        kernels = collections.defaultdict(list)
        routed = []
        for e in ops:
            category = e.get("args", {}).get("hlo_category", "")
            priority = 1 if category in ("while", "conditional") else 2
            label = category or "device_runtime"
            if e["name"].startswith("region."):
                # These regions lack HLO provenance. They include custom
                # execution, but must not all be called Pallas/GEMM kernels.
                label, priority = "unmapped_device_regions", 2
            if category in ("all-reduce", "all-gather"):
                label, priority = category.replace("-", "_") + "_including_wait", 4
            regions.append((e["ts"], e["ts"] + e["dur"], label, priority))
            stage = event_stage(e, ranges)
            if stage:
                stages.append((e["ts"], e["ts"] + e["dur"], stage, 1))
            if e["name"].startswith(("checkpoint_", "mhc-")):
                kernels[(stage or "unknown") + "/" + e["name"].split(".")[0]].append(
                    (e["ts"], e["ts"] + e["dur"])
                )
            if category == "while" and "/low_bit/moe.py:" in e.get("args", {}).get("source", ""):
                routed.append((e["ts"], e["ts"] + e["dur"]))
        named = collections.defaultdict(list)
        for e in lines.get("XLA TraceMe", []):
            if e["name"].startswith("V4_"):
                named[e["name"]].append((e["ts"], e["ts"] + e["dur"]))
        counts = collections.Counter(e["name"].split("(")[0] for e in modules)
        result["cores"].append(
            {
                "name": name,
                "module_counts": dict(counts),
                "complete_layer_coverage": (
                    None if framework else bool(markers) and counts["jit_run"] == 43 * len(markers)
                ),
                "complete_model_dispatch_coverage": (
                    bool(markers) and counts["jit_jitted_run_model"] == len(markers)
                    if framework
                    else None
                ),
                "partition_ms": {
                    key: value / 1000
                    for key, value in partition_intervals(windows, regions).items()
                },
                "stage_partition_ms": {
                    key: value / 1000 for key, value in partition_intervals(windows, stages).items()
                },
                "kernel_inclusive_ms": {
                    key: interval_union(value) / 1000 for key, value in kernels.items()
                },
                "module_union_ms": interval_union([(e["ts"], e["ts"] + e["dur"]) for e in modules])
                / 1000,
                "routed_loop_inclusive_ms": interval_union(routed) / 1000,
                "named_region_ms": {k: interval_union(v) / 1000 for k, v in named.items()},
                "named_region_union_ms": interval_union([v for vs in named.values() for v in vs])
                / 1000,
            }
        )
    return result


def summarize_trace(trace):
    processes, threads, events = {}, {}, collections.defaultdict(list)
    for event in trace["traceEvents"]:
        key = (event.get("pid"), event.get("tid"))
        if event.get("ph") == "M":
            if event.get("name") == "process_name":
                processes[key[0]] = event["args"]["name"]
            elif event.get("name") == "thread_name":
                threads[key] = event["args"]["name"]
        elif event.get("ph") == "X" and "dur" in event:
            events[key].append(event)
    planes = {}
    for key, group in events.items():
        pid, tid = key
        if pid not in planes:
            planes[pid] = {"name": processes.get(pid, str(pid)), "lines": []}
        intervals = [(v["ts"], v["ts"] + v["dur"]) for v in group]
        sums = collections.defaultdict(lambda: {"count": 0, "duration_us": 0})
        for event in group:
            sums[event["name"]]["count"] += 1
            sums[event["name"]]["duration_us"] += event["dur"]
        top = sorted(sums.items(), key=lambda item: item[1]["duration_us"], reverse=True)[:30]
        planes[pid]["lines"].append(
            {
                "name": threads.get(key, str(tid)),
                "id": tid,
                "events": len(group),
                "union_ms": interval_union(intervals) / 1000,
                "summed_ms": sum(v["dur"] for v in group) / 1000,
                "span_ms": (max(b for _, b in intervals) - min(a for a, _ in intervals)) / 1000,
                "top_events": [{"name": name, **value} for name, value in top],
                "samples": group[:5],
            }
        )
    return {
        "scope": "measured trace intervals; do not add nested/parallel line totals",
        "planes": list(planes.values()),
    }


def raw_summary(path, converter, ranges=()):
    recovered = path.parent / "xprof-recovered.trace.json.gz"
    traces = [recovered] if recovered.exists() else list(path.parent.glob("*.trace.json.gz"))
    issue = None
    try:
        if not traces:
            raise FileNotFoundError("no Chrome JSON trace export")
        with gzip.open(traces[0], "rt") as stream:
            trace = json.load(stream)
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as error:
        issue = str(error)
        if converter is None:
            raise RuntimeError(
                "invalid/missing JSON; rerun with XProf to recover from XPlane"
            ) from error
        data, _ = converter.xspace_to_tool_data([str(path)], "trace_viewer", {})
        trace = json.loads(data)
        with gzip.open(recovered, "wt") as stream:
            json.dump(trace, stream)
        traces = [recovered]
    summary, breakdown = summarize_trace(trace), workload_breakdown(trace, ranges)
    for result in (summary, breakdown):
        result["trace_source"] = str(traces[0])
        result["json_export_issue"] = issue
    return summary, breakdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--capture", action="append")
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    raw_to_tool_data = None
    if not args.raw_only:
        from xprof.convert import raw_to_tool_data

    output = args.profile / "analysis"
    ranges = source_ranges(args.profile)
    # The original profiler records repeated control rounds in ``captures``.
    # The 8K worker instead creates one trace directory per selected chunk, so
    # each directory is one workload step and does not need a duplicate list in
    # its report.
    capture_steps = {
        c["label"]: len(c["seconds"])
        for c in json.loads((args.profile / "report.json").read_text()).get("captures", [])
    }
    output.mkdir(exist_ok=True)
    index = output / "exports.json"
    result = (
        json.loads(index.read_text())
        if index.exists()
        else {
            "scope": "XPlane measured times; XProf roofline FLOPs/bytes are compiler estimates",
            "captures": {},
        }
    )
    for root in sorted((args.profile / "traces").iterdir()):
        if args.capture and root.name not in args.capture:
            continue
        paths = sorted(root.rglob("*.xplane.pb"))
        if not paths:
            continue
        folder = output / root.name
        folder.mkdir(exist_ok=True)
        capture = result["captures"].get(root.name, {"tools": {}})
        capture["xplane_paths"] = [str(p) for p in paths]
        result["captures"][root.name] = capture
        raw, breakdown = raw_summary(paths[0], raw_to_tool_data, ranges)
        (folder / "raw_summary.json").write_text(json.dumps(raw, indent=2) + "\n")
        (folder / "workload_breakdown.json").write_text(json.dumps(breakdown, indent=2) + "\n")

        def export_hlo_summary(parsed):
            summary = summarize_hlo_table(
                parsed,
                cores=len(breakdown["cores"]),
                steps=capture_steps.get(root.name, 1),
                ranges=ranges,
            )
            (folder / "hlo_breakdown.json").write_text(json.dumps(summary, indent=2) + "\n")

        if (folder / "hlo_stats.json").exists():
            export_hlo_summary(json.loads((folder / "hlo_stats.json").read_text()))
        if args.raw_only:
            print(json.dumps({"capture": root.name, "raw_analysis_complete": True}), flush=True)
            continue
        tool_names = raw_to_tool_data.xspace_to_tool_names([str(p) for p in paths])
        capture["available_tools"] = tool_names
        print(json.dumps({"capture": root.name, "available_tools": tool_names}), flush=True)
        for tool in (
            "overview_page",
            "hlo_stats",
            "op_profile",
            "framework_op_stats",
            "roofline_model",
            "memory_profile",
            "pod_viewer",
            "kernel_utilization",
            "utilization_viewer",
            "perf_counters",
        ):
            if tool not in {name.rstrip("^") for name in tool_names}:
                capture["tools"][tool] = {"available": False}
                continue
            try:
                data, content_type = raw_to_tool_data.xspace_to_tool_data(
                    [str(p) for p in paths], tool, {"use_saved_result": False}
                )
                if data is None:
                    raise RuntimeError("tool returned no data")
                if isinstance(data, bytes):
                    data = data.decode()
                parsed = json.loads(data)
                (folder / f"{tool}.json").write_text(json.dumps(parsed, indent=2) + "\n")
                if tool == "hlo_stats":
                    export_hlo_summary(parsed)
                capture["tools"][tool] = {"available": True, "content_type": content_type}
            except Exception as error:
                capture["tools"][tool] = {"available": True, "error": str(error)}
            print(
                json.dumps({"capture": root.name, "tool": tool, **capture["tools"][tool]}),
                flush=True,
            )
            (output / "exports.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "exports.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
