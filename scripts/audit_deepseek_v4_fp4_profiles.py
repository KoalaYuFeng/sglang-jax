"""Cross-check XProf GMM/collective timings against raw TensorCore XLA events.

Only the XLA Ops line is counted, never its duplicate Async Ops line. No
reconstructed HLO module, compiler FLOP estimate or bandwidth counter is used.
Collective durations include waiting; per-core averages are not chip sums.
For whole-model captures, exact typed instructions and source/caller identity
from HLO Stats distinguish MoE sums from sampler sums. No HLO timing is used
to select or measure raw events, and unrecognized reductions fail closed.
"""

import argparse
import hashlib
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import jax


def collective_owners(table_path):
    table = json.loads(table_path.read_text())
    columns = [column["id"] for column in table["cols"]]
    owners = {}
    for record in table["rows"]:
        row = dict(zip(columns, [cell.get("v") for cell in record["c"]], strict=True))
        if row["category"] != "all-reduce":
            continue
        source, operation = row.get("source_info") or "", row.get("tf_op_name") or ""
        moe = "/deepseek_v4/moe_gmm.py:" in source
        sampler = "jit(jitted_sampler)/Sampler/" in operation
        if moe == sampler:
            raise ValueError("unknown or ambiguous all-reduce source ownership")
        expression = row["hlo_op_expression"]
        family = (
            "all_reduce_including_wait" if moe else "sampler_all_reduce_including_wait"
        )
        if expression in owners and owners[expression] != family:
            raise ValueError("ambiguous raw collective instruction")
        owners[expression] = family
    if not owners:
        raise ValueError("no source-attributed collectives")
    return owners


def audit(profile, *, layers=1):
    if layers not in (1, 43):
        raise ValueError("expected one isolated layer or the full 43-layer model")
    report = json.loads((profile / "report.json").read_text())
    if not report.get("complete"):
        raise ValueError("capture is not complete")
    captures = {capture["label"]: capture for capture in report["captures"]}
    results = {}
    for summary_path in sorted((profile / "analysis").glob("*/fp4_moe_breakdown.json")):
        label = summary_path.parent.name
        summary = json.loads(summary_path.read_text())
        table_path = summary_path.with_name("hlo_stats.json")
        owners = collective_owners(table_path) if layers == 43 else None
        traces = list((profile / "traces" / label).rglob("*.xplane.pb"))
        if len(traces) != 1:
            raise ValueError("expected one host trace")
        data = jax.profiler.ProfileData.from_file(str(traces[0]))
        steps = len(captures[label]["seconds"])
        cores = []
        for lane in range(8):
            plane = data.find_plane_with_name(f"/device:TPU:{lane}")
            if plane is None:
                raise ValueError(f"missing TensorCore {lane}")
            durations, counts = Counter(), Counter()
            for line in plane.lines:
                if line.name != "XLA Ops":
                    continue
                for event in line.events:
                    if re.match(r"%gmm_checkpoint_fp4[-\w.]* =", event.name):
                        family = "gmm"
                    elif " all-reduce(" in event.name:
                        if owners is not None and event.name not in owners:
                            raise ValueError(
                                "raw collective has no exact source witness"
                            )
                        family = (
                            owners[event.name]
                            if owners is not None
                            else "all_reduce_including_wait"
                        )
                    else:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        stats = dict(event.stats)
                    durations[family] += (
                        float(stats["device_duration_ps"])
                        * float(stats["Time Scale Multiplier"])
                        / (steps * 1e9)
                    )
                    counts[family] += 1
            if (
                counts["gmm"] != 3 * layers * steps
                or counts["all_reduce_including_wait"] != layers * steps
            ):
                raise AssertionError(
                    f"incomplete raw coverage on lane {lane}: {counts}"
                )
            cores.append(
                {
                    "tensorcore": lane,
                    "ms_per_call": dict(durations),
                    "counts": dict(counts),
                }
            )
        averaged = {
            family: sum(c["ms_per_call"][family] for c in cores) / len(cores)
            for family in ("gmm", "all_reduce_including_wait")
        }
        expected = {
            "gmm": sum(
                value
                for name, value in summary["stage_ms"].items()
                if re.fullmatch(
                    r"moe_(gate_up|gate|up|down)_gmm_including_conversion", name
                )
            ),
            "all_reduce_including_wait": summary["stage_ms"][
                "moe_all_reduce_including_wait"
            ],
        }
        for family, value in expected.items():
            if abs(value - averaged[family]) > 0.000001:
                raise AssertionError(
                    f"raw/XProf timing disagreement for {label}/{family}"
                )
        results[label] = {
            "raw_ms_per_call": averaged,
            "xprof_ms_per_call": expected,
            "per_tensorcore": cores,
            "xplane_sha256": hashlib.sha256(traces[0].read_bytes()).hexdigest(),
        }
        if owners is not None:
            results[label]["collective_identity_witness"] = {
                "hlo_stats_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
                "exact_instruction_owners": owners,
                "sampler_raw_ms_per_call": sum(
                    c["ms_per_call"].get("sampler_all_reduce_including_wait", 0)
                    for c in cores
                )
                / len(cores),
            }
    if not results:
        raise ValueError("no exported profiles to audit")
    if set(results) != set(captures):
        raise ValueError("missing or unexpected exported captures")
    output = {
        "complete": True,
        "layers": layers,
        "scope": __doc__,
        "captures": results,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capture_report_sha256": hashlib.sha256(
            (profile / "report.json").read_bytes()
        ).hexdigest(),
    }
    (profile / "raw_timing_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    return {label: data["raw_ms_per_call"] for label, data in results.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=(1, 43), default=1)
    args = parser.parse_args()
    print(json.dumps(audit(args.profile, layers=args.layers), indent=2))
