"""Read exact selected HLO durations from the independent parallel experiments.

Only XLA Ops are counted (not duplicate Async Ops). Selected instruction
families exclude control-flow envelopes; this is not a complete additive
roofline/resource breakdown. Collectives include waiting. No hardware memory
bandwidth or isolated in-Pallas dequantization time is inferred.
"""

import argparse
import collections
import hashlib
import json
import re
import warnings
from pathlib import Path


def event_family(instruction):
    if " all-gather(" in instruction:
        return "all_gather_including_wait"
    if " all-reduce(" in instruction:
        return "all_reduce_including_wait"
    if " slice(" in instruction:
        return "slices"
    if " copy(" in instruction:
        return "copies"
    if " custom-call(" not in instruction:
        return None
    name = re.match(r"%([\w.-]+)\s*=", instruction)
    if name is None:
        return None
    for prefix, family in (
        ("checkpoint_fp8_online_dequant_matmul", "fp8_projection_including_dequant"),
        ("gmm_checkpoint_fp8", "fp8_projection_including_dequant"),
        ("v4_inverse_rope_checkpoint_fp8_wo_a", "inverse_rope_fp8_wo_a"),
        ("v4_exact_rms_norm", "v4_fused_normalization_and_rope"),
        ("v4_exact_qnorm_rope", "v4_fused_normalization_and_rope"),
        ("csa-compressor-project-decode-batched", "csa_batched_decode_projection"),
        ("csa-compressor-project-gemv", "csa_single_row_decode_projection"),
        ("csa-compressor-project-b8", "csa_mxu_projection"),
        ("hca-state-project", "hca_projection"),
        ("csa-compressor-snapshot", "csa_emission"),
        ("hca-boundary-snapshot", "hca_emission"),
        ("csa-joint-attention", "csa_attention"),
        ("hca-paged-stream", "hca_attention"),
        ("StreamIdxTC", "stream_index"),
    ):
        if name[1].startswith(prefix):
            return family
    return None


def analyze(capture):
    import jax

    path = Path(capture["xplane"])
    data = jax.profiler.ProfileData.from_file(str(path))
    totals, counts = collections.Counter(), collections.Counter()
    witnesses = {}
    expected_lanes = [lane["name"] for lane in capture["active_lanes"]]
    iterations = capture["iterations"]
    for name in expected_lanes:
        plane = data.find_plane_with_name(name)
        if plane is None:
            raise ValueError(f"missing active lane {name}")
        modules = [
            e for line in plane.lines if line.name == "XLA Modules" for e in line.events
        ]
        if len(modules) != iterations:
            raise ValueError(f"unexpected module coverage on {name}")
        for line in plane.lines:
            if line.name != "XLA Ops":
                continue
            for event in line.events:
                family = event_family(event.name)
                if family is None:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    stats = dict(event.stats)
                if (
                    "device_duration_ps" not in stats
                    or "Time Scale Multiplier" not in stats
                ):
                    raise ValueError(
                        "selected instruction lacks exact device duration metadata"
                    )
                duration = float(stats["device_duration_ps"]) * float(
                    stats["Time Scale Multiplier"]
                )
                totals[family] += duration
                counts[family] += 1
                witnesses.setdefault(family, event.name[:1800])
    factor = len(expected_lanes) * iterations * 1e9
    for family in ("csa_attention", "hca_attention"):
        if counts[family] and counts[family] != len(expected_lanes) * iterations:
            raise ValueError(f"incomplete single-layer {family} coverage")
    return {
        "xplane_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "active_tensorcores": len(expected_lanes),
        "iterations": iterations,
        "selected_family_ms_per_core_per_call": {
            key: value / factor for key, value in totals.items()
        },
        "occurrences": dict(counts),
        "instruction_witnesses": witnesses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.report.read_text())
    if not original["complete"]:
        raise ValueError("requires a completed experiment")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {
        "scope": __doc__,
        "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in original["cases"]:
        variants = {}
        for name, variant in case["variants"].items():
            if "profile" in variant:
                variants[name] = analyze(variant["profile"])
        if variants:
            result["cases"].append({"label": case["label"], "variants": variants})
            print(
                json.dumps(
                    {
                        "case": case["label"],
                        "families": {
                            name: value["selected_family_ms_per_core_per_call"]
                            for name, value in variants.items()
                        },
                    }
                ),
                flush=True,
            )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
