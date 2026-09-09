"""Verify actual shared-GMM custom calls in every layer of an accepted V4 HLO."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from deepseek_v4_kernel_evidence import _hlo_graph, _owner


def compiled_moe_evidence(text, *, layers=43, backend="gmm"):
    if backend not in ("gmm", "gmm_tuned"):
        raise ValueError("GMM evidence requires an explicit grouped backend")
    family = (
        "gmm_checkpoint_fp4_candidate_scale_kn_packed_scale-"
        if backend == "gmm_tuned"
        else "gmm_checkpoint_fp4-"
    )
    nodes, consumers = _hlo_graph(text)
    counts = {layer: {"gate_up": 0, "down": 0} for layer in range(layers)}
    witnesses = []
    for identity, node in nodes.items():
        if node["opcode"] != "custom-call" or not identity[1].startswith(
            "gmm_checkpoint_fp4"
        ):
            continue
        if not identity[1].startswith(family):
            raise AssertionError(
                f"wrong FP4 adapter for requested {backend}: {identity}"
            )
        shape = re.search(r"-k_(\d+)-n_(\d+)-", identity[1])
        if shape is None:
            raise AssertionError(f"missing logical GMM dimensions: {identity}")
        projection = {(4096, 2048): "gate_up", (2048, 4096): "down"}.get(
            tuple(map(int, shape.groups()))
        )
        if projection is None:
            raise AssertionError(f"unexpected V4 GMM projection: {identity}")
        owner = _owner(identity, nodes, consumers)
        if owner["layer"] not in counts:
            raise AssertionError(f"out-of-range GMM layer: {owner}")
        counts[owner["layer"]][projection] += 1
        witnesses.append(
            {"instruction": "/%".join(identity), "projection": projection, **owner}
        )
    if any(value != {"gate_up": 2, "down": 1} for value in counts.values()):
        raise AssertionError(
            f"shared FP4 GMM must cover W1/W3/W2 in every layer: {counts}"
        )
    forbidden = []
    # Full expert *weight or scale* shapes, not the allowed smaller VMEM tiles.
    shapes = r"1,(?:2048,(?:2048|128)|4096,(?:1024|64)|128,2048|64,4096)"
    for line in text.splitlines():
        instruction = line.partition("backend_config=")[0]
        if re.match(
            r"\s+(?:ROOT )?%[-\w.]+ = u8\[" + shapes + r"\]", instruction
        ) and re.search(r"\b(?:fusion|dynamic-slice)\(", instruction):
            forbidden.append(instruction.strip())
    if forbidden:
        raise AssertionError(
            f"whole-expert weight/scale slices remain: {forbidden[:4]}"
        )
    scale_copies = [
        line.strip()
        for line in text.splitlines()
        if re.search(
            r"= u8\[64,(?:4096,64|64,4096|2048,128|128,2048)\].*? copy\(", line
        )
    ]
    if backend == "gmm_tuned" and scale_copies:
        raise AssertionError(
            "tuned model still contains full expert-scale layout copies"
        )
    return {
        "complete": True,
        "hlo_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "layer_count": layers,
        "backend": backend,
        "full_scale_layout_copies": scale_copies,
        "custom_call_instructions": len(witnesses),
        "per_layer_projection_counts": counts,
        "whole_expert_weight_or_scale_slices": forbidden,
        "witnesses": witnesses,
        "scope": "compiled SSA call ownership, not dynamic invocation counts or measured HBM traffic",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if not receipt.get("complete") or receipt.get("moe_backend") not in (
        "gmm",
        "gmm_tuned",
    ):
        raise ValueError("requires a completed native GMM correctness receipt")
    saved = receipt["kernel_evidence"]
    path = args.receipt.parent / saved["hlo_file"]
    text = path.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != saved["hlo_sha256"]:
        raise ValueError("optimized HLO does not match the executed native receipt")
    report = compiled_moe_evidence(text, backend=receipt["moe_backend"])
    report["framework_source_fingerprint"] = receipt["framework_source_fingerprint"]
    report["native_receipt_sha256"] = hashlib.sha256(
        args.receipt.read_bytes()
    ).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("witnesses", "per_layer_projection_counts")
            }
        )
    )


if __name__ == "__main__":
    main()
