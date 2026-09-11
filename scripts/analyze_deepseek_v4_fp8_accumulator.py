"""Re-adjudicate saved accumulator experiments using direct FP64/BF16 rounding."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from deepseek_v4_bf16_reference import round_bf16_direct
from deepseek_v4_numerical_acceptance import tensor_metrics
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run / "direct-rounding"
    out.mkdir(exist_ok=False)
    original = json.loads((args.run / "report.json").read_text())
    assert original["complete"]
    baseline = np.load(args.run / "baseline.npz")
    target = round_bf16_direct(baseline["fp64"])
    bad0 = baseline["rounded"][:63] != target[:63]
    union = bad0.copy()
    report = {
        "complete": False,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "source_fingerprint": original["source_fingerprint"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "valid_elements": 63 * 32768,
        "torch_reference_different_elements": int(
            np.count_nonzero(target[:63] != baseline["fp64_bf16"][:63])
        ),
        "baseline_wrong_bf16_elements": int(bad0.sum()),
        "variants": [],
    }
    for entry in original["variants"]:
        data = np.load(args.run / (entry["name"] + ".npz"))
        value = data["accumulator"]
        np.testing.assert_array_equal(round_bf16_direct(value), data["output"])
        bad = data["output"][:63] != target[:63]
        union |= bad
        report["variants"].append(
            {
                "name": entry["name"],
                "wrong_bf16_elements": int(bad.sum()),
                "fixed_elements": int((bad0 & ~bad).sum()),
                "new_wrong_elements": int((~bad0 & bad).sum()),
                "bf16": tensor_metrics(data["output"][:63], target[:63]),
            }
        )
    # Validate FP64 reduction and direct rounding using exact rational sums at
    # every coordinate where any tested implementation disagrees with reference.
    prior = json.loads(
        (args.profiles / "v4-cpu-trajectory-20260911-02/report.json").read_text()
    )
    host = DeepSeekV4Checkpoint(prior["checkpoint"]).load_linear("layers.0.attn.wq_b")
    checks = []
    for row, channel in np.argwhere(union):
        raw = host.data[channel]
        exp, mantissa = (raw & 127) >> 3, raw & 7
        weight = np.where(
            exp == 0,
            mantissa * 2.0**-9,
            (1 + mantissa / 8) * np.exp2(exp.astype(np.float64) - 7),
        )
        weight *= np.where(raw >> 7, -1, 1)
        weight *= np.repeat(
            np.exp2(host.scales[channel // 128].astype(np.float64) - 127), 128
        )
        products = baseline["dequant_input"][row].astype(np.float64) * weight
        exact = sum((Fraction.from_float(float(p)) for p in products), Fraction())
        # Require representability as well as agreement: no FP64 reduction error.
        assert Fraction.from_float(float(exact)) == exact
        assert float(exact) == baseline["fp64"][row, channel]
        rounded = float(target[row, channel])
        bits = np.asarray(rounded, np.float32).view(np.uint32).item() >> 16
        # Check both adjacent finite BF16 values and nearest-even tie ownership.
        distance = abs(exact - Fraction.from_float(rounded))
        for neighbor in (bits - 1, bits + 1):
            other = np.asarray(neighbor << 16, np.uint32).view(np.float32).item()
            other_distance = abs(exact - Fraction.from_float(other))
            assert distance < other_distance or (
                distance == other_distance and bits % 2 == 0
            )
        checks.append(
            {
                "position": int(row + 64),
                "channel": int(channel),
                "fp64": float(exact),
                "bf16": rounded,
            }
        )
    report["rational_checks"] = checks
    report["rational_checked_coordinates"] = len(checks)
    report["complete"] = True
    np.savez_compressed(out / "reference.npz", direct_bf16=target)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "rational_checks"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
