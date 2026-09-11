"""Replay and exact-rational adjudication of residual matrix artifacts."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from deepseek_v4_attention_diagnostics import fp8_roundtrip_cpu
from deepseek_v4_bf16_reference import round_bf16_direct
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args()
    base = args.profiles
    output = base / "v4-fp8-residuals-20260911-01/matrix-verification.json"
    assert not output.exists()
    prior = json.loads((base / "v4-cpu-trajectory-20260911-02/report.json").read_text())
    host = DeepSeekV4Checkpoint(prior["checkpoint"]).load_linear("layers.0.attn.wq_b")
    results = []
    for ordinal, label, offset, valid in ((1, "p0", 0, 64), (2, "p64", 64, 63)):
        folder = base / f"v4-fp8-residual-matrix-20260911-0{ordinal}"
        reference = np.load(folder / "reference.npz")
        qx, _ = fp8_roundtrip_cpu(reference["input"])
        target = round_bf16_direct(reference["fp64"])
        for k in (16, 8):
            data = np.load(folder / f"residual{k}-m64.npz")
            if k == 16:
                replay = np.load(
                    base / f"v4-fp8-residuals-20260911-01/{label}-k16-residual1.npz"
                )
                np.testing.assert_array_equal(data["accumulator"], replay["carrier"])
                np.testing.assert_array_equal(data["output"], replay["output"])
            checks = []
            for row, channel in np.argwhere(data["output"][:valid] != target[:valid]):
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
                exact = sum(
                    (
                        Fraction.from_float(float(a)) * Fraction.from_float(float(b))
                        for a, b in zip(qx[row], weight)
                    ),
                    Fraction(),
                )
                assert (
                    Fraction.from_float(float(reference["fp64"][row, channel])) == exact
                )
                checks.append(
                    {
                        "position": int(row + offset),
                        "channel": int(channel),
                        "fp64": float(exact),
                        "candidate_bf16": float(data["output"][row, channel]),
                        "reference_bf16": float(target[row, channel]),
                    }
                )
            results.append({"chunk": label, "split_k": k, "remaining": checks})
    report = {
        "complete": True,
        "diagnostic_only": True,
        "k16_replay_bitwise": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
