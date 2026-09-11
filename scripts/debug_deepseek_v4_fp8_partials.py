"""Isolate partial-dot error from compensated-summation error on frozen inputs.

Diagnostic adapters only. Production sources and archived experiments stay
unchanged. FP64 comparisons are arithmetic controls, not model acceptance.
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from analyze_deepseek_v4_native_profile import fingerprint
from debug_deepseek_v4_fp8_accumulator import SplitKDiagnostic
from deepseek_v4_bf16_reference import round_bf16_direct
from sgl_jax.srt.kernels.deepseek_v4.fp8 import CheckpointFP8Rhs, decode_weight_tile
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


@dataclass(frozen=True)
class PartialDiagnostic(CheckpointFP8Rhs):
    index: int = 0
    split_k: int = 128
    name = "diagnostic_partial_dot"

    def dot(self, lhs, rhs, scales):
        weight = decode_weight_tile(rhs, scales)
        lhs = activation_fp8_roundtrip(lhs)
        begin = self.index * self.split_k
        return jax.lax.dot_general(
            lhs[:, begin : begin + self.split_k],
            weight[:, begin : begin + self.split_k],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )


def compensated_host(partials):
    """Explicit FP32 NumPy operations: no fused matmul/accumulation."""
    high = np.zeros_like(partials[0], dtype=np.float32)
    low = np.zeros_like(high)
    for partial in partials:
        total = high + partial
        virtual = total - high
        low = low + ((high - (total - virtual)) + (partial - virtual))
        high = total
    return high + low


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, out = args.profiles, args.output
    previous = base / "v4-fp8-accumulator-20260911-02"
    prior = json.loads((previous / "report.json").read_text())
    assert prior["complete"] and fingerprint() == prior["source_fingerprint"]
    assert jax.default_backend() == "tpu" and len(jax.devices()) == 4
    out.mkdir(exist_ok=False)
    baseline = np.load(previous / "baseline.npz")
    candidate = np.load(previous / "split128_compensated.npz")
    x, qx = baseline["input"], baseline["dequant_input"]
    exact = baseline["fp64"]
    target = round_bf16_direct(exact)
    staged = round_bf16_direct(exact.astype(np.float32))
    wrong0 = baseline["rounded"][:63] != target[:63]
    new_bad = (~wrong0) & (candidate["output"][:63] != target[:63])
    coordinates = [(int(r), int(c)) for r, c in np.argwhere(new_bad)]
    assert coordinates == [(1, 12873), (38, 20773)]
    report = {
        "complete": False,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "valid_elements": 63 * 32768,
        "variants": [],
        "regressions": [],
    }

    def emit(event, **values):
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": event, "time": time.time(), **values}), flush=True)

    trajectory = json.loads(
        (base / "v4-cpu-trajectory-20260911-02/report.json").read_text()
    )
    host = DeepSeekV4Checkpoint(trajectory["checkpoint"]).load_linear(
        "layers.0.attn.wq_b"
    )
    raw, scales = host.data, host.scales
    device = (jnp.asarray(x, jnp.bfloat16), jnp.asarray(raw), jnp.asarray(scales))

    def compute(value, weight, scale, *, adapter):
        return gmm(
            value,
            weight[None],
            jnp.asarray([value.shape[0]], jnp.int32),
            preferred_element_type=jnp.float32,
            rhs_scale=scale[None, :, None, :],
            tiling=(32, x.shape[1], 128),
            rhs_adapter=adapter,
            interpret=False,
        )

    run = jax.jit(compute, static_argnames=("adapter",))
    try:
        emit("replay")
        for adapter, expected in (
            (CheckpointFP8Rhs(), baseline["accumulator"]),
            (SplitKDiagnostic(), candidate["accumulator"]),
        ):
            np.testing.assert_array_equal(
                np.asarray(run(*device, adapter=adapter)), expected
            )
        report["baseline_and_candidate_fp32_replay_bitwise"] = True
        magnitude = raw & 127
        exponent, mantissa = magnitude >> 3, magnitude & 7
        weight64 = np.where(
            exponent == 0,
            mantissa * 2.0**-9,
            (1 + mantissa / 8) * np.exp2(exponent.astype(np.float64) - 7),
        )
        weight64 *= np.where(raw >> 7, -1, 1)
        weight64 *= np.repeat(
            np.repeat(np.exp2(scales.astype(np.float64) - 127), 128, axis=0),
            128,
            axis=1,
        )
        partials, reference_parts = [], []
        for index in range(8):
            emit("partial", index=index)
            partials.append(
                np.asarray(
                    run(*device, adapter=PartialDiagnostic(index=index)), np.float32
                )
            )
            begin = index * 128
            reference_parts.append(
                qx[:, begin : begin + 128].astype(np.float64)
                @ weight64[:, begin : begin + 128].T
            )
        partials, reference_parts = np.stack(partials), np.stack(reference_parts)
        np.testing.assert_array_equal(reference_parts.sum(0), exact)
        combined = compensated_host(partials)
        sum_of_partial64 = partials.astype(np.float64).sum(0)
        report["host_compensated_vs_candidate_fp32_differences"] = int(
            np.count_nonzero(combined != candidate["accumulator"])
        )
        report["host_compensated_vs_rounded_partial_sum_differences"] = int(
            np.count_nonzero(combined != sum_of_partial64.astype(np.float32))
        )
        report["partial_fp32_vs_correctly_rounded_differences"] = int(
            np.count_nonzero(partials != reference_parts.astype(np.float32))
        )
        report["partial_fp32_vs_exact_differences"] = int(
            np.count_nonzero(partials.astype(np.float64) != reference_parts)
        )
        for row, channel in coordinates + [(44, 7656)]:
            products = qx[row].astype(np.float64) * weight64[channel]
            rational = [
                sum(
                    (Fraction.from_float(float(v)) for v in products[i : i + 128]),
                    Fraction(),
                )
                for i in range(0, 1024, 128)
            ]
            for i, value in enumerate(rational):
                assert Fraction.from_float(float(value)) == value
                assert float(value) == reference_parts[i, row, channel]
            report["regressions"].append(
                {
                    "position": row + 64,
                    "channel": channel,
                    "baseline_fp32": float(baseline["accumulator"][row, channel]),
                    "candidate_fp32": float(candidate["accumulator"][row, channel]),
                    "host_combined_fp32": float(combined[row, channel]),
                    "exact": float(exact[row, channel]),
                    "sum_of_traced_partial64": float(sum_of_partial64[row, channel]),
                    "partial_fp32": partials[:, row, channel].tolist(),
                    "partial_fp64": reference_parts[:, row, channel].tolist(),
                    "partial_errors": (
                        partials[:, row, channel].astype(np.float64)
                        - reference_parts[:, row, channel]
                    ).tolist(),
                    "rational_partials_verified": True,
                }
            )
        np.savez_compressed(
            out / "partials.npz",
            partial_fp32=partials,
            partial_fp64=reference_parts,
            host_combined=combined,
        )
        emit(
            "attribution",
            **{
                k: v
                for k, v in report.items()
                if k.startswith(("host_", "partial_"))
            },
            coordinates=report["regressions"],
        )
        for split_k in (64, 32):
            emit("candidate", split_k=split_k)
            value = np.asarray(
                run(*device, adapter=SplitKDiagnostic(split_k=split_k)), np.float32
            )
            output = round_bf16_direct(value)
            bad = output[:63] != target[:63]
            entry = {
                "split_k": split_k,
                "different_bf16_elements": int(bad.sum()),
                "fixed_elements": int((wrong0 & ~bad).sum()),
                "new_differences": int((~wrong0 & bad).sum()),
                "versus_correct_fp32_then_bf16": int(
                    np.count_nonzero(output[:63] != staged[:63])
                ),
                "coordinate_fp32": [
                    float(value[r, c]) for r, c in coordinates + [(44, 7656)]
                ],
            }
            report["variants"].append(entry)
            np.savez_compressed(
                out / f"split{split_k}.npz", accumulator=value, output=output
            )
            emit("candidate_complete", **entry)
        assert fingerprint() == prior["source_fingerprint"]
        report["complete"] = True
        emit("complete")
    except BaseException:
        import traceback

        report["error"] = traceback.format_exc()
        emit("failed", error=report["error"])
        raise


if __name__ == "__main__":
    main()
