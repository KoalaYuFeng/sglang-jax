"""Read-only-runtime attribution of the last two residual-aware Q projections."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from analyze_deepseek_v4_native_profile import fingerprint
from debug_deepseek_v4_fp8_partials import PartialDiagnostic
from deepseek_v4_attention_diagnostics import fp8_roundtrip_cpu
from deepseek_v4_bf16_reference import round_bf16_direct
from deepseek_v4_fp8_residual_candidate import ResidualDiagnostic
from sgl_jax.srt.kernels.deepseek_v4.fp8 import decode_weight_tile
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, out = args.profiles, args.output
    prior = json.loads((base / "v4-cpu-trajectory-20260911-02/report.json").read_text())
    assert fingerprint() == prior["source_fingerprint"]
    assert jax.default_backend() == "tpu" and len(jax.devices()) == 4
    out.mkdir(exist_ok=False)
    report = {
        "complete": False,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coordinates": [],
    }

    def emit(event, **values):
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": event, **values}), flush=True)

    def compute(x, w, s, *, adapter):
        return gmm(
            x,
            w[None],
            jnp.asarray([64], jnp.int32),
            preferred_element_type=jnp.float32,
            rhs_scale=s[None, :, None, :],
            tiling=(32, 1024, 128),
            rhs_adapter=adapter,
            interpret=False,
        )

    run = jax.jit(compute, static_argnames=("adapter",))
    host = DeepSeekV4Checkpoint(prior["checkpoint"]).load_linear("layers.0.attn.wq_b")
    try:
        for ordinal, position, channel, row in ((1, 61, 10345, 61), (2, 88, 26881, 24)):
            emit("start", position=position, channel=channel)
            folder = base / f"v4-fp8-residual-matrix-20260911-0{ordinal}"
            reference = np.load(folder / "reference.npz")
            frozen = np.load(folder / "residual8-m64.npz")
            x = reference["input"]
            start = channel // 128 * 128
            col = channel - start
            raw = host.data[start : start + 128]
            scales = host.scales[start // 128 : start // 128 + 1]
            inputs = (
                jnp.asarray(x, jnp.bfloat16),
                jnp.asarray(raw),
                jnp.asarray(scales),
            )
            qx, _ = fp8_roundtrip_cpu(x)
            np.testing.assert_array_equal(
                np.asarray(activation_fp8_roundtrip(inputs[0]), np.float32), qx
            )
            exp, mantissa = (raw & 127) >> 3, raw & 7
            w64 = np.where(
                exp == 0,
                mantissa * 2.0**-9,
                (1 + mantissa / 8) * np.exp2(exp.astype(np.float64) - 7),
            )
            w64 *= np.where(raw >> 7, -1, 1)
            w64 *= np.repeat(np.exp2(scales.astype(np.float64) - 127), 128, axis=1)
            np.testing.assert_array_equal(
                np.asarray(decode_weight_tile(inputs[1], inputs[2]), np.float32), w64
            )
            carrier = np.asarray(
                run(
                    *inputs,
                    adapter=ResidualDiagnostic(split_k=8, preserve_bf16_residual=True),
                ),
                np.float32,
            )
            np.testing.assert_array_equal(
                carrier, frozen["accumulator"][:, start : start + 128]
            )
            high, low = [
                np.asarray(
                    run(*inputs, adapter=ResidualDiagnostic(split_k=8, component=c)),
                    np.float32,
                )
                for c in ("high", "low")
            ]
            pair = high.astype(np.float64) + low.astype(np.float64)
            np.testing.assert_array_equal(
                round_bf16_direct(pair), round_bf16_direct(carrier)
            )
            partials = []
            for index in range(128):
                if index % 16 == 0:
                    emit("partial", position=position, index=index)
                partials.append(
                    np.asarray(
                        run(*inputs, adapter=PartialDiagnostic(index=index, split_k=8)),
                        np.float32,
                    )
                )
            partials = np.stack(partials)
            h = np.zeros_like(high)
            l = np.zeros_like(low)
            target_steps = []
            for index, p in enumerate(partials):
                total = h + p
                virtual = total - h
                error = (h - (total - virtual)) + (p - virtual)
                next_l = l + error
                target_steps.append(
                    [
                        float(total[row, col]),
                        float(error[row, col]),
                        float(next_l[row, col]),
                        float(
                            np.float64(next_l[row, col])
                            - np.float64(l[row, col])
                            - np.float64(error[row, col])
                        ),
                    ]
                )
                h, l = total, next_l
            np.testing.assert_array_equal(h, high)
            np.testing.assert_array_equal(l, low)
            exact_products = [
                Fraction.from_float(float(a)) * Fraction.from_float(float(b))
                for a, b in zip(qx[row], w64[col])
            ]
            exact_parts = [
                sum(exact_products[i : i + 8], Fraction()) for i in range(0, 1024, 8)
            ]
            exact = sum(exact_parts, Fraction())
            assert Fraction.from_float(float(reference["fp64"][row, channel])) == exact
            traced_sum = sum(
                (Fraction.from_float(float(v)) for v in partials[:, row, col]),
                Fraction(),
            )
            pair_exact = Fraction.from_float(
                float(high[row, col])
            ) + Fraction.from_float(float(low[row, col]))
            errors = []
            for i, (p, e) in enumerate(zip(partials[:, row, col], exact_parts)):
                difference = Fraction.from_float(float(p)) - e
                if difference:
                    errors.append(
                        {
                            "index": i,
                            "k_range": [i * 8, i * 8 + 8],
                            "partial_fp32": float(p),
                            "partial_exact": float(e),
                            "error": float(difference),
                            "correctly_rounded_fp32": float(np.float32(float(e)))
                            == float(p),
                            "products": [
                                float(v) for v in exact_products[i * 8 : i * 8 + 8]
                            ],
                        }
                    )
            entry = {
                "position": position,
                "channel": channel,
                "carrier_replay_bitwise": True,
                "activation_and_weight_decode_bitwise": True,
                "host_high_low_reconstruction_bitwise": True,
                "exact_sum": float(exact),
                "sum_of_partial_fp32": float(traced_sum),
                "high": float(high[row, col]),
                "low": float(low[row, col]),
                "high_plus_low_exact": float(pair_exact),
                "partial_error_total": float(traced_sum - exact),
                "compensation_error": float(pair_exact - traced_sum),
                "final_pair_error": float(pair_exact - exact),
                "partial_errors": errors,
                "low_sum_rounding_steps": [
                    {
                        "index": i,
                        "high": v[0],
                        "error_term": v[1],
                        "low": v[2],
                        "low_rounding_error": v[3],
                    }
                    for i, v in enumerate(target_steps)
                    if v[3]
                ],
                "exact_bf16": float(round_bf16_direct(float(exact))),
                "rounded_partial_sum_bf16": float(round_bf16_direct(float(traced_sum))),
                "candidate_bf16": float(round_bf16_direct(pair[row, col])),
            }
            report["coordinates"].append(entry)
            np.savez_compressed(
                out / f"position{position}.npz",
                input=x,
                quantized_input=qx,
                raw=raw,
                scales=scales,
                partials=partials,
                high=high,
                low=low,
                carrier=carrier,
                steps=np.asarray(target_steps),
            )
            emit("attribution", **entry)
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
