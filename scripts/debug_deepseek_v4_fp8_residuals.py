"""Trace K64 regressions and test retained residuals on both frozen chunks."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from analyze_deepseek_v4_native_profile import fingerprint
from debug_deepseek_v4_fp8_partials import PartialDiagnostic, compensated_host
from deepseek_v4_attention_diagnostics import fp8_roundtrip_cpu
from deepseek_v4_bf16_reference import round_bf16_direct
from deepseek_v4_fp8_residual_candidate import ResidualDiagnostic
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
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
        "attribution": [],
        "cases": [],
    }

    def emit(event, **values):
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": event, **values}), flush=True)

    def compute(x, w, s, *, adapter):
        return gmm(
            x,
            w[None],
            jnp.asarray([x.shape[0]], jnp.int32),
            preferred_element_type=jnp.float32,
            rhs_scale=s[None, :, None, :],
            tiling=(32, x.shape[1], 128),
            rhs_adapter=adapter,
            interpret=False,
        )

    run = jax.jit(compute, static_argnames=("adapter",))
    host = DeepSeekV4Checkpoint(prior["checkpoint"]).load_linear("layers.0.attn.wq_b")
    raw, scales = host.data, host.scales
    exp, mantissa = (raw & 127) >> 3, raw & 7
    weight64 = np.where(
        exp == 0,
        mantissa * 2.0**-9,
        (1 + mantissa / 8) * np.exp2(exp.astype(np.float64) - 7),
    )
    weight64 *= np.where(raw >> 7, -1, 1)
    weight64 *= np.repeat(
        np.repeat(np.exp2(scales.astype(np.float64) - 127), 128, 0), 128, 1
    )
    second = np.load(base / "v4-fp8-accumulator-20260911-02/baseline.npz")
    first = np.load(base / "v4-fp8-split-matrix-20260911-01/reference.npz")
    old64_second = np.load(base / "v4-fp8-partials-20260911-01/split64.npz")[
        "accumulator"
    ]
    old64_first = np.load(base / "v4-fp8-split-matrix-20260911-01/split64-m64.npz")[
        "accumulator"
    ]
    try:
        emit("target_partials")
        channels = np.r_[np.arange(50 * 128, 51 * 128), np.arange(175 * 128, 176 * 128)]
        args_small = (
            jnp.asarray(second["input"], jnp.bfloat16),
            jnp.asarray(raw[channels]),
            jnp.asarray(scales[[50, 175]]),
        )
        replay = np.asarray(run(*args_small, adapter=ResidualDiagnostic()), np.float32)
        np.testing.assert_array_equal(replay, old64_second[:, channels])
        partials = np.stack(
            [
                np.asarray(
                    run(*args_small, adapter=PartialDiagnostic(index=i, split_k=64)),
                    np.float32,
                )
                for i in range(16)
            ]
        )
        combined = compensated_host(partials)
        np.testing.assert_array_equal(combined, replay)
        for row, channel in ((43, 6431), (10, 22505)):
            col = int(np.flatnonzero(channels == channel)[0])
            products = (
                second["dequant_input"][row].astype(np.float64) * weight64[channel]
            )
            exacts = []
            for i in range(0, 1024, 64):
                exact = sum(
                    (Fraction.from_float(float(v)) for v in products[i : i + 64]),
                    Fraction(),
                )
                assert Fraction.from_float(float(exact)) == exact
                exacts.append(float(exact))
            errors = partials[:, row, col].astype(np.float64) - exacts
            report["attribution"].append(
                {
                    "position": row + 64,
                    "channel": channel,
                    "partial_fp32": partials[:, row, col].tolist(),
                    "partial_exact": exacts,
                    "partial_errors": errors.tolist(),
                    "nonzero_partial_errors": np.flatnonzero(errors).tolist(),
                    "fp32_sum": float(replay[row, col]),
                    "fp64_sum": float(sum(exacts)),
                    "host_reconstruction_bitwise": True,
                    "rational_partials_verified": True,
                }
            )
        np.savez_compressed(
            out / "target-partials.npz",
            partials=partials,
            channels=channels,
            replay=replay,
        )
        emit("attribution", coordinates=report["attribution"])
        for label, data, baseline_key, old64, valid in (
            ("p64", second, "rounded", old64_second, 63),
            ("p0", first, "baseline_bf16", old64_first, 64),
        ):
            x, exact = data["input"], data["fp64"]
            qx, _ = fp8_roundtrip_cpu(x)
            target = round_bf16_direct(exact)
            staged = round_bf16_direct(exact.astype(np.float32))
            bad0 = data[baseline_key][:valid] != target[:valid]
            disputed = bad0.copy()
            inputs = (
                jnp.asarray(x, jnp.bfloat16),
                jnp.asarray(raw),
                jnp.asarray(scales),
            )
            for k in (64, 16):
                emit("components", chunk=label, split_k=k)
                high, low = [
                    np.asarray(
                        run(
                            *inputs,
                            adapter=ResidualDiagnostic(split_k=k, component=component),
                        ),
                        np.float32,
                    )
                    for component in ("high", "low")
                ]
                pair64 = high.astype(np.float64) + low.astype(np.float64)
                for preserve in (False, True):
                    name = f"{label}-k{k}-residual{int(preserve)}"
                    value = np.asarray(
                        run(
                            *inputs,
                            adapter=ResidualDiagnostic(
                                split_k=k, preserve_bf16_residual=preserve
                            ),
                        ),
                        np.float32,
                    )
                    if k == 64 and not preserve:
                        np.testing.assert_array_equal(value, old64)
                    expected = round_bf16_direct(
                        pair64 if preserve else pair64.astype(np.float32)
                    )
                    output = round_bf16_direct(value)
                    np.testing.assert_array_equal(output, expected)
                    np.testing.assert_array_equal(
                        np.asarray(jnp.asarray(value).astype(jnp.bfloat16), np.float32),
                        output,
                    )
                    bad = output[:valid] != target[:valid]
                    disputed |= bad
                    entry = {
                        "name": name,
                        "valid_elements": valid * 32768,
                        "different_bf16_elements": int(bad.sum()),
                        "fixed_elements": int((bad0 & ~bad).sum()),
                        "new_differences": int((~bad0 & bad).sum()),
                        "versus_correct_fp32_then_bf16": int(
                            np.count_nonzero(output[:valid] != staged[:valid])
                        ),
                        "high_low_rounding_matches_cpu_bitwise": True,
                        "remaining_coordinates": [
                            [int(r + (64 if label == "p64" else 0)), int(c)]
                            for r, c in np.argwhere(bad)
                        ],
                    }
                    report["cases"].append(entry)
                    np.savez_compressed(
                        out / (name + ".npz"), carrier=value, output=output
                    )
                    emit("candidate", **entry)
                np.savez_compressed(
                    out / f"{label}-k{k}-components.npz", high=high, low=low
                )
            # Independently adjudicate every disputed coordinate in both chunks.
            for row, col in np.argwhere(disputed):
                products = qx[row].astype(np.float64) * weight64[col]
                exact_r = sum(
                    (Fraction.from_float(float(v)) for v in products), Fraction()
                )
                assert Fraction.from_float(float(exact[row, col])) == exact_r
            report[label + "_rational_coordinates_verified"] = int(disputed.sum())
            emit("chunk_complete", chunk=label)
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
