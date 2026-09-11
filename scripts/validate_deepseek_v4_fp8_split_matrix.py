"""Held-out projection-only controls for split-K arithmetic; not a model gate."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from analyze_deepseek_v4_native_profile import fingerprint
from debug_deepseek_v4_fp8_accumulator import SplitKDiagnostic
from deepseek_v4_attention_diagnostics import fp8_roundtrip_cpu
from deepseek_v4_bf16_reference import round_bf16_direct
from sgl_jax.srt.kernels.deepseek_v4.fp8 import CheckpointFP8Rhs
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(
        (args.profiles / "v4-cpu-trajectory-20260911-02/report.json").read_text()
    )
    assert fingerprint() == prior["source_fingerprint"]
    assert jax.default_backend() == "tpu" and len(jax.devices()) == 4
    out = args.output
    out.mkdir(exist_ok=False)
    report = {
        "complete": False,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "positions": [0, 63],
        "cases": [],
    }

    def emit(event, **values):
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": event, **values}), flush=True)

    def compute(x, w, s, *, adapter, tile_m):
        return gmm(
            x,
            w[None],
            jnp.asarray([x.shape[0]], jnp.int32),
            preferred_element_type=jnp.float32,
            rhs_scale=s[None, :, None, :],
            tiling=(tile_m, x.shape[1], 128),
            rhs_adapter=adapter,
            interpret=False,
        )

    run = jax.jit(compute, static_argnames=("adapter", "tile_m"))
    native = np.load(args.profiles / "v4-cpu-attention-stages-20260911-02/native.npz")
    x = native["qr"][:64]
    qx, _ = fp8_roundtrip_cpu(x)
    host = DeepSeekV4Checkpoint(prior["checkpoint"]).load_linear("layers.0.attn.wq_b")
    raw, scales = host.data, host.scales
    inputs = (jnp.asarray(x, jnp.bfloat16), jnp.asarray(raw), jnp.asarray(scales))
    try:
        emit("baseline")
        np.testing.assert_array_equal(
            np.asarray(activation_fp8_roundtrip(inputs[0]), np.float32), qx
        )
        baseline = np.asarray(
            run(*inputs, adapter=CheckpointFP8Rhs(), tile_m=32), np.float32
        )
        baseline_bf16 = round_bf16_direct(baseline)
        np.testing.assert_array_equal(baseline_bf16, native["attn.wq_b.output"][:64])
        report["historical_baseline_bf16_bitwise"] = True
        exponent, mantissa = (raw & 127) >> 3, raw & 7
        weight = np.where(
            exponent == 0,
            mantissa * 2.0**-9,
            (1 + mantissa / 8) * np.exp2(exponent.astype(np.float64) - 7),
        )
        weight *= np.where(raw >> 7, -1, 1)
        weight *= np.repeat(
            np.repeat(np.exp2(scales.astype(np.float64) - 127), 128, 0), 128, 1
        )
        exact = qx.astype(np.float64) @ weight.T
        direct = round_bf16_direct(exact)
        staged = round_bf16_direct(exact.astype(np.float32))
        np.savez_compressed(
            out / "reference.npz",
            input=x,
            fp64=exact,
            direct_bf16=direct,
            staged_bf16=staged,
            baseline_accumulator=baseline,
            baseline_bf16=baseline_bf16,
        )
        variants = [
            ("original", CheckpointFP8Rhs()),
            *[(f"split{k}", SplitKDiagnostic(split_k=k)) for k in (128, 64, 32)],
        ]
        for name, adapter in variants:
            full = None
            for rows, tile_m in ((64, 32), (8, 8)):
                emit("variant", name=name, rows=rows)
                values = np.asarray(
                    run(
                        inputs[0][:rows],
                        inputs[1],
                        inputs[2],
                        adapter=adapter,
                        tile_m=tile_m,
                    ),
                    np.float32,
                )
                outputs = round_bf16_direct(values)
                bad0 = baseline_bf16[:rows] != direct[:rows]
                bad = outputs != direct[:rows]
                if rows == 64:
                    full = values
                entry = {
                    "name": name,
                    "rows": rows,
                    "tile_m": tile_m,
                    "different_bf16_elements": int(bad.sum()),
                    "fixed_elements": int((bad0 & ~bad).sum()),
                    "new_differences": int((~bad0 & bad).sum()),
                    "versus_correct_fp32_then_bf16": int(
                        np.count_nonzero(outputs != staged[:rows])
                    ),
                    "fp32_row_subset_matches_full": bool(
                        np.array_equal(values, full[:rows])
                    ),
                }
                np.savez_compressed(
                    out / f"{name}-m{rows}.npz", accumulator=values, output=outputs
                )
                report["cases"].append(entry)
                emit("variant_complete", **entry)
            # wq_b output-channel partition: independent shards on four physical
            # devices. No all-reduce, scheduler, or full-model TP4 claim.
            emit("four_device_projection", name=name)
            pieces = []
            for index, device in enumerate(jax.devices()):
                start, end = index * 8192, (index + 1) * 8192
                args_on_device = (
                    jax.device_put(x, device).astype(jnp.bfloat16),
                    jax.device_put(raw[start:end], device),
                    jax.device_put(scales[start // 128 : end // 128], device),
                )
                piece = run(*args_on_device, adapter=adapter, tile_m=32)
                assert piece.devices() == {device}
                pieces.append(piece)
            assembled = np.concatenate(
                [np.asarray(value, np.float32) for value in pieces], axis=1
            )
            np.testing.assert_array_equal(assembled, full)
            report["cases"][-1]["four_device_projection_bitwise"] = True
            emit("four_device_complete", name=name)
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
