"""Shared-GMM FP32-output trace and isolated split-K numerical experiments.

Same adapter/dot/tiling as production, but FP32 output exposes the accumulator.
BF16 replay is mandatory. FP64 adjudication is not an official acceptance gate.
No serving sources, defaults, or reference arithmetic are changed.
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_attention_diagnostics import fp8_roundtrip_cpu
from deepseek_v4_numerical_acceptance import tensor_metrics
from sgl_jax.srt.kernels.deepseek_v4.fp8 import (
    CheckpointFP8Rhs,
    decode_weight_tile,
    fp8_linear,
)
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


@dataclass(frozen=True)
class SplitKDiagnostic(CheckpointFP8Rhs):
    """Numerical prototype only: short dot products plus compensated FP32 sum."""

    split_k: int = 128
    compensate: bool = True
    name = "diagnostic_split_k_fp8"

    def dot(self, lhs, rhs, scales):
        weight = decode_weight_tile(rhs, scales)
        if self.quantize_activation:
            lhs = activation_fp8_roundtrip(lhs)
        high = jnp.zeros((lhs.shape[0], weight.shape[0]), jnp.float32)
        low = jnp.zeros_like(high)
        for begin in range(0, lhs.shape[1], self.split_k):
            partial = jax.lax.dot_general(
                lhs[:, begin : begin + self.split_k],
                weight[:, begin : begin + self.split_k],
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            # Pallas TPU does not lower optimization_barrier. This prototype
            # must be judged from measured outputs, not assumed TwoSum semantics.
            total = high + partial
            if self.compensate:
                virtual = total - high
                error = (high - (total - virtual)) + (partial - virtual)
                low = low + error
            high = total
        return high + low if self.compensate else high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    base, out = options.profiles, options.output
    prior = json.loads((base / "v4-cpu-trajectory-20260911-02/report.json").read_text())
    assert fingerprint() == prior["source_fingerprint"]
    assert jax.default_backend() == "tpu" and len(jax.devices()) == 4
    out.mkdir(exist_ok=False)
    native = np.load(base / "v4-cpu-attention-stages-20260911-02/native.npz")
    cp = DeepSeekV4Checkpoint(prior["checkpoint"])
    x = np.pad(native["qr"][64:127], ((0, 1), (0, 0)))
    host = cp.load_linear("layers.0.attn.wq_b")
    raw, scales = host.data, host.scales
    k = x.shape[1]
    device_x, device_w, device_s = (
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(raw),
        jnp.asarray(scales),
    )
    torch.set_num_threads(8)
    report = {
        "complete": False,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_shape": list(x.shape),
        "valid_positions": [64, 126],
        "variants": [],
        "instrumentation": "unchanged shared GMM/adapter with FP32 output; BF16 replay required",
    }

    def emit(event, **values):
        print(json.dumps(dict(event=event, time=time.time(), **values)), flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def bf16(value):
        return torch.from_numpy(np.array(value)).to(torch.bfloat16).float().numpy()

    def compute(value, weight, scale, *, tile_m, adapter):
        padded = (value.shape[0] + tile_m - 1) // tile_m * tile_m
        lhs = jnp.pad(value, ((0, padded - value.shape[0]), (0, 0)))
        result = gmm(
            lhs,
            weight[None],
            jnp.asarray([padded], jnp.int32),
            preferred_element_type=jnp.float32,
            rhs_scale=scale[None, :, None, :],
            tiling=(tile_m, k, 128),
            rhs_adapter=adapter,
            interpret=False,
        )
        return result[: value.shape[0]]

    run = jax.jit(compute, static_argnames=("tile_m", "adapter"))
    try:
        emit("baseline")
        ordinary = np.asarray(fp8_linear(device_x, device_w, device_s), np.float32)
        np.testing.assert_array_equal(ordinary[:63], native["attn.wq_b.output"][64:127])
        accumulator = np.asarray(
            run(device_x, device_w, device_s, tile_m=32, adapter=CheckpointFP8Rhs()),
            np.float32,
        )
        rounded = bf16(accumulator)
        np.testing.assert_array_equal(rounded, ordinary)
        tpu_cast = np.asarray(jnp.asarray(accumulator).astype(jnp.bfloat16), np.float32)
        np.testing.assert_array_equal(tpu_cast, rounded)
        report["baseline_bf16_replay_bitwise"] = True
        report["standalone_tpu_cast_matches_cpu_bitwise"] = True
        emit("fp64_reference")
        qx, _ = fp8_roundtrip_cpu(x)
        np.testing.assert_array_equal(
            np.asarray(activation_fp8_roundtrip(device_x), np.float32), qx
        )
        magnitude = raw & 127
        exponent, mantissa = magnitude >> 3, magnitude & 7
        assert not np.any(magnitude == 127)
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
        for block in (16, 59):
            decoded = decode_weight_tile(
                device_w[block * 128 : (block + 1) * 128], device_s[block : block + 1]
            )
            np.testing.assert_array_equal(
                np.asarray(decoded, np.float32),
                weight64[block * 128 : (block + 1) * 128].astype(np.float32),
            )
        report["input_quantization_bitwise"] = True
        report["target_weight_tiles_bitwise"] = True
        exact = qx.astype(np.float64) @ weight64.T
        exact_bf16 = bf16(exact)
        np.savez_compressed(
            out / "baseline.npz",
            input=x,
            dequant_input=qx,
            accumulator=accumulator,
            rounded=rounded,
            fp64=exact,
            fp64_bf16=exact_bf16,
        )
        row = 108 - 64
        report["coordinates"] = []
        for channel in (2108, 7656):
            value = accumulator[row, channel]
            report["coordinates"].append(
                {
                    "position": 108,
                    "channel": channel,
                    "fp32": float(value),
                    "fp32_bits": int(value.view(np.uint32)),
                    "fp64": float(exact[row, channel]),
                    "fp32_error": float(np.float64(value) - exact[row, channel]),
                    "bf16": float(rounded[row, channel]),
                    "fp64_bf16": float(exact_bf16[row, channel]),
                }
            )
        baseline_bad = rounded[:63] != exact_bf16[:63]
        report["baseline_wrong_bf16_elements"] = int(baseline_bad.sum())
        emit(
            "accumulator",
            coordinates=report["coordinates"],
            baseline_wrong=report["baseline_wrong_bf16_elements"],
        )
        variants = [
            (f"original_m{m}", m, CheckpointFP8Rhs()) for m in (8, 16, 32, 64, 128)
        ]
        variants += [
            ("split128_plain", 32, SplitKDiagnostic(split_k=128, compensate=False)),
            ("split128_compensated", 32, SplitKDiagnostic(split_k=128)),
            ("split256_compensated", 32, SplitKDiagnostic(split_k=256)),
        ]
        for name, tile_m, adapter in variants:
            emit("variant", name=name)
            value = np.asarray(
                run(device_x, device_w, device_s, tile_m=tile_m, adapter=adapter),
                np.float32,
            )
            output = bf16(value)
            bad = output[:63] != exact_bf16[:63]
            entry = {
                "name": name,
                "tile_m": tile_m,
                "wrong_bf16_elements": int(bad.sum()),
                "fixed_elements": int((baseline_bad & ~bad).sum()),
                "new_wrong_elements": int((~baseline_bad & bad).sum()),
                "fp32": tensor_metrics(value[:63], exact[:63]),
                "bf16": tensor_metrics(output[:63], exact_bf16[:63]),
                "fp32_bitwise_baseline": bool(np.array_equal(value, accumulator)),
                "coordinate_fp32": [float(value[row, c]) for c in (2108, 7656)],
                "coordinate_bf16": [float(output[row, c]) for c in (2108, 7656)],
            }
            report["variants"].append(entry)
            np.savez_compressed(out / (name + ".npz"), accumulator=value, output=output)
            emit("variant_complete", **entry)
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
