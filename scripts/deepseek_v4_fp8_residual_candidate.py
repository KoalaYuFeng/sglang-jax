"""Isolated numerical candidates, NOT serving adapters or default changes."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from sgl_jax.srt.kernels.deepseek_v4.fp8 import CheckpointFP8Rhs, decode_weight_tile
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip


def bf16_rounding_carrier(high, low):
    """FP32 carrier that retains a sum residual at a BF16 midpoint.

    Output is intended ONLY for subsequent BF16 conversion, not as a correctly
    rounded FP32 result. Requires finite, non-overflowing inputs/intermediates.
    A one-ULP step in the residual's direction avoids double rounding when the
    rounded FP32 sum is exactly at a BF16 midpoint.
    """
    total = high + low
    virtual = total - high
    residual = (high - (total - virtual)) + (low - virtual)
    bits = jax.lax.bitcast_convert_type(total, jnp.uint32)
    midpoint = (bits & jnp.uint32(65535)) == jnp.uint32(32768)
    step_up_bits = (residual > 0) == (total >= 0)
    step = jnp.where(step_up_bits, jnp.uint32(1), jnp.uint32(0xFFFFFFFF))
    shifted = jax.lax.bitcast_convert_type(bits + step, jnp.float32)
    return jnp.where(midpoint & (residual != 0), shifted, total)


@dataclass(frozen=True)
class ResidualDiagnostic(CheckpointFP8Rhs):
    split_k: int = 64
    preserve_bf16_residual: bool = False
    component: str = "result"
    name = "diagnostic_fp8_residual"

    def dot(self, lhs, rhs, scales):
        weight = decode_weight_tile(rhs, scales)
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
            total = high + partial
            virtual = total - high
            low = low + ((high - (total - virtual)) + (partial - virtual))
            high = total
        if self.component == "high":
            return high
        if self.component == "low":
            return low
        if self.preserve_bf16_residual:
            return bf16_rounding_carrier(high, low)
        return high + low
