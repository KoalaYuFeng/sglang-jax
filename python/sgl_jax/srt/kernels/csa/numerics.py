"""FP32 pooling primitives for low-bit-sensitive V4 index compression."""

import jax
import jax.numpy as jnp


def _exp_nonpositive(x):
    """Range-reduced FP32 exp for nonpositive softmax shifts, including -inf.

    The degree-eight polynomial is evaluated on [-ln(2)/2, ln(2)/2]. Splitting
    ln(2) keeps the range-reduction product exact at these exponents. Values
    below -87 contribute less than 1.65e-38 and are flushed explicitly, avoiding
    platform-dependent subnormal behavior. No optimization barriers or FP64
    arithmetic are required in the Pallas program.
    """
    safe = jnp.maximum(x, jnp.float32(-87))
    exponent = jnp.rint(safe * jnp.float32(1.4426950408889634))
    remainder = (safe - exponent * jnp.float32(0.693145751953125)) - exponent * jnp.float32(
        1.428606765330187e-6
    )
    value = jnp.full_like(remainder, 1 / 40320)
    for coefficient in (1 / 5040, 1 / 720, 1 / 120, 1 / 24, 1 / 6, 1 / 2, 1, 1):
        value = value * remainder + jnp.float32(coefficient)
    # These integral exponents are in [-126, 0]. Construct powers of two
    # exactly; exp2 lowering can introduce error/underflow even at integers.
    scale = jax.lax.bitcast_convert_type((exponent.astype(jnp.int32) + 127) << 23, jnp.float32)
    return jnp.where(x >= -87, value * scale, 0.0)


def _sum8_compensated(values):
    """Pairwise FP32 sum with the rounding residual retained at each node.

    The eight weighted values can nearly cancel. An ordinary reduction lost
    enough low bits to cross a BF16 midpoint even after correcting exp. TwoSum
    recovers each addition's residual without FP64 or unsupported TPU barriers.
    Do not reassociate its subtractions or enable unsafe fast-math here.
    """

    def two_sum(a, b):
        high = a + b
        virtual_b = high - a
        low = (a - (high - virtual_b)) + (b - virtual_b)
        return high, low

    parts = [values[:, i, :] for i in range(8)]
    high0, low0 = two_sum(parts[0], parts[1])
    high1, low1 = two_sum(parts[2], parts[3])
    high2, low2 = two_sum(parts[4], parts[5])
    high3, low3 = two_sum(parts[6], parts[7])
    high4, low4 = two_sum(high0, high1)
    high5, low5 = two_sum(high2, high3)
    high6, low6 = two_sum(high4, high5)
    return high6 + (((low0 + low1) + (low2 + low3)) + ((low4 + low5) + low6))


def index_pool_v4(values, scores):
    """Softmax-weighted FP32 [groups,8,128] pooling before BF16 rounding.

    Compensate the eight unnormalized terms, then divide once. This avoids
    rounding every normalized probability before its value multiplication.
    The exponent path addresses v5p exp approximation observed at the frozen
    group-1643/group-1928 BF16 midpoint regressions. The caller still owns
    BF16 rounding, RMSNorm, RoPE, Hadamard, and FP4 quantization.
    """
    if values.shape != scores.shape or values.shape[1:] != (8, 128):
        raise ValueError("V4 index pooling requires matching [groups,8,128] windows")
    shifted = scores.astype(jnp.float32) - jnp.max(scores, axis=1, keepdims=True)
    exponentials = _exp_nonpositive(shifted)
    numerator = _sum8_compensated(values.astype(jnp.float32) * exponentials)
    denominator = _sum8_compensated(exponentials)
    # A nonempty softmax window contains a maximum with exp(0)=1. Empty,
    # fully masked windows emit zero; the caller also masks invalid groups.
    return numerator / jnp.maximum(denominator, jnp.float32(1))
