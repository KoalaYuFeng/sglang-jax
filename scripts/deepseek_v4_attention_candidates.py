"""Standalone numerical candidates; never imported by the serving runtime.

These preserve the original operator's layout and BF16 boundaries, except
for the explicitly tested summation/exp alternatives. No default is changed.
"""

import types

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
from sgl_jax.srt.kernels.csa.numerics import _exp_nonpositive
from sgl_jax.srt.kernels.deepseek_v4 import normalization
from sgl_jax.srt.kernels.low_bit.formats import round_bf16


def compensated_pair_sum(x):
    high, low = x, jnp.zeros_like(x)
    stride = 1
    while stride < x.shape[-1]:
        other = pltpu.roll(high, x.shape[-1] - stride, axis=1)
        other_low = pltpu.roll(low, x.shape[-1] - stride, axis=1)
        total = high + other
        virtual = total - high
        error = (high - (total - virtual)) + (other - virtual)
        low = (low + other_low) + error
        high = total
        stride *= 2
    return (high + low)[:, :1]


def make_compensated_qnorm():
    """Isolate the Q-norm prototype without altering other normalization calls."""
    original = normalization.qnorm_rope.__wrapped__
    candidate = types.FunctionType(
        original.__code__,
        {**original.__globals__, "pair_sum": compensated_pair_sum},
        name="diagnostic_compensated_qnorm",
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )
    return jax.jit(candidate, static_argnames=("eps",))


def precise_single_query_attention(q, kv, indices, sink, scale):
    """Original 64-key online-softmax structure with only exp substituted."""
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("indices must be a nonempty key-slot vector")
    count = (indices.shape[0] + 63) // 64 * 64
    indices = jnp.pad(indices, ((0, count - indices.shape[0]),), constant_values=-1)
    initial = (
        jnp.full(q.shape[:-1], -1e30, jnp.float32),
        jnp.zeros(q.shape[:-1], jnp.float32),
        jnp.zeros(q.shape, jnp.float32),
    )

    def block(i, state):
        maximum, denominator, numerator = state
        ids = jax.lax.dynamic_slice_in_dim(indices, i * 64, 64)
        keys = kv[jnp.maximum(ids, 0)]
        score = jnp.matmul(q, keys.T, preferred_element_type=jnp.float32) * scale
        valid = ids[None, :] >= 0
        score = jnp.where(valid, score, -1e30)
        next_max = jnp.maximum(maximum, jnp.max(score, axis=-1))
        alpha = _exp_nonpositive(maximum - next_max)
        probability = jnp.where(
            valid, _exp_nonpositive(score - next_max[..., None]), 0.0
        )
        numerator = numerator * alpha[..., None] + jnp.matmul(
            round_bf16(probability), keys, preferred_element_type=jnp.float32
        )
        denominator = denominator * alpha + jnp.sum(probability, axis=-1)
        return next_max, denominator, numerator

    maximum, denominator, numerator = jax.lax.fori_loop(0, count // 64, block, initial)
    final_max = jnp.maximum(maximum, sink)
    rescale = _exp_nonpositive(maximum - final_max)
    denom = denominator * rescale + _exp_nonpositive(sink - final_max)
    return round_bf16(numerator * rescale[..., None] / denom[..., None])
