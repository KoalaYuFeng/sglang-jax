"""V4 numerical primitives for the native serving kernels.

Keep checkpoint arithmetic explicit. The independent bring-up reference stays
unchanged so serving adapters can be checked against it.
"""

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.low_bit.formats import round_bf16
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul


@dataclass(frozen=True)
class V4LayerConfig:
    hidden: int = 4096
    heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    groups: int = 8
    o_rank: int = 1024
    index_heads: int = 64
    index_dim: int = 128
    index_topk: int = 512
    window: int = 128
    hc: int = 4
    sinkhorn_iters: int = 20
    eps: float = 1e-6
    hc_eps: float = 1e-6
    active_experts: int = 6
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    max_context: int = 256
    ratio: int = 0
    hash_routing: bool = False
    rope_base: float = 10000.0
    original_seq_len: int = 0
    rope_factor: float = 16.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0


def _fixed_tree_sum_last(x):
    """Deterministic last-axis reduction independent of leading token shape."""
    value = x
    while value.shape[-1] > 1:
        if value.shape[-1] % 2:
            value = jnp.pad(value, ((0, 0),) * (value.ndim - 1) + ((0, 1),))
        value = value[..., 0::2] + value[..., 1::2]
    return value[..., 0]


def _fixed_tree_mean_last(x):
    return _fixed_tree_sum_last(x) / jnp.float32(x.shape[-1])


def rms_norm(x, weight, eps):
    xf = x.astype(jnp.float32)
    variance = _fixed_tree_mean_last(xf * xf)[..., None]
    normalized = xf * jax.lax.rsqrt(variance + eps)
    return round_bf16(normalized * weight.astype(jnp.float32))


@functools.lru_cache(maxsize=32)
def _rope_frequencies(rd, base, original_seq_len, factor, beta_fast, beta_slow):
    """Host FP32 coefficients with the checkpoint's explicit rounding order.

    Do not trace power/YaRN construction into the TPU program. Its angle
    differences can cross BF16/FP4 rounding boundaries (the CSA-index
    position-219 regression). Only the
    small, immutable frequency vector is cached, independent of layer, batch,
    and context length; execution still multiplies dynamic positions on device.
    No PyTorch dependency or host callback is introduced.
    """
    if rd <= 0 or rd % 2:
        raise ValueError("RoPE dimension must be positive and even")
    if base <= 1 or factor <= 0 or beta_fast <= 0 or beta_slow <= 0:
        raise ValueError("invalid RoPE base/scaling parameters")
    exponent = np.arange(0, rd, 2, dtype=np.float32) / np.float32(rd)
    # Scalar libm followed by FP32 rounding avoids SIMD pow implementation
    # differences. The reciprocal and all YaRN operations round in FP32.
    powers = np.asarray([math.pow(base, float(x)) for x in exponent], dtype=np.float32)
    frequency = np.float32(1) / powers
    if original_seq_len > 0:

        def correction(rotations):
            return (
                rd * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
            )

        low = max(math.floor(correction(beta_fast)), 0)
        high = min(math.ceil(correction(beta_slow)), rd - 1)
        ramp = np.clip(
            (np.arange(rd // 2, dtype=np.float32) - np.float32(low))
            / np.float32(high - low if high != low else 0.001),
            np.float32(0),
            np.float32(1),
        )
        smooth = np.float32(1) - ramp
        # Preserve both subtractions from the official FP32 recipe. Replacing
        # 1 - smooth with ramp is algebraically, but not numerically, equivalent.
        frequency = frequency / np.float32(factor) * (np.float32(1) - smooth) + frequency * smooth
    frequency.flags.writeable = False
    return frequency


def rope_angles(positions, config):
    """Shared V4 phase: immutable host frequencies × dynamic FP32 positions."""
    frequency = jnp.asarray(
        _rope_frequencies(
            config.rope_dim,
            config.rope_base,
            config.original_seq_len,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
    )
    return positions.astype(jnp.float32)[:, None] * frequency[None, :]


def rope(x, positions, config, *, inverse=False):
    rd = config.rope_dim
    phase = rope_angles(positions, config)
    phase = phase.reshape((positions.shape[0],) + (1,) * (x.ndim - 2) + (rd // 2,))
    cosine, sine = jnp.cos(phase), jnp.sin(phase) * (-1 if inverse else 1)
    paired = x[..., -rd:].astype(jnp.float32).reshape(*x.shape[:-1], rd // 2, 2)
    a, b = paired[..., 0], paired[..., 1]
    rotated = jnp.stack((a * cosine - b * sine, a * sine + b * cosine), axis=-1)
    return jnp.concatenate((x[..., :-rd], round_bf16(rotated.reshape(*x.shape[:-1], rd))), axis=-1)


def hadamard_rotate(x):
    # F32 butterfly and one final BF16 rounding, matching the CUDA reference.
    width = x.shape[-1]
    if width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    value = x.astype(jnp.float32)
    stride = 1
    while stride < width:
        grouped = value.reshape(*x.shape[:-1], -1, 2, stride)
        a, b = grouped[..., 0, :], grouped[..., 1, :]
        value = jnp.stack((a + b, a - b), axis=-2).reshape(x.shape)
        stride *= 2
    return round_bf16(value * width**-0.5)


def linear(x, weights, prefix, *, quantize=True):
    weight = weights[prefix + ".weight"]
    scales = weights.get(prefix + ".scale")
    fmt = "bf16" if scales is None else "fp8"
    return low_bit_matmul(
        x, weight, scales, weight_format=fmt, quantize_activation=quantize and scales is not None
    )


def _single_query_attention(q, kv, indices, sink, scale):
    """Fixed matrix/reduction shapes for both prefill and cached decode."""
    count = (indices.shape[-1] + 63) // 64 * 64
    indices = jnp.pad(indices, ((0, count - indices.shape[-1]),), constant_values=-1)
    initial = (
        jnp.full(q.shape[:-1], -1e30, jnp.float32),
        jnp.zeros(q.shape[:-1], jnp.float32),
        jnp.zeros(q.shape, jnp.float32),
    )

    def block(i, state):
        maximum, denominator, numerator = state
        ids = jax.lax.dynamic_slice_in_dim(indices, i * 64, 64, axis=0)
        keys = kv[jnp.maximum(ids, 0)]
        score = jnp.matmul(q, keys.T, preferred_element_type=jnp.float32) * scale
        valid = ids[None, :] >= 0
        score = jnp.where(valid, score, -1e30)
        next_max = jnp.maximum(maximum, jnp.max(score, axis=-1))
        alpha = jnp.exp(maximum - next_max)
        probability = jnp.where(valid, jnp.exp(score - next_max[..., None]), 0.0)
        numerator = numerator * alpha[..., None] + jnp.matmul(
            round_bf16(probability),
            keys,
            preferred_element_type=jnp.float32,
        )
        denominator = denominator * alpha + jnp.sum(probability, axis=-1)
        return next_max, denominator, numerator

    maximum, denominator, numerator = jax.lax.fori_loop(0, count // 64, block, initial)
    # Include the learned sink only in the denominator, with stable scaling.
    final_max = jnp.maximum(maximum, sink)
    rescale = jnp.exp(maximum - final_max)
    denom = denominator * rescale + jnp.exp(sink - final_max)
    return round_bf16(numerator * rescale[..., None] / denom[..., None])


def official_head_collapse(streams, fn, scale, base, *, eps=1e-6, hc_eps=1e-6):
    """Official F32 projection -> RMS scalar -> gates, with no BF16 pre-round."""
    flat = streams.astype(jnp.float32).reshape(streams.shape[0], -1)
    inverse_rms = jax.lax.rsqrt(_fixed_tree_mean_last(flat * flat)[:, None] + eps)
    mixes = jnp.matmul(flat, fn.T, precision=jax.lax.Precision.HIGHEST) * inverse_rms
    pre = jax.nn.sigmoid(mixes * scale + base) + hc_eps
    values = [pre[:, i, None] * streams[:, i].astype(jnp.float32) for i in range(streams.shape[1])]
    return round_bf16(functools.reduce(jnp.add, values))


def config_for_layer(source, layer_id, max_context):
    ratio = source["compress_ratios"][layer_id]
    yarn = source["rope_scaling"]
    return V4LayerConfig(
        hidden=source["hidden_size"],
        heads=source["num_attention_heads"],
        head_dim=source["head_dim"],
        rope_dim=source["qk_rope_head_dim"],
        groups=source["o_groups"],
        o_rank=source["o_lora_rank"],
        index_heads=source["index_n_heads"],
        index_dim=source["index_head_dim"],
        index_topk=source["index_topk"],
        window=source["sliding_window"],
        hc=source["hc_mult"],
        sinkhorn_iters=source["hc_sinkhorn_iters"],
        eps=source["rms_norm_eps"],
        hc_eps=source["hc_eps"],
        active_experts=source["num_experts_per_tok"],
        route_scale=source["routed_scaling_factor"],
        swiglu_limit=source["swiglu_limit"],
        max_context=max_context,
        ratio=ratio,
        hash_routing=layer_id < source["num_hash_layers"],
        rope_base=source["compress_rope_theta"] if ratio else source["rope_theta"],
        original_seq_len=yarn["original_max_position_embeddings"] if ratio else 0,
        rope_factor=yarn["factor"],
        beta_fast=yarn["beta_fast"],
        beta_slow=yarn["beta_slow"],
    )
