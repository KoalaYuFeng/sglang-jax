"""Correctness-first native-checkpoint matmul on TPU v5p.

Only a 128-output-channel weight tile is decoded in VMEM. HBM inputs retain
packed FP4/FP8 bytes and compact block scales. This deliberately simple
full-K tile is a baseline, not a tuned kernel. It exposes load/dequant cost
without adding a persistent or full-layer BF16 weight tensor.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .formats import activation_fp8_roundtrip, decode_e8m0, decode_fp4_codes, decode_fp8


def _expand_columns(values, count):
    # Avoid reshape/broadcast into a new minor dimension, unsupported by the
    # v5p Mosaic vector-layout pass. All expansion remains inside VMEM.
    return jnp.concatenate(
        [
            jnp.broadcast_to(values[:, i : i + 1], (values.shape[0], count))
            for i in range(values.shape[1])
        ],
        axis=1,
    )


def _unpack_fp4_vmem(packed):
    lo = jax.lax.bitcast_convert_type(decode_fp4_codes(packed & 15), jnp.uint16)
    hi = jax.lax.bitcast_convert_type(decode_fp4_codes(packed >> 4), jnp.uint16)
    words = lo.astype(jnp.uint32) | (hi.astype(jnp.uint32) << jnp.uint32(16))
    # TPU bitcast expands the second minor dimension, not the last one.
    return pltpu.bitcast(words.T, jnp.bfloat16).T


@functools.partial(jax.jit, static_argnames=("weight_format", "quantize_activation"))
def low_bit_matmul(x, weight, scales, *, weight_format, quantize_activation=False):
    """BF16 [M,K] @ checkpoint [N,K]^T -> BF16 [M,N].

    ``weight`` is uint8 [N,K/2] for fp4, uint8 [N,K] for fp8, or BF16
    [N,K] for bf16. Scales are uint8 [N,K/32] or [ceil(N/128),K/128].
    BF16 uses None for scales. The caller handles tensor/expert sharding.
    K is currently a multiple of 128 and <=8192; M/N tails are padded.
    """
    if x.ndim != 2 or weight.ndim != 2 or x.dtype != jnp.bfloat16:
        raise ValueError("expected 2D BF16 activations and a 2D weight")
    if weight_format not in ("fp4", "fp8", "bf16"):
        raise ValueError(f"unsupported weight format: {weight_format}")
    m, k = x.shape
    n, stored_k = weight.shape
    if m < 1 or n < 1 or k % 128 or k > 8192:
        raise ValueError("requires positive M/N and K divisible by 128, at most 8192")
    if stored_k != (k // 2 if weight_format == "fp4" else k):
        raise ValueError("weight shape does not match logical K")
    if weight_format == "bf16":
        if weight.dtype != jnp.bfloat16 or scales is not None:
            raise ValueError("BF16 weights require BF16 data and no scales")
    else:
        if weight.dtype != jnp.uint8 or scales is None or scales.dtype != jnp.uint8:
            raise ValueError("low-bit weights and E8M0 scales must be raw uint8 bytes")
        expected = (n, k // 32) if weight_format == "fp4" else ((n + 127) // 128, k // 128)
        if scales.shape != expected:
            raise ValueError(f"invalid {weight_format} scale shape: {scales.shape} != {expected}")

    bm, bn = 8, 128
    padded_m, padded_n = (m + bm - 1) // bm * bm, (n + bn - 1) // bn * bn
    x = jnp.pad(x, ((0, padded_m - m), (0, 0)))
    weight = jnp.pad(weight, ((0, padded_n - n), (0, 0)))
    if weight_format == "fp4":
        scales = jnp.pad(scales, ((0, padded_n - n), (0, 0)), constant_values=127)

    def kernel(x_ref, w_ref, s_ref, y_ref):
        if weight_format == "fp4":
            values = _unpack_fp4_vmem(w_ref[...]).astype(jnp.float32)
            w_bf16 = (values * _expand_columns(decode_e8m0(s_ref[...]), 32)).astype(jnp.bfloat16)
        elif weight_format == "fp8":
            # FP8 scales are tiny; keep their checkpoint layout and read the
            # output-block's row in VMEM. No host/HBM scale expansion.
            all_scales = decode_e8m0(s_ref[...])
            selected = jnp.arange(all_scales.shape[0])[:, None] == pl.program_id(1)
            scale = jnp.sum(jnp.where(selected, all_scales, 0.0), axis=0, keepdims=True)
            w_bf16 = (decode_fp8(w_ref[...]) * _expand_columns(scale, 128)).astype(jnp.bfloat16)
        else:
            w_bf16 = w_ref[...]
        lhs = x_ref[...]
        if quantize_activation:
            lhs = activation_fp8_roundtrip(lhs)
        y_ref[...] = jax.lax.dot_general(
            lhs,
            w_bf16,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)

    if weight_format == "fp4":
        scale_spec = pl.BlockSpec((bn, k // 32), lambda mi, ni: (ni, 0))
    elif weight_format == "fp8":
        scale_spec = pl.BlockSpec(scales.shape, lambda mi, ni: (0, 0))
    else:
        scale_spec = None
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded_m, padded_n), jnp.bfloat16),
        grid=(padded_m // bm, padded_n // bn),
        in_specs=(
            pl.BlockSpec((bm, k), lambda mi, ni: (mi, 0)),
            pl.BlockSpec((bn, stored_k), lambda mi, ni: (ni, 0)),
            scale_spec,
        ),
        out_specs=pl.BlockSpec((bm, bn), lambda mi, ni: (mi, ni)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=16 * 1024 * 1024 - 64 * 1024,
        ),
        interpret=jax.default_backend() != "tpu",
        name=f"checkpoint_{weight_format}_online_dequant_matmul",
    )(x, weight, scales)
    return result[:m, :n]
