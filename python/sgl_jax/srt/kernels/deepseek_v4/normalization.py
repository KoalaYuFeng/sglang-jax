"""V4 exact-tree normalization/rotary kernels, local to one TP shard.

The adjacent-pair reduction lives in VMEM, not a chain of HBM slice programs.
Q normalization deliberately has more BF16 rounding boundaries than RMSNorm.
Do not replace either recipe with a generic mean or a single final cast.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip, round_bf16


def pair_sum(x):
    """Same adjacent-pair tree as numerics._fixed_tree_sum_last, in VMEM."""
    # At stage s, lane i holds the adjacent-tree sum of [i, i + 2**s).
    # Only lane zero is returned. Keeping a 2D lane layout avoids both the
    # unsupported stride-2 slice and padding a new minor dimension of size 2.
    stride = 1
    while stride < x.shape[-1]:
        x = x + pltpu.roll(x, x.shape[-1] - stride, axis=1)
        stride *= 2
    return x[:, :1]


def rotate_tile(x, cosine, sine):
    """Rotate adjacent pairs and retain the explicit BF16 output boundary."""
    rd = 2 * cosine.shape[-1]
    # Reuse the FP4 adapter's word interleave; [..., 32, 2] would pad to 128 lanes.
    words = pltpu.bitcast(x[:, -rd:].T, jnp.uint32).T
    a = jax.lax.bitcast_convert_type((words & 0xFFFF).astype(jnp.uint16), jnp.bfloat16).astype(
        jnp.float32
    )
    b = jax.lax.bitcast_convert_type((words >> 16).astype(jnp.uint16), jnp.bfloat16).astype(
        jnp.float32
    )
    real = jax.lax.bitcast_convert_type(round_bf16(a * cosine - b * sine), jnp.uint16)
    imag = jax.lax.bitcast_convert_type(round_bf16(a * sine + b * cosine), jnp.uint16)
    packed = real.astype(jnp.uint32) | (imag.astype(jnp.uint32) << 16)
    rotated = pltpu.bitcast(packed.T, jnp.bfloat16).T
    return jnp.concatenate((x[:, :-rd], rotated), axis=-1)


def _validate(x, width):
    if x.dtype != jnp.bfloat16 or x.size == 0:
        raise ValueError("V4 normalization requires nonempty BF16 activations")
    if width < 128 or width > 8192 or width & (width - 1):
        raise ValueError("V4 normalization width must be a power of two in [128, 8192]")


@functools.partial(jax.jit, static_argnames=("eps", "quantize_nope"))
def rms_norm(x, weight, eps=1e-6, *, cosine=None, sine=None, quantize_nope=False):
    """Weighted RMSNorm, optionally followed by RoPE and KV's A8 roundtrip.

    x is [tokens, width]; cosine/sine are precomputed FP32 [tokens, rope/2].
    Raw weights/scales elsewhere in the model are unaffected by this kernel.
    """
    if x.ndim != 2 or weight.shape != (x.shape[-1],):
        raise ValueError("expected x[M,D], weight[D]")
    m, width = x.shape
    _validate(x, width)
    if (cosine is None) != (sine is None):
        raise ValueError("cosine and sine must be supplied together")
    if cosine is not None:
        if cosine.shape != (m, 32) or sine.shape != cosine.shape:
            raise ValueError("V4 requires cosine/sine[M,32] for 64 rotary channels")
        if cosine.dtype != jnp.float32 or sine.dtype != jnp.float32:
            raise ValueError("V4 rotary angles must remain FP32")
    elif quantize_nope:
        raise ValueError("KV quantization requires rotary angles")
    bm = 8
    padded = pl.cdiv(m, bm) * bm
    x = jnp.pad(x, ((0, padded - m), (0, 0)))
    if cosine is not None:
        cosine = jnp.pad(cosine, ((0, padded - m), (0, 0)))
        sine = jnp.pad(sine, ((0, padded - m), (0, 0)))

    def kernel(x_ref, w_ref, c_ref, s_ref, out_ref):
        value = x_ref[...].astype(jnp.float32)
        variance = pair_sum(value * value) / jnp.float32(width)
        normalized = value * jax.lax.rsqrt(variance + eps)
        result = round_bf16(normalized * w_ref[...].astype(jnp.float32)[None])
        if c_ref is not None:
            result = rotate_tile(result, c_ref[...], s_ref[...])
            if quantize_nope:
                result = jnp.concatenate(
                    (activation_fp8_roundtrip(result[:, :-64], 64), result[:, -64:]), axis=-1
                )
        out_ref[...] = result

    spec = pl.BlockSpec((bm, width), lambda i: (i, 0))
    phase_spec = None if cosine is None else pl.BlockSpec((bm, 32), lambda i: (i, 0))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded, width), jnp.bfloat16),
        grid=(padded // bm,),
        in_specs=(spec, pl.BlockSpec(weight.shape, lambda i: (0,)), phase_spec, phase_spec),
        out_specs=spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=32 * 1024 * 1024 - 64 * 1024,
        ),
        interpret=jax.default_backend() != "tpu",
        name="v4_exact_rms_norm" + ("_rope_kv" if cosine is not None else ""),
    )(x, weight, cosine, sine)[:m]


@functools.partial(jax.jit, static_argnames=("eps",))
def qnorm_rope(q, cosine, sine, eps=1e-6):
    """Q[T,H,D] -> normalized/rotated Q, with all five V4 BF16 boundaries."""
    if q.ndim != 3:
        raise ValueError("Q must have shape [tokens, heads, head_dim]")
    tokens, heads, width = q.shape
    _validate(q, width)
    if cosine.shape != (tokens, 32) or sine.shape != cosine.shape:
        raise ValueError("expected cosine/sine[tokens,32]")
    if cosine.dtype != jnp.float32 or sine.dtype != jnp.float32:
        raise ValueError("V4 rotary angles must remain FP32")
    padded_heads = pl.cdiv(heads, 8) * 8
    bt = min(8, 1 << max(0, (128 // padded_heads).bit_length() - 1), 1 << (tokens - 1).bit_length())
    phase_rows = max(8, bt)
    padded_tokens = pl.cdiv(tokens, bt) * bt
    phase_tokens = pl.cdiv(tokens, phase_rows) * phase_rows
    q = jnp.pad(q, ((0, padded_tokens - tokens), (0, padded_heads - heads), (0, 0)))
    q = q.reshape(padded_tokens * padded_heads, width)
    cosine = jnp.pad(cosine, ((0, phase_tokens - tokens), (0, 0)))
    sine = jnp.pad(sine, ((0, phase_tokens - tokens), (0, 0)))

    def kernel(q_ref, c_ref, s_ref, out_ref):
        value = q_ref[...].astype(jnp.float32)
        squared = round_bf16(value * value).astype(jnp.float32)
        variance = round_bf16(pair_sum(squared) / jnp.float32(width))
        variance = round_bf16(variance.astype(jnp.float32) + eps)
        inverse = round_bf16(jax.lax.rsqrt(variance.astype(jnp.float32)))
        normed = round_bf16(value * inverse.astype(jnp.float32))
        # Batch token/head rows; select phases in VMEM without HBM head expansion.
        rows = jnp.arange(bt * padded_heads) // padded_heads + pl.program_id(0) * bt % phase_rows
        indices = jnp.broadcast_to(rows[:, None], (bt * padded_heads, 32))
        cos = jnp.take_along_axis(c_ref[...], indices, axis=0, mode="promise_in_bounds")
        sin = jnp.take_along_axis(s_ref[...], indices, axis=0, mode="promise_in_bounds")
        out_ref[...] = rotate_tile(normed, cos, sin)

    spec = pl.BlockSpec((bt * padded_heads, width), lambda t: (t, 0))
    phase_spec = pl.BlockSpec((phase_rows, 32), lambda t: (t * bt // phase_rows, 0))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(padded_tokens // bt,),
        in_specs=(spec, phase_spec, phase_spec),
        out_specs=spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=jax.default_backend() != "tpu",
        name="v4_exact_qnorm_rope",
    )(q, cosine, sine).reshape(padded_tokens, padded_heads, width)[:tokens, :heads]
