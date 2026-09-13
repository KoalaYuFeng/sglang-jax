"""V4 inverse-RoPE/grouped output projection fusion; tensor-only interface."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.deepseek_v4.normalization import rotate_tile
from sgl_jax.srt.kernels.low_bit.fp8 import decode_weight_tile, projection_block_m


@functools.partial(jax.jit, static_argnames=("groups", "head_dim", "block_m"))
def inverse_rope_fp8_wo_a(
    value, weight, scales, cosine, sine, *, groups, head_dim=512, block_m=None
):
    """[T,H,D] -> [T,G*rank], without an HBM inverse-RoPE/group intermediate.

    wo_a intentionally does NOT quantize activations: this preserves the
    currently accepted V4 arithmetic, unlike some upstream fused kernels.
    Complete groups are local to each TP chip; no collective runs here.
    """
    if value.ndim != 3 or value.dtype != jnp.bfloat16 or value.shape[-1] != head_dim:
        raise ValueError("wo_a requires BF16[T,H,head_dim]")
    m, heads, _ = value.shape
    block_m = projection_block_m(m) if block_m is None else block_m
    if m < 1 or groups < 1 or heads % groups or weight.shape[0] % groups:
        raise ValueError("wo_a sharding must keep complete head/output groups")
    k, rank = heads // groups * head_dim, weight.shape[0] // groups
    if head_dim % 128 or rank % 128 or k > 8192 or weight.shape != (groups * rank, k):
        raise ValueError("invalid grouped wo_a checkpoint shape")
    if (
        weight.dtype != jnp.uint8
        or scales.dtype != jnp.uint8
        or scales.shape != (groups * rank // 128, k // 128)
    ):
        raise ValueError("wo_a requires raw FP8/E8M0 checkpoint bytes")
    if (
        cosine.shape != (m, 32)
        or sine.shape != cosine.shape
        or cosine.dtype != jnp.float32
        or sine.dtype != jnp.float32
    ):
        raise ValueError("wo_a requires FP32 cosine/sine[T,32]")
    if block_m not in (8, 16, 32, 64, 128):
        raise ValueError("unsupported wo_a M tile")
    padded = pl.cdiv(m, block_m) * block_m
    value = jnp.pad(value.reshape(m, groups * k), ((0, padded - m), (0, 0)))
    cosine = jnp.pad(cosine, ((0, padded - m), (0, 0)))
    sine = jnp.pad(sine, ((0, padded - m), (0, 0)))

    def kernel(x_ref, w_ref, scale_ref, c_ref, s_ref, out_ref):
        x = x_ref[...]
        rotated = [
            rotate_tile(
                x[:, h * head_dim : (h + 1) * head_dim], c_ref[...], -s_ref[...]
            )
            for h in range(heads // groups)
        ]
        lhs = jnp.concatenate(rotated, axis=-1)
        weight_bf16 = decode_weight_tile(w_ref[...], scale_ref[...])
        out_ref[...] = jax.lax.dot_general(
            lhs,
            weight_bf16,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded, groups * rank), jnp.bfloat16),
        grid=(groups, rank // 128, padded // block_m),
        in_specs=(
            pl.BlockSpec((block_m, k), lambda g, n, t: (t, g)),
            pl.BlockSpec((128, k), lambda g, n, t: (g * (rank // 128) + n, 0)),
            pl.BlockSpec(
                (None, 1, k // 128), lambda g, n, t: (g * (rank // 128) + n, 0, 0)
            ),
            pl.BlockSpec((block_m, 32), lambda g, n, t: (t, 0)),
            pl.BlockSpec((block_m, 32), lambda g, n, t: (t, 0)),
        ),
        out_specs=pl.BlockSpec(
            (block_m, 128), lambda g, n, t: (t, g * (rank // 128) + n)
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=32 * 1024 * 1024 - 64 * 1024,
        ),
        interpret=jax.default_backend() != "tpu",
        name="v4_inverse_rope_checkpoint_fp8_wo_a",
    )(value, weight, scales[:, None, :], cosine, sine)
    return result[:m].reshape(m, groups * rank)
