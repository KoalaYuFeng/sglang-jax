"""V4 merged projections and inverse-RoPE/grouped-wo_a fusion.

Merged checkpoint packing is a load-time operation, never a decode-time
concatenation. The caller retains the existing replicated/head-TP placement.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.deepseek_v4.normalization import rotate_tile
from sgl_jax.srt.kernels.deepseek_v4.fp8 import decode_weight_tile, fp8_linear, projection_block_m

MERGED_PROJECTIONS = {
    "attn.wqkv_a": ("attn.wq_a", "attn.wkv"),
    "ffn.shared_experts.gate_up": ("ffn.shared_experts.w1", "ffn.shared_experts.w3"),
}


def pack_merged_weights(weights):
    """Host-only, byte-preserving packing; replace (not duplicate) originals.

    The V4 components all end on 128-output-channel scale boundaries. Reject
    other layouts rather than silently assigning the wrong E8M0 block scale.
    """
    import numpy as np

    result = dict(weights)
    for target, sources in MERGED_PROJECTIONS.items():
        raw = [result[name + ".weight"] for name in sources]
        scales = [result[name + ".scale"] for name in sources]
        if any(not isinstance(value, np.ndarray) for value in (*raw, *scales)):
            raise TypeError("merged packing must run on host checkpoint arrays at load time")
        k = raw[0].shape[-1]
        if target == "ffn.shared_experts.gate_up" and raw[0].shape != raw[1].shape:
            raise ValueError("V4 shared expert gate/up must have identical shapes")
        for w, s in zip(raw, scales, strict=True):
            if w.ndim != 2 or w.dtype != np.uint8 or w.shape[1] != k or w.shape[0] % 128:
                raise ValueError("merged FP8 components require raw bytes and aligned N/equal K")
            if k % 128 or s.dtype != np.uint8 or s.shape != (w.shape[0] // 128, k // 128):
                raise ValueError("merged components require compact E8M0 block scales")
        for suffix, arrays in ((".weight", raw), (".scale", scales)):
            result[target + suffix] = np.concatenate(arrays, axis=0)
            for source in sources:
                del result[source + suffix]
    return result


def unpack_merged_weights(weights):
    """Read-only raw-byte views for diagnostics using the unchanged oracle.

    Never used in the production decode path. No dequantization/requantization.
    """
    result = dict(weights)
    for target, sources in MERGED_PROJECTIONS.items():
        if target + ".weight" not in result:
            continue
        split = (
            result["attn.q_norm.weight"].shape[0]
            if target == "attn.wqkv_a"
            else result[target + ".weight"].shape[0] // 2
        )
        for suffix, boundary in ((".weight", split), (".scale", split // 128)):
            value = result.pop(target + suffix)
            result[sources[0] + suffix], result[sources[1] + suffix] = (
                value[:boundary],
                value[boundary:],
            )
    return result


def merged_linear(x, weights, prefix, split, *, block_m=None):
    result = fp8_linear(x, weights[prefix + ".weight"], weights[prefix + ".scale"], block_m=block_m)
    if not 0 < split < result.shape[1] or split % 128:
        raise ValueError("merged projection split must be an interior scale-block boundary")
    return result[:, :split], result[:, split:]


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
            rotate_tile(x[:, h * head_dim : (h + 1) * head_dim], c_ref[...], -s_ref[...])
            for h in range(heads // groups)
        ]
        lhs = jnp.concatenate(rotated, axis=-1)
        weight_bf16 = decode_weight_tile(w_ref[...], scale_ref[...])
        out_ref[...] = jax.lax.dot_general(
            lhs, weight_bf16, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((padded, groups * rank), jnp.bfloat16),
        grid=(groups, rank // 128, padded // block_m),
        in_specs=(
            pl.BlockSpec((block_m, k), lambda g, n, t: (t, g)),
            pl.BlockSpec((128, k), lambda g, n, t: (g * (rank // 128) + n, 0)),
            pl.BlockSpec((None, 1, k // 128), lambda g, n, t: (g * (rank // 128) + n, 0, 0)),
            pl.BlockSpec((block_m, 32), lambda g, n, t: (t, 0)),
            pl.BlockSpec((block_m, 32), lambda g, n, t: (t, 0)),
        ),
        out_specs=pl.BlockSpec((block_m, 128), lambda g, n, t: (t, g * (rank // 128) + n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=32 * 1024 * 1024 - 64 * 1024,
        ),
        interpret=jax.default_backend() != "tpu",
        name="v4_inverse_rope_checkpoint_fp8_wo_a",
    )(value, weight, scales[:, None, :], cosine, sine)
    return result[:m].reshape(m, groups * rank)
