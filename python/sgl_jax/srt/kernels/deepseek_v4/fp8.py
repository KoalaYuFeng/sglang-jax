"""V4 raw E4M3/E8M0 Linear on SGLang-JAX's shared GMM execution structure.

Full-K FP32 accumulation follows the accepted V4 checkpoint arithmetic.
Only the current RHS tile is decoded to BF16, inside Pallas VMEM. Neither
weight requantization nor full BF16 weights are part of this interface.
"""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip, decode_e8m0, decode_fp8
from sgl_jax.srt.kernels.low_bit.matmul import _expand_columns


def decode_weight_tile(raw, scales):
    """A single output block's compact [1,K/128] scales, DMA-selected in GMM."""
    return (decode_fp8(raw) * _expand_columns(decode_e8m0(scales), 128)).astype(jnp.bfloat16)


@dataclass(frozen=True)
class CheckpointFP8Rhs:
    quantize_activation: bool = True
    name = "checkpoint_fp8"

    @property
    def compiler_params(self):
        return {"vmem_limit_bytes": 32 * 1024 * 1024 - 64 * 1024}

    def logical_shape(self, rhs):
        groups, n, k = rhs.shape
        return groups, k, n

    def validate(self, *, lhs, rhs, group_sizes, rhs_scale, rhs_bias):
        if lhs.ndim != 2 or lhs.dtype != jnp.bfloat16:
            raise ValueError("FP8 GMM requires BF16[M,K]")
        if rhs.ndim != 3 or rhs.dtype != jnp.uint8:
            raise ValueError("FP8 GMM requires raw uint8[G,N,K]")
        groups, k, n = self.logical_shape(rhs)
        if not groups or lhs.shape[1] != k or k % 128 or not 128 <= k <= 8192:
            raise ValueError("FP8 GMM requires matching K divisible by 128, up to 8192")
        if not lhs.shape[0] or lhs.shape[0] % 8 or not n or n % 128:
            raise ValueError("FP8 GMM requires M padded to 8 and N aligned to 128")
        if group_sizes.ndim != 1 or group_sizes.dtype != jnp.int32 or group_sizes.size < groups:
            raise ValueError("FP8 GMM requires int32 global group sizes")
        if (
            rhs_scale is None
            or rhs_scale.dtype != jnp.uint8
            or rhs_scale.shape != (groups, n // 128, 1, k // 128)
        ):
            raise ValueError("FP8 GMM requires compact uint8[G,N/128,1,K/128] E8M0 scales")
        if rhs_bias is not None:
            raise ValueError("checkpoint FP8 projections do not have a bias")

    def validate_tiling(self, *, tm, tk, tn, k):
        if tm not in (8, 16, 32, 64, 128) or tk != k or tn != 128:
            raise ValueError("FP8 GMM requires M=8/16/32/64/128, full-K, N=128")

    def block_specs(self, *, tk, tn, indices):
        def rhs_indices(*args):
            group, k_tile, n_tile = indices(*args)
            return group, n_tile, k_tile

        # A unit axis lets TPU DMA select an output block without a sub-eight
        # row slice in the second-minor dimension. It adds no storage bytes.
        def scale_indices(*args):
            group, _, n_tile = indices(*args)
            return group, n_tile, 0, 0

        return (
            pl.BlockSpec((None, tn, tk), rhs_indices),
            pl.BlockSpec((None, None, 1, tk // 128), scale_indices),
        )

    def dot(self, lhs, rhs, scales):
        weight = decode_weight_tile(rhs, scales)
        if self.quantize_activation:
            lhs = activation_fp8_roundtrip(lhs)
        return jax.lax.dot_general(
            lhs, weight, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )


def projection_block_m(tokens):
    """v5p gate: small decode rows; amortize conversion across prefill rows."""
    return 8 if tokens < 32 else 32


@functools.partial(jax.jit, static_argnames=("quantize_activation", "block_m"))
def fp8_linear(x, weight, scales, *, quantize_activation=True, block_m=None):
    """Dense group-of-one GMM. Caller owns TP; M/N tails are zero padded."""
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("expected two-dimensional activations and checkpoint weights")
    m, k = x.shape
    block_m = projection_block_m(m) if block_m is None else block_m
    n = weight.shape[0]
    if scales is None or scales.shape != (pl.cdiv(n, 128), k // 128):
        raise ValueError("invalid compact FP8 scale shape")
    if block_m not in (8, 16, 32, 64, 128):
        raise ValueError("unsupported FP8 M tile")
    pm, pn = pl.cdiv(m, block_m) * block_m, pl.cdiv(n, 128) * 128
    x = jnp.pad(x, ((0, pm - m), (0, 0)))
    weight = jnp.pad(weight, ((0, pn - n), (0, 0)))
    result = gmm(
        x,
        weight[None],
        jnp.asarray([pm], jnp.int32),
        preferred_element_type=jnp.bfloat16,
        rhs_scale=scales[None, :, None, :],
        tiling=(block_m, k, 128),
        interpret=jax.default_backend() != "tpu",
        rhs_adapter=CheckpointFP8Rhs(quantize_activation),
    )
    return result[:m, :n]
