"""Checkpoint FP4 storage adapter for the existing MegaBlocks GMM driver.

Only the RHS storage layout and tile arithmetic are specialized. The shared
driver owns nonempty-group scheduling, group-offset EP, scalar prefetch,
weight-tile DMA/pipelining and masked writes. No complete expert weight is
sliced or expanded before entering Pallas. Full-K FP32 accumulation and the
eight-row tile deliberately retain the accepted V4 arithmetic for this gate.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.low_bit.formats import (
    activation_fp8_roundtrip,
    decode_e8m0,
    decode_fp4_codes,
)
from sgl_jax.srt.kernels.low_bit.matmul import _expand_columns, _unpack_fp4_vmem


@dataclass(frozen=True)
class CheckpointFP4Rhs:
    quantize_activation: bool = True
    name = "checkpoint_fp4"

    @property
    def compiler_params(self):
        return {"vmem_limit_bytes": 16 * 1024 * 1024 - 64 * 1024}

    def logical_shape(self, rhs):
        experts, n, packed_k = rhs.shape
        return experts, packed_k * 2, n

    def validate(self, *, lhs, rhs, group_sizes, rhs_scale, rhs_bias):
        if lhs.ndim != 2 or lhs.dtype != jnp.bfloat16:
            raise ValueError("FP4 GMM requires BF16[M,K] activations")
        if rhs.ndim != 3 or rhs.dtype != jnp.uint8:
            raise ValueError("FP4 GMM requires raw uint8[E,N,K/2] weights")
        experts, k, n = self.logical_shape(rhs)
        if experts <= 0 or lhs.shape[1] != k or k % 128 or not 128 <= k <= 8192:
            raise ValueError(
                "FP4 GMM requires matching K, a multiple of 128 up to 8192"
            )
        if lhs.shape[0] <= 0 or lhs.shape[0] % 8 or n <= 0 or n % 128:
            raise ValueError("FP4 GMM requires M padded to 8 and N aligned to 128")
        if group_sizes.ndim != 1 or group_sizes.dtype != jnp.int32:
            raise ValueError("FP4 GMM requires int32 group sizes")
        if group_sizes.size < experts:
            raise ValueError("FP4 GMM has more local experts than global groups")
        if (
            rhs_scale is None
            or rhs_scale.dtype != jnp.uint8
            or rhs_scale.shape != (experts, n, k // 32)
        ):
            raise ValueError("FP4 GMM requires compact uint8[E,N,K/32] E8M0 scales")
        if rhs_bias is not None:
            raise ValueError("checkpoint FP4 GMM does not have a projection bias")

    def validate_tiling(self, *, tm, tk, tn, k):
        if (tm, tk, tn) != (8, k, 128):
            raise ValueError(
                "checkpoint FP4 numerical gate requires an 8/full-K/128 tile"
            )

    def block_specs(self, *, tk, tn, indices):
        def packed_indices(*args):
            expert, k_tile, n_tile = indices(*args)
            return expert, n_tile, k_tile

        return (
            pl.BlockSpec((None, tn, tk // 2), packed_indices),
            pl.BlockSpec((None, tn, tk // 32), packed_indices),
        )

    def dot(self, lhs, rhs, scales):
        decoded = _unpack_fp4_vmem(rhs).astype(jnp.float32)
        weight = (decoded * _expand_columns(decode_e8m0(scales), 32)).astype(
            jnp.bfloat16
        )
        if self.quantize_activation:
            lhs = activation_fp8_roundtrip(lhs)
        return jax.lax.dot_general(
            lhs, weight, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )


def grouped_fp4_matmul(
    x, weights, scales, group_sizes, *, first_expert, interpret=False
):
    """Invoke shared GMM on sorted route rows, retaining raw checkpoint storage.

    group_sizes sums to the padded row count. A final, non-local sentinel group
    owns padding/inactive routes. GMM zeros rows outside this chip's experts.
    """
    return gmm(
        x,
        weights,
        group_sizes,
        preferred_element_type=jnp.bfloat16,
        rhs_scale=scales,
        tiling=(8, x.shape[1], 128),
        group_offset=first_expert,
        interpret=interpret,
        rhs_adapter=CheckpointFP4Rhs(),
    )


def dequantize_packed_pairs(rhs, scales):
    """Scale nibble streams before interleaving, entirely inside tile VMEM.

    Each FP32 multiply and BF16 rounding is unchanged. The expanded FP32 scale
    has K/2 rather than K columns. No BF16 scale approximation is introduced.
    """
    expanded = _expand_columns(decode_e8m0(scales), 16)
    lo = (decode_fp4_codes(rhs & 15).astype(jnp.float32) * expanded).astype(
        jnp.bfloat16
    )
    hi = (decode_fp4_codes(rhs >> 4).astype(jnp.float32) * expanded).astype(
        jnp.bfloat16
    )
    lo_bits = jax.lax.bitcast_convert_type(lo, jnp.uint16).astype(jnp.uint32)
    hi_bits = jax.lax.bitcast_convert_type(hi, jnp.uint16).astype(jnp.uint32)
    return pltpu.bitcast((lo_bits | (hi_bits << jnp.uint32(16))).T, jnp.bfloat16).T


@dataclass(frozen=True)
class TiledCheckpointFP4Rhs(CheckpointFP4Rhs):
    transpose_scales: bool = False
    tile_m: int = 8
    tile_n: int = 128
    packed_scale: bool = False

    @property
    def name(self):
        return (
            "checkpoint_fp4_tiled"
            + ("_scale_kn" if self.transpose_scales else "")
            + ("_packed_scale" if self.packed_scale else "")
        )

    def validate(self, *, lhs, rhs, group_sizes, rhs_scale, rhs_bias):
        # Validation uses shape/dtype only; do not transpose actual HBM operands.
        if self.transpose_scales and rhs_scale is not None:
            if rhs_scale.ndim != 3:
                raise ValueError("transposed scales must be uint8[E,K/32,N]")
            rhs_scale = jax.ShapeDtypeStruct(
                (rhs_scale.shape[0], rhs_scale.shape[2], rhs_scale.shape[1]),
                rhs_scale.dtype,
            )
        super().validate(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            rhs_scale=rhs_scale,
            rhs_bias=rhs_bias,
        )
        if self.tile_m not in (8, 16, 32) or self.tile_n not in (128, 256, 512):
            raise ValueError("unsupported FP4 tile configuration")
        if lhs.shape[0] % self.tile_m:
            raise ValueError("route rows must be padded to configured tile_m")
        if rhs.shape[1] % self.tile_n:
            raise ValueError("output width must divide configured tile_n")

    def validate_tiling(self, *, tm, tk, tn, k):
        if (tm, tk, tn) != (self.tile_m, k, self.tile_n):
            raise ValueError(
                "FP4 adapter requires its declared M/N and full-K accumulation"
            )

    def block_specs(self, *, tk, tn, indices):
        weight_spec, scale_spec = super().block_specs(tk=tk, tn=tn, indices=indices)
        if self.transpose_scales:
            scale_spec = pl.BlockSpec((None, tk // 32, tn), indices)
        return weight_spec, scale_spec

    def dot(self, lhs, rhs, scales):
        if self.transpose_scales:
            scales = scales.T  # Tile-local VMEM transpose, never a full expert tensor.
        if self.packed_scale:
            weight = dequantize_packed_pairs(rhs, scales)
            if self.quantize_activation:
                lhs = activation_fp8_roundtrip(lhs)
            return jax.lax.dot_general(
                lhs,
                weight,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        return super().dot(lhs, rhs, scales)


def transpose_compact_scales(scales):
    """Lossless E8M0 byte permutation; caller performs this once before timing."""
    if scales.ndim != 3 or scales.dtype != jnp.uint8:
        raise ValueError("expected compact uint8[E,N,K/32] scales")
    return jnp.swapaxes(scales, 1, 2)


def tuned_fp4_tile_m(tokens):
    """Conservative shape policy: keep M8 below the measured M128 prefill gate."""
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("FP4 token capacity must be a positive integer")
    return 32 if tokens >= 128 else 8


def grouped_fp4_matmul_tuned(
    x, weights, scales, group_sizes, *, first_expert, tile_m, interpret=False
):
    """Explicit compact-KN scale ABI over the unchanged shared GMM driver."""
    tile_n = min(256, weights.shape[1])
    return gmm(
        x,
        weights,
        group_sizes,
        rhs_scale=scales,
        preferred_element_type=jnp.bfloat16,
        tiling=(tile_m, x.shape[1], tile_n),
        group_offset=first_expert,
        interpret=interpret,
        rhs_adapter=TiledCheckpointFP4Rhs(
            transpose_scales=True, packed_scale=True, tile_m=tile_m, tile_n=tile_n
        ),
    )
