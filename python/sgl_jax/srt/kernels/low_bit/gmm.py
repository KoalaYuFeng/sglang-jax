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

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip, decode_e8m0
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
            raise ValueError("FP4 GMM requires matching K, a multiple of 128 up to 8192")
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
            raise ValueError("checkpoint FP4 numerical gate requires an 8/full-K/128 tile")

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
        weight = (decoded * _expand_columns(decode_e8m0(scales), 32)).astype(jnp.bfloat16)
        if self.quantize_activation:
            lhs = activation_fp8_roundtrip(lhs)
        return jax.lax.dot_general(
            lhs, weight, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )


def grouped_fp4_matmul(x, weights, scales, group_sizes, *, first_expert, interpret=False):
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
