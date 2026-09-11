"""V4 numerical contract for the FP32 expert-parallel reduction."""

import jax
import jax.numpy as jnp


def ordered_ep_sum(local, axis_name="tensor"):
    """Sum rank-ordered partials with batch/layout-independent FP32 rounding.

    A hardware psum may change its accumulation order across buffer chunks.
    Even a one-ULP difference can cross the subsequent shared-add/BF16
    midpoint and make identical requests diverge. Gather rank-ordered values
    without arithmetic, then explicitly round each ascending-rank addition.
    For EP4 the contract is ((rank0 + rank1) + rank2) + rank3.

    This specifies reproducible arithmetic, not exact real-number summation.
    Local expert accumulation, packed weights and GEMM are unchanged.
    """
    if local.dtype != jnp.float32:
        raise ValueError("V4 expert-parallel partials must be FP32")
    ranks = jax.lax.axis_size(axis_name)
    if ranks == 1:
        return local
    gathered = jax.lax.all_gather(local, axis_name, axis=0)
    total = gathered[0]
    for rank in range(1, ranks):
        total = jax.lax.optimization_barrier(total + gathered[rank])
    return total
