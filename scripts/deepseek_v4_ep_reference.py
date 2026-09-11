"""Independent deterministic EP arithmetic for acceptance, never serving dispatch.

The candidate gathers all partials. This reference passes one accumulator
between ranks, then broadcasts its bits with point-to-point permutations.
Neither this module nor its NumPy oracle imports the candidate helper.
"""

import jax
import jax.numpy as jnp
import numpy as np


def numpy_ep_sum(partials):
    """Ascending-rank FP32 rounding, not exact real-number summation."""
    values = np.asarray(partials)
    if values.dtype != np.float32 or values.shape[0] < 1:
        raise ValueError("requires nonempty rank-major FP32 partials")
    result = values[0].copy()
    for value in values[1:]:
        np.add(result, value, out=result, dtype=np.float32)
    return result


def ring_ep_sum_reference(local, axis_name="tensor"):
    """EP rank 0 -> 1 -> ... -> N-1, explicit FP32 rounding at each owner."""
    if local.dtype != jnp.float32:
        raise ValueError("reference requires FP32 partials")
    ranks = jax.lax.axis_size(axis_name)
    if ranks == 1:
        return local
    rank = jax.lax.axis_index(axis_name)
    carry = jnp.where(rank == 0, local, jnp.zeros_like(local))
    for owner in range(1, ranks):
        incoming = jax.lax.ppermute(carry, axis_name, [(owner - 1, owner)])
        rounded = jax.lax.optimization_barrier(incoming + local)
        carry = jnp.where(rank == owner, rounded, jnp.zeros_like(local))
    result = carry
    for destination in range(ranks - 1):
        received = jax.lax.ppermute(carry, axis_name, [(ranks - 1, destination)])
        result = jnp.where(rank == destination, received, result)
    return result


def numpy_checkpoint_partials(checkpoint, layer, x, ids, mixing, *, ranks=4):
    """Selected real FP4 experts only, decoded/calculated independently on CPU.

Imports the pre-existing byte/activation NumPy test oracle, not a TPU GEMM.
Preserves rank ownership and local ascending-expert FP32 accumulation before
the independently specified EP sum. Does not expand the full checkpoint.
"""
    from sgl_jax.test.kernels.test_deepseek_v4_low_bit import (
        _bf16,
        _reference_activation,
        _reference_weight,
    )

    if x.shape[0] != 1 or ids.shape != mixing.shape or ids.shape[0] != 1:
        raise ValueError("independent byte oracle expects one captured token")
    experts = checkpoint.config["n_routed_experts"]
    if experts % ranks:
        raise ValueError("experts must partition evenly")
    limit = checkpoint.config["swiglu_limit"]
    partials = np.zeros((ranks, *x.shape), np.float32)
    for expert in sorted(set(ids[0].tolist())):
        scale = np.sum(np.where(ids[0] == expert, mixing[0], 0), dtype=np.float32)
        if scale == 0:
            continue

        def project(value, projection, expert=expert):
            host = checkpoint.load_linear(f"layers.{layer}.ffn.experts.{expert}.w{projection}")
            weight = _reference_weight(host.data, host.scales, "fp4")
            return _bf16(_reference_activation(value) @ weight.T)

        gate = np.minimum(project(x, 1), limit)
        up = np.clip(project(x, 3), -limit, limit)
        hidden = _bf16(scale * (gate / (1 + np.exp(-gate))) * up)
        owner = expert // (experts // ranks)
        np.add(partials[owner], project(hidden, 2), out=partials[owner], dtype=np.float32)
    return partials
