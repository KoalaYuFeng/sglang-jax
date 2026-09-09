"""Shared stable expert grouping for EPMoE and checkpoint-native adapters."""

import jax.numpy as jnp


def expert_permutation(expert_ids, num_experts):
    """Return flattened route permutation and global expert histogram.

    IDs must be in [0, num_experts). Callers own invalid/padding sentinels and
    numerical combine semantics; grouping never changes routing weights.
    """
    flat = jnp.ravel(expert_ids)
    return jnp.argsort(flat, stable=True), jnp.bincount(flat, length=num_experts)
