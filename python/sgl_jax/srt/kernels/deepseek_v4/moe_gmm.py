"""V4 math over shared EPMoE grouping and MegaBlocks GMM execution.

The router is unchanged. Packed checkpoint weights feed the shared GMM driver
through CheckpointFP4Rhs, without a Python/JAX loop selecting whole experts.
Route scaling precedes W2's activation QAT; each projection rounds to BF16;
local outputs combine in ascending expert order, as in the retained baseline.
"""

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.gmm.routing import expert_permutation
from sgl_jax.srt.kernels.low_bit.gmm import grouped_fp4_matmul


def pack_routes(expert_ids, routing_weights, num_experts):
    """Coalesce duplicate choices, then group routes with the shared helper.

    A non-local sentinel expert owns inactive routes and padding. Coalescing
    before activation quantization preserves the legacy semantics even for a
    repeated expert ID (two separately quantized W2 inputs are not equivalent).
    """
    if expert_ids.ndim != 2 or expert_ids.shape != routing_weights.shape:
        raise ValueError("expected matching [tokens, top_k] expert IDs and routing weights")
    if not jnp.issubdtype(expert_ids.dtype, jnp.integer):
        raise ValueError("expert IDs must be integers")
    tokens, top_k = expert_ids.shape
    if tokens <= 0 or top_k <= 0 or num_experts <= 0:
        raise ValueError("token, top-k and expert counts must be positive")
    same = expert_ids[:, :, None] == expert_ids[:, None, :]
    mixing = jnp.sum(jnp.where(same, routing_weights[:, None, :], 0), axis=-1)
    earlier = jnp.arange(top_k)[None, :] < jnp.arange(top_k)[:, None]
    first = ~jnp.any(same & earlier[None], axis=-1)
    active = first & (mixing != 0) & (expert_ids >= 0) & (expert_ids < num_experts)
    ids = jnp.where(active, expert_ids, num_experts).reshape(-1)
    mixing = jnp.where(active, mixing, 0).reshape(-1)
    padding = (-ids.size) % 8
    ids = jnp.pad(ids, (0, padding), constant_values=num_experts)
    mixing = jnp.pad(mixing, (0, padding))
    permutation, sizes = expert_permutation(ids, num_experts + 1)
    return permutation, sizes, ids[permutation], mixing[permutation]


def gmm_fp4_experts(
    x,
    w1,
    w3,
    w2,
    s1,
    s3,
    s2,
    expert_ids,
    routing_weights,
    *,
    num_experts=256,
    axis_name="tensor",
    swiglu_limit=10.0,
    interpret=False,
    tuned=False,
):
    if expert_ids.shape[0] != x.shape[0]:
        raise ValueError("route rows must match input tokens")
    if w1.shape[0] * jax.lax.axis_size(axis_name) != num_experts:
        raise ValueError("local expert count times EP size must equal num_experts")
    if w1.shape != w3.shape or w2.shape != (w1.shape[0], x.shape[1], w1.shape[1] // 2):
        raise ValueError("packed W1/W3/W2 expert projection shapes do not agree")
    permutation, sizes, sorted_ids, mixing = pack_routes(expert_ids, routing_weights, num_experts)
    top_k = expert_ids.shape[1]
    rows = permutation // top_k
    valid = (sorted_ids < num_experts) & (rows < x.shape[0])
    inputs = jnp.where(valid[:, None], x[jnp.minimum(rows, x.shape[0] - 1)], 0)
    first_expert = jax.lax.axis_index(axis_name) * w1.shape[0]
    if tuned:
        from sgl_jax.srt.kernels.low_bit.fp4_tuning import (
            grouped_fp4_matmul_tuned,
            tuned_fp4_tile_m,
        )

        tile_m = tuned_fp4_tile_m(x.shape[0])
        padding = (-inputs.shape[0]) % tile_m
        if padding:
            inputs = jnp.pad(inputs, ((0, padding), (0, 0)))
            mixing = jnp.pad(mixing, (0, padding))
            sizes = sizes.at[-1].add(padding)

    def project(value, weight, scale):
        if tuned:
            return grouped_fp4_matmul_tuned(
                value,
                weight,
                scale,
                sizes,
                first_expert=first_expert,
                tile_m=tile_m,
                interpret=interpret,
            )
        return grouped_fp4_matmul(
            value, weight, scale, sizes, first_expert=first_expert, interpret=interpret
        )

    gate = jnp.minimum(project(inputs, w1, s1).astype(jnp.float32), swiglu_limit)
    up = jnp.clip(project(inputs, w3, s3).astype(jnp.float32), -swiglu_limit, swiglu_limit)
    hidden = (mixing[:, None] * jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    projected = project(hidden, w2, s2).astype(jnp.float32)
    inverse = jnp.zeros_like(permutation).at[permutation].set(jnp.arange(permutation.size))
    unsorted = projected[inverse][: expert_ids.size].reshape(*expert_ids.shape, x.shape[1])
    expert_order = jnp.argsort(expert_ids, axis=-1, stable=True)
    ordered = jnp.take_along_axis(unsorted, expert_order[..., None], axis=1)
    local = jnp.zeros(x.shape, jnp.float32)
    for choice in range(top_k):
        local = local + ordered[:, choice]
    return jax.lax.psum(local, axis_name)
