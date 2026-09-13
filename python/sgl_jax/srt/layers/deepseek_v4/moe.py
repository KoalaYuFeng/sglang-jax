"""Packed-token V4 routing and FP4 expert dispatch.

Keep the original pre-down-projection route scaling. Expert weights remain
packed in HBM; the low-bit GEMM expands only its current tile in VMEM.
"""

import functools

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.gmm.routing import expert_permutation
from sgl_jax.srt.kernels.low_bit.fp4 import grouped_fp4_matmul
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul
from sgl_jax.srt.layers.deepseek_v4.collectives import ordered_ep_sum
from sgl_jax.srt.layers.deepseek_v4.linear import DenseKernels


def validate_backend(backend):
    if backend not in ("legacy", "gmm", "gmm_tuned"):
        raise ValueError(
            "V4 MoE backend must be legacy, gmm or gmm_tuned; no automatic fallback"
        )
    return backend


def route(x, token_ids, weights, config, metadata):
    rows = metadata.router_rows
    blocks = jnp.where((rows >= 0)[..., None], x[jnp.maximum(rows, 0)], 0)
    score = jax.lax.map(
        lambda block: jnp.matmul(
            block.astype(jnp.float32),
            weights["ffn.gate.weight"].T.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        ),
        blocks,
    ).reshape(-1, weights["ffn.gate.weight"].shape[0])[metadata.router_output_rows]
    score = jnp.sqrt(jax.nn.softplus(score))
    if config.hash_routing:
        indices = weights["ffn.gate.tid2eid"][token_ids]
    else:
        _, indices = jax.lax.top_k(
            score + weights["ffn.gate.bias"], config.active_experts
        )
    selected = jnp.take_along_axis(score, indices, axis=-1)
    denominator = functools.reduce(
        jnp.add, (selected[:, i] for i in range(config.active_experts))
    )[:, None]
    routing = jnp.where(
        metadata.token_valid[:, None], selected / denominator * config.route_scale, 0
    )
    return indices, routing


def grouped_fp4_experts(
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
    axis_name="tensor",
    swiglu_limit=10.0,
):
    """Gather active tokens per local expert, compute 8-row tiles, scatter back.

    Fixed-size index storage cannot overflow even if all tokens route to one
    expert. Dynamic loop bounds skip unused tiles instead of computing a full
    padded batch for every expert. The outer expert order matches the oracle.
    """
    tokens, local_experts = x.shape[0], w1.shape[0]
    first_expert = jax.lax.axis_index(axis_name) * local_experts
    padded = (tokens + 7) // 8 * 8

    def accumulate(expert, total):
        route_weights = jnp.sum(
            jnp.where(expert_ids == first_expert + expert, routing_weights, 0), axis=-1
        )
        active = route_weights != 0
        count = jnp.sum(active.astype(jnp.int32))
        permutation = jnp.nonzero(active, size=padded, fill_value=tokens)[0]

        def tile(tile_id, total):
            rows = jax.lax.dynamic_slice_in_dim(permutation, tile_id * 8, 8)
            valid = rows < tokens
            safe = jnp.minimum(rows, tokens - 1)
            inputs = jnp.where(valid[:, None], x[safe], 0)

            def project(value, weight, scale):
                return low_bit_matmul(
                    value,
                    weight[expert],
                    scale[expert],
                    weight_format="fp4",
                    quantize_activation=True,
                )

            gate = jnp.minimum(
                project(inputs, w1, s1).astype(jnp.float32), swiglu_limit
            )
            up = jnp.clip(
                project(inputs, w3, s3).astype(jnp.float32), -swiglu_limit, swiglu_limit
            )
            mixing = jnp.where(valid, route_weights[safe], 0)
            hidden = (mixing[:, None] * jax.nn.silu(gate) * up).astype(jnp.bfloat16)
            output = project(hidden, w2, s2).astype(jnp.float32)
            return total.at[rows].add(output, mode="drop")

        return jax.lax.fori_loop(0, (count + 7) // 8, tile, total)

    local = jax.lax.fori_loop(
        0, local_experts, accumulate, jnp.zeros(x.shape, jnp.float32)
    )
    return ordered_ep_sum(local, axis_name)


def moe(
    x,
    token_ids,
    weights,
    config,
    metadata,
    *,
    backend="legacy",
    dense_kernels=DenseKernels(),
):
    validate_backend(backend)
    indices, routing = route(x, token_ids, weights, config, metadata)
    experts = grouped_fp4_experts
    if backend in ("gmm", "gmm_tuned"):
        experts = gmm_fp4_experts
        if backend == "gmm_tuned":
            experts = functools.partial(gmm_fp4_experts, tuned=True)
    routed = experts(
        x,
        *(weights["experts." + key] for key in ("w1", "w3", "w2", "s1", "s3", "s2")),
        indices,
        routing,
        swiglu_limit=config.swiglu_limit,
    )
    linear = dense_kernels.linear
    if dense_kernels.merged_projections:
        from sgl_jax.srt.layers.deepseek_v4.linear import merged_linear

        prefix = "ffn.shared_experts.gate_up"
        gate, up = merged_linear(
            x, weights, prefix, weights[prefix + ".weight"].shape[0] // 2
        )
    else:
        gate, up = (
            linear(x, weights, "ffn.shared_experts.w1"),
            linear(x, weights, "ffn.shared_experts.w3"),
        )
    gate = jnp.minimum(gate.astype(jnp.float32), config.swiglu_limit)
    up = jnp.clip(
        up.astype(jnp.float32),
        -config.swiglu_limit,
        config.swiglu_limit,
    )
    hidden = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    shared = linear(hidden, weights, "ffn.shared_experts.w2")
    return jnp.where(
        metadata.token_valid[:, None],
        (routed + shared.astype(jnp.float32)).astype(jnp.bfloat16),
        0,
    )


def pack_routes(expert_ids, routing_weights, num_experts):
    """Coalesce duplicate choices, then group routes with the shared helper.

    A non-local sentinel expert owns inactive routes and padding. Coalescing
    before activation quantization preserves the legacy semantics even for a
    repeated expert ID (two separately quantized W2 inputs are not equivalent).
    """
    if expert_ids.ndim != 2 or expert_ids.shape != routing_weights.shape:
        raise ValueError(
            "expected matching [tokens, top_k] expert IDs and routing weights"
        )
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
    permutation, sizes, sorted_ids, mixing = pack_routes(
        expert_ids, routing_weights, num_experts
    )
    top_k = expert_ids.shape[1]
    rows = permutation // top_k
    valid = (sorted_ids < num_experts) & (rows < x.shape[0])
    inputs = jnp.where(valid[:, None], x[jnp.minimum(rows, x.shape[0] - 1)], 0)
    first_expert = jax.lax.axis_index(axis_name) * w1.shape[0]
    if tuned:
        from sgl_jax.srt.kernels.low_bit.fp4 import (
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
    up = jnp.clip(
        project(inputs, w3, s3).astype(jnp.float32), -swiglu_limit, swiglu_limit
    )
    hidden = (mixing[:, None] * jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    projected = project(hidden, w2, s2).astype(jnp.float32)
    inverse = (
        jnp.zeros_like(permutation).at[permutation].set(jnp.arange(permutation.size))
    )
    unsorted = projected[inverse][: expert_ids.size].reshape(
        *expert_ids.shape, x.shape[1]
    )
    expert_order = jnp.argsort(expert_ids, axis=-1, stable=True)
    ordered = jnp.take_along_axis(unsorted, expert_order[..., None], axis=1)
    local = jnp.zeros(x.shape, jnp.float32)
    for choice in range(top_k):
        local = local + ordered[:, choice]
    return ordered_ep_sum(local, axis_name)
