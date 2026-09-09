"""Packed-token V4 routing and FP4 expert dispatch.

Keep the original pre-down-projection route scaling. Expert weights remain
packed in HBM; the low-bit GEMM expands only its current tile in VMEM.
"""

import functools

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.deepseek_v4.dense import DenseKernels
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul


def validate_backend(backend):
    if backend not in ("legacy", "gmm", "gmm_tuned"):
        raise ValueError("V4 MoE backend must be legacy, gmm or gmm_tuned; no automatic fallback")
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
        _, indices = jax.lax.top_k(score + weights["ffn.gate.bias"], config.active_experts)
    selected = jnp.take_along_axis(score, indices, axis=-1)
    denominator = functools.reduce(jnp.add, (selected[:, i] for i in range(config.active_experts)))[
        :, None
    ]
    routing = jnp.where(
        metadata.token_valid[:, None], selected / denominator * config.route_scale, 0
    )
    return indices, routing


def grouped_fp4_experts(
    x, w1, w3, w2, s1, s3, s2, expert_ids, routing_weights, *, axis_name="tensor", swiglu_limit=10.0
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

            gate = jnp.minimum(project(inputs, w1, s1).astype(jnp.float32), swiglu_limit)
            up = jnp.clip(project(inputs, w3, s3).astype(jnp.float32), -swiglu_limit, swiglu_limit)
            mixing = jnp.where(valid, route_weights[safe], 0)
            hidden = (mixing[:, None] * jax.nn.silu(gate) * up).astype(jnp.bfloat16)
            output = project(hidden, w2, s2).astype(jnp.float32)
            return total.at[rows].add(output, mode="drop")

        return jax.lax.fori_loop(0, (count + 7) // 8, tile, total)

    local = jax.lax.fori_loop(0, local_experts, accumulate, jnp.zeros(x.shape, jnp.float32))
    return jax.lax.psum(local, axis_name)


def moe(x, token_ids, weights, config, metadata, *, backend="legacy", dense_kernels=DenseKernels()):
    validate_backend(backend)
    indices, routing = route(x, token_ids, weights, config, metadata)
    experts = grouped_fp4_experts
    if backend in ("gmm", "gmm_tuned"):
        from sgl_jax.srt.kernels.deepseek_v4.moe_gmm import gmm_fp4_experts

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
        from sgl_jax.srt.kernels.deepseek_v4.projections import merged_linear

        prefix = "ffn.shared_experts.gate_up"
        gate, up = merged_linear(x, weights, prefix, weights[prefix + ".weight"].shape[0] // 2)
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
        metadata.token_valid[:, None], (routed + shared.astype(jnp.float32)).astype(jnp.bfloat16), 0
    )
