"""Unoptimized expert-parallel FP4 SwiGLU reference execution on TPU.

Inputs to shard_map are replicated activations/routes and expert-axis-sharded
packed weights. A used expert processes the complete token block with zero
route weights for unused rows. This is intentionally correctness-first, not
an efficient token dispatcher. Expert selection may copy a packed tensor in
HBM; it never dequantizes the full expert weight to HBM.
"""

import jax
import jax.numpy as jnp

from .matmul import low_bit_matmul


def routed_fp4_experts(
    x, w1, w3, w2, s1, s3, s2, expert_ids, routing_weights, *, axis_name="tensor", swiglu_limit=10.0
):
    """Local experts + cross-chip sum; return FP32 before shared-expert add."""
    local_experts = w1.shape[0]
    first_expert = jax.lax.axis_index(axis_name) * local_experts

    def accumulate(expert, total):
        route = jnp.sum(
            jnp.where(expert_ids == first_expert + expert, routing_weights, 0.0), axis=-1
        )

        def compute(total):
            def linear(value, weight, scales):
                return low_bit_matmul(
                    value,
                    weight[expert],
                    scales[expert],
                    weight_format="fp4",
                    quantize_activation=True,
                )

            gate = jnp.minimum(linear(x, w1, s1).astype(jnp.float32), swiglu_limit)
            up = jnp.clip(linear(x, w3, s3).astype(jnp.float32), -swiglu_limit, swiglu_limit)
            # In official V4, route scaling precedes the down projection and
            # its activation quantization, unlike post-GEMM mixture weighting.
            hidden = (route[:, None] * jax.nn.silu(gate) * up).astype(jnp.bfloat16)
            return total + linear(hidden, w2, s2).astype(jnp.float32)

        return jax.lax.cond(jnp.any(route != 0), compute, lambda total: total, total)

    local = jax.lax.fori_loop(0, local_experts, accumulate, jnp.zeros(x.shape, jnp.float32))
    return jax.lax.psum(local, axis_name)
