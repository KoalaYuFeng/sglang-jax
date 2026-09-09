"""V4 attention over ordinary physical pages and dynamic ragged requests."""

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.deepseek_v4.compressor import compress, physical_locations
from sgl_jax.srt.kernels.deepseek_v4.dense import DenseKernels
from sgl_jax.srt.kernels.deepseek_v4.numerics import (
    _fixed_tree_mean_last,
    _single_query_attention,
    hadamard_rotate,
    rope,
    rope_angles,
)
from sgl_jax.srt.kernels.low_bit.formats import (
    activation_fp4_roundtrip,
    activation_fp8_roundtrip,
    round_bf16,
)
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul


def attention(
    x,
    positions,
    weights,
    cache,
    config,
    metadata,
    locations,
    *,
    trace=None,
    hca_backend="reference",
    csa_backend="reference",
    output_tensor_axis=None,
    csa_decode_batch=False,
    dense_kernels=DenseKernels(),
):
    """V4 caller-owned optional head TP and pure-decode CSA projection batching.

    With output_tensor_axis, Q/sink and complete wo_a groups are caller-sharded,
    while index heads, shared KV and compressor state remain replicated. Gather
    BF16 wo_a results before the unchanged full-K wo_b to avoid a new floating
    point collective/reduction order. Serving defaults retain replicated
    attention; the opt-in model path selects TP for CSA/HCA layers only.
    """
    from sgl_jax.srt.kernels.deepseek_v4 import csa, hca

    hca.validate_backend(hca_backend)
    csa.validate_backend(csa_backend)
    linear = dense_kernels.linear
    rms_norm = dense_kernels.norm
    cache = dict(cache)
    from sgl_jax.srt.kernels.deepseek_v4 import normalization, projections

    phase = rope_angles(positions, config)
    cosine, sine = jnp.cos(phase), jnp.sin(phase)
    if dense_kernels.merged_projections:
        qa, kv = projections.merged_linear(
            x, weights, "attn.wqkv_a", weights["attn.q_norm.weight"].shape[0]
        )
    else:
        qa, kv = linear(x, weights, "attn.wq_a"), linear(x, weights, "attn.wkv")
    qr = rms_norm(qa, weights["attn.q_norm.weight"], config.eps)
    q = linear(qr, weights, "attn.wq_b").reshape(x.shape[0], config.heads, config.head_dim)
    if dense_kernels.fused_norm:
        q = normalization.qnorm_rope(q, cosine, sine, config.eps)
        kv = normalization.rms_norm(
            kv,
            weights["attn.kv_norm.weight"],
            config.eps,
            cosine=cosine,
            sine=sine,
            quantize_nope=True,
        )
    else:
        square = round_bf16(q.astype(jnp.float32) ** 2)
        variance = round_bf16(_fixed_tree_mean_last(square.astype(jnp.float32))[..., None])
        variance = round_bf16(variance.astype(jnp.float32) + config.eps)
        inverse = round_bf16(jax.lax.rsqrt(variance.astype(jnp.float32)))
        q = rope(round_bf16(q.astype(jnp.float32) * inverse.astype(jnp.float32)), positions, config)
        kv = rms_norm(kv, weights["attn.kv_norm.weight"], config.eps)
        kv = rope(kv, positions, config)
        kv = jnp.concatenate(
            (activation_fp8_roundtrip(kv[:, : -config.rope_dim], 64), kv[:, -config.rope_dim :]),
            axis=-1,
        )
    destination = jnp.where(metadata.token_valid, locations, cache["window"].shape[0])
    cache["window"] = cache["window"].at[destination].set(kv, mode="drop")
    logical_window = (
        jnp.maximum(positions[:, None] - config.window + 1, 0) + jnp.arange(config.window)[None]
    )
    window = physical_locations(metadata, metadata.token_requests, logical_window)
    window = jnp.where(
        (logical_window <= positions[:, None]) & metadata.token_valid[:, None], window, -1
    )
    indices = window
    all_kv = cache["window"]
    if config.ratio:
        cache = compress(
            x,
            weights,
            cache,
            config,
            metadata,
            hca_backend=hca_backend,
            csa_backend=csa_backend,
            csa_decode_batch=csa_decode_batch,
        )
        candidate_count = config.max_context // config.ratio
        logical_candidates = jnp.arange(candidate_count)
        terminal_positions = (logical_candidates + 1) * config.ratio - 1
        available = (positions + 1) // config.ratio
        if config.ratio == 4:
            cache = compress(
                x,
                weights,
                cache,
                config,
                metadata,
                index=True,
                csa_backend=csa_backend,
                csa_decode_batch=csa_decode_batch,
            )
            iq = linear(qr, weights, "attn.indexer.wq_b").reshape(
                x.shape[0], config.index_heads, config.index_dim
            )
            iq = activation_fp4_roundtrip(hadamard_rotate(rope(iq, positions, config)))
            iw = round_bf16(
                linear(x, weights, "attn.indexer.weights_proj").astype(jnp.float32)
                * (config.index_dim**-0.5 * config.index_heads**-0.5)
            )

            def select(item):
                query, mixing, request, length = item
                physical = (
                    physical_locations(metadata, request[None], terminal_positions[None])[0]
                    // config.ratio
                )
                keys = cache["index.compressed"][physical]
                score = round_bf16(
                    jnp.einsum("hd,td->ht", query, keys, preferred_element_type=jnp.float32)
                )
                score = round_bf16(
                    jnp.maximum(score.astype(jnp.float32), 0) * mixing.astype(jnp.float32)[:, None]
                )
                score = round_bf16(jnp.sum(score.astype(jnp.float32)[None], axis=1)[0]).astype(
                    jnp.float32
                )
                score = jnp.where(logical_candidates < length, score, -jnp.inf)
                return score, jax.lax.top_k(score, min(config.index_topk, candidate_count))[1]

            if csa_backend == "pallas":
                index_score, selected = csa.select(
                    iq, iw, cache["index.compressed"], metadata, config
                )
            else:
                index_score, selected = jax.lax.map(
                    select, (iq, iw, metadata.token_requests, available)
                )
            if trace is not None:
                trace.update(index_q=iq, index_score=index_score, index_selected=selected)
        else:
            selected = jnp.broadcast_to(logical_candidates, (x.shape[0], candidate_count))
        physical = (
            physical_locations(metadata, metadata.token_requests, (selected + 1) * config.ratio - 1)
            // config.ratio
        )
        compressed_valid = (
            (selected >= 0) & (selected < available[:, None]) & metadata.token_valid[:, None]
        )
        compressed = jnp.where(
            compressed_valid,
            physical + cache["window"].shape[0],
            -1,
        )
        indices = jnp.concatenate((window, compressed), axis=-1)
        if not (
            (config.ratio == 128 and hca_backend == "pallas")
            or (config.ratio == 4 and csa_backend == "pallas")
        ):
            all_kv = jnp.concatenate((cache["window"], cache["main.compressed"]), axis=0)
    if config.ratio == 128 and hca_backend == "pallas":
        value = hca.attend(
            q,
            cache["window"],
            cache["main.compressed"],
            window,
            positions,
            weights["attn.attn_sink"],
            metadata,
            config,
        )
    elif config.ratio == 4 and csa_backend == "pallas":
        value = csa.attend(
            q,
            cache["window"][jnp.maximum(window, 0)],
            window >= 0,
            cache["main.compressed"][physical],
            jnp.sum(compressed_valid, axis=1, dtype=jnp.int32),
            weights["attn.attn_sink"],
            config,
        )
    else:
        value = jax.lax.map(
            lambda row: _single_query_attention(
                row[0], all_kv, row[1], weights["attn.attn_sink"], config.head_dim**-0.5
            ),
            (q, indices),
        )
    if trace is not None:
        trace.update(q=q, qr=qr, kv=kv, indices=indices, attention_value=value)
    weight, scales = weights["attn.wo_a.weight"], weights["attn.wo_a.scale"]
    if dense_kernels.fused_wo_a:
        projected = projections.inverse_rope_fp8_wo_a(
            value,
            weight,
            scales,
            cosine,
            sine,
            groups=config.groups,
            head_dim=config.head_dim,
        )
    else:
        value = rope(value, positions, config, inverse=True).reshape(x.shape[0], config.groups, -1)
        projected = [
            low_bit_matmul(
                value[:, group],
                weight[group * config.o_rank : (group + 1) * config.o_rank],
                scales[group * config.o_rank // 128 : (group + 1) * config.o_rank // 128],
                weight_format="fp8",
                quantize_activation=False,
            )
            for group in range(config.groups)
        ]
        projected = jnp.concatenate(projected, axis=-1)
    if output_tensor_axis is not None:
        projected = jax.lax.all_gather(projected, output_tensor_axis, axis=1, tiled=True)
    output = linear(projected, weights, "attn.wo_b")
    return jnp.where(metadata.token_valid[:, None], output, 0), cache
