"""Thin V4 ABI adapters to the original CSA Pallas programs.

Request/page ownership and persistent FP32 state remain in the V4 framework.
Main/index KV still uses BF16 with FP8/FP4 QAT; checkpoint storage is unchanged.
"""

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.csa.compressor import (
    csa_emit_overlap_pallas,
    csa_emit_selected_pallas,
    csa_project_decode_pallas,
    csa_project_pallas,
)
from sgl_jax.srt.kernels.csa.indexer import paged_lightning_topk
from sgl_jax.srt.kernels.csa.joint_attention import joint_attention_pallas
from sgl_jax.srt.kernels.deepseek_v4.numerics import rope_angles


def validate_backend(backend):
    if backend not in ("pallas", "reference"):
        raise ValueError("V4 CSA backend must be pallas or reference; no automatic fallback")
    return backend


def validate_config(config):
    if (config.ratio, config.head_dim, config.index_dim, config.rope_dim, config.window) != (
        4,
        512,
        128,
        64,
        128,
    ):
        raise ValueError("original CSA V4 adapter requires r4/d512/index128/rope64/window128")


def project(x, wkv, wgate, metadata, config, *, decode_batch=False):
    validate_config(config)
    fused = jnp.concatenate((wkv.T, wgate.T), axis=1)
    if decode_batch:
        # The V4 model selects this only for ForwardMode.DECODE, never MIXED.
        # Invalid slot-aligned rows compute zero and never own request state.
        if x.shape[0] != metadata.query_lens.shape[0]:
            raise ValueError("batched CSA projection requires slot-aligned pure decode")
        projected = csa_project_decode_pallas(jnp.where(metadata.token_valid[:, None], x, 0), fused)
        return projected[:, : wkv.shape[0]], projected[:, wkv.shape[0] :]
    rows = metadata.router_rows
    blocks = jnp.where((rows >= 0)[..., None], x[jnp.maximum(rows, 0)], 0)
    requests = metadata.token_requests[jnp.maximum(rows, 0)]
    single = jnp.any((rows >= 0) & (metadata.query_lens[requests] == 1), axis=1)
    live_row = jnp.argmax(rows >= 0, axis=1).astype(jnp.int32)

    def project_block(item):
        block, is_single, row = item

        def decode(block):
            # A single-query router block has exactly one live row. Compute
            # only that GEMV in the original Pallas kernel; padded projections
            # are zero and never own request state. No reference fallback.
            query = jax.lax.dynamic_slice_in_dim(block, row, 1, axis=0)
            value = csa_project_pallas(query, fused, projection_mode="v4_gemv")
            return jnp.zeros((8, fused.shape[1]), jnp.float32).at[row].set(value[0])

        return jax.lax.cond(
            is_single, decode, lambda block: csa_project_pallas(block, fused), block
        )

    projected = jax.lax.map(project_block, (blocks, single, live_row))
    projected = projected.reshape(-1, fused.shape[1])[metadata.router_output_rows]
    return projected[:, : wkv.shape[0]], projected[:, wkv.shape[0] :]


def emit(values, scores, norm, starts, valid, config, *, raw_overlap=False):
    """Select the independently validated raw-window ABI for production V4.

    Retain the selected-window entry for existing diagnostics and callers.
    Both entries have exactly the same V4 rounding and output contract.
    """
    validate_config(config)
    phase = rope_angles(jnp.maximum(starts, 0), config)
    emitter = csa_emit_overlap_pallas if raw_overlap else csa_emit_selected_pallas
    return emitter(
        values,
        scores,
        norm,
        jnp.concatenate((jnp.cos(phase), jnp.sin(phase)), axis=-1),
        valid,
        norm_eps=config.eps,
    )


def select(query, mixing, index_cache, metadata, config):
    """Pack dynamic ragged requests into the original StreamIndex ABI.

    Decode batches can contain inert slots between live queries. Compact only
    the input views, not request/cache ownership, and restore caller row order.
    Shapes stay static and no request length becomes a compilation argument.
    """
    validate_config(config)
    tokens = query.shape[0]
    lengths = metadata.query_lens
    order = jnp.argsort(lengths == 0, stable=True)
    inverse = jnp.zeros_like(order).at[order].set(jnp.arange(len(order)))
    cumulative = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(lengths[order])))
    request = metadata.token_requests
    packed_row = cumulative[inverse[request]] + jnp.arange(tokens) - metadata.query_starts[request]
    destination = jnp.where(metadata.token_valid, packed_row, tokens)
    packed_query = jnp.zeros_like(query).at[destination].set(query, mode="drop")
    packed_mixing = jnp.zeros_like(mixing).at[destination].set(mixing, mode="drop")
    pages = metadata.page_table[order]
    pages = jnp.pad(pages, ((0, 0), (0, (-pages.shape[1]) % 4)))
    scores, selected = paged_lightning_topk(
        packed_query,
        packed_mixing,
        index_cache.reshape(-1, 16, 2, 128),
        metadata.seq_lens[order],
        pages.reshape(-1),
        cumulative,
        jnp.array([0, 0, jnp.sum(lengths > 0)], jnp.int32),
        k=min(config.index_topk, config.max_context // 4),
        numerical_mode="v4",
        candidate_count=config.max_context // 4,
        return_scores=True,
    )
    rows = jnp.clip(packed_row, 0, tokens - 1)
    return jnp.where(metadata.token_valid[:, None], scores[rows], -jnp.inf), selected[rows]


def attend(query, window, window_valid, selected, selected_lengths, sink, config):
    validate_config(config)
    return joint_attention_pallas(
        query,
        window,
        window_valid,
        selected,
        None,
        selected_lengths,
        sink,
        scale=config.head_dim**-0.5,
        selected_tile=64,
        tokens_per_program=1,
        numerical_mode="v4",
    )
