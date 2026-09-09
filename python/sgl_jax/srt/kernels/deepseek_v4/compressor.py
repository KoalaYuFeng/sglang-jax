"""Ragged V4 compression with page-owned snapshots and request-owned scratch."""

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.deepseek_v4.numerics import hadamard_rotate, rms_norm, rope
from sgl_jax.srt.kernels.low_bit.formats import (
    activation_fp4_roundtrip,
    activation_fp8_roundtrip,
    round_bf16,
)


def physical_locations(metadata, requests, positions):
    requests = requests.reshape(requests.shape + (1,) * (positions.ndim - requests.ndim))
    logical_page = jnp.clip(positions // 128, 0, metadata.page_table.shape[1] - 1)
    pages = metadata.page_table[requests, logical_page]
    return pages * 128 + positions % 128


def _initial_state(cache, prefix, metadata):
    """Page-aligned radix hits and fresh requests restore inside the donated call."""
    lengths = metadata.prefix_lens
    prior = (
        physical_locations(metadata, jnp.arange(lengths.size), jnp.maximum(lengths - 1, 0)) // 128
    )
    restore = (lengths > 0) & (lengths % 128 == 0)
    result = []
    for suffix, empty in (("kv", 0), ("score", -jnp.inf)):
        live = cache[f"{prefix}.{suffix}"][metadata.req_slots]
        saved = cache[f"{prefix}.snapshot_{suffix}"][prior]
        state = jnp.where(restore[:, None, None], saved, live)
        result.append(jnp.where((lengths == 0)[:, None, None], empty, state))
    return tuple(result)


def _project(x, weight, metadata):
    """Keep FP32 projection arithmetic independent of packed batch shape.

    On v5p, a BF16-input/F32-output dot changes by a few ulps when M or a
    token's lane changes. Those ulps persist in compressor scratch and can
    cross the later BF16 pooling boundary (the real position-8023 failure).
    Reuse the router's request/absolute-position-aligned eight-row blocks.
    Preserve the original single-request arithmetic: one-token queries use
    GEMV, multi-token queries use GEMM. A fixed eight-iteration inner map is
    essential: a map over just the packed tokens still changes its lowering
    between B1 and B2. Do not round projections/persistent scratch to BF16.
    """
    rows = metadata.router_rows
    blocks = jnp.where((rows >= 0)[..., None], x[jnp.maximum(rows, 0)], 0)
    requests = metadata.token_requests[jnp.maximum(rows, 0)]
    single = jnp.any((rows >= 0) & (metadata.query_lens[requests] == 1), axis=1)

    def project_block(item):
        # Each router block belongs to one request, including partial/padded
        # blocks, so the same rule works for mixed prefill and decode batches.
        return jax.lax.cond(
            item[1],
            lambda block: jax.lax.map(
                lambda row: jnp.matmul(row[None], weight.T, preferred_element_type=jnp.float32)[0],
                block,
            ),
            lambda block: jnp.matmul(block, weight.T, preferred_element_type=jnp.float32),
            item[0],
        )

    projected = jax.lax.map(project_block, (blocks, single))
    return projected.reshape(-1, weight.shape[0])[metadata.router_output_rows]


def compress(
    x,
    weights,
    cache,
    config,
    metadata,
    *,
    index=False,
    trace=None,
    hca_backend="reference",
    csa_backend="reference",
    csa_decode_batch=False,
):
    ratio = config.ratio
    overlap = ratio == 4
    dim = config.index_dim if index else config.head_dim
    prefix = "index" if index else "main"
    weight_prefix = "attn.indexer.compressor" if index else "attn.compressor"
    initial_values, initial_scores = _initial_state(cache, prefix, metadata)
    from sgl_jax.srt.kernels.deepseek_v4 import csa, hca

    hca.validate_backend(hca_backend)
    csa.validate_backend(csa_backend)
    original_hca = ratio == 128 and hca_backend == "pallas"
    original_csa = ratio == 4 and csa_backend == "pallas"
    adapter = hca if original_hca else csa
    if original_hca or original_csa:
        projection_options = {"decode_batch": csa_decode_batch} if original_csa else {}
        kv, scores = adapter.project(
            x,
            weights[weight_prefix + ".wkv.weight"],
            weights[weight_prefix + ".wgate.weight"],
            metadata,
            config,
            **projection_options,
        )
    else:
        kv = _project(x, weights[weight_prefix + ".wkv.weight"], metadata)
        scores = _project(x, weights[weight_prefix + ".wgate.weight"], metadata)

    def projected_rows(requests, positions):
        start = metadata.prefix_lens[requests, None]
        offsets = metadata.query_starts[requests, None] + positions - start
        rows = jnp.clip(offsets, 0, x.shape[0] - 1)
        from_batch = (positions >= start) & (positions < metadata.seq_lens[requests, None])
        state_rows = positions % ratio
        if overlap:
            state_rows += jnp.where(positions >= (start // ratio) * ratio, ratio, 0)
        old_values = initial_values[requests[:, None], state_rows]
        old_scores = initial_scores[requests[:, None], state_rows]
        values = jnp.where(from_batch[..., None], kv[rows], old_values)
        biased = scores[rows] + weights[weight_prefix + ".ape"][positions % ratio]
        values = jnp.where((positions >= 0)[..., None], values, 0)
        score = jnp.where(from_batch[..., None], biased, old_scores)
        return values, jnp.where((positions >= 0)[..., None], score, -jnp.inf)

    group_requests = metadata.group4_requests if overlap else metadata.group128_requests
    group_starts = metadata.group4_starts if overlap else metadata.group128_starts
    span = ratio * (2 if overlap else 1)
    positions = group_starts[:, None] + jnp.arange(span)[None] - (ratio if overlap else 0)
    group_values, group_scores = projected_rows(group_requests, positions)
    raw_values, raw_scores = group_values, group_scores
    if trace is not None:
        trace.update(
            raw_group_values=group_values, raw_group_scores=group_scores, group_positions=positions
        )
    if overlap and (not original_csa or trace is not None):
        # v5p can mis-lower a concatenate of two doubly-sliced rank-3
        # tensors (silently selecting the first channel half for both).
        # The retained reference uses this explicit 2-D gather. Production
        # Pallas selects inside VMEM; a traced call only reconstructs a
        # diagnostic view here, not an observed intermediate of the kernel.
        columns = jnp.arange(dim)[None] + (jnp.arange(span) >= ratio)[:, None] * dim
        columns = jnp.broadcast_to(columns, (group_starts.size, span, dim)).reshape(-1, dim)
        group_values = jnp.take_along_axis(
            group_values.reshape(-1, 2 * dim), columns, axis=1
        ).reshape(-1, span, dim)
        group_scores = jnp.take_along_axis(
            group_scores.reshape(-1, 2 * dim), columns, axis=1
        ).reshape(-1, span, dim)
    valid_groups = group_starts >= 0
    if not original_csa or trace is not None:
        group_scores = jnp.where(valid_groups[:, None, None], group_scores, 0)
    if trace is not None:
        trace.update(kv=kv, scores=scores)
        if original_csa:
            # Raw inputs and csa_emitted are actual boundary observations.
            # Give the reconstructed views distinct names to avoid claiming
            # that a trace exported internal Pallas values.
            trace.update(selected_view_values=group_values, selected_view_scores=group_scores)
        else:
            trace.update(group_values=group_values, group_scores=group_scores)

    # Match the official vectorized prefill reduction, including rounding.
    # Group construction removes the dynamic sliced state reduction that
    # previously required a different single-token decoder implementation.
    if original_hca or original_csa:
        pooled = adapter.emit(
            raw_values if original_csa else group_values,
            raw_scores if original_csa else group_scores,
            weights[weight_prefix + ".norm.weight"],
            group_starts,
            valid_groups,
            config,
            **({"raw_overlap": True} if original_csa else {}),
        )
        # The fused emitter does not export internal intermediates. Do not
        # label recomputed reference intermediates as actual Pallas values.
        if trace is not None:
            trace["hca_emitted" if original_hca else "csa_emitted"] = pooled
    else:
        pooled = round_bf16(jnp.sum(group_values * jax.nn.softmax(group_scores, axis=1), axis=1))
        if trace is not None:
            trace["pooled"] = pooled
        pooled = rms_norm(pooled, weights[weight_prefix + ".norm.weight"], config.eps)
        if trace is not None:
            trace["normalized"] = pooled
        pooled = rope(pooled, jnp.maximum(group_starts, 0), config)
    if index:
        pooled = activation_fp4_roundtrip(hadamard_rotate(pooled))
    else:
        pooled = jnp.concatenate(
            (
                activation_fp8_roundtrip(pooled[:, : -config.rope_dim], 64),
                pooled[:, -config.rope_dim :],
            ),
            axis=-1,
        )
    terminal = physical_locations(metadata, group_requests, group_starts + ratio - 1)
    target = cache[prefix + ".compressed"]
    destination = jnp.where(valid_groups, terminal // ratio, target.shape[0])
    cache[prefix + ".compressed"] = target.at[destination].set(pooled, mode="drop")
    if trace is not None:
        trace.update(final=pooled, destination=destination)

    # Save the terminal state at every complete physical page. These snapshots
    # follow the ordinary radix page ownership, including fork/eviction/reuse.
    page_requests = metadata.group128_requests
    page_end = metadata.group128_starts + 128
    snapshot_positions = page_end[:, None] - ratio + jnp.arange(ratio)[None]
    snapshot_values, snapshot_scores = projected_rows(page_requests, snapshot_positions)
    if overlap:
        snapshot_values = jnp.concatenate((snapshot_values, snapshot_values), axis=1)
        snapshot_scores = jnp.concatenate((snapshot_scores, snapshot_scores), axis=1)
    page_ids = physical_locations(metadata, page_requests, page_end - 1) // 128
    for suffix, value in (("kv", snapshot_values), ("score", snapshot_scores)):
        key = f"{prefix}.snapshot_{suffix}"
        destinations = jnp.where(metadata.group128_starts >= 0, page_ids, cache[key].shape[0])
        cache[key] = cache[key].at[destinations].set(value, mode="drop")

    # Materialize each request's partial state independently. Zero-length padded
    # requests use dropped writes and never alter another request's state.
    lengths = metadata.seq_lens
    rows = jnp.arange(ratio)[None]
    last_complete = (lengths[:, None] // ratio - 1) * ratio + rows
    current = (lengths[:, None] // ratio) * ratio + rows
    tail = jnp.where(rows < (lengths % ratio)[:, None], current, last_complete)
    state_positions = jnp.concatenate((last_complete, tail), axis=1) if overlap else tail
    live_values, live_scores = projected_rows(jnp.arange(lengths.size), state_positions)
    for suffix, value in (("kv", live_values), ("score", live_scores)):
        key = f"{prefix}.{suffix}"
        destinations = jnp.where(metadata.query_lens > 0, metadata.req_slots, cache[key].shape[0])
        cache[key] = cache[key].at[destinations].set(value, mode="drop")
    return cache
