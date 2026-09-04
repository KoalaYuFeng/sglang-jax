"""End-to-end execution for the CSA kernel family."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.csa.compressor import (
    csa_dual_state_step_cache,
    csa_dual_uniform_prefill_pallas,
)
from sgl_jax.srt.kernels.csa.indexer import (
    paged_lightning_topk,
    paged_sparsecore_gather_packed,
)
from sgl_jax.srt.kernels.csa.joint_attention import (
    blocked_ragged_joint_attention_pallas,
    joint_attention_pallas,
)
from sgl_jax.srt.kernels.csa.tune import (
    CSA_ATTENTION_DIM,
    CSA_ATTENTION_HEADS,
    CSA_CACHE_PACKING,
    CSA_COMPRESSION_RATIO,
    CSA_DEFAULT_PAGE_SIZE,
    CSA_ROPE_RECORD_BYTES,
    CSA_TOP_K,
    CSA_WINDOW_SIZE,
    TPU_V6E,
    csa_topk_is_identity,
    get_csa_attention_schedule,
    get_csa_indexer_schedule,
)


def _install_compressed_records(
    emitted_positions,
    request_pages,
    nope,
    rope,
    index_records,
    main_nope,
    main_rope,
    index_cache,
):
    logical = jnp.floor_divide(emitted_positions, CSA_COMPRESSION_RATIO)
    logical_page = jnp.floor_divide(logical, CSA_DEFAULT_PAGE_SIZE)
    physical_page = jnp.take_along_axis(request_pages, logical_page, axis=1)
    physical = physical_page * CSA_DEFAULT_PAGE_SIZE + jnp.mod(logical, CSA_DEFAULT_PAGE_SIZE)
    physical = physical.reshape(-1)
    main_nope = (
        main_nope.reshape(-1, CSA_CACHE_PACKING, TPU_V6E.vector_lanes)
        .at[physical]
        .set(nope.reshape(-1, CSA_CACHE_PACKING, TPU_V6E.vector_lanes))
        .reshape(main_nope.shape)
    )
    main_rope = (
        main_rope.reshape(-1, CSA_ROPE_RECORD_BYTES)
        .at[physical]
        .set(rope.reshape(-1, CSA_ROPE_RECORD_BYTES))
        .reshape(main_rope.shape)
    )
    index_cache = (
        index_cache.reshape(-1, index_records.shape[-2] * index_records.shape[-1])
        .at[physical]
        .set(index_records.reshape(physical.shape[0], -1))
        .reshape(index_cache.shape)
    )
    return main_nope, main_rope, index_cache


def _compressed_locations(positions, request_pages):
    logical = jnp.floor_divide(positions, CSA_COMPRESSION_RATIO)
    logical_page = jnp.floor_divide(logical, CSA_DEFAULT_PAGE_SIZE)
    physical_page = jnp.take_along_axis(
        request_pages,
        logical_page[:, None],
        axis=1,
    )[:, 0]
    return physical_page * CSA_DEFAULT_PAGE_SIZE + jnp.mod(
        logical,
        CSA_DEFAULT_PAGE_SIZE,
    )


def _compress_token_range(
    group_x,
    group_positions,
    start,
    stop,
    request_pages,
    weight,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    main_state,
    index_state,
    main_nope,
    main_rope,
    index_cache,
):
    for offset in range(start, stop):
        positions = group_positions[:, offset]
        result = csa_dual_state_step_cache(
            group_x[:, offset],
            main_state,
            index_state,
            weight,
            main_ape,
            index_ape,
            main_norm,
            index_norm,
            cos,
            sin,
            positions,
            _compressed_locations(positions, request_pages),
            main_nope,
            main_rope,
            index_cache,
        )
        main_state, index_state = result[1], result[2]
        main_nope, main_rope, index_cache = result[3:]
    return main_state, index_state, main_nope, main_rope, index_cache


def _compress_chunk_partitioned(
    x,
    weight,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    positions,
    compressed_pages,
    main_state,
    index_state,
    main_nope,
    main_rope,
    index_cache,
    *,
    query_lengths: tuple[int, ...],
    query_start_slots: tuple[int, ...],
):
    token_starts = np.asarray((0, *np.cumsum(query_lengths)[:-1]), np.int32)
    partitions = tuple(zip(query_lengths, query_start_slots, strict=True))
    for query_length, start_slot in sorted(set(partitions)):
        request_ids_np = np.asarray(
            [
                request
                for request, partition in enumerate(partitions)
                if partition == (query_length, start_slot)
            ],
            np.int32,
        )
        request_ids = jnp.asarray(request_ids_np)
        starts = token_starts[request_ids_np]
        token_indices_np = starts[:, None] + np.arange(query_length, dtype=np.int32)[None]
        token_indices = jnp.asarray(token_indices_np)
        group_x = x[token_indices]
        group_positions = positions[token_indices]
        request_main_state = main_state[request_ids]
        request_index_state = index_state[request_ids]
        request_pages = compressed_pages[request_ids]
        if query_length == 1:
            (
                request_main_state,
                request_index_state,
                main_nope,
                main_rope,
                index_cache,
            ) = _compress_token_range(
                group_x,
                group_positions,
                0,
                1,
                request_pages,
                weight,
                main_ape,
                index_ape,
                main_norm,
                index_norm,
                cos,
                sin,
                request_main_state,
                request_index_state,
                main_nope,
                main_rope,
                index_cache,
            )
        else:
            (
                request_main_state,
                request_index_state,
                main_nope,
                main_rope,
                index_cache,
            ) = _compress_token_range(
                group_x,
                group_positions,
                0,
                (-start_slot) % CSA_COMPRESSION_RATIO,
                request_pages,
                weight,
                main_ape,
                index_ape,
                main_norm,
                index_norm,
                cos,
                sin,
                request_main_state,
                request_index_state,
                main_nope,
                main_rope,
                index_cache,
            )
            leading = min(
                query_length,
                (-start_slot) % CSA_COMPRESSION_RATIO,
            )
            bulk_stop = (
                leading
                + ((query_length - leading) // CSA_COMPRESSION_RATIO) * CSA_COMPRESSION_RATIO
            )
            if bulk_stop > leading:
                group_starts = group_positions[:, leading:bulk_stop:CSA_COMPRESSION_RATIO]
                (
                    nope,
                    rope,
                    index_records,
                    request_main_state,
                    request_index_state,
                ) = csa_dual_uniform_prefill_pallas(
                    group_x[:, leading:bulk_stop],
                    request_main_state,
                    request_index_state,
                    weight,
                    main_ape,
                    index_ape,
                    main_norm,
                    index_norm,
                    cos[group_starts],
                    sin[group_starts],
                )
                main_nope, main_rope, index_cache = _install_compressed_records(
                    group_positions[
                        :,
                        leading + CSA_COMPRESSION_RATIO - 1 : bulk_stop : CSA_COMPRESSION_RATIO,
                    ],
                    request_pages,
                    nope,
                    rope,
                    index_records,
                    main_nope,
                    main_rope,
                    index_cache,
                )
            (
                request_main_state,
                request_index_state,
                main_nope,
                main_rope,
                index_cache,
            ) = _compress_token_range(
                group_x,
                group_positions,
                bulk_stop,
                query_length,
                request_pages,
                weight,
                main_ape,
                index_ape,
                main_norm,
                index_norm,
                cos,
                sin,
                request_main_state,
                request_index_state,
                main_nope,
                main_rope,
                index_cache,
            )
        main_state = main_state.at[request_ids].set(request_main_state)
        index_state = index_state.at[request_ids].set(request_index_state)
    return (
        main_state,
        index_state,
        main_nope,
        main_rope,
        index_cache,
    )


def _window_rows(cache, page_table, positions, cu_q, seq_ids, new_kv):
    prefix = positions - (jnp.arange(positions.shape[0]) - cu_q[seq_ids])
    key_positions = positions[:, None] - (CSA_WINDOW_SIZE - 1) + jnp.arange(CSA_WINDOW_SIZE)[None]
    valid = key_positions >= 0
    from_chunk = key_positions >= prefix[:, None]
    local_index = cu_q[seq_ids, None] + key_positions - prefix[:, None]
    safe_new = jnp.clip(local_index, 0, new_kv.shape[0] - 1)
    page = jnp.floor_divide(jnp.maximum(key_positions, 0), CSA_DEFAULT_PAGE_SIZE)
    physical_page = jnp.take_along_axis(page_table[seq_ids], page, axis=1)
    physical = physical_page * CSA_DEFAULT_PAGE_SIZE + jnp.mod(
        jnp.maximum(key_positions, 0), CSA_DEFAULT_PAGE_SIZE
    )
    history = cache.reshape(-1, CSA_ATTENTION_DIM)[physical]
    rows = jnp.where(from_chunk[..., None], new_kv[safe_new], history)
    return jnp.where(valid[..., None], rows, 0), valid


def _request_window_context(
    cache,
    page_table,
    prefix_lens,
    cu_q,
    seq_ids,
    new_kv,
    *,
    maximum_query: int,
):
    history_positions = (
        prefix_lens[:, None]
        - CSA_WINDOW_SIZE
        + jnp.arange(CSA_WINDOW_SIZE, dtype=jnp.int32)[None, :]
    )
    history_valid = history_positions >= 0
    safe_positions = jnp.maximum(history_positions, 0)
    logical_pages = jnp.floor_divide(safe_positions, CSA_DEFAULT_PAGE_SIZE)
    physical_pages = jnp.take_along_axis(page_table, logical_pages, axis=1)
    physical = physical_pages * CSA_DEFAULT_PAGE_SIZE + jnp.mod(
        safe_positions, CSA_DEFAULT_PAGE_SIZE
    )
    history = cache.reshape(-1, CSA_ATTENTION_DIM)[physical]
    history = jnp.where(history_valid[..., None], history, 0)

    local_queries = jnp.arange(new_kv.shape[0], dtype=jnp.int32) - cu_q[seq_ids]
    current = (
        jnp.zeros((prefix_lens.shape[0], maximum_query, CSA_ATTENTION_DIM), new_kv.dtype)
        .at[seq_ids, local_queries]
        .set(new_kv)
    )
    return jnp.concatenate((history, current), axis=1)


def _update_window_cache(cache, page_table, positions, seq_ids, new_kv):
    logical_page = jnp.floor_divide(positions, CSA_DEFAULT_PAGE_SIZE)
    physical_page = jnp.take_along_axis(page_table[seq_ids], logical_page[:, None], axis=1)[:, 0]
    physical = physical_page * CSA_DEFAULT_PAGE_SIZE + jnp.mod(positions, CSA_DEFAULT_PAGE_SIZE)
    return cache.reshape(-1, CSA_ATTENTION_DIM).at[physical].set(new_kv).reshape(cache.shape)


def build_csa_step(
    query_lengths: tuple[int, ...],
    *,
    query_start_slots: tuple[int, ...],
    uniform_prefill: bool,
    softmax_scale: float = CSA_ATTENTION_DIM**-0.5,
):
    """Specialize and JIT one complete CSA step for static request lengths."""
    if not query_lengths or len(query_start_slots) != len(query_lengths):
        raise ValueError("query lengths and start slots must be nonempty and aligned")
    if any(length <= 0 for length in query_lengths):
        raise ValueError("query lengths must be positive")
    if any(slot < 0 or slot >= CSA_COMPRESSION_RATIO for slot in query_start_slots):
        raise ValueError("query start slots must be in [0, compression_ratio)")
    if uniform_prefill and any(query_start_slots):
        raise ValueError("uniform prefill must start at a compression-group boundary")
    maximum_query = max(query_lengths)
    workload_case = "decode" if maximum_query == 1 else ("prefill" if uniform_prefill else "mixed")
    _, attention_block = get_csa_attention_schedule(sum(query_lengths))
    indexer_schedule = get_csa_indexer_schedule(
        prefill_query_length=maximum_query,
        mixed_max_query_length=maximum_query,
    )
    token_starts = np.asarray((0, *np.cumsum(query_lengths)[:-1]), np.int32)
    query_block_requests = []
    query_block_offsets = []
    query_block_tokens = []
    query_block_valid = []
    for request, query_length in enumerate(query_lengths):
        for offset in range(0, query_length, attention_block):
            query_block_requests.append(request)
            query_block_offsets.append(offset)
            local = offset + np.arange(attention_block, dtype=np.int32)
            valid = local < query_length
            query_block_tokens.append(np.where(valid, token_starts[request] + local, 0))
            query_block_valid.append(valid)
    query_block_requests = jnp.asarray(query_block_requests, jnp.int32)
    query_block_offsets = jnp.asarray(query_block_offsets, jnp.int32)
    query_block_tokens = jnp.asarray(np.asarray(query_block_tokens), jnp.int32)
    query_block_valid = jnp.asarray(np.asarray(query_block_valid))
    valid_block_rows = jnp.asarray(
        np.flatnonzero(np.asarray(query_block_valid).reshape(-1)),
        jnp.int32,
    )

    def run(
        x,
        weight,
        main_ape,
        index_ape,
        main_norm,
        index_norm,
        cos,
        sin,
        positions,
        cu_q,
        seq_ids,
        compressed_pages,
        window_pages,
        seq_lens,
        distribution,
        index_q_fp8,
        index_weights,
        attention_q,
        new_kv,
        sink,
        main_state,
        index_state,
        main_nope,
        main_rope,
        index_cache,
        window_cache,
    ):
        (
            main_state,
            index_state,
            main_nope,
            main_rope,
            index_cache,
        ) = _compress_chunk_partitioned(
            x,
            weight,
            main_ape,
            index_ape,
            main_norm,
            index_norm,
            cos,
            sin,
            positions,
            compressed_pages,
            main_state,
            index_state,
            main_nope,
            main_rope,
            index_cache,
            query_lengths=query_lengths,
            query_start_slots=query_start_slots,
        )
        candidate_lengths = jnp.floor_divide(
            positions + 1,
            CSA_COMPRESSION_RATIO,
        )
        candidate_capacity = compressed_pages.shape[1] * CSA_DEFAULT_PAGE_SIZE
        if csa_topk_is_identity(candidate_capacity, selected=CSA_TOP_K):
            logical_rows = jnp.arange(CSA_TOP_K, dtype=jnp.int32)[None]
            topk = jnp.where(logical_rows < candidate_lengths[:, None], logical_rows, -1)
        else:
            topk = paged_lightning_topk(
                index_q_fp8,
                index_weights,
                index_cache,
                seq_lens,
                compressed_pages.reshape(-1),
                cu_q,
                distribution,
                k=CSA_TOP_K,
                completed_groups_only=True,
                prefill_query_length=maximum_query,
                mixed_max_query_length=maximum_query,
                num_kv_pages_per_block=indexer_schedule.num_kv_pages_per_block,
                num_queries_per_block=indexer_schedule.num_queries_per_block,
                decode_request_batch=indexer_schedule.decode_request_batch,
                workload_case=workload_case,
            )
        cu_compressed = (
            jnp.arange(compressed_pages.shape[0] + 1, dtype=jnp.int32) * candidate_capacity
        )
        use_shared_window = maximum_query > 1
        if use_shared_window:
            gather_topk = topk[query_block_tokens]
            gather_topk = jnp.where(query_block_valid[..., None], gather_topk, -1)
            gather_topk = gather_topk.reshape(-1, CSA_TOP_K)
            gather_seq_ids = jnp.broadcast_to(
                query_block_requests[:, None], query_block_valid.shape
            ).reshape(-1)
        else:
            gather_topk = topk
            gather_seq_ids = seq_ids
        selected_nope, selected_rope = paged_sparsecore_gather_packed(
            main_nope,
            main_rope,
            gather_topk,
            compressed_pages.reshape(-1),
            cu_compressed,
            gather_seq_ids,
            page_size=CSA_DEFAULT_PAGE_SIZE,
            page_table_stride=compressed_pages.shape[1],
        )
        selected_tile, token_tile = get_csa_attention_schedule(
            attention_q.shape[0],
            shared_window=use_shared_window,
        )
        selected_tile = min(selected_tile, CSA_TOP_K)
        if use_shared_window:
            prefix_lens = seq_lens - jnp.diff(cu_q)
            combined_window = _request_window_context(
                window_cache,
                window_pages,
                prefix_lens,
                cu_q,
                seq_ids,
                new_kv,
                maximum_query=maximum_query,
            )
            q_blocks = attention_q[query_block_tokens]
            q_blocks = jnp.where(query_block_valid[..., None, None], q_blocks, 0)
            selected_nope = selected_nope.reshape(
                query_block_valid.shape[0],
                attention_block,
                CSA_TOP_K,
                TPU_V6E.vector_lanes,
            )
            selected_rope = selected_rope.reshape(
                query_block_valid.shape[0],
                attention_block,
                CSA_TOP_K // CSA_CACHE_PACKING,
                TPU_V6E.vector_lanes,
            )
            block_output = blocked_ragged_joint_attention_pallas(
                q_blocks,
                combined_window,
                jnp.diff(cu_q),
                prefix_lens,
                query_block_requests,
                query_block_offsets,
                selected_nope,
                selected_rope,
                jnp.sum(gather_topk >= 0, axis=1, dtype=jnp.int32).reshape(query_block_valid.shape),
                sink,
                scale=softmax_scale,
                selected_tile=selected_tile,
            )
            output = block_output.reshape(-1, CSA_ATTENTION_HEADS, CSA_ATTENTION_DIM)[
                valid_block_rows
            ]
        else:
            window, window_valid = _window_rows(
                window_cache, window_pages, positions, cu_q, seq_ids, new_kv
            )
            output = joint_attention_pallas(
                attention_q,
                window,
                window_valid,
                selected_nope,
                selected_rope,
                jnp.sum(topk >= 0, axis=1, dtype=jnp.int32),
                sink,
                scale=softmax_scale,
                selected_tile=selected_tile,
                tokens_per_program=token_tile,
            )
        window_cache = _update_window_cache(window_cache, window_pages, positions, seq_ids, new_kv)
        return (
            output,
            topk,
            main_state,
            index_state,
            main_nope,
            main_rope,
            index_cache,
            window_cache,
        )

    return jax.jit(run, donate_argnums=(20, 21, 22, 23, 24, 25))


__all__ = ["build_csa_step"]
