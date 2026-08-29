"""Lightning Indexer, Top-K, and selected-record gathering for CSA."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from sgl_jax.srt.kernels.csa.tune import (
    CSA_ATTENTION_DIM,
    CSA_CACHE_PACKING,
    CSA_COMPRESSION_RATIO,
    CSA_INDEX_DIM,
    CSA_INDEX_HEADS,
    CSA_INDEX_RECORD_BYTES,
    TPU_V6E,
    get_csa_fused_page_mapping_capacity,
    get_csa_gather_schedule,
    get_csa_indexer_schedule,
    get_csa_paged_gather_schedule,
)
from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk


def paged_lightning_topk(
    q,
    weights,
    index_cache,
    seq_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    k: int,
    prefill_query_length: int | None = None,
    mixed_max_query_length: int | None = None,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    decode_request_batch: int | None = None,
    completed_groups_only: bool = True,
    workload_case: str | None = None,
):
    """Adapt CSA metadata to the shared DSA Top-K implementation."""
    if q.ndim != 3 or q.shape[1:] != (CSA_INDEX_HEADS, CSA_INDEX_DIM):
        raise ValueError("CSA index queries must be [T,64,128]")
    if q.dtype != jnp.bfloat16:
        raise ValueError("CSA index queries must use BF16")
    if weights.shape != q.shape[:2]:
        raise ValueError("CSA index weights must be [T,64]")
    if (
        index_cache.dtype != jnp.uint8
        or index_cache.ndim != 4
        or index_cache.shape[-2:] != (CSA_CACHE_PACKING, CSA_INDEX_RECORD_BYTES)
    ):
        raise ValueError("CSA index cache must be uint8[pages,page_size/4,4,256]")
    if prefill_query_length is not None and prefill_query_length <= 0:
        raise ValueError("prefill_query_length must be positive")
    if mixed_max_query_length is not None and mixed_max_query_length <= 0:
        raise ValueError("mixed_max_query_length must be positive")
    schedule = get_csa_indexer_schedule(
        prefill_query_length=(prefill_query_length if prefill_query_length is not None else 1),
        mixed_max_query_length=(
            mixed_max_query_length if mixed_max_query_length is not None else max(q.shape[0], 1)
        ),
    )
    if num_kv_pages_per_block is None:
        num_kv_pages_per_block = schedule.num_kv_pages_per_block
    if num_queries_per_block is None:
        num_queries_per_block = schedule.num_queries_per_block
    if decode_request_batch is None:
        maximum_batch = min(schedule.decode_request_batch, seq_lens.shape[0])
        decode_request_batch = 1 << (maximum_batch.bit_length() - 1)

    def select_case(value):
        if isinstance(value, int):
            return value
        case = {"decode": 0, "prefill": 1, "mixed": 2}.get(workload_case)
        return value[case] if case is not None else value

    if workload_case is not None and workload_case not in ("decode", "prefill", "mixed"):
        raise ValueError("workload_case must be 'decode', 'prefill', 'mixed', or None")
    if workload_case is not None:
        num_kv_pages_per_block = (select_case(num_kv_pages_per_block),) * 3
        num_queries_per_block = (select_case(num_queries_per_block),) * 3

    pages_per_request = page_indices.shape[0] // seq_lens.shape[0]
    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_block = min(num_kv_pages_per_block, pages_per_request)
    else:
        num_kv_pages_per_block = tuple(
            min(block, pages_per_request) for block in num_kv_pages_per_block
        )

    # The shared kernel's middle segment is a one-token path. CSA sends every
    # multi-token request through its dynamic-query path instead.
    distribution = distribution.at[1].set(distribution[0])
    retrieval_k = k + int(completed_groups_only)
    topk = streamindex_topk(
        q,
        weights,
        index_cache,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=retrieval_k,
        compression_ratio=CSA_COMPRESSION_RATIO,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        decode_req_batch_size=decode_request_batch,
    )
    if not completed_groups_only:
        return topk[:, :k]

    token = jnp.arange(q.shape[0], dtype=jnp.int32)
    request = jnp.searchsorted(cu_q_lens[1:], token, side="right")
    query_lengths = jnp.diff(cu_q_lens)
    position = seq_lens[request] - query_lengths[request] + token - cu_q_lens[request]
    completed = (position + 1) // CSA_COMPRESSION_RATIO
    valid = (topk >= 0) & (topk < completed[:, None])
    head = topk[:, :k]
    replacement = jnp.where(valid[:, k:], topk[:, k:], -1)
    return jnp.where(valid[:, :k], head, replacement)


_PAGE_ID_CHUNK = TPU_V6E.vector_lanes


def _page_id_kernel(windows_ref, logical_ref, out_ref, *, num_chunks: int):
    logical = logical_ref[...]
    physical = jnp.zeros_like(logical)
    for chunk in range(num_chunks):
        pages = windows_ref[:, chunk * _PAGE_ID_CHUNK : (chunk + 1) * _PAGE_ID_CHUNK]
        local = logical - chunk * _PAGE_ID_CHUNK
        selected = jnp.take_along_axis(
            pages,
            jnp.clip(local, 0, _PAGE_ID_CHUNK - 1),
            axis=1,
        )
        physical = jnp.where((local >= 0) & (local < _PAGE_ID_CHUNK), selected, physical)
    out_ref[...] = physical


def _resolve_page_ids(
    page_indices,
    logical_pages,
    query_seq_ids,
    *,
    page_table_stride: int,
    block_tokens: int = TPU_V6E.sublanes,
):
    """Stage fixed-stride page-table rows, then resolve logical pages in VMEM."""
    if page_table_stride <= 0 or page_indices.shape[0] % page_table_stride:
        raise ValueError("page_table_stride must divide page_indices")
    tokens, topk = logical_pages.shape
    padded_capacity = (page_table_stride + _PAGE_ID_CHUNK - 1) // _PAGE_ID_CHUNK * _PAGE_ID_CHUNK
    page_table = page_indices.reshape(-1, page_table_stride)
    if padded_capacity > page_table_stride:
        page_table = jnp.pad(
            page_table,
            ((0, 0), (0, padded_capacity - page_table_stride)),
        )
    windows = page_table[query_seq_ids]
    padded_tokens = (tokens + block_tokens - 1) // block_tokens * block_tokens
    if padded_tokens > tokens:
        windows = jnp.pad(windows, ((0, padded_tokens - tokens), (0, 0)))
        logical_pages = jnp.pad(
            logical_pages,
            ((0, padded_tokens - tokens), (0, 0)),
        )
    physical = pl.pallas_call(
        functools.partial(
            _page_id_kernel,
            num_chunks=padded_capacity // _PAGE_ID_CHUNK,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(
                pl.BlockSpec((block_tokens, padded_capacity), lambda block: (block, 0)),
                pl.BlockSpec((block_tokens, topk), lambda block: (block, 0)),
            ),
            out_specs=pl.BlockSpec((block_tokens, topk), lambda block: (block, 0)),
            grid=(padded_tokens // block_tokens,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        name=f"csa-page-ids-p{padded_capacity}",
    )(windows, logical_pages)
    return physical[:tokens]


def _gather_kernel(
    nope_in_hbm_ref: Any,
    rope_in_hbm_ref: Any,
    indices_hbm_ref: Any,
    page_indices_hbm_ref: Any,
    page_starts_hbm_ref: Any,
    nope_out_hbm_ref: Any,
    rope_out_hbm_ref: Any,
    physical_indices_vmem_ref: Any,
    *,
    core_axis_name: str,
    subcore_axis_name: str,
    num_row_subchunks: int,
    num_streams: int,
    page_size: int,
    paged: bool,
    record_count: int,
    topk: int,
):
    sparse_core = pltpu.get_tpu_info().sparse_core
    assert sparse_core is not None
    lanes = sparse_core.num_lanes
    workers = jax.lax.axis_size((core_axis_name, subcore_axis_name))
    rows_per_worker = lanes * num_row_subchunks
    block_size = rows_per_worker * workers
    num_blocks = pl.cdiv(indices_hbm_ref.shape[0], block_size)

    in_bits = jax.dtypes.itemsize_bits(nope_in_hbm_ref.dtype)
    in_packing = jax.dtypes.itemsize_bits(jnp.int32) // in_bits
    in_mask = (1 << in_bits) - 1
    worker_index = lax.axis_index((core_axis_name, subcore_axis_name))

    nope_in_i32 = nope_in_hbm_ref.bitcast(jnp.int32)
    rope_in_i32 = rope_in_hbm_ref.bitcast(jnp.int32)
    nope_out_i32 = nope_out_hbm_ref
    rope_out_i32 = rope_out_hbm_ref
    nope_in_cols = nope_in_i32.shape[1]
    rope_in_cols = rope_in_i32.shape[1]

    def store_packed_nope(gather_ref, out_ref, out_row_base):
        out_ref[pl.ds(out_row_base, lanes), pl.ds(0, TPU_V6E.vector_lanes)] = gather_ref[
            pl.ds(0, lanes), pl.ds(0, TPU_V6E.vector_lanes)
        ]

    def store_packed_rope(gather_ref, out_ref, sub_indices, out_row_base):
        for output_row in range(lanes // in_packing):
            packed = jnp.zeros((1, TPU_V6E.vector_lanes), jnp.int32)
            for offset in range(in_packing):
                row = output_row * in_packing + offset
                sub = lax.rem(sub_indices[row], in_packing)
                value = jnp.bitwise_and(
                    jnp.bitwise_right_shift(
                        gather_ref[pl.ds(row, 1), pl.ds(0, TPU_V6E.vector_lanes)],
                        in_bits * sub,
                    ),
                    in_mask,
                )
                packed = jnp.bitwise_or(packed, jnp.left_shift(value, in_bits * offset))
            out_ref[
                pl.ds(out_row_base + output_row, 1),
                pl.ds(0, TPU_V6E.vector_lanes),
            ] = packed

    def gather_cache(indices_ref, output_base):
        """Gather one worker's physical rows into the final flat output."""

        def subchunk(step, stream):
            return step * num_streams + stream

        def index_window(step, stream):
            return indices_ref[pl.ds(subchunk(step, stream) * lanes, lanes)]

        nope_rows = lanes
        rope_rows = lanes // in_packing

        def body(*refs):
            step = pl.program_id(0)
            nope_gathers = refs[:num_streams]
            rope_gathers = refs[num_streams : 2 * num_streams]
            nope_output = refs[2 * num_streams]
            rope_output = refs[2 * num_streams + 1]
            for stream in range(num_streams):
                store_packed_nope(
                    nope_gathers[stream],
                    nope_output,
                    stream * nope_rows,
                )
                store_packed_rope(
                    rope_gathers[stream],
                    rope_output,
                    index_window(step, stream),
                    stream * rope_rows,
                )

        nope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(lanes), nope_in_cols),
                lambda step, stream=stream: (index_window(step, stream), 0),
            )
            for stream in range(num_streams)
        )
        rope_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(lanes), rope_in_cols),
                lambda step, stream=stream: (
                    lax.div(index_window(step, stream), in_packing),
                    0,
                ),
            )
            for stream in range(num_streams)
        )
        nope_output_spec = pl.BlockSpec(
            (num_streams * nope_rows, TPU_V6E.vector_lanes),
            lambda step: (output_base // num_streams + step, 0),
        )
        rope_output_spec = pl.BlockSpec(
            (num_streams * rope_rows, TPU_V6E.vector_lanes),
            lambda step: (output_base // num_streams + step, 0),
        )
        pltpu.emit_pipeline(
            body,
            grid=(num_row_subchunks // num_streams,),
            in_specs=nope_specs + rope_specs,
            out_specs=(nope_output_spec, rope_output_spec),
        )(
            *([nope_in_i32] * num_streams),
            *([rope_in_i32] * num_streams),
            nope_out_i32,
            rope_out_i32,
        )

    def direct_outer_pipeline(indices_ref):
        block = pl.program_id(0)
        output_base = (block * workers + worker_index) * num_row_subchunks
        gather_cache(indices_ref, output_base)

    def paged_outer_pipeline(logical_indices_ref, page_starts_ref):
        block = pl.program_id(0)
        global_row_base = (block * workers + worker_index) * rows_per_worker
        output_base = (block * workers + worker_index) * num_row_subchunks
        query_index = global_row_base // topk
        page_starts = page_starts_ref[pl.ds(0, lanes)]
        page_start_index = lax.rem(query_index, lanes)
        page_start = page_starts[0]
        for lane in range(1, lanes):
            page_start = lax.select(
                page_start_index == lane,
                page_starts[lane],
                page_start,
            )

        def logical_window(step):
            return logical_indices_ref[pl.ds(step * lanes, lanes)]

        def page_locations(step):
            logical = jnp.maximum(logical_window(step), 0)
            return page_start + lax.div(logical, page_size)

        def resolve_page(page_gather_ref):
            step = pl.program_id(0)
            logical = logical_window(step)
            safe_logical = jnp.maximum(logical, 0)
            physical_page = page_gather_ref[pl.ds(0, lanes)]
            physical = physical_page * page_size + lax.rem(safe_logical, page_size)
            global_rows = global_row_base + step * lanes + jnp.arange(lanes, dtype=jnp.int32)
            dummy = lax.rem(global_rows, record_count)
            physical_indices_vmem_ref[pl.ds(step * lanes, lanes)] = jnp.where(
                logical >= 0, physical, dummy
            )

        pltpu.emit_pipeline(
            resolve_page,
            grid=(num_row_subchunks,),
            in_specs=pl.BlockSpec(
                (pl.Indirect(lanes),),
                lambda step: (page_locations(step),),
            ),
        )(page_indices_hbm_ref)
        gather_cache(physical_indices_vmem_ref, output_base)

    if paged:

        def query_index(block):
            global_row_base = (block * workers + worker_index) * rows_per_worker
            return global_row_base // topk

        pltpu.emit_pipeline(
            paged_outer_pipeline,
            grid=(num_blocks,),
            in_specs=(
                pl.BlockSpec(
                    (rows_per_worker,),
                    lambda block: (block * workers + worker_index,),
                ),
                pl.BlockSpec(
                    (lanes,),
                    lambda block: (lax.div(query_index(block), lanes),),
                ),
            ),
        )(indices_hbm_ref, page_starts_hbm_ref)
    else:
        pltpu.emit_pipeline(
            direct_outer_pipeline,
            grid=(num_blocks,),
            in_specs=pl.BlockSpec(
                (rows_per_worker,),
                lambda block: (block * workers + worker_index,),
            ),
        )(indices_hbm_ref)


@functools.partial(
    jax.jit,
    static_argnames=(
        "page_size",
        "num_streams",
        "num_row_subchunks",
        "page_table_stride",
    ),
)
def paged_sparsecore_gather_packed(
    nope_cache,
    rope_cache,
    topk_indices,
    page_indices,
    cu_kv_lens,
    query_seq_ids,
    *,
    page_size: int,
    num_streams: int | None = None,
    num_row_subchunks: int | None = None,
    page_table_stride: int | None = None,
):
    """Resolve page tables inside SparseCore and gather cache records."""
    if topk_indices.ndim != 2 or topk_indices.dtype != jnp.int32:
        raise ValueError("topk_indices must be rank-2 int32")
    if any(value.dtype != jnp.int32 for value in (page_indices, cu_kv_lens, query_seq_ids)):
        raise ValueError("paged gather metadata must use int32")
    tokens, topk = topk_indices.shape
    if query_seq_ids.shape != (tokens,):
        raise ValueError("query_seq_ids must be [T]")
    if page_indices.ndim != 1 or cu_kv_lens.ndim != 1:
        raise ValueError("page_indices and cu_kv_lens must be one-dimensional")
    if not page_indices.size:
        raise ValueError("page_indices must be nonempty")
    if nope_cache.dtype != jnp.uint8 or rope_cache.dtype != jnp.uint8:
        raise ValueError("packed CSA caches must use uint8 storage")
    if nope_cache.ndim != 4 or nope_cache.shape[-2:] != (
        CSA_CACHE_PACKING,
        TPU_V6E.vector_lanes,
    ):
        raise ValueError("NoPE cache must be [pages,page_size,4,128]")
    if rope_cache.ndim != 4 or rope_cache.shape[-2:] != (
        CSA_CACHE_PACKING,
        TPU_V6E.vector_lanes,
    ):
        raise ValueError("RoPE cache must be [pages,page_size/4,4,128]")
    if (
        page_size != nope_cache.shape[1]
        or CSA_CACHE_PACKING * rope_cache.shape[1] != page_size
        or rope_cache.shape[0] != nope_cache.shape[0]
    ):
        raise ValueError("NoPE and RoPE cache page layouts do not match")

    sparse_core = pltpu.get_tpu_info().sparse_core
    if sparse_core is None:
        raise RuntimeError("SparseCore is unavailable")
    lanes = sparse_core.num_lanes
    workers = sparse_core.num_cores * sparse_core.num_subcores
    selected_rows = tokens * topk
    mapping_capacity = get_csa_fused_page_mapping_capacity(
        topk,
        sparse_core_lanes=lanes,
        sparse_core_workers=workers,
    )
    fuse_page_mapping = tokens <= mapping_capacity
    if num_streams is None and num_row_subchunks is None:
        if fuse_page_mapping:
            schedule = get_csa_paged_gather_schedule(
                tokens,
                topk,
                sparse_core_lanes=lanes,
                sparse_core_workers=workers,
            )
        else:
            schedule = get_csa_gather_schedule(
                selected_rows,
                sparse_core_lanes=lanes,
                sparse_core_workers=workers,
            )
        num_streams = schedule.num_streams
        num_row_subchunks = schedule.num_row_subchunks
    elif num_streams is None or num_row_subchunks is None:
        raise ValueError("set both SparseCore schedule values or neither")
    if num_streams <= 0 or num_row_subchunks <= 0 or num_row_subchunks % num_streams:
        raise ValueError("num_streams must divide num_row_subchunks")
    rows_per_worker = lanes * num_row_subchunks
    if fuse_page_mapping and topk % rows_per_worker:
        raise ValueError("each SparseCore worker chunk must divide topk")

    if selected_rows % CSA_CACHE_PACKING:
        raise ValueError("packed SparseCore output requires a multiple of four rows")
    block_size = rows_per_worker * workers
    padded_rows = (selected_rows + block_size - 1) // block_size * block_size
    record_count = nope_cache.size // CSA_ATTENTION_DIM
    if fuse_page_mapping:
        gather_indices = topk_indices.reshape(-1)
        if padded_rows > selected_rows:
            gather_indices = jnp.pad(
                gather_indices,
                (0, padded_rows - selected_rows),
                constant_values=-1,
            )
        query_rows = padded_rows // topk
        page_starts = (cu_kv_lens[:-1] // jnp.int32(page_size))[query_seq_ids]
        if query_rows > tokens:
            page_starts = jnp.pad(page_starts, (0, query_rows - tokens))
        page_starts = jnp.pad(page_starts, (0, (-query_rows) % lanes))
        kernel_page_indices = page_indices
    else:
        valid = topk_indices >= 0
        logical = jnp.maximum(topk_indices, 0)
        page_starts = (cu_kv_lens[:-1] // jnp.int32(page_size))[query_seq_ids]
        logical_pages = logical // jnp.int32(page_size)
        if page_table_stride is None:
            physical_pages = page_indices[page_starts[:, None] + logical_pages]
        else:
            physical_pages = _resolve_page_ids(
                page_indices,
                logical_pages,
                query_seq_ids,
                page_table_stride=page_table_stride,
            )
        physical = physical_pages * jnp.int32(page_size) + logical % jnp.int32(page_size)
        flat_rows = jnp.arange(selected_rows, dtype=jnp.int32)
        dummy = flat_rows % jnp.int32(record_count)
        gather_indices = jnp.where(valid.reshape(-1), physical.reshape(-1), dummy)
        if padded_rows > selected_rows:
            pad_rows = jnp.arange(selected_rows, padded_rows, dtype=jnp.int32)
            gather_indices = jnp.concatenate((gather_indices, pad_rows % record_count))
        kernel_page_indices = jnp.zeros((1,), jnp.int32)
        page_starts = jnp.zeros((lanes,), jnp.int32)

    nope_cache = nope_cache.reshape(-1, nope_cache.shape[-1])
    rope_cache = rope_cache.reshape(-1, rope_cache.shape[-1])
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sparse_core.num_cores,
        num_subcores=sparse_core.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out_type = (
        jax.ShapeDtypeStruct((padded_rows, TPU_V6E.vector_lanes), jnp.int32),
        jax.ShapeDtypeStruct(
            (padded_rows // CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
            jnp.int32,
        ),
    )
    nope, rope = pl.kernel(
        functools.partial(
            _gather_kernel,
            core_axis_name=mesh.core_axis_name,
            subcore_axis_name=mesh.subcore_axis_name,
            num_row_subchunks=num_row_subchunks,
            num_streams=num_streams,
            page_size=page_size,
            paged=fuse_page_mapping,
            record_count=record_count,
            topk=topk,
        ),
        out_type=out_type,
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        mesh=mesh,
        scratch_types=(pltpu.VMEM((rows_per_worker,), jnp.int32),),
        name=(
            f"sc-csa-{'fused-map' if fuse_page_mapping else 'premapped'}-"
            f"s{num_streams}-r{num_row_subchunks}"
        ),
    )(
        nope_cache,
        rope_cache,
        gather_indices,
        kernel_page_indices,
        page_starts,
    )
    return nope[:selected_rows], rope[: selected_rows // CSA_CACHE_PACKING]


__all__ = [
    "paged_lightning_topk",
    "paged_sparsecore_gather_packed",
]
