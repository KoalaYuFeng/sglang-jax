"""CSA model constants, TPU layout constraints, and derived schedules."""

from __future__ import annotations

from dataclasses import dataclass

CSA_COMPRESSION_RATIO = 4
CSA_STATE_SLOTS = 2 * CSA_COMPRESSION_RATIO

CSA_HIDDEN_DIM = 4096
CSA_ATTENTION_HEADS = 64
CSA_INDEX_HEADS = 64
CSA_INDEX_DIM = 128
CSA_ATTENTION_DIM = 512
CSA_ROPE_DIM = 64
CSA_ROPE_FREQUENCY_DIM = CSA_ROPE_DIM // 2
CSA_MAIN_NOPE_DIM = CSA_ATTENTION_DIM - CSA_ROPE_DIM
CSA_MAIN_PROJECTED_DIM = 2 * CSA_ATTENTION_DIM
CSA_INDEX_PROJECTED_DIM = 2 * CSA_INDEX_DIM
CSA_DUAL_PROJECTION_DIM = 2 * (CSA_MAIN_PROJECTED_DIM + CSA_INDEX_PROJECTED_DIM)

CSA_TOP_K = 512
CSA_WINDOW_SIZE = 128
CSA_DEFAULT_PAGE_SIZE = 128
CSA_CACHE_PACKING = 4
CSA_PAGE_TABLE_COUNT = 2
CSA_PAGE_INDEX_BYTES = 4

CSA_FP8_BLOCK_SIZE = 64
CSA_FP8_AMAX_FLOOR = 1e-4
CSA_NORM_EPS = 1e-6
CSA_MAIN_NOPE_SCALE_COUNT = CSA_MAIN_NOPE_DIM // CSA_FP8_BLOCK_SIZE
CSA_MAIN_NOPE_RECORD_BYTES = CSA_ATTENTION_DIM
CSA_MAIN_NOPE_PADDING_BYTES = (
    CSA_MAIN_NOPE_RECORD_BYTES - CSA_MAIN_NOPE_DIM - CSA_MAIN_NOPE_SCALE_COUNT
)
CSA_ROPE_RECORD_BYTES = 2 * CSA_ROPE_DIM
CSA_MAIN_RECORD_BYTES = CSA_MAIN_NOPE_RECORD_BYTES + CSA_ROPE_RECORD_BYTES
CSA_INDEX_SCALE_COUNT = 1
CSA_INDEX_RECORD_BYTES = 2 * CSA_INDEX_DIM
CSA_INDEX_PADDING_BYTES = CSA_INDEX_RECORD_BYTES - CSA_INDEX_DIM - CSA_INDEX_SCALE_COUNT


@dataclass(frozen=True)
class TPULayout:
    vector_lanes: int
    sublanes: int
    uint8_row_tile: int
    sparsecore_lanes: int
    sparsecore_workers: int
    scalar_prefetch_bytes: int

    @property
    def sparsecore_wave_rows(self) -> int:
        return self.sparsecore_lanes * self.sparsecore_workers


@dataclass(frozen=True)
class CSAV6eCalibration:
    projection_k_tiles: tuple[int, int]
    sparsecore_program_rows: int
    indexer_pages: tuple[int, int, int]
    indexer_decode_request_batch: int


TPU_V6E = TPULayout(
    vector_lanes=128,
    sublanes=8,
    uint8_row_tile=32,
    sparsecore_lanes=8,
    sparsecore_workers=32,
    scalar_prefetch_bytes=1 << 20,
)

# Values in this table are benchmark-selected on TPU v6e; they are not
# presented as capacity formulas.
V6E_CALIBRATION = CSAV6eCalibration(
    projection_k_tiles=(2048, 4096),
    sparsecore_program_rows=8192,
    indexer_pages=(3, 2, 2),
    indexer_decode_request_batch=4,
)

PIPELINE_BUFFERS = 2


@dataclass(frozen=True)
class CSAIndexerSchedule:
    num_kv_pages_per_block: tuple[int, int, int]
    num_queries_per_block: tuple[int, int, int]
    decode_request_batch: int


@dataclass(frozen=True)
class CSAGatherSchedule:
    num_streams: int
    num_row_subchunks: int


def get_csa_compressor_projection_k_tile(hidden: int, batch: int) -> int:
    if hidden <= 0 or hidden % TPU_V6E.vector_lanes:
        raise ValueError("hidden must be a positive multiple of the TPU lane count")
    if batch <= 0:
        raise ValueError("batch must be positive")
    small_tile, large_tile = V6E_CALIBRATION.projection_k_tiles
    preferred = large_tile if batch > TPU_V6E.uint8_row_tile else small_tile
    tile = min(hidden, preferred)
    while hidden % tile:
        tile -= TPU_V6E.vector_lanes
    return tile


def _indexer_query_tile(query_length: int, maximum: int) -> int:
    if query_length <= 0 or maximum <= 0:
        raise ValueError("query length and maximum must be positive")
    target = min(query_length, maximum)
    return 1 << (target - 1).bit_length()


def get_csa_indexer_schedule(
    *,
    prefill_query_length: int = 1,
    mixed_max_query_length: int = 1,
) -> CSAIndexerSchedule:
    return CSAIndexerSchedule(
        num_kv_pages_per_block=V6E_CALIBRATION.indexer_pages,
        num_queries_per_block=(
            1,
            _indexer_query_tile(prefill_query_length, TPU_V6E.vector_lanes),
            _indexer_query_tile(mixed_max_query_length, TPU_V6E.vector_lanes),
        ),
        decode_request_batch=V6E_CALIBRATION.indexer_decode_request_batch,
    )


def _maximum_sparsecore_subchunks(
    *,
    sparse_core_lanes: int,
    sparse_core_workers: int,
) -> int:
    rows_per_wave_step = sparse_core_lanes * sparse_core_workers
    if rows_per_wave_step <= 0:
        raise ValueError("SparseCore lanes and workers must be positive")
    return V6E_CALIBRATION.sparsecore_program_rows // rows_per_wave_step


def get_csa_gather_schedule(
    selected_rows: int,
    *,
    sparse_core_lanes: int,
    sparse_core_workers: int,
) -> CSAGatherSchedule:
    if selected_rows <= 0:
        raise ValueError("selected_rows must be positive")
    base_block = sparse_core_lanes * sparse_core_workers
    maximum = _maximum_sparsecore_subchunks(
        sparse_core_lanes=sparse_core_lanes,
        sparse_core_workers=sparse_core_workers,
    )
    required = (selected_rows + base_block - 1) // base_block
    row_subchunks = min(required, maximum)
    streams = PIPELINE_BUFFERS if required > maximum else 1
    return CSAGatherSchedule(streams, row_subchunks)


def get_csa_paged_gather_schedule(
    query_count: int,
    topk: int,
    *,
    sparse_core_lanes: int,
    sparse_core_workers: int,
) -> CSAGatherSchedule:
    if query_count <= 0 or topk <= 0:
        raise ValueError("query_count and topk must be positive")
    rows_per_wave_step = sparse_core_lanes * sparse_core_workers
    if topk % rows_per_wave_step:
        raise ValueError("topk must be divisible by one SparseCore wave")
    queries_per_wave = 1
    maximum_queries = get_csa_fused_page_mapping_capacity(
        topk,
        sparse_core_lanes=sparse_core_lanes,
        sparse_core_workers=sparse_core_workers,
    )
    while queries_per_wave < query_count and PIPELINE_BUFFERS * queries_per_wave <= maximum_queries:
        queries_per_wave *= PIPELINE_BUFFERS
    row_subchunks = queries_per_wave * topk // rows_per_wave_step
    streams = (
        PIPELINE_BUFFERS
        if (query_count == 1 and row_subchunks >= PIPELINE_BUFFERS)
        or query_count > queries_per_wave
        else 1
    )
    return CSAGatherSchedule(streams, row_subchunks)


def get_csa_fused_page_mapping_capacity(
    topk: int,
    *,
    sparse_core_lanes: int,
    sparse_core_workers: int,
) -> int:
    if topk <= 0:
        raise ValueError("topk must be positive")
    maximum = _maximum_sparsecore_subchunks(
        sparse_core_lanes=sparse_core_lanes,
        sparse_core_workers=sparse_core_workers,
    )
    rows_per_wave_step = sparse_core_lanes * sparse_core_workers
    if topk % rows_per_wave_step:
        raise ValueError("topk must be divisible by one SparseCore wave")
    return maximum * rows_per_wave_step // (PIPELINE_BUFFERS * topk)


def get_csa_attention_schedule(
    query_count: int,
    *,
    shared_window: bool = False,
) -> tuple[int, int]:
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    selected_tile = CSA_TOP_K if shared_window else CSA_TOP_K // PIPELINE_BUFFERS
    return selected_tile, min(query_count, TPU_V6E.sublanes)


def csa_topk_is_identity(candidate_count: int, *, selected: int = CSA_TOP_K) -> bool:
    if candidate_count <= 0 or selected <= 0:
        raise ValueError("candidate_count and selected must be positive")
    return candidate_count <= selected


def get_csa_max_running_requests(max_context_len: int, page_size: int) -> int:
    if max_context_len <= 0 or page_size <= 0:
        raise ValueError("context length and page size must be positive")
    pages_per_request = (max_context_len + page_size - 1) // page_size
    metadata_bytes = CSA_PAGE_TABLE_COUNT * pages_per_request * CSA_PAGE_INDEX_BYTES
    return max(1, TPU_V6E.scalar_prefetch_bytes // metadata_bytes)


__all__ = [
    "CSA_ATTENTION_DIM",
    "CSA_ATTENTION_HEADS",
    "CSA_CACHE_PACKING",
    "CSA_COMPRESSION_RATIO",
    "CSA_DEFAULT_PAGE_SIZE",
    "CSA_DUAL_PROJECTION_DIM",
    "CSA_FP8_AMAX_FLOOR",
    "CSA_FP8_BLOCK_SIZE",
    "CSA_HIDDEN_DIM",
    "CSA_INDEX_DIM",
    "CSA_INDEX_HEADS",
    "CSA_INDEX_PADDING_BYTES",
    "CSA_INDEX_PROJECTED_DIM",
    "CSA_INDEX_RECORD_BYTES",
    "CSA_INDEX_SCALE_COUNT",
    "CSA_MAIN_NOPE_DIM",
    "CSA_MAIN_NOPE_PADDING_BYTES",
    "CSA_MAIN_NOPE_RECORD_BYTES",
    "CSA_MAIN_NOPE_SCALE_COUNT",
    "CSA_MAIN_PROJECTED_DIM",
    "CSA_MAIN_RECORD_BYTES",
    "CSA_NORM_EPS",
    "CSA_PAGE_INDEX_BYTES",
    "CSA_PAGE_TABLE_COUNT",
    "CSA_ROPE_DIM",
    "CSA_ROPE_FREQUENCY_DIM",
    "CSA_ROPE_RECORD_BYTES",
    "CSA_STATE_SLOTS",
    "CSA_TOP_K",
    "CSA_WINDOW_SIZE",
    "CSAGatherSchedule",
    "CSAIndexerSchedule",
    "CSAV6eCalibration",
    "PIPELINE_BUFFERS",
    "TPULayout",
    "TPU_V6E",
    "V6E_CALIBRATION",
    "csa_topk_is_identity",
    "get_csa_attention_schedule",
    "get_csa_compressor_projection_k_tile",
    "get_csa_fused_page_mapping_capacity",
    "get_csa_gather_schedule",
    "get_csa_indexer_schedule",
    "get_csa_max_running_requests",
    "get_csa_paged_gather_schedule",
]
