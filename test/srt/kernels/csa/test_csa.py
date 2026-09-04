"""End-to-end correctness test for the CSA kernel family."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.csa.csa import build_csa_step
from sgl_jax.srt.kernels.csa.csa_memory import CSAKVPool, CSARecurrentStatePool
from sgl_jax.srt.kernels.csa.tune import (
    CSA_ATTENTION_DIM,
    CSA_CACHE_PACKING,
    CSA_COMPRESSION_RATIO,
    CSA_DEFAULT_PAGE_SIZE,
    CSA_DUAL_PROJECTION_DIM,
    CSA_HIDDEN_DIM,
    CSA_INDEX_DIM,
    CSA_INDEX_HEADS,
    CSA_INDEX_PROJECTED_DIM,
    CSA_INDEX_RECORD_BYTES,
    CSA_MAIN_PROJECTED_DIM,
    CSA_ROPE_FREQUENCY_DIM,
    CSA_ROPE_RECORD_BYTES,
    CSA_STATE_SLOTS,
    CSA_TOP_K,
    CSA_WINDOW_SIZE,
    TPU_V6E,
)
from sgl_jax.srt.layers.attention.csa_backend import CSABackend
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

from . import ref

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU")


def _validate_backend(mesh):
    sequence = CSA_TOP_K
    with jax.set_mesh(mesh):
        kv_pool = CSAKVPool(
            sequence,
            CSA_DEFAULT_PAGE_SIZE,
            jnp.bfloat16,
            1,
            mesh,
            max_num_requests=1,
            max_context_len=sequence,
        )
        state_pool = CSARecurrentStatePool([0], 1, mesh)

    class PageTables:
        def page_tables(self, req_pool_indices, seq_lens):
            batch = len(req_pool_indices)
            compressed_entries = int(np.max(seq_lens)) // CSA_COMPRESSION_RATIO
            compressed_pages = max(
                1,
                (compressed_entries + CSA_DEFAULT_PAGE_SIZE - 1) // CSA_DEFAULT_PAGE_SIZE,
            )
            window_pages = max(
                1,
                (int(np.max(seq_lens)) + CSA_DEFAULT_PAGE_SIZE - 1) // CSA_DEFAULT_PAGE_SIZE,
            )
            return (
                np.ones((batch, compressed_pages), np.int32),
                np.ones((batch, window_pages), np.int32),
            )

    worker_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=np.asarray([0], np.int32),
        seq_lens=np.asarray([sequence], np.int32),
        positions=np.asarray([sequence - 1], np.int32),
        extend_seq_lens=None,
        extend_prefix_lens=None,
        recurrent_indices=np.asarray([1], np.int32),
    )
    backend = CSABackend(mesh=mesh)
    backend.page_table_provider = PageTables()
    decode_metadata = backend.get_forward_metadata(worker_batch)
    uniform_metadata = backend.get_forward_metadata(
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=np.asarray([0, 1], np.int32),
            seq_lens=np.asarray([128, 128], np.int32),
            positions=np.tile(np.arange(128, dtype=np.int32), 2),
            extend_seq_lens=np.asarray([128, 128], np.int32),
            extend_prefix_lens=np.asarray([0, 0], np.int32),
            recurrent_indices=np.asarray([1, 2], np.int32),
        )
    )
    ragged_metadata = backend.get_forward_metadata(
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=np.asarray([0, 1], np.int32),
            seq_lens=np.asarray([129, 136], np.int32),
            positions=np.concatenate(
                (np.asarray([128], np.int32), np.arange(128, 136, dtype=np.int32))
            ),
            extend_seq_lens=np.asarray([1, 8], np.int32),
            extend_prefix_lens=np.asarray([128, 128], np.int32),
            recurrent_indices=np.asarray([1, 2], np.int32),
        )
    )
    assert uniform_metadata.uniform_prefill
    assert uniform_metadata.query_lengths == (128, 128)
    assert uniform_metadata.query_start_slots == (0, 0)
    assert not ragged_metadata.uniform_prefill
    assert ragged_metadata.query_lengths == (1, 8)
    assert ragged_metadata.query_start_slots == (0, 0)
    backend.forward_metadata = decode_metadata
    assert backend.forward_metadata.query_lengths == (1,)

    q = jnp.ones((1, CSA_INDEX_HEADS, CSA_ATTENTION_DIM), jnp.bfloat16)
    new_kv = jnp.ones((1, CSA_ATTENTION_DIM), jnp.bfloat16)
    with jax.set_mesh(mesh):
        output, updates = backend(
            q,
            new_kv,
            new_kv,
            SimpleNamespace(layer_id=0, scaling=CSA_ATTENTION_DIM**-0.5),
            SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                positions=jnp.asarray([sequence - 1], jnp.int32),
            ),
            kv_pool,
            recurrent_state_pool=state_pool,
            compressor_input=jnp.ones((1, CSA_HIDDEN_DIM), jnp.bfloat16),
            dual_weight=jnp.zeros((CSA_HIDDEN_DIM, CSA_DUAL_PROJECTION_DIM), jnp.bfloat16),
            main_ape=jnp.zeros((CSA_COMPRESSION_RATIO, CSA_MAIN_PROJECTED_DIM), jnp.float32),
            index_ape=jnp.zeros((CSA_COMPRESSION_RATIO, CSA_INDEX_PROJECTED_DIM), jnp.float32),
            main_norm=jnp.ones((CSA_ATTENTION_DIM,), jnp.float32),
            index_norm=jnp.ones((CSA_INDEX_DIM,), jnp.float32),
            cos=jnp.ones((sequence, CSA_ROPE_FREQUENCY_DIM), jnp.float32),
            sin=jnp.zeros((sequence, CSA_ROPE_FREQUENCY_DIM), jnp.float32),
            index_query=jnp.zeros((1, CSA_INDEX_HEADS, CSA_INDEX_DIM), jnp.bfloat16),
            index_weights=jnp.zeros((1, CSA_INDEX_HEADS), jnp.float32),
            attention_sink=jnp.zeros((CSA_INDEX_HEADS,), jnp.float32),
        )
        jax.block_until_ready((output, updates))
        pools = MemoryPools(
            token_to_kv_pool=kv_pool,
            recurrent_state_pool=state_pool,
        )
        pools.replace_all(backend.pack_pool_updates([updates]))
    assert output.shape == (1, CSA_INDEX_HEADS * CSA_ATTENTION_DIM)
    assert np.isfinite(np.asarray(output, np.float32)).all()
    assert kv_pool.window_buffer[0] is updates[2]
    assert state_pool.get_csa_states(0)[0] is updates[0]


def _validate_complete_step(query_lengths, seq_lens, *, uniform_prefill, seed):
    rng = np.random.default_rng(seed)
    batch = len(query_lengths)
    hidden = TPU_V6E.vector_lanes
    tokens = sum(query_lengths)
    prefixes = np.asarray(seq_lens, np.int32) - np.asarray(query_lengths, np.int32)
    positions = np.concatenate(
        [
            np.arange(prefix, sequence, dtype=np.int32)
            for prefix, sequence in zip(prefixes, seq_lens, strict=True)
        ]
    )
    cu_q_lens = np.asarray((0, *np.cumsum(query_lengths)), np.int32)
    query_seq_ids = np.repeat(np.arange(batch, dtype=np.int32), query_lengths)

    entries = np.asarray(seq_lens, np.int32) // CSA_COMPRESSION_RATIO
    pages_per_request = max(
        1,
        (int(entries.max()) + CSA_DEFAULT_PAGE_SIZE - 1) // CSA_DEFAULT_PAGE_SIZE,
    )
    compressed_pages = np.arange(
        1,
        batch * pages_per_request + 1,
        dtype=np.int32,
    ).reshape(batch, pages_per_request)
    physical_pages = 1 + compressed_pages.size
    candidate_capacity = pages_per_request * CSA_DEFAULT_PAGE_SIZE

    logical_main = 0.1 * rng.standard_normal(
        (batch, candidate_capacity, CSA_ATTENTION_DIM), dtype=np.float32
    )
    logical_index = np.zeros((batch, candidate_capacity, CSA_INDEX_DIM), np.float32)
    encoded_bits = min((candidate_capacity - 1).bit_length(), CSA_INDEX_HEADS)
    logical_rows = np.arange(candidate_capacity, dtype=np.int32)
    for bit in range(encoded_bits):
        logical_index[:, :, bit] = (logical_rows >> bit) & 1

    main_nope_rows = np.zeros((physical_pages * CSA_DEFAULT_PAGE_SIZE, CSA_ATTENTION_DIM), np.uint8)
    main_rope_rows = np.zeros(
        (physical_pages * CSA_DEFAULT_PAGE_SIZE, CSA_ROPE_RECORD_BYTES), np.uint8
    )
    index_rows = np.zeros(
        (physical_pages * CSA_DEFAULT_PAGE_SIZE, CSA_INDEX_RECORD_BYTES), np.uint8
    )
    for request in range(batch):
        packed_nope, packed_rope = ref.pack_main(logical_main[request])
        packed_index = ref.pack_index(logical_index[request])
        for logical_page, physical_page in enumerate(compressed_pages[request]):
            logical = slice(
                logical_page * CSA_DEFAULT_PAGE_SIZE,
                (logical_page + 1) * CSA_DEFAULT_PAGE_SIZE,
            )
            physical = slice(
                physical_page * CSA_DEFAULT_PAGE_SIZE,
                (physical_page + 1) * CSA_DEFAULT_PAGE_SIZE,
            )
            main_nope_rows[physical] = packed_nope[logical]
            main_rope_rows[physical] = packed_rope[logical]
            index_rows[physical] = packed_index[logical]

    raw_pages = max(
        1,
        (max(seq_lens) + CSA_DEFAULT_PAGE_SIZE - 1) // CSA_DEFAULT_PAGE_SIZE,
    )
    window_pages = np.broadcast_to(
        np.arange(1, batch + 1, dtype=np.int32)[:, None],
        (batch, raw_pages),
    ).copy()
    window_cache = np.asarray(
        jnp.asarray(
            0.1
            * rng.standard_normal(
                (batch + 1, CSA_DEFAULT_PAGE_SIZE, CSA_ATTENTION_DIM),
                dtype=np.float32,
            ),
            jnp.bfloat16,
        )
    )

    def initial_state(width):
        state = np.zeros((batch, CSA_STATE_SLOTS, 2, width), np.float32)
        state[:, :, 1] = -np.inf
        return state

    main_state = initial_state(CSA_MAIN_PROJECTED_DIM)
    index_state = initial_state(CSA_INDEX_PROJECTED_DIM)
    compressor_input = np.zeros((tokens, hidden), np.float32)
    dual_weight = np.zeros((hidden, CSA_DUAL_PROJECTION_DIM), np.float32)
    main_ape = np.zeros((CSA_COMPRESSION_RATIO, CSA_MAIN_PROJECTED_DIM), np.float32)
    index_ape = np.zeros((CSA_COMPRESSION_RATIO, CSA_INDEX_PROJECTED_DIM), np.float32)
    main_norm = np.ones((CSA_ATTENTION_DIM,), np.float32)
    index_norm = np.ones((CSA_INDEX_DIM,), np.float32)
    cos = np.ones((max(seq_lens), CSA_ROPE_FREQUENCY_DIM), np.float32)
    sin = np.zeros_like(cos)
    index_query = np.zeros((tokens, CSA_INDEX_HEADS, CSA_INDEX_DIM), np.float32)
    index_weights = np.zeros((tokens, CSA_INDEX_HEADS), np.float32)
    for bit in range(encoded_bits):
        index_query[:, bit, bit] = 1
        index_weights[:, bit] = 1 << bit
    attention_query = np.asarray(
        jnp.asarray(
            rng.standard_normal((tokens, CSA_INDEX_HEADS, CSA_ATTENTION_DIM), dtype=np.float32),
            jnp.bfloat16,
        )
    )
    new_kv = np.asarray(
        jnp.asarray(
            rng.standard_normal((tokens, CSA_ATTENTION_DIM), dtype=np.float32),
            jnp.bfloat16,
        )
    )
    sink = rng.standard_normal((CSA_INDEX_HEADS,), dtype=np.float32)

    main_emitted_positions, main_emitted, expected_main_state = ref.compressor_ragged(
        compressor_input,
        main_state,
        dual_weight[:, : 2 * CSA_MAIN_PROJECTED_DIM],
        main_ape,
        main_norm,
        cos,
        sin,
        positions,
        query_lengths,
    )
    index_emitted_positions, index_emitted, expected_index_state = ref.compressor_ragged(
        compressor_input,
        index_state,
        dual_weight[:, 2 * CSA_MAIN_PROJECTED_DIM :],
        index_ape,
        index_norm,
        cos,
        sin,
        positions,
        query_lengths,
    )
    expected_nope = main_nope_rows.copy()
    expected_rope = main_rope_rows.copy()
    expected_index = index_rows.copy()
    for request in range(batch):
        if main_emitted_positions[request].size:
            logical = main_emitted_positions[request] // CSA_COMPRESSION_RATIO
            logical_page = logical // CSA_DEFAULT_PAGE_SIZE
            physical = (
                compressed_pages[request, logical_page] * CSA_DEFAULT_PAGE_SIZE
                + logical % CSA_DEFAULT_PAGE_SIZE
            )
            packed_nope, packed_rope = ref.pack_main(main_emitted[request])
            expected_nope[physical] = packed_nope
            expected_rope[physical] = packed_rope
            np.testing.assert_array_equal(
                main_emitted_positions[request], index_emitted_positions[request]
            )
            expected_index[physical] = ref.pack_index(index_emitted[request])

    decoded_index = ref.decode_index(expected_index)
    logical_keys = np.zeros((batch, candidate_capacity, CSA_INDEX_DIM), np.float32)
    for request in range(batch):
        for logical_page, physical_page in enumerate(compressed_pages[request]):
            logical = slice(
                logical_page * CSA_DEFAULT_PAGE_SIZE,
                (logical_page + 1) * CSA_DEFAULT_PAGE_SIZE,
            )
            physical = slice(
                physical_page * CSA_DEFAULT_PAGE_SIZE,
                (physical_page + 1) * CSA_DEFAULT_PAGE_SIZE,
            )
            logical_keys[request, logical] = decoded_index[physical]
    if candidate_capacity <= CSA_TOP_K:
        rows = np.arange(CSA_TOP_K, dtype=np.int32)[None]
        available = (positions + 1) // CSA_COMPRESSION_RATIO
        expected_topk = np.where(rows < available[:, None], rows, -1)
    else:
        expected_topk = ref.lightning_topk(
            index_query,
            logical_keys,
            index_weights,
            positions,
            query_seq_ids,
            entries,
            selected=CSA_TOP_K,
        )

    decoded_main = ref.decode_main(expected_nope, expected_rope)
    selected = ref.gather_paged(
        decoded_main,
        expected_topk,
        compressed_pages,
        query_seq_ids,
        page_size=CSA_DEFAULT_PAGE_SIZE,
    )
    window, window_valid = ref.sliding_window(
        window_cache,
        window_pages,
        positions,
        cu_q_lens,
        query_seq_ids,
        new_kv,
        window_size=CSA_WINDOW_SIZE,
        page_size=CSA_DEFAULT_PAGE_SIZE,
    )
    expected_output = ref.joint_attention(
        attention_query,
        window,
        window_valid,
        selected,
        np.sum(expected_topk >= 0, axis=1, dtype=np.int32),
        sink,
    )

    distribution = (
        np.asarray((batch, batch, batch), np.int32)
        if max(query_lengths) == 1
        else np.asarray((0, 0, batch), np.int32)
    )
    step = build_csa_step(
        query_lengths,
        query_start_slots=tuple(int(value) for value in prefixes % CSA_COMPRESSION_RATIO),
        uniform_prefill=uniform_prefill,
    )
    actual = step(
        jnp.asarray(compressor_input, jnp.bfloat16),
        jnp.asarray(dual_weight, jnp.bfloat16),
        jnp.asarray(main_ape),
        jnp.asarray(index_ape),
        jnp.asarray(main_norm),
        jnp.asarray(index_norm),
        jnp.asarray(cos),
        jnp.asarray(sin),
        jnp.asarray(positions),
        jnp.asarray(cu_q_lens),
        jnp.asarray(query_seq_ids),
        jnp.asarray(compressed_pages),
        jnp.asarray(window_pages),
        jnp.asarray(seq_lens, jnp.int32),
        jnp.asarray(distribution),
        jnp.asarray(index_query, jnp.bfloat16),
        jnp.asarray(index_weights),
        jnp.asarray(attention_query, jnp.bfloat16),
        jnp.asarray(new_kv, jnp.bfloat16),
        jnp.asarray(sink),
        jnp.asarray(main_state),
        jnp.asarray(index_state),
        jnp.asarray(
            main_nope_rows.reshape(
                physical_pages,
                CSA_DEFAULT_PAGE_SIZE,
                CSA_CACHE_PACKING,
                TPU_V6E.vector_lanes,
            )
        ),
        jnp.asarray(
            main_rope_rows.reshape(
                physical_pages,
                CSA_DEFAULT_PAGE_SIZE // CSA_CACHE_PACKING,
                CSA_CACHE_PACKING,
                TPU_V6E.vector_lanes,
            )
        ),
        jnp.asarray(
            index_rows.reshape(
                physical_pages,
                CSA_DEFAULT_PAGE_SIZE // CSA_CACHE_PACKING,
                CSA_CACHE_PACKING,
                CSA_INDEX_RECORD_BYTES,
            )
        ),
        jnp.asarray(window_cache, jnp.bfloat16),
    )
    jax.block_until_ready(actual)
    output, topk, actual_main_state, actual_index_state = actual[:4]
    actual_nope, actual_rope, actual_index, actual_window = actual[4:]
    np.testing.assert_array_equal(
        np.sort(np.asarray(topk), axis=-1),
        np.sort(expected_topk, axis=-1),
    )
    np.testing.assert_allclose(
        np.asarray(output, np.float32), expected_output, rtol=2e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(actual_main_state), expected_main_state, rtol=2e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(actual_index_state), expected_index_state, rtol=2e-2, atol=1e-2
    )
    np.testing.assert_array_equal(
        np.asarray(actual_nope).reshape(expected_nope.shape), expected_nope
    )
    np.testing.assert_array_equal(
        np.asarray(actual_rope).reshape(expected_rope.shape), expected_rope
    )
    np.testing.assert_array_equal(
        np.asarray(actual_index).reshape(expected_index.shape), expected_index
    )
    expected_window = window_cache.copy()
    for token, (position, request) in enumerate(zip(positions, query_seq_ids, strict=True)):
        physical_page = window_pages[request, position // CSA_DEFAULT_PAGE_SIZE]
        physical = physical_page * CSA_DEFAULT_PAGE_SIZE + position % CSA_DEFAULT_PAGE_SIZE
        expected_window.reshape(-1, CSA_ATTENTION_DIM)[physical] = new_kv[token]
    np.testing.assert_array_equal(np.asarray(actual_window), expected_window)


def test_csa_end_to_end_matches_numpy():
    """Validate compressor, Indexer, gather, and joint attention end to end."""
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()[:1], object).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    _validate_complete_step((1,), (4096,), uniform_prefill=False, seed=2030)
    _validate_complete_step((128,), (128,), uniform_prefill=True, seed=2031)
    _validate_complete_step((1, 4), (4096, 4096), uniform_prefill=False, seed=2032)
    _validate_complete_step((4,), (4097,), uniform_prefill=False, seed=2033)
    _validate_complete_step((2, 5), (4097, 4102), uniform_prefill=False, seed=2034)
    _validate_complete_step((7,), (7,), uniform_prefill=True, seed=2035)
    _validate_backend(mesh)
