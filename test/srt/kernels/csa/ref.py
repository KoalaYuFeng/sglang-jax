"""Independent NumPy specification for DeepSeek-V4 CSA."""

from __future__ import annotations

import ml_dtypes
import numpy as np

COMPRESSION_RATIO = 4
STATE_SLOTS = 8
ATTENTION_DIM = 512
INDEX_DIM = 128
ROPE_DIM = 64
ROPE_FREQUENCY_DIM = 32
NOPE_DIM = 448
FP8_BLOCK_SIZE = 64
NOPE_SCALE_COUNT = 7
NOPE_PADDING_BYTES = 57
INDEX_PADDING_BYTES = 127
FP8_AMAX_FLOOR = 1e-4
NORM_EPS = 1e-6


def _pool(state, norm, cos, sin):
    head_dim = norm.shape[0]
    kv_halves = state[:, :, 0].reshape(state.shape[0], STATE_SLOTS, 2, head_dim)
    score_halves = state[:, :, 1].reshape(state.shape[0], STATE_SLOTS, 2, head_dim)
    values = np.concatenate(
        (
            kv_halves[:, :COMPRESSION_RATIO, 0],
            kv_halves[:, COMPRESSION_RATIO:, 1],
        ),
        axis=1,
    )
    logits = np.concatenate(
        (
            score_halves[:, :COMPRESSION_RATIO, 0],
            score_halves[:, COMPRESSION_RATIO:, 1],
        ),
        axis=1,
    )
    coefficients = np.exp(logits - logits.max(axis=1, keepdims=True))
    coefficients /= coefficients.sum(axis=1, keepdims=True)
    pooled = np.sum(values * coefficients, axis=1)
    rms = np.sqrt(np.mean(np.square(pooled), axis=-1, keepdims=True) + NORM_EPS)
    pooled = pooled / rms * norm
    rope = pooled[:, -ROPE_DIM:].reshape(-1, ROPE_FREQUENCY_DIM, 2)
    real = rope[..., 0].copy()
    imaginary = rope[..., 1].copy()
    rope[..., 0] = real * cos - imaginary * sin
    rope[..., 1] = real * sin + imaginary * cos
    pooled[:, -ROPE_DIM:] = rope.reshape(-1, ROPE_DIM)
    return pooled


def compressor_step(x, state, weight, ape, norm, cos, sin, positions):
    """Project one token per request, update overlap state, and pool."""
    state = np.array(state, np.float32, copy=True)
    projection = np.asarray(x, np.float32) @ np.asarray(weight, np.float32)
    projected_dim = 2 * norm.shape[0]
    values = projection[:, :projected_dim]
    scores = projection[:, projected_dim:]
    emit = np.mod(positions + 1, COMPRESSION_RATIO) == 0
    for row, position in enumerate(positions):
        slot = int(position % COMPRESSION_RATIO) + COMPRESSION_RATIO
        state[row, slot, 0] = values[row]
        state[row, slot, 1] = scores[row] + ape[position % COMPRESSION_RATIO]
    source = np.maximum(positions + 1 - COMPRESSION_RATIO, 0)
    pooled = _pool(state, norm, cos[source], sin[source])
    for row in np.flatnonzero(emit):
        current = state[row, COMPRESSION_RATIO:].copy()
        state[row] = np.concatenate((current, current), axis=0)
    return pooled, emit, state


def compressor_ragged(
    x,
    state,
    weight,
    ape,
    norm,
    cos,
    sin,
    positions,
    query_lengths,
):
    """Sequential specification for a packed ragged compressor chunk."""
    state = np.asarray(state, np.float32).copy()
    emitted_positions = []
    emitted_values = []
    token_start = 0
    for request, query_length in enumerate(query_lengths):
        request_positions = []
        request_values = []
        request_state = state[request : request + 1]
        for token in range(token_start, token_start + query_length):
            pooled, emit, request_state = compressor_step(
                x[token : token + 1],
                request_state,
                weight,
                ape,
                norm,
                cos,
                sin,
                positions[token : token + 1],
            )
            if emit[0]:
                request_positions.append(int(positions[token]))
                request_values.append(pooled[0])
        state[request] = request_state[0]
        emitted_positions.append(np.asarray(request_positions, np.int32))
        emitted_values.append(
            np.asarray(request_values, np.float32).reshape(-1, norm.shape[0])
        )
        token_start += query_length
    return tuple(emitted_positions), tuple(emitted_values), state


def pack_main(pooled):
    """Encode one CSA main record as FP8 NoPE plus big-endian BF16 RoPE."""
    fp8_max = float(ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).max)
    values = []
    scales = []
    for block in range(NOPE_SCALE_COUNT):
        block_values = pooled[:, block * FP8_BLOCK_SIZE : (block + 1) * FP8_BLOCK_SIZE]
        amax = np.maximum(
            np.max(np.abs(block_values), axis=-1, keepdims=True),
            FP8_AMAX_FLOOR,
        )
        scale = np.exp2(np.ceil(np.log2(amax / fp8_max))).astype(np.float32)
        values.append((block_values / scale).astype(ml_dtypes.float8_e4m3fn))
        scales.append(scale)
    scale_bytes = (
        np.concatenate(scales, axis=-1).view(np.uint32) >> np.uint32(23)
    ).astype(np.uint8)
    nope = np.concatenate(
        (
            np.concatenate(values, axis=-1).view(np.uint8),
            scale_bytes,
            np.zeros((pooled.shape[0], NOPE_PADDING_BYTES), np.uint8),
        ),
        axis=-1,
    )
    rope_bits = pooled[:, -ROPE_DIM:].astype(ml_dtypes.bfloat16).view(np.uint16)
    rope = np.concatenate(
        (
            np.right_shift(rope_bits, 8).astype(np.uint8),
            np.bitwise_and(rope_bits, np.iinfo(np.uint8).max).astype(np.uint8),
        ),
        axis=-1,
    )
    return nope, rope


def pack_index(pooled):
    """Encode one Lightning-Indexer key and its E8M0 scale."""
    fp8_max = float(ml_dtypes.finfo(ml_dtypes.float8_e4m3fn).max)
    amax = np.maximum(
        np.max(np.abs(pooled), axis=-1, keepdims=True),
        FP8_AMAX_FLOOR,
    )
    scale = np.exp2(np.ceil(np.log2(amax / fp8_max))).astype(np.float32)
    values = (pooled / scale).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    scale_byte = np.right_shift(scale.view(np.uint32), 23).astype(np.uint8)
    return np.concatenate(
        (
            values,
            scale_byte,
            np.zeros((pooled.shape[0], INDEX_PADDING_BYTES), np.uint8),
        ),
        axis=-1,
    )


def decode_main(nope, rope):
    values = nope[..., :NOPE_DIM].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    scales = (
        nope[..., NOPE_DIM : NOPE_DIM + NOPE_SCALE_COUNT]
        .view(ml_dtypes.float8_e8m0fnu)
        .astype(np.float32)
    )
    nope_values = values * np.repeat(scales, FP8_BLOCK_SIZE, axis=-1)
    bits = np.left_shift(rope[..., :ROPE_DIM].astype(np.uint16), 8) | rope[
        ..., ROPE_DIM:
    ].astype(np.uint16)
    return np.concatenate(
        (nope_values, bits.view(ml_dtypes.bfloat16).astype(np.float32)),
        axis=-1,
    )


def decode_index(records):
    values = records[..., :INDEX_DIM].view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    scales = (
        records[..., INDEX_DIM : INDEX_DIM + 1]
        .view(ml_dtypes.float8_e8m0fnu)
        .astype(np.float32)
    )
    return values * scales


def lightning_topk(
    queries,
    keys,
    weights,
    query_positions,
    query_seq_ids,
    kv_lens,
    *,
    selected=512,
):
    """Exact causal Lightning-Indexer selection with -1 padding."""
    queries = np.asarray(queries, np.float32)
    keys = np.asarray(keys, np.float32)
    weights = np.asarray(weights, np.float32)
    selected_keys = keys[np.asarray(query_seq_ids, np.int32)]
    dots = np.einsum("thd,tkd->thk", queries, selected_keys)
    scores = np.einsum("thk,th->tk", np.maximum(dots, 0), weights)
    result = np.full((queries.shape[0], selected), -1, np.int32)
    for token, (position, request) in enumerate(
        zip(query_positions, query_seq_ids, strict=True)
    ):
        available = min(
            int(kv_lens[request]),
            (int(position) + 1) // COMPRESSION_RATIO,
        )
        count = min(selected, available)
        if count:
            order = np.argsort(-scores[token, :available], kind="stable")[:count]
            result[token, :count] = order.astype(np.int32)
    return result


def gather_paged(records, topk, page_indices, query_seq_ids, *, page_size=128):
    """Gather logical per-request rows through a fixed-width page table."""
    records = np.asarray(records)
    page_indices = np.asarray(page_indices, np.int32)
    result = np.zeros((*topk.shape, records.shape[-1]), records.dtype)
    for token, request in enumerate(query_seq_ids):
        for output_row, logical in enumerate(topk[token]):
            if logical < 0:
                continue
            physical_page = page_indices[request, logical // page_size]
            result[token, output_row] = records[
                physical_page * page_size + logical % page_size
            ]
    return result


def sliding_window(
    cache,
    page_indices,
    positions,
    cu_q_lens,
    query_seq_ids,
    new_values,
    *,
    window_size=128,
    page_size=128,
):
    """Materialize the causal SWA rows for a packed ragged query batch."""
    cache = np.asarray(cache).reshape(-1, cache.shape[-1])
    rows = np.zeros((len(positions), window_size, cache.shape[-1]), np.float32)
    valid = np.zeros((len(positions), window_size), np.bool_)
    for token, (position, request) in enumerate(
        zip(positions, query_seq_ids, strict=True)
    ):
        local_query = token - int(cu_q_lens[request])
        prefix = int(position) - local_query
        for column, key_position in enumerate(
            range(int(position) - window_size + 1, int(position) + 1)
        ):
            if key_position < 0:
                continue
            valid[token, column] = True
            if key_position >= prefix:
                source = int(cu_q_lens[request]) + key_position - prefix
                rows[token, column] = new_values[source]
            else:
                physical_page = page_indices[request, key_position // page_size]
                physical = physical_page * page_size + key_position % page_size
                rows[token, column] = cache[physical]
    return rows, valid


def joint_attention(q, window, window_valid, selected, selected_valid, sink):
    """Stable FP32 softmax over SWA, selected CSA records, and sink logits."""
    q = np.asarray(q, np.float32)
    output = np.zeros_like(q)
    scale = ATTENTION_DIM**-0.5
    for token in range(q.shape[0]):
        window_rows = np.asarray(window[token], np.float32)[window_valid[token]]
        selected_rows = np.asarray(selected[token], np.float32)[: selected_valid[token]]
        values = np.concatenate((window_rows, selected_rows), axis=0)
        scores = np.einsum("hd,kd->hk", q[token], values) * scale
        maximum = np.maximum(scores.max(axis=-1), sink)
        probability = np.exp(scores - maximum[:, None])
        denominator = probability.sum(axis=-1) + np.exp(sink - maximum)
        output[token] = (
            np.einsum("hk,kd->hd", probability, values) / denominator[:, None]
        )
    return output


__all__ = [
    "compressor_ragged",
    "compressor_step",
    "decode_index",
    "decode_main",
    "gather_paged",
    "joint_attention",
    "lightning_topk",
    "pack_index",
    "pack_main",
    "sliding_window",
]
