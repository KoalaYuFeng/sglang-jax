"""Joint attention over CSA's raw window and selected compressed KV."""

from __future__ import annotations

import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from .tune import (
    CSA_ATTENTION_DIM,
    CSA_CACHE_PACKING,
    CSA_FP8_BLOCK_SIZE,
    CSA_MAIN_NOPE_DIM,
    CSA_MAIN_NOPE_SCALE_COUNT,
    CSA_ROPE_DIM,
    CSA_WINDOW_SIZE,
    PIPELINE_BUFFERS,
    TPU_V6E,
    get_csa_attention_schedule,
)

ATTENTION_SELECTED_TILE, _ = get_csa_attention_schedule(1)
ATTENTION_SHARED_WINDOW_SELECTED_TILE, ATTENTION_TOKEN_TILE = get_csa_attention_schedule(
    TPU_V6E.sublanes,
    shared_window=True,
)
ATTENTION_GATHER_TILE = TPU_V6E.vector_lanes


def _interpret_pallas() -> bool:
    requested = os.environ.get("PALLAS_INTERPRET", "").strip().lower()
    return requested in ("1", "true") or jax.default_backend() != "tpu"


def _decode_selected_nope(nope_bytes):
    """Decode 448 FP8 values and seven E8M0 block scales in VMEM."""
    values = pltpu.bitcast(nope_bytes[..., :CSA_MAIN_NOPE_DIM], jnp.float8_e4m3fn).astype(
        jnp.bfloat16
    )
    scales = pltpu.bitcast(
        nope_bytes[
            ...,
            CSA_MAIN_NOPE_DIM : CSA_MAIN_NOPE_DIM + CSA_MAIN_NOPE_SCALE_COUNT,
        ],
        jnp.float8_e8m0fnu,
    ).astype(jnp.bfloat16)
    expanded = jnp.concatenate(
        tuple(
            jnp.broadcast_to(
                scales[..., block : block + 1],
                (*scales.shape[:-1], CSA_FP8_BLOCK_SIZE),
            )
            for block in range(CSA_MAIN_NOPE_SCALE_COUNT)
        ),
        axis=-1,
    )
    return (values * expanded).astype(jnp.bfloat16)


def _decode_packed_selected(nope_words, rope_words):
    """Relayout cache-native SparseCore output only after it reaches VMEM."""
    nope_bytes = pltpu.bitcast(nope_words, jnp.uint8)
    nope_bytes = nope_bytes.reshape(nope_words.shape[0], nope_words.shape[1], CSA_ATTENTION_DIM)
    rope_bytes = pltpu.bitcast(rope_words, jnp.uint8)
    rope_bytes = rope_bytes.reshape(
        rope_words.shape[0],
        CSA_CACHE_PACKING * rope_words.shape[1],
        2 * CSA_ROPE_DIM,
    )
    high = rope_bytes[..., :CSA_ROPE_DIM].astype(jnp.int32)
    low = rope_bytes[..., CSA_ROPE_DIM:].astype(jnp.int32)
    rope_bits = (jnp.left_shift(high, 8) | low).astype(jnp.uint16)
    rope = pltpu.bitcast(rope_bits, jnp.bfloat16)
    return _decode_selected_nope(nope_bytes), rope


def _joint_attention_kernel(
    q_ref,
    window_ref,
    window_valid_ref,
    selected_nope_hbm_ref,
    selected_rope_hbm_ref,
    selected_lengths_ref,
    sink_ref,
    out_ref,
    maximum_ref,
    denominator_ref,
    accumulator_ref,
    *,
    block_tokens: int,
    selected_steps: int,
    selected_tile: int,
    softmax_scale: float,
):
    """Stream SWA and selected compressed records through one softmax state."""
    program = pl.program_id(0)
    heads = q_ref.shape[1]
    head_dim = q_ref.shape[2]
    q = q_ref[...].astype(jnp.bfloat16).reshape(block_tokens * heads, head_dim)
    sink = jnp.broadcast_to(
        sink_ref[...].astype(jnp.float32)[None, :, None],
        (block_tokens, heads, 1),
    ).reshape(block_tokens * heads, 1)
    negative_finite = jnp.finfo(jnp.float32).min
    maximum_ref[...] = jnp.full(maximum_ref.shape, negative_finite, jnp.float32)
    denominator_ref[...] = jnp.zeros(denominator_ref.shape, jnp.float32)
    accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)
    selected_lengths = selected_lengths_ref[:, 0, 0].astype(jnp.int32)

    def consume(kv, valid):
        valid = valid != 0
        previous = None
        previous_value = None

        def finish(token, probability, token_kv, alpha, next_maximum, next_denominator):
            rows = pl.ds(token * heads, heads)
            value = jax.lax.dot_general(
                probability,
                token_kv.astype(jnp.bfloat16),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            accumulator_ref[rows] = alpha * accumulator_ref[rows] + value
            maximum_storage = maximum_ref[rows]
            denominator_storage = denominator_ref[rows]
            maximum_ref[rows] = jnp.broadcast_to(next_maximum, maximum_storage.shape)
            denominator_ref[rows] = jnp.broadcast_to(next_denominator, denominator_storage.shape)
            return value

        for token in range(block_tokens):
            rows = pl.ds(token * heads, heads)
            token_q = q[token * heads : (token + 1) * heads]
            if previous_value is not None:
                token_q = jnp.where(previous_value == jnp.inf, previous_value, token_q)
            scores = jax.lax.dot_general(
                token_q,
                kv[token].astype(jnp.bfloat16),
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ) * jnp.float32(softmax_scale)
            scores = jnp.where(valid[token][None, :], scores, negative_finite)
            block_maximum = jnp.max(scores, axis=1, keepdims=True)
            maximum_storage = maximum_ref[rows]
            maximum = maximum_storage[:, :1]
            next_maximum = jnp.maximum(maximum, block_maximum)
            alpha = jnp.exp(maximum - next_maximum)
            probability = jnp.where(valid[token][None, :], jnp.exp(scores - next_maximum), 0.0)
            denominator_storage = denominator_ref[rows]
            next_denominator = alpha * denominator_storage[:, :1] + jnp.sum(
                probability, axis=1, keepdims=True
            )
            if previous is not None:
                previous_value = finish(*previous)
            previous = (
                token,
                probability,
                kv[token],
                alpha,
                next_maximum,
                next_denominator,
            )

        finish(*previous)

    consume(window_ref[...].astype(jnp.bfloat16), window_valid_ref[:, 0, ...])

    def selected_step(nope_ref, rope_ref):
        step = pl.program_id(0)
        nope, rope = _decode_packed_selected(nope_ref[...], rope_ref[...])
        kv = jnp.concatenate((nope, rope), axis=-1)
        selected_index = step * selected_tile + jnp.arange(selected_tile, dtype=jnp.int32)
        consume(kv, selected_index[None, :] < selected_lengths[:, None])

    selected_specs = (
        pl.BlockSpec(
            (block_tokens, selected_tile, TPU_V6E.vector_lanes),
            lambda step: (program, step, 0),
        ),
        pl.BlockSpec(
            (
                block_tokens,
                selected_tile // CSA_CACHE_PACKING,
                TPU_V6E.vector_lanes,
            ),
            lambda step: (program, step, 0),
        ),
    )

    pltpu.emit_pipeline(
        selected_step,
        grid=(selected_steps,),
        in_specs=selected_specs,
        dimension_semantics=("arbitrary",),
    )(
        selected_nope_hbm_ref,
        selected_rope_hbm_ref,
    )
    maximum = maximum_ref[...][:, :1]
    denominator = denominator_ref[...][:, :1] + jnp.exp(sink - maximum)
    out_ref[...] = (
        (accumulator_ref[...] * pl.reciprocal(denominator, approx=True))
        .reshape(block_tokens, heads, head_dim)
        .astype(out_ref.dtype)
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "scale",
        "selected_tile",
        "tokens_per_program",
        "interpret",
    ),
)
def joint_attention_pallas(
    q,
    window_kv,
    window_valid,
    selected_nope,
    selected_rope,
    selected_valid,
    sink,
    *,
    scale: float,
    selected_tile: int = ATTENTION_SELECTED_TILE,
    tokens_per_program: int = ATTENTION_TOKEN_TILE,
    interpret: bool | None = None,
):
    """Fuse SWA and cache-native selected attention into one online softmax."""
    if q.ndim != 3 or q.dtype != jnp.bfloat16:
        raise ValueError("q must be BF16 [tokens,heads,512]")
    tokens, heads, head_dim = q.shape
    if head_dim != CSA_ATTENTION_DIM or tokens < 1 or heads < 1:
        raise ValueError("CSA attention requires a nonempty [tokens,heads,512] query")
    if window_kv.ndim != 3 or window_kv.shape[0] != tokens:
        raise ValueError("window_kv must be [tokens,window,512]")
    if window_kv.shape[-1] != head_dim or window_kv.dtype != jnp.bfloat16:
        raise ValueError("window_kv must use BF16 with width 512")
    if window_kv.shape[1] > CSA_WINDOW_SIZE or window_valid.shape != window_kv.shape[:2]:
        raise ValueError("window_valid must match a window of at most 128 rows")
    if selected_nope.ndim == 2:
        if selected_nope.shape[0] % tokens:
            raise ValueError("packed NoPE rows must be divisible by tokens")
        selected = selected_nope.shape[0] // tokens
        selected_nope = selected_nope.reshape(tokens, selected, TPU_V6E.vector_lanes)
    elif selected_nope.ndim == 3 and selected_nope.shape[0] == tokens:
        selected = selected_nope.shape[1]
    else:
        raise ValueError("packed NoPE must be int32 [tokens*selected,128]")
    if (
        selected_nope.dtype != jnp.int32
        or selected < 1
        or selected_nope.shape[2] != TPU_V6E.vector_lanes
        or selected % CSA_CACHE_PACKING
    ):
        raise ValueError("packed NoPE requires int32 records and selected divisible by four")
    if selected_rope.ndim == 2:
        if selected_rope.shape != (
            tokens * selected // CSA_CACHE_PACKING,
            TPU_V6E.vector_lanes,
        ):
            raise ValueError("packed RoPE must be int32 [tokens*selected/4,128]")
        selected_rope = selected_rope.reshape(
            tokens,
            selected // CSA_CACHE_PACKING,
            TPU_V6E.vector_lanes,
        )
    if (
        selected_rope.shape
        != (
            tokens,
            selected // CSA_CACHE_PACKING,
            TPU_V6E.vector_lanes,
        )
        or selected_rope.dtype != jnp.int32
    ):
        raise ValueError("packed RoPE shape must match packed NoPE")
    if selected_valid.shape != (tokens,) or not jnp.issubdtype(selected_valid.dtype, jnp.integer):
        raise ValueError("selected lengths must be integer [tokens]")
    selected_lengths = selected_valid.astype(jnp.int32)
    if sink.shape != (heads,):
        raise ValueError("sink must be [heads]")
    if selected_tile < TPU_V6E.vector_lanes or selected_tile % TPU_V6E.vector_lanes:
        raise ValueError("selected_tile must be a positive multiple of 128")
    maximum_token_tile = PIPELINE_BUFFERS * TPU_V6E.sublanes
    if tokens_per_program < 1 or tokens_per_program > maximum_token_tile:
        raise ValueError(f"tokens_per_program must be in [1,{maximum_token_tile}]")

    block_tokens = min(tokens, tokens_per_program)
    padded_tokens = (tokens + block_tokens - 1) // block_tokens * block_tokens
    padded_heads = (heads + TPU_V6E.sublanes - 1) // TPU_V6E.sublanes * TPU_V6E.sublanes
    padded_selected = (selected + selected_tile - 1) // selected_tile * selected_tile
    selected_steps = padded_selected // selected_tile
    token_pad = padded_tokens - tokens
    head_pad = padded_heads - heads
    selected_pad = padded_selected - selected
    window_pad = CSA_WINDOW_SIZE - window_kv.shape[1]
    q = jnp.pad(q, ((0, token_pad), (0, head_pad), (0, 0)))
    window_kv = jnp.pad(
        window_kv,
        ((0, token_pad), (0, window_pad), (0, 0)),
    )
    window_valid = jnp.pad(
        window_valid.astype(jnp.uint8),
        ((0, token_pad), (0, window_pad)),
        constant_values=False,
    ).reshape(padded_tokens, 1, TPU_V6E.vector_lanes)
    selected_nope = jnp.pad(
        selected_nope,
        ((0, token_pad), (0, selected_pad), (0, 0)),
    )
    selected_rope = jnp.pad(
        selected_rope,
        (
            (0, token_pad),
            (
                0,
                selected_pad // CSA_CACHE_PACKING,
            ),
            (0, 0),
        ),
    )
    selected_lengths = jnp.pad(selected_lengths, (0, token_pad))
    selected_lengths = jnp.broadcast_to(
        selected_lengths[:, None, None],
        (padded_tokens, 1, TPU_V6E.vector_lanes),
    )
    sink = jnp.pad(
        sink.astype(jnp.float32),
        (0, head_pad),
        constant_values=-jnp.inf,
    )
    if interpret is None:
        interpret = _interpret_pallas()

    out = pl.pallas_call(
        functools.partial(
            _joint_attention_kernel,
            block_tokens=block_tokens,
            selected_steps=selected_steps,
            selected_tile=selected_tile,
            softmax_scale=float(scale),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(padded_tokens // block_tokens,),
            in_specs=(
                pl.BlockSpec(
                    (block_tokens, padded_heads, head_dim),
                    lambda program: (program, 0, 0),
                ),
                pl.BlockSpec(
                    (block_tokens, TPU_V6E.vector_lanes, head_dim),
                    lambda program: (program, 0, 0),
                ),
                pl.BlockSpec(
                    (block_tokens, 1, TPU_V6E.vector_lanes),
                    lambda program: (program, 0, 0),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(
                    (block_tokens, 1, TPU_V6E.vector_lanes),
                    lambda program: (program, 0, 0),
                ),
                pl.BlockSpec((padded_heads,), lambda program: (0,)),
            ),
            out_specs=pl.BlockSpec(
                (block_tokens, padded_heads, head_dim),
                lambda program: (program, 0, 0),
            ),
            scratch_shapes=(
                pltpu.VMEM(
                    (block_tokens * padded_heads, TPU_V6E.vector_lanes),
                    jnp.float32,
                ),
                pltpu.VMEM(
                    (block_tokens * padded_heads, TPU_V6E.vector_lanes),
                    jnp.float32,
                ),
                pltpu.VMEM((block_tokens * padded_heads, head_dim), jnp.float32),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, padded_heads, head_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(f"csa-joint-attention-b{block_tokens}-k{selected_tile}-h{padded_heads}-d{head_dim}"),
    )(
        q,
        window_kv,
        window_valid,
        selected_nope,
        selected_rope,
        selected_lengths,
        sink,
    )
    return out[:tokens, :heads]


def _blocked_ragged_joint_attention_kernel(
    q_ref,
    window_ref,
    window_valid_ref,
    selected_nope_hbm_ref,
    selected_rope_hbm_ref,
    selected_lengths_ref,
    sink_ref,
    out_ref,
    maximum_ref,
    denominator_ref,
    accumulator_ref,
    *,
    queries_per_block: int,
    selected: int,
    selected_tile: int,
    softmax_scale: float,
):
    """Reuse one query-block SWA tile while streaming token-specific selected KV."""
    query_block = pl.program_id(0)
    q = q_ref[0].astype(jnp.bfloat16)
    heads = q.shape[1]
    negative_finite = jnp.finfo(jnp.float32).min
    maximum_ref[...] = jnp.full(maximum_ref.shape, negative_finite, jnp.float32)
    denominator_ref[...] = jnp.zeros(denominator_ref.shape, jnp.float32)
    accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)
    selected_lengths = selected_lengths_ref[0, 0, :queries_per_block].astype(jnp.int32)

    def consume_shared(kv, valid):
        rows = queries_per_block * heads
        scores = jax.lax.dot_general(
            q.reshape(rows, q.shape[-1]),
            kv.astype(jnp.bfloat16),
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ) * jnp.float32(softmax_scale)
        scores = scores.reshape(queries_per_block, heads, -1)
        scores = jnp.where(valid[:, None, :], scores, negative_finite).reshape(rows, -1)
        block_maximum = jnp.max(scores, axis=1, keepdims=True)
        previous_maximum = maximum_ref[...][:, :1]
        next_maximum = jnp.maximum(previous_maximum, block_maximum)
        alpha = jnp.exp(previous_maximum - next_maximum)
        probabilities = jnp.where(
            valid.repeat(heads, axis=0),
            jnp.exp(scores - next_maximum),
            0.0,
        )
        next_denominator = alpha * denominator_ref[...][:, :1] + jnp.sum(
            probabilities, axis=1, keepdims=True
        )
        value = jax.lax.dot_general(
            probabilities,
            kv.astype(jnp.bfloat16),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        accumulator_ref[...] = alpha * accumulator_ref[...] + value
        maximum_ref[...] = jnp.broadcast_to(next_maximum, maximum_ref.shape)
        denominator_ref[...] = jnp.broadcast_to(next_denominator, denominator_ref.shape)

    def consume(kv, valid):
        previous = None
        previous_value = None

        def finish(token, probability, token_kv, alpha, next_maximum, next_denominator):
            rows = pl.ds(token * heads, heads)
            value = jax.lax.dot_general(
                probability,
                token_kv.astype(jnp.bfloat16),
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            accumulator_ref[rows] = alpha * accumulator_ref[rows] + value
            maximum_storage = maximum_ref[rows]
            denominator_storage = denominator_ref[rows]
            maximum_ref[rows] = jnp.broadcast_to(next_maximum, maximum_storage.shape)
            denominator_ref[rows] = jnp.broadcast_to(next_denominator, denominator_storage.shape)
            return value

        for token in range(queries_per_block):
            rows = pl.ds(token * heads, heads)
            token_q = q[token]
            if previous_value is not None:
                token_q = jnp.where(previous_value == jnp.inf, previous_value, token_q)
            token_kv = kv[token]
            scores = jax.lax.dot_general(
                token_q,
                token_kv.astype(jnp.bfloat16),
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            ) * jnp.float32(softmax_scale)
            scores = jnp.where(valid[token][None, :], scores, negative_finite)
            block_maximum = jnp.max(scores, axis=1, keepdims=True)
            previous_maximum = maximum_ref[rows, :1]
            next_maximum = jnp.maximum(previous_maximum, block_maximum)
            alpha = jnp.exp(previous_maximum - next_maximum)
            probability = jnp.where(
                valid[token][None, :],
                jnp.exp(scores - next_maximum),
                0.0,
            )
            next_denominator = alpha * denominator_ref[rows, :1] + jnp.sum(
                probability, axis=1, keepdims=True
            )
            if previous is not None:
                previous_value = finish(*previous)
            previous = (
                token,
                probability,
                token_kv,
                alpha,
                next_maximum,
                next_denominator,
            )

        finish(*previous)

    consume_shared(
        window_ref[0].astype(jnp.bfloat16),
        window_valid_ref[0].reshape(queries_per_block, -1) != 0,
    )

    def consume_selected(nope_ref, rope_ref, step):
        nope, rope = _decode_packed_selected(nope_ref[...], rope_ref[...])
        selected_indices = step * selected_tile + jnp.arange(selected_tile, dtype=jnp.int32)
        valid = selected_indices[None, :] < selected_lengths[:, None]
        consume(jnp.concatenate((nope, rope), axis=-1), valid)

    selected_steps = selected // selected_tile
    if selected_steps == 1:
        consume_selected(selected_nope_hbm_ref, selected_rope_hbm_ref, 0)
    else:

        def selected_step(nope_ref, rope_ref):
            consume_selected(nope_ref, rope_ref, pl.program_id(0))

        pltpu.emit_pipeline(
            selected_step,
            grid=(selected_steps,),
            in_specs=(
                pl.BlockSpec(
                    (queries_per_block, selected_tile, TPU_V6E.vector_lanes),
                    lambda step: (query_block, step, 0),
                ),
                pl.BlockSpec(
                    (
                        queries_per_block,
                        selected_tile // CSA_CACHE_PACKING,
                        TPU_V6E.vector_lanes,
                    ),
                    lambda step: (query_block, step, 0),
                ),
            ),
            dimension_semantics=("arbitrary",),
        )(
            selected_nope_hbm_ref,
            selected_rope_hbm_ref,
        )

    sink = sink_ref[...].astype(jnp.float32)
    for query in range(queries_per_block):
        rows = pl.ds(query * heads, heads)
        denominator = denominator_ref[rows, :1] + jnp.exp(sink[:, None] - maximum_ref[rows, :1])
        out_ref[0, query] = (
            accumulator_ref[rows] * pl.reciprocal(denominator, approx=True)
        ).astype(jnp.bfloat16)


@functools.partial(
    jax.jit,
    static_argnames=(
        "scale",
        "selected_tile",
        "window_size",
        "interpret",
    ),
)
def blocked_ragged_joint_attention_pallas(
    q_blocks,
    combined_window,
    q_lens,
    prefix_lens,
    query_block_request_ids,
    query_block_offsets,
    selected_nope,
    selected_rope,
    selected_lengths,
    sink,
    *,
    scale: float,
    selected_tile: int = ATTENTION_SHARED_WINDOW_SELECTED_TILE,
    window_size: int = CSA_WINDOW_SIZE,
    interpret: bool | None = None,
):
    """Block-major ragged joint attention with request-shared SWA storage."""
    if q_blocks.ndim != 4 or q_blocks.dtype != jnp.bfloat16:
        raise ValueError("q_blocks must be BF16 [blocks,queries,heads,512]")
    blocks, queries_per_block, heads, head_dim = q_blocks.shape
    if head_dim != CSA_ATTENTION_DIM or blocks < 1 or queries_per_block < 1:
        raise ValueError("q_blocks must be nonempty with width 512")
    if window_size != CSA_WINDOW_SIZE:
        raise ValueError("CSA uses a fixed 128-row sliding window")
    if q_lens.ndim != 1 or prefix_lens.shape != q_lens.shape:
        raise ValueError("q_lens and prefix_lens must be [batch]")
    batch = q_lens.shape[0]
    if combined_window.ndim != 3 or combined_window.shape[0] != batch:
        raise ValueError("combined_window must be [batch,window+max_query,512]")
    if combined_window.shape[-1] != head_dim or combined_window.dtype != jnp.bfloat16:
        raise ValueError("combined_window must use BF16 with width 512")
    if query_block_request_ids.shape != (blocks,) or query_block_offsets.shape != (blocks,):
        raise ValueError("query-block metadata must be [blocks]")
    if selected_nope.ndim != 4 or selected_nope.shape[:2] != (blocks, queries_per_block):
        raise ValueError("selected_nope must be [blocks,queries,selected,128]")
    selected = selected_nope.shape[2]
    if selected_nope.shape[3] != TPU_V6E.vector_lanes or selected_nope.dtype != jnp.int32:
        raise ValueError("selected_nope must use cache-native int32 words")
    if selected_rope.shape != (
        blocks,
        queries_per_block,
        selected // CSA_CACHE_PACKING,
        TPU_V6E.vector_lanes,
    ):
        raise ValueError("selected_rope shape does not match selected_nope")
    if selected_rope.dtype != jnp.int32:
        raise ValueError("selected_rope must use cache-native int32 words")
    if selected_lengths.shape != (blocks, queries_per_block):
        raise ValueError("selected_lengths must be [blocks,queries]")
    if (
        selected % selected_tile
        or selected_tile < TPU_V6E.vector_lanes
        or selected_tile % TPU_V6E.vector_lanes
    ):
        raise ValueError("selected_tile must divide selected and be 128-aligned")
    if sink.shape != (heads,):
        raise ValueError("sink must be [heads]")

    padded_heads = (heads + TPU_V6E.sublanes - 1) // TPU_V6E.sublanes * TPU_V6E.sublanes
    q_blocks = jnp.pad(q_blocks, ((0, 0), (0, 0), (0, padded_heads - heads), (0, 0)))
    selected_nope = selected_nope.reshape(
        blocks * queries_per_block,
        selected,
        TPU_V6E.vector_lanes,
    )
    selected_rope = selected_rope.reshape(
        blocks * queries_per_block,
        selected // CSA_CACHE_PACKING,
        TPU_V6E.vector_lanes,
    )
    selected_lengths = jnp.pad(
        selected_lengths.astype(jnp.int32),
        ((0, 0), (0, TPU_V6E.vector_lanes - queries_per_block)),
    )[:, None, :]
    window_rows = 2 * window_size
    combined_window = jnp.pad(
        combined_window,
        ((0, 0), (0, window_rows), (0, 0)),
    )
    block_requests = query_block_request_ids.astype(jnp.int32)
    block_offsets = query_block_offsets.astype(jnp.int32)
    block_q_lens = q_lens.astype(jnp.int32)[block_requests]
    block_prefix_lens = prefix_lens.astype(jnp.int32)[block_requests]
    window_starts = block_offsets + 1
    window_positions = window_starts[:, None] + jnp.arange(window_rows, dtype=jnp.int32)
    shared_window = jnp.take_along_axis(
        combined_window[block_requests],
        window_positions[..., None],
        axis=1,
    )
    query_locals = block_offsets[:, None] + jnp.arange(queries_per_block, dtype=jnp.int32)
    shared_offsets = jnp.arange(window_rows, dtype=jnp.int32)
    relative_offsets = (
        shared_offsets[None, :] - jnp.arange(queries_per_block, dtype=jnp.int32)[:, None]
    )
    shared_positions = window_starts[:, None] + shared_offsets[None, :]
    window_valid = (
        (query_locals[..., None] < block_q_lens[:, None, None])
        & (relative_offsets[None] >= 0)
        & (relative_offsets[None] < window_size)
        & (
            shared_positions[:, None, :]
            >= jnp.maximum(window_size - block_prefix_lens, 0)[:, None, None]
        )
    ).astype(jnp.uint8)
    window_valid = window_valid.reshape(
        blocks,
        queries_per_block,
        window_rows // TPU_V6E.vector_lanes,
        TPU_V6E.vector_lanes,
    )
    sink = jnp.pad(
        sink.astype(jnp.float32),
        (0, padded_heads - heads),
        constant_values=-jnp.inf,
    )
    if interpret is None:
        interpret = _interpret_pallas()

    if selected == selected_tile:
        selected_specs = (
            pl.BlockSpec(
                (queries_per_block, selected, TPU_V6E.vector_lanes),
                lambda block, *_: (block, 0, 0),
            ),
            pl.BlockSpec(
                (
                    queries_per_block,
                    selected // CSA_CACHE_PACKING,
                    TPU_V6E.vector_lanes,
                ),
                lambda block, *_: (block, 0, 0),
            ),
        )
    else:
        selected_specs = (
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        )

    out = pl.pallas_call(
        functools.partial(
            _blocked_ragged_joint_attention_kernel,
            queries_per_block=queries_per_block,
            selected=selected,
            selected_tile=selected_tile,
            softmax_scale=float(scale),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(blocks,),
            in_specs=(
                pl.BlockSpec(
                    (1, queries_per_block, padded_heads, head_dim),
                    lambda block, *_: (block, 0, 0, 0),
                ),
                pl.BlockSpec(
                    (1, window_rows, head_dim),
                    lambda block: (block, 0, 0),
                ),
                pl.BlockSpec(
                    (
                        1,
                        queries_per_block,
                        window_rows // TPU_V6E.vector_lanes,
                        TPU_V6E.vector_lanes,
                    ),
                    lambda block, *_: (block, 0, 0, 0),
                ),
                *selected_specs,
                pl.BlockSpec(
                    (1, 1, TPU_V6E.vector_lanes),
                    lambda block, *_: (block, 0, 0),
                ),
                pl.BlockSpec((padded_heads,), lambda block, *_: (0,)),
            ),
            out_specs=pl.BlockSpec(
                (1, queries_per_block, padded_heads, head_dim),
                lambda block, *_: (block, 0, 0, 0),
            ),
            scratch_shapes=(
                pltpu.VMEM(
                    (queries_per_block * padded_heads, TPU_V6E.vector_lanes),
                    jnp.float32,
                ),
                pltpu.VMEM(
                    (queries_per_block * padded_heads, TPU_V6E.vector_lanes),
                    jnp.float32,
                ),
                pltpu.VMEM(
                    (queries_per_block * padded_heads, head_dim),
                    jnp.float32,
                ),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(q_blocks.shape, jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            f"csa-blocked-joint-q{queries_per_block}-k{selected_tile}-h{padded_heads}-d{head_dim}"
        ),
    )(
        q_blocks,
        shared_window,
        window_valid,
        selected_nope,
        selected_rope,
        selected_lengths,
        sink,
    )
    return out[:, :, :heads]


__all__ = [
    "blocked_ragged_joint_attention_pallas",
    "joint_attention_pallas",
]
