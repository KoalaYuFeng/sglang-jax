"""Ratio-4 overlap compression shared by CSA KV and index keys."""

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
    CSA_COMPRESSION_RATIO,
    CSA_DEFAULT_PAGE_SIZE,
    CSA_FP8_AMAX_FLOOR,
    CSA_FP8_BLOCK_SIZE,
    CSA_INDEX_DIM,
    CSA_INDEX_PROJECTED_DIM,
    CSA_INDEX_PADDING_BYTES,
    CSA_INDEX_RECORD_BYTES,
    CSA_MAIN_NOPE_PADDING_BYTES,
    CSA_MAIN_NOPE_RECORD_BYTES,
    CSA_MAIN_NOPE_SCALE_COUNT,
    CSA_MAIN_PROJECTED_DIM,
    CSA_MAIN_RECORD_BYTES,
    CSA_NORM_EPS,
    CSA_ROPE_DIM,
    CSA_ROPE_FREQUENCY_DIM,
    CSA_ROPE_RECORD_BYTES,
    CSA_STATE_SLOTS,
    TPU_V6E,
    get_csa_compressor_projection_k_tile,
)

COMPRESS_RATIO = CSA_COMPRESSION_RATIO
STATE_SLOTS = CSA_STATE_SLOTS


def _interpret_pallas() -> bool:
    requested = os.environ.get("PALLAS_INTERPRET", "").strip().lower()
    return requested in ("1", "true") or jax.default_backend() != "tpu"


def _csa_state_step_kernel(
    x_ref,
    state_ref,
    weight_ref,
    ape_ref,
    norm_ref,
    cos_ref,
    sin_ref,
    positions_ref,
    value_ref,
    emit_ref,
    state_out_ref,
    projection_ref,
    *,
    head_dim: int,
    k_steps: int,
    norm_eps: float,
    record_kind: str,
):
    """Project, update the overlap state, and pool a completed ratio-4 window."""
    k_step = pl.program_id(1)

    @pl.when(k_step == 0)
    def _zero_projection():
        projection_ref[...] = jnp.zeros_like(projection_ref)

    projection_ref[...] += jax.lax.dot_general(
        x_ref[...].astype(jnp.bfloat16),
        weight_ref[...].astype(jnp.bfloat16),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    @pl.when(k_step == k_steps - 1)
    def _finish():
        batch = x_ref.shape[0]
        head_tiles = head_dim // TPU_V6E.vector_lanes
        projection_tiles = 2 * head_tiles
        positions = positions_ref[:, 0]
        slots = jnp.mod(positions, COMPRESS_RATIO).astype(jnp.int32)
        projected = projection_ref[...].reshape(
            batch,
            COMPRESS_RATIO,
            head_tiles,
            TPU_V6E.vector_lanes,
        )
        kv = projected[:, :2].reshape(batch, projection_tiles, TPU_V6E.vector_lanes)
        score = projected[:, 2:].reshape(batch, projection_tiles, TPU_V6E.vector_lanes)

        state_out_ref[...] = state_ref[...]
        for row in range(batch):
            destination = slots[row] + COMPRESS_RATIO
            state_out_ref[row, destination, 0, ...] = kv[row]
            state_out_ref[row, destination, 1, ...] = score[row] + ape_ref[row].astype(jnp.float32)

        state = state_out_ref[...]
        kv_halves = state[:, :, 0].reshape(batch, STATE_SLOTS, 2, head_dim)
        score_halves = state[:, :, 1].reshape(batch, STATE_SLOTS, 2, head_dim)
        previous_kv = kv_halves[:, :COMPRESS_RATIO, 0]
        current_kv = kv_halves[:, COMPRESS_RATIO:, 1]
        previous_score = score_halves[:, :COMPRESS_RATIO, 0]
        current_score = score_halves[:, COMPRESS_RATIO:, 1]
        window_kv = jnp.concatenate((previous_kv, current_kv), axis=1)
        window_score = jnp.concatenate((previous_score, current_score), axis=1)
        pooled = jnp.sum(window_kv * jax.nn.softmax(window_score, axis=1), axis=1)
        pooled *= jax.lax.rsqrt(jnp.mean(jnp.square(pooled), axis=-1, keepdims=True) + norm_eps)
        flat = pooled * norm_ref[...].reshape(1, head_dim).astype(jnp.float32)
        nope = flat[:, : head_dim - CSA_ROPE_DIM]
        pairs = flat[:, head_dim - CSA_ROPE_DIM :].reshape(
            batch,
            CSA_ROPE_FREQUENCY_DIM,
            2,
        )
        real, imag = pairs[..., 0], pairs[..., 1]
        cos = cos_ref[...].astype(jnp.float32)
        sin = sin_ref[...].astype(jnp.float32)
        rope = jnp.stack((real * cos - imag * sin, real * sin + imag * cos), axis=-1).reshape(
            batch, CSA_ROPE_DIM
        )
        pooled = jnp.concatenate((nope, rope), axis=-1)
        if record_kind == "main":
            fp8_max = jnp.float32(jnp.finfo(jnp.float8_e4m3fn).max)
            quantized = []
            scales = []
            for block in range(CSA_MAIN_NOPE_SCALE_COUNT):
                values = pooled[
                    :,
                    block * CSA_FP8_BLOCK_SIZE : (block + 1) * CSA_FP8_BLOCK_SIZE,
                ]
                amax = jnp.maximum(
                    jnp.max(jnp.abs(values), axis=-1, keepdims=True),
                    CSA_FP8_AMAX_FLOOR,
                )
                scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
                quantized.append((values / scale).astype(jnp.float8_e4m3fn))
                scales.append(scale)
            quantized_bytes = pltpu.bitcast(jnp.concatenate(quantized, axis=-1), jnp.uint8)
            scale_values = jnp.concatenate(scales, axis=-1)
            scale_bytes = jnp.right_shift(pltpu.bitcast(scale_values, jnp.uint32), 23).astype(
                jnp.uint8
            )
            nope_record = jnp.concatenate(
                (
                    quantized_bytes,
                    scale_bytes,
                    jnp.zeros(
                        (
                            batch,
                            CSA_MAIN_NOPE_PADDING_BYTES,
                        ),
                        jnp.uint8,
                    ),
                ),
                axis=-1,
            )
            rope_bits = pltpu.bitcast(rope.astype(jnp.bfloat16), jnp.uint16).astype(jnp.int32)
            rope_bytes = jnp.concatenate(
                (
                    jnp.right_shift(rope_bits, 8).astype(jnp.uint8),
                    (rope_bits & 0xFF).astype(jnp.uint8),
                ),
                axis=-1,
            )
            value_ref[...] = jnp.concatenate((nope_record, rope_bytes), axis=-1).reshape(
                batch,
                CSA_MAIN_RECORD_BYTES // TPU_V6E.vector_lanes,
                TPU_V6E.vector_lanes,
            )
        else:
            fp8_max = jnp.float32(jnp.finfo(jnp.float8_e4m3fn).max)
            amax = jnp.maximum(
                jnp.max(jnp.abs(pooled), axis=-1, keepdims=True),
                CSA_FP8_AMAX_FLOOR,
            )
            scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
            values = pltpu.bitcast((pooled / scale).astype(jnp.float8_e4m3fn), jnp.uint8)
            scale_byte = jnp.right_shift(pltpu.bitcast(scale, jnp.uint32), 23).astype(jnp.uint8)
            record = jnp.concatenate(
                (
                    values,
                    scale_byte,
                    jnp.zeros((batch, CSA_INDEX_PADDING_BYTES), jnp.uint8),
                ),
                axis=-1,
            )
            value_ref[...] = record.reshape(
                batch,
                CSA_INDEX_RECORD_BYTES // TPU_V6E.vector_lanes,
                TPU_V6E.vector_lanes,
            )
        emit = (jnp.mod(positions + 1, COMPRESS_RATIO) == 0).astype(jnp.int32)
        emit_ref[:, 0] = emit.astype(jnp.bool_)
        current = state[:, COMPRESS_RATIO:]
        rolled = jnp.concatenate((current, current), axis=1)
        for row in range(batch):
            state_out_ref[row, ...] = jnp.where(emit[row] != 0, rolled[row], state[row])


def _pack_main_cache_rows(pooled):
    batch = pooled.shape[0]
    fp8_max = jnp.float32(jnp.finfo(jnp.float8_e4m3fn).max)
    quantized = []
    scales = []
    for block in range(CSA_MAIN_NOPE_SCALE_COUNT):
        values = pooled[
            :,
            block * CSA_FP8_BLOCK_SIZE : (block + 1) * CSA_FP8_BLOCK_SIZE,
        ]
        amax = jnp.maximum(
            jnp.max(jnp.abs(values), axis=-1, keepdims=True),
            CSA_FP8_AMAX_FLOOR,
        )
        scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
        quantized.append((values / scale).astype(jnp.float8_e4m3fn))
        scales.append(scale)
    values = pltpu.bitcast(jnp.concatenate(quantized, axis=-1), jnp.uint8)
    scale_bits = pltpu.bitcast(jnp.concatenate(scales, axis=-1), jnp.uint32)
    scale_bytes = jnp.right_shift(scale_bits, 23).astype(jnp.uint8)
    nope = jnp.concatenate(
        (
            values,
            scale_bytes,
            jnp.zeros((batch, CSA_MAIN_NOPE_PADDING_BYTES), jnp.uint8),
        ),
        axis=-1,
    )
    rope_bits = pltpu.bitcast(pooled[:, -CSA_ROPE_DIM:].astype(jnp.bfloat16), jnp.uint16).astype(
        jnp.int32
    )
    rope = jnp.concatenate(
        (
            jnp.right_shift(rope_bits, 8).astype(jnp.uint8),
            (rope_bits & 0xFF).astype(jnp.uint8),
        ),
        axis=-1,
    )
    return nope.reshape(batch, CSA_CACHE_PACKING, TPU_V6E.vector_lanes), rope


def _pack_index_cache_rows(pooled):
    """Pack TPU-Inference-compatible FP8 values and one E8M0 scale."""
    batch = pooled.shape[0]
    fp8_max = jnp.float32(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.maximum(
        jnp.max(jnp.abs(pooled), axis=-1, keepdims=True),
        CSA_FP8_AMAX_FLOOR,
    )
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    values = pltpu.bitcast((pooled / scale).astype(jnp.float8_e4m3fn), jnp.uint8)
    scale_byte = jnp.right_shift(pltpu.bitcast(scale, jnp.uint32), 23).astype(jnp.uint8)
    return jnp.concatenate(
        (
            values,
            scale_byte,
            jnp.zeros((batch, CSA_INDEX_PADDING_BYTES), jnp.uint8),
        ),
        axis=-1,
    ).reshape(
        batch,
        CSA_INDEX_RECORD_BYTES // TPU_V6E.vector_lanes,
        TPU_V6E.vector_lanes,
    )


def _pool_uniform_groups(
    initial_state,
    kv,
    score,
    norm_ref,
    cos,
    sin,
    *,
    head_dim: int,
    norm_eps: float,
):
    """Pool aligned ratio-4 groups, carrying the preceding group forward."""
    groups = kv.shape[0]
    initial_kv = initial_state[0, :COMPRESS_RATIO, 0].reshape(COMPRESS_RATIO, 2, head_dim)[:, 0]
    initial_score = initial_state[0, :COMPRESS_RATIO, 1].reshape(COMPRESS_RATIO, 2, head_dim)[:, 0]
    if groups == 1:
        previous_kv = initial_kv[None]
        previous_score = initial_score[None]
    else:
        previous_kv = jnp.concatenate((initial_kv[None], kv[:-1, :, :head_dim]), axis=0)
        previous_score = jnp.concatenate((initial_score[None], score[:-1, :, :head_dim]), axis=0)
    window_kv = jnp.concatenate((previous_kv, kv[:, :, head_dim:]), axis=1)
    window_score = jnp.concatenate((previous_score, score[:, :, head_dim:]), axis=1)
    pooled = jnp.sum(window_kv * jax.nn.softmax(window_score, axis=1), axis=1)
    pooled *= jax.lax.rsqrt(jnp.mean(jnp.square(pooled), axis=-1, keepdims=True) + norm_eps)
    pooled *= norm_ref[...].reshape(1, head_dim).astype(jnp.float32)
    pairs = pooled[:, -CSA_ROPE_DIM:].reshape(groups, CSA_ROPE_FREQUENCY_DIM, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rope = jnp.stack(
        (real * cos - imag * sin, real * sin + imag * cos),
        axis=-1,
    ).reshape(groups, CSA_ROPE_DIM)
    return jnp.concatenate((pooled[:, :-CSA_ROPE_DIM], rope), axis=-1)


def _csa_dual_uniform_prefill_kernel(
    x_ref,
    main_state_ref,
    index_state_ref,
    fused_weight_ref,
    main_ape_ref,
    index_ape_ref,
    main_norm_ref,
    index_norm_ref,
    cos_ref,
    sin_ref,
    main_nope_ref,
    main_rope_ref,
    index_ref,
    main_state_out_ref,
    index_state_out_ref,
    projection_ref,
    *,
    k_steps: int,
    norm_eps: float,
):
    """Compress one aligned prompt chunk per request without projection spills."""
    k_step = pl.program_id(1)
    product = jax.lax.dot_general(
        x_ref[0].astype(jnp.bfloat16),
        fused_weight_ref[...].astype(jnp.bfloat16),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    projection_ref[...] = jnp.where(
        k_step == 0,
        product,
        projection_ref[...] + product,
    )

    @pl.when(k_step == k_steps - 1)
    def _finish():
        groups = x_ref.shape[1] // COMPRESS_RATIO
        projected = projection_ref[...]
        main_kv = projected[:, :CSA_MAIN_PROJECTED_DIM].reshape(
            groups,
            COMPRESS_RATIO,
            CSA_MAIN_PROJECTED_DIM,
        )
        main_score = projected[:, CSA_MAIN_PROJECTED_DIM : 2 * CSA_MAIN_PROJECTED_DIM].reshape(
            groups, COMPRESS_RATIO, CSA_MAIN_PROJECTED_DIM
        )
        main_score += (
            main_ape_ref[...]
            .astype(jnp.float32)
            .reshape(
                1,
                COMPRESS_RATIO,
                CSA_MAIN_PROJECTED_DIM,
            )
        )
        index_start = 2 * CSA_MAIN_PROJECTED_DIM
        index_kv = projected[:, index_start : index_start + CSA_INDEX_PROJECTED_DIM].reshape(
            groups, COMPRESS_RATIO, CSA_INDEX_PROJECTED_DIM
        )
        index_score = projected[:, index_start + CSA_INDEX_PROJECTED_DIM :].reshape(
            groups,
            COMPRESS_RATIO,
            CSA_INDEX_PROJECTED_DIM,
        )
        index_score += (
            index_ape_ref[...]
            .astype(jnp.float32)
            .reshape(
                1,
                COMPRESS_RATIO,
                CSA_INDEX_PROJECTED_DIM,
            )
        )

        main = _pool_uniform_groups(
            main_state_ref,
            main_kv,
            main_score,
            main_norm_ref,
            cos_ref[0].astype(jnp.float32),
            sin_ref[0].astype(jnp.float32),
            head_dim=CSA_ATTENTION_DIM,
            norm_eps=norm_eps,
        )
        index = _pool_uniform_groups(
            index_state_ref,
            index_kv,
            index_score,
            index_norm_ref,
            cos_ref[0].astype(jnp.float32),
            sin_ref[0].astype(jnp.float32),
            head_dim=CSA_INDEX_DIM,
            norm_eps=norm_eps,
        )
        main_nope, main_rope = _pack_main_cache_rows(main)
        index_record = _pack_index_cache_rows(index)
        main_nope_ref[0] = main_nope
        main_rope_ref[0] = main_rope
        index_ref[0] = index_record

        main_last = jnp.stack((main_kv[-1], main_score[-1]), axis=1)
        index_last = jnp.stack((index_kv[-1], index_score[-1]), axis=1)
        main_state_out_ref[0] = jnp.concatenate((main_last, main_last), axis=0).reshape(
            STATE_SLOTS,
            2,
            CSA_MAIN_PROJECTED_DIM // TPU_V6E.vector_lanes,
            TPU_V6E.vector_lanes,
        )
        index_state_out_ref[0] = jnp.concatenate((index_last, index_last), axis=0).reshape(
            STATE_SLOTS,
            2,
            CSA_INDEX_PROJECTED_DIM // TPU_V6E.vector_lanes,
            TPU_V6E.vector_lanes,
        )


@functools.partial(
    jax.jit,
    static_argnames=("norm_eps", "interpret"),
)
def csa_dual_uniform_prefill_pallas(
    x,
    main_state,
    index_state,
    fused_weight,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    *,
    norm_eps: float = CSA_NORM_EPS,
    interpret: bool | None = None,
):
    """Compress aligned uniform ratio-4 chunks and return cache-native records."""
    if x.ndim != 3 or x.dtype != jnp.bfloat16:
        raise ValueError("x must be BF16 [batch,sequence,hidden]")
    batch, sequence, hidden = x.shape
    if sequence < COMPRESS_RATIO or sequence % COMPRESS_RATIO:
        raise ValueError("sequence must contain complete ratio-4 groups")
    groups = sequence // COMPRESS_RATIO
    main_projected = 2 * CSA_ATTENTION_DIM
    index_projected = 2 * CSA_INDEX_DIM
    if main_state.shape != (batch, STATE_SLOTS, 2, main_projected):
        raise ValueError("main_state must be [batch,8,2,1024]")
    if index_state.shape != (batch, STATE_SLOTS, 2, index_projected):
        raise ValueError("index_state must be [batch,8,2,256]")
    fused_width = 2 * (main_projected + index_projected)
    if fused_weight.shape != (hidden, fused_width):
        raise ValueError("fused_weight must be [hidden,2560]")
    if main_ape.shape != (COMPRESS_RATIO, main_projected) or index_ape.shape != (
        COMPRESS_RATIO,
        index_projected,
    ):
        raise ValueError("compressor APE shapes are invalid")
    if main_norm.shape != (CSA_ATTENTION_DIM,) or index_norm.shape != (CSA_INDEX_DIM,):
        raise ValueError("compressor norm shapes are invalid")
    if cos.shape != (batch, groups, CSA_ROPE_FREQUENCY_DIM) or sin.shape != cos.shape:
        raise ValueError("cos and sin must be [batch,groups,32]")
    tile_k = get_csa_compressor_projection_k_tile(hidden, batch * sequence)
    k_steps = hidden // tile_k
    main_tiles = main_projected // TPU_V6E.vector_lanes
    index_tiles = index_projected // TPU_V6E.vector_lanes
    main_norm_tiles = CSA_ATTENTION_DIM // TPU_V6E.vector_lanes
    index_norm_tiles = CSA_INDEX_DIM // TPU_V6E.vector_lanes
    if interpret is None:
        interpret = _interpret_pallas()

    outputs = pl.pallas_call(
        functools.partial(
            _csa_dual_uniform_prefill_kernel,
            k_steps=k_steps,
            norm_eps=float(norm_eps),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, k_steps),
            in_specs=(
                pl.BlockSpec((1, sequence, tile_k), lambda request, k: (request, 0, k)),
                pl.BlockSpec(
                    (1, STATE_SLOTS, 2, main_tiles, TPU_V6E.vector_lanes),
                    lambda request, k: (request, 0, 0, 0, 0),
                ),
                pl.BlockSpec(
                    (1, STATE_SLOTS, 2, index_tiles, TPU_V6E.vector_lanes),
                    lambda request, k: (request, 0, 0, 0, 0),
                ),
                pl.BlockSpec((tile_k, fused_width), lambda request, k: (k, 0)),
                pl.BlockSpec(
                    (COMPRESS_RATIO, main_tiles, TPU_V6E.vector_lanes),
                    lambda request, k: (0, 0, 0),
                ),
                pl.BlockSpec(
                    (COMPRESS_RATIO, index_tiles, TPU_V6E.vector_lanes),
                    lambda request, k: (0, 0, 0),
                ),
                pl.BlockSpec(
                    (main_norm_tiles, TPU_V6E.vector_lanes),
                    lambda request, k: (0, 0),
                ),
                pl.BlockSpec(
                    (index_norm_tiles, TPU_V6E.vector_lanes),
                    lambda request, k: (0, 0),
                ),
                pl.BlockSpec(
                    (1, groups, CSA_ROPE_FREQUENCY_DIM),
                    lambda request, k: (request, 0, 0),
                ),
                pl.BlockSpec(
                    (1, groups, CSA_ROPE_FREQUENCY_DIM),
                    lambda request, k: (request, 0, 0),
                ),
            ),
            out_specs=(
                pl.BlockSpec(
                    (1, groups, CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
                    lambda request, k: (request, 0, 0, 0),
                ),
                pl.BlockSpec(
                    (1, groups, TPU_V6E.vector_lanes),
                    lambda request, k: (request, 0, 0),
                ),
                pl.BlockSpec(
                    (
                        1,
                        groups,
                        CSA_INDEX_RECORD_BYTES // TPU_V6E.vector_lanes,
                        TPU_V6E.vector_lanes,
                    ),
                    lambda request, k: (request, 0, 0, 0),
                ),
                pl.BlockSpec(
                    (1, STATE_SLOTS, 2, main_tiles, TPU_V6E.vector_lanes),
                    lambda request, k: (request, 0, 0, 0, 0),
                ),
                pl.BlockSpec(
                    (1, STATE_SLOTS, 2, index_tiles, TPU_V6E.vector_lanes),
                    lambda request, k: (request, 0, 0, 0, 0),
                ),
            ),
            scratch_shapes=(pltpu.VMEM((sequence, fused_width), jnp.float32),),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (batch, groups, CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
                jnp.uint8,
            ),
            jax.ShapeDtypeStruct((batch, groups, TPU_V6E.vector_lanes), jnp.uint8),
            jax.ShapeDtypeStruct(
                (
                    batch,
                    groups,
                    CSA_INDEX_RECORD_BYTES // TPU_V6E.vector_lanes,
                    TPU_V6E.vector_lanes,
                ),
                jnp.uint8,
            ),
            jax.ShapeDtypeStruct(
                (batch, STATE_SLOTS, 2, main_tiles, TPU_V6E.vector_lanes),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (batch, STATE_SLOTS, 2, index_tiles, TPU_V6E.vector_lanes),
                jnp.float32,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=f"csa-dual-prefill-s{sequence}-k{tile_k}",
    )(
        x,
        main_state.reshape(batch, STATE_SLOTS, 2, main_tiles, TPU_V6E.vector_lanes),
        index_state.reshape(batch, STATE_SLOTS, 2, index_tiles, TPU_V6E.vector_lanes),
        fused_weight.astype(jnp.bfloat16),
        main_ape.reshape(COMPRESS_RATIO, main_tiles, TPU_V6E.vector_lanes),
        index_ape.reshape(COMPRESS_RATIO, index_tiles, TPU_V6E.vector_lanes),
        main_norm.reshape(main_norm_tiles, TPU_V6E.vector_lanes),
        index_norm.reshape(index_norm_tiles, TPU_V6E.vector_lanes),
        cos,
        sin,
    )
    return (
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3].reshape(batch, STATE_SLOTS, 2, CSA_MAIN_PROJECTED_DIM),
        outputs[4].reshape(batch, STATE_SLOTS, 2, CSA_INDEX_PROJECTED_DIM),
    )


@functools.partial(
    jax.jit,
    donate_argnames=("state_pool",),
    static_argnames=(
        "norm_eps",
        "record_kind",
        "rope_preselected",
        "interpret",
    ),
)
def csa_state_step_fused_pallas(
    x_t,
    state_pool,
    fused_weight,
    ape,
    norm_weight,
    cos,
    sin,
    positions,
    *,
    record_kind: str,
    norm_eps: float = CSA_NORM_EPS,
    rope_preselected: bool = False,
    interpret: bool | None = None,
):
    """Fuse decode projection, state update, pooling, norm, RoPE, and packing."""
    if x_t.ndim != 2:
        raise ValueError("x_t must be [batch,hidden]")
    batch, hidden = x_t.shape
    if record_kind not in ("main", "index"):
        raise ValueError("record_kind must be 'main' or 'index'")
    head_dim = CSA_ATTENTION_DIM if record_kind == "main" else CSA_INDEX_DIM
    projected_dim = 2 * head_dim
    if state_pool.shape != (batch, STATE_SLOTS, 2, projected_dim):
        raise ValueError("state_pool must be [batch,8,2,2*head_dim]")
    if state_pool.dtype != jnp.float32:
        raise ValueError("state_pool must use FP32")
    if fused_weight.shape != (hidden, 2 * projected_dim):
        raise ValueError("fused_weight must be [hidden,4*head_dim]")
    if ape.shape != (COMPRESS_RATIO, projected_dim) or ape.dtype != jnp.float32:
        raise ValueError("ape must be FP32 [4,2*head_dim]")
    if norm_weight.shape != (head_dim,):
        raise ValueError("norm_weight must be [head_dim]")
    if positions.shape != (batch,) or not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("positions must be integer [batch]")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[1] != CSA_ROPE_FREQUENCY_DIM:
        raise ValueError("cos and sin must both be [max_position,32]")
    if rope_preselected and cos.shape[0] != batch:
        raise ValueError("preselected cos and sin must both be [batch,32]")

    tile_b = TPU_V6E.sublanes
    padded_batch = (batch + tile_b - 1) // tile_b * tile_b
    pad = padded_batch - batch
    head_tiles = head_dim // TPU_V6E.vector_lanes
    projection_tiles = projected_dim // TPU_V6E.vector_lanes
    tile_k = get_csa_compressor_projection_k_tile(hidden, batch)
    k_steps = hidden // tile_k
    x_t = jnp.pad(x_t.astype(jnp.bfloat16), ((0, pad), (0, 0)))
    state_pool = jnp.pad(
        state_pool,
        ((0, pad), (0, 0), (0, 0), (0, 0)),
        constant_values=0,
    )
    if pad:
        state_pool = state_pool.at[batch:, :, 1].set(
            -jnp.inf,
            out_sharding=jax.typeof(state_pool).sharding,
        )
    positions = jnp.pad(positions.astype(jnp.int32), (0, pad))
    slots = jnp.mod(positions, COMPRESS_RATIO)
    ape_selected = jnp.take(ape, slots, axis=0).reshape(
        padded_batch, projection_tiles, TPU_V6E.vector_lanes
    )
    if rope_preselected:
        cos_selected = jnp.pad(cos, ((0, pad), (0, 0)))
        sin_selected = jnp.pad(sin, ((0, pad), (0, 0)))
    else:
        source_position = jnp.maximum(positions + 1 - COMPRESS_RATIO, 0)
        cos_selected = jnp.take(cos, source_position, axis=0, mode="clip")
        sin_selected = jnp.take(sin, source_position, axis=0, mode="clip")
    state_tiled = state_pool.reshape(
        padded_batch,
        STATE_SLOTS,
        2,
        projection_tiles,
        TPU_V6E.vector_lanes,
    )
    if interpret is None:
        interpret = _interpret_pallas()

    record_bytes = CSA_MAIN_RECORD_BYTES if record_kind == "main" else CSA_INDEX_RECORD_BYTES
    output_tiles = record_bytes // TPU_V6E.vector_lanes
    outputs = pl.pallas_call(
        functools.partial(
            _csa_state_step_kernel,
            head_dim=head_dim,
            k_steps=k_steps,
            norm_eps=float(norm_eps),
            record_kind=record_kind,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(padded_batch // tile_b, k_steps),
            in_specs=(
                pl.BlockSpec((tile_b, tile_k), lambda block, k: (block, k)),
                pl.BlockSpec(
                    (tile_b, STATE_SLOTS, 2, projection_tiles, TPU_V6E.vector_lanes),
                    lambda block, k: (block, 0, 0, 0, 0),
                ),
                pl.BlockSpec((tile_k, 2 * projected_dim), lambda block, k: (k, 0)),
                pl.BlockSpec(
                    (tile_b, projection_tiles, TPU_V6E.vector_lanes),
                    lambda block, k: (block, 0, 0),
                ),
                pl.BlockSpec((head_tiles, TPU_V6E.vector_lanes), lambda block, k: (0, 0)),
                pl.BlockSpec((tile_b, CSA_ROPE_FREQUENCY_DIM), lambda block, k: (block, 0)),
                pl.BlockSpec((tile_b, CSA_ROPE_FREQUENCY_DIM), lambda block, k: (block, 0)),
                pl.BlockSpec((tile_b, 1), lambda block, k: (block, 0)),
            ),
            out_specs=(
                pl.BlockSpec(
                    (tile_b, output_tiles, TPU_V6E.vector_lanes),
                    lambda block, k: (block, 0, 0),
                ),
                pl.BlockSpec((tile_b, 1), lambda block, k: (block, 0)),
                pl.BlockSpec(
                    (tile_b, STATE_SLOTS, 2, projection_tiles, TPU_V6E.vector_lanes),
                    lambda block, k: (block, 0, 0, 0, 0),
                ),
            ),
            scratch_shapes=(pltpu.VMEM((tile_b, 2 * projected_dim), jnp.float32),),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (padded_batch, output_tiles, TPU_V6E.vector_lanes),
                jnp.uint8,
            ),
            jax.ShapeDtypeStruct((padded_batch, 1), jnp.bool_),
            jax.ShapeDtypeStruct(state_tiled.shape, jnp.float32),
        ),
        input_output_aliases={1: 2},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(f"csa-compressor-decode-b{tile_b}-k{tile_k}-d{head_dim}-{record_kind}"),
    )(
        x_t,
        state_tiled,
        fused_weight.astype(jnp.bfloat16),
        ape_selected,
        norm_weight.reshape(head_tiles, TPU_V6E.vector_lanes),
        cos_selected,
        sin_selected,
        positions[:, None],
    )
    value, emit, state_out = outputs
    return (
        value[:batch].reshape(batch, record_bytes),
        emit[:batch, 0],
        state_out[:batch].reshape(batch, STATE_SLOTS, 2, projected_dim),
    )


def _write_fp8_cache_rows_kernel(
    locations_ref,
    valid_ref,
    main_nope_ref,
    main_rope_ref,
    index_ref,
    main_nope_cache_hbm_ref,
    main_rope_cache_hbm_ref,
    index_cache_hbm_ref,
    _,
    __,
    ___,
    rope_groups_ref,
    index_groups_ref,
    dma_semaphores,
    *,
    page_size: int,
    tile_n: int,
):
    """DMA complete compressor records into input-output-aliased HBM caches."""
    block = pl.program_id(0)

    for row in range(tile_n):
        token = block * tile_n + row

        @pl.when(valid_ref[token])
        def _start_row(row=row, token=token):
            location = locations_ref[token]
            page = jnp.floor_divide(location, page_size)
            slot = jnp.mod(location, page_size)
            group = jnp.floor_divide(slot, CSA_CACHE_PACKING)
            pltpu.make_async_copy(
                main_nope_ref.at[row],
                main_nope_cache_hbm_ref.at[page, slot],
                dma_semaphores.at[0, row],
            ).start()
            pltpu.make_async_copy(
                main_rope_cache_hbm_ref.at[page, group],
                rope_groups_ref.at[row],
                dma_semaphores.at[1, row],
            ).start()
            pltpu.make_async_copy(
                index_cache_hbm_ref.at[page, group],
                index_groups_ref.at[row],
                dma_semaphores.at[2, row],
            ).start()

    for row in range(tile_n):
        token = block * tile_n + row

        @pl.when(valid_ref[token])
        def _wait_row(row=row, token=token):
            location = locations_ref[token]
            page = jnp.floor_divide(location, page_size)
            slot = jnp.mod(location, page_size)
            group = jnp.floor_divide(slot, CSA_CACHE_PACKING)
            lane = jnp.mod(slot, CSA_CACHE_PACKING)
            main_nope_destination = main_nope_cache_hbm_ref.at[page, slot]
            pltpu.make_async_copy(
                main_nope_destination,
                main_nope_destination,
                dma_semaphores.at[0, row],
            ).wait()
            buffers = (
                rope_groups_ref.at[row],
                index_groups_ref.at[row],
            )
            for cache_index, buffer in enumerate(buffers, start=1):
                pltpu.make_async_copy(
                    buffer,
                    buffer,
                    dma_semaphores.at[cache_index, row],
                ).wait()
            lanes = jnp.arange(CSA_CACHE_PACKING, dtype=jnp.int32)[:, None]
            rope_groups_ref[row] = jnp.where(
                lanes == lane,
                jnp.broadcast_to(
                    main_rope_ref[row],
                    (CSA_CACHE_PACKING, CSA_ROPE_RECORD_BYTES),
                ),
                rope_groups_ref[row],
            )
            index_groups_ref[row] = jnp.where(
                lanes == lane,
                jnp.broadcast_to(
                    index_ref[row],
                    (CSA_CACHE_PACKING, CSA_INDEX_RECORD_BYTES),
                ),
                index_groups_ref[row],
            )
            pltpu.make_async_copy(
                rope_groups_ref.at[row],
                main_rope_cache_hbm_ref.at[page, group],
                dma_semaphores.at[1, row],
            ).start()
            pltpu.make_async_copy(
                index_groups_ref.at[row],
                index_cache_hbm_ref.at[page, group],
                dma_semaphores.at[2, row],
            ).start()

    for row in range(tile_n):
        token = block * tile_n + row

        @pl.when(valid_ref[token])
        def _wait_group_stores(row=row, token=token):
            location = locations_ref[token]
            page = jnp.floor_divide(location, page_size)
            slot = jnp.mod(location, page_size)
            group = jnp.floor_divide(slot, CSA_CACHE_PACKING)
            destinations = (
                main_rope_cache_hbm_ref.at[page, group],
                index_cache_hbm_ref.at[page, group],
            )
            for cache_index, destination in enumerate(destinations, start=1):
                pltpu.make_async_copy(
                    destination,
                    destination,
                    dma_semaphores.at[cache_index, row],
                ).wait()


@functools.partial(jax.jit, static_argnames=("page_size", "interpret"))
def _write_fp8_cache_rows(
    main_nope,
    main_rope,
    index_records,
    emit,
    locations,
    main_nope_cache,
    main_rope_cache,
    index_cache,
    *,
    page_size: int,
    interpret: bool | None = None,
):
    """Write selected rows without materializing each full updated cache."""
    tokens = main_nope.shape[0]
    tile_n = TPU_V6E.uint8_row_tile
    padded = (tokens + tile_n - 1) // tile_n * tile_n
    pad = padded - tokens
    main_nope = jnp.pad(main_nope, ((0, pad), (0, 0), (0, 0)))
    main_rope = jnp.pad(main_rope, ((0, pad), (0, 0)))
    index_records = jnp.pad(index_records, ((0, pad), (0, 0)))
    locations = jnp.pad(locations.astype(jnp.int32), (0, pad))
    valid = jnp.pad((emit & (locations[:tokens] >= 0)).astype(jnp.bool_), (0, pad))
    if interpret is None:
        interpret = _interpret_pallas()

    return pl.pallas_call(
        functools.partial(
            _write_fp8_cache_rows_kernel,
            page_size=page_size,
            tile_n=tile_n,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(padded // tile_n,),
            in_specs=(
                pl.BlockSpec(
                    (tile_n, CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
                    lambda block, *_: (block, 0, 0),
                ),
                pl.BlockSpec(
                    (tile_n, CSA_ROPE_RECORD_BYTES),
                    lambda block, *_: (block, 0),
                ),
                pl.BlockSpec(
                    (tile_n, CSA_INDEX_RECORD_BYTES),
                    lambda block, *_: (block, 0),
                ),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            scratch_shapes=(
                pltpu.VMEM(
                    (tile_n, CSA_CACHE_PACKING, CSA_ROPE_RECORD_BYTES),
                    jnp.uint8,
                ),
                pltpu.VMEM(
                    (tile_n, CSA_CACHE_PACKING, CSA_INDEX_RECORD_BYTES),
                    jnp.uint8,
                ),
                pltpu.SemaphoreType.DMA((3, tile_n)),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(main_nope_cache.shape, main_nope_cache.dtype),
            jax.ShapeDtypeStruct(main_rope_cache.shape, main_rope_cache.dtype),
            jax.ShapeDtypeStruct(index_cache.shape, index_cache.dtype),
        ),
        input_output_aliases={5: 0, 6: 1, 7: 2},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=f"csa-fp8-cache-write-n{tile_n}",
    )(
        locations,
        valid,
        main_nope,
        main_rope,
        index_records,
        main_nope_cache,
        main_rope_cache,
        index_cache,
    )


@functools.partial(
    jax.jit,
    donate_argnames=(
        "main_state_pool",
        "index_state_pool",
        "main_nope_cache",
        "main_rope_cache",
        "index_cache",
    ),
    static_argnames=(
        "page_size",
        "norm_eps",
        "rope_preselected",
        "interpret",
    ),
)
def csa_dual_state_step_cache(
    x_t,
    main_state_pool,
    index_state_pool,
    fused_weight,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    positions,
    cache_locations,
    main_nope_cache,
    main_rope_cache,
    index_cache,
    *,
    page_size: int = CSA_DEFAULT_PAGE_SIZE,
    norm_eps: float = CSA_NORM_EPS,
    rope_preselected: bool = False,
    interpret: bool | None = None,
):
    """Compress 512-wide KV and 128-wide index keys, then write paged caches.

    ``fused_weight`` is prepacked once as ``[main_wkv_wgate | index_wkv_wgate]``.
    ``cache_locations`` contains physical compressed-record rows and uses -1 for
    tokens that must not write a cache entry. Emitted rows in one call must map
    to distinct four-record cache groups, as guaranteed by the page allocator.
    """
    if x_t.ndim != 2 or x_t.dtype != jnp.bfloat16:
        raise ValueError("x_t must be BF16 [batch,hidden]")
    batch, hidden = x_t.shape
    main_projected = 2 * CSA_ATTENTION_DIM
    index_projected = 2 * CSA_INDEX_DIM
    if fused_weight.shape != (hidden, 2 * main_projected + 2 * index_projected):
        raise ValueError("fused_weight must be [hidden,2560]")
    if fused_weight.dtype != jnp.bfloat16:
        raise ValueError("fused_weight must use BF16")
    if main_state_pool.shape != (batch, STATE_SLOTS, 2, main_projected):
        raise ValueError("main_state_pool must be [batch,8,2,1024]")
    if index_state_pool.shape != (batch, STATE_SLOTS, 2, index_projected):
        raise ValueError("index_state_pool must be [batch,8,2,256]")
    if main_state_pool.dtype != jnp.float32 or index_state_pool.dtype != jnp.float32:
        raise ValueError("compressor states must use FP32")
    if main_ape.shape != (COMPRESS_RATIO, main_projected):
        raise ValueError("main_ape must be [4,1024]")
    if index_ape.shape != (COMPRESS_RATIO, index_projected):
        raise ValueError("index_ape must be [4,256]")
    if main_ape.dtype != jnp.float32 or index_ape.dtype != jnp.float32:
        raise ValueError("compressor APE tensors must use FP32")
    if main_norm.shape != (CSA_ATTENTION_DIM,) or index_norm.shape != (CSA_INDEX_DIM,):
        raise ValueError("main_norm and index_norm must be [512] and [128]")
    if positions.shape != (batch,) or cache_locations.shape != (batch,):
        raise ValueError("positions and cache_locations must be [batch]")
    if positions.dtype != jnp.int32 or cache_locations.dtype != jnp.int32:
        raise ValueError("positions and cache_locations must use int32")
    rope_frequency_dim = CSA_ROPE_DIM // 2
    rope_shape = (batch, rope_frequency_dim) if rope_preselected else None
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[1] != rope_frequency_dim:
        raise ValueError("cos and sin must both have width 32")
    if rope_shape is not None and cos.shape != rope_shape:
        raise ValueError("preselected cos and sin must both be [batch,32]")
    if page_size <= 0 or page_size % CSA_CACHE_PACKING:
        raise ValueError("page_size must be a positive multiple of four")
    if main_nope_cache.dtype != jnp.uint8 or main_nope_cache.shape[1:] != (
        page_size,
        CSA_CACHE_PACKING,
        TPU_V6E.vector_lanes,
    ):
        raise ValueError("main_nope_cache must be uint8[pages,page_size,4,128]")
    grouped_shape = (
        main_nope_cache.shape[0],
        page_size // CSA_CACHE_PACKING,
        CSA_CACHE_PACKING,
        TPU_V6E.vector_lanes,
    )
    if main_rope_cache.dtype != jnp.uint8 or main_rope_cache.shape != grouped_shape:
        raise ValueError("main_rope_cache layout does not match main_nope_cache")
    index_shape = (
        main_nope_cache.shape[0],
        page_size // CSA_CACHE_PACKING,
        CSA_CACHE_PACKING,
        CSA_INDEX_RECORD_BYTES,
    )
    if index_cache.dtype != jnp.uint8 or index_cache.shape != index_shape:
        raise ValueError("index cache must be uint8[pages,page_size/4,4,256]")

    main_record, main_emit, main_state_pool = csa_state_step_fused_pallas(
        x_t,
        main_state_pool,
        fused_weight[:, : 2 * CSA_MAIN_PROJECTED_DIM],
        main_ape,
        main_norm,
        cos,
        sin,
        positions,
        record_kind="main",
        norm_eps=norm_eps,
        rope_preselected=rope_preselected,
        interpret=interpret,
    )
    index_value, index_emit, index_state_pool = csa_state_step_fused_pallas(
        x_t,
        index_state_pool,
        fused_weight[:, 2 * CSA_MAIN_PROJECTED_DIM :],
        index_ape,
        index_norm,
        cos,
        sin,
        positions,
        record_kind="index",
        norm_eps=norm_eps,
        rope_preselected=rope_preselected,
        interpret=interpret,
    )
    emit = main_emit & index_emit
    caches = _write_fp8_cache_rows(
        main_record[:, :CSA_MAIN_NOPE_RECORD_BYTES].reshape(
            batch,
            CSA_CACHE_PACKING,
            TPU_V6E.vector_lanes,
        ),
        main_record[:, CSA_MAIN_NOPE_RECORD_BYTES:],
        index_value,
        emit,
        cache_locations,
        main_nope_cache,
        main_rope_cache,
        index_cache,
        page_size=page_size,
        interpret=interpret,
    )
    return (
        emit,
        main_state_pool,
        index_state_pool,
        *caches,
    )


__all__ = [
    "COMPRESS_RATIO",
    "STATE_SLOTS",
    "csa_dual_state_step_cache",
    "csa_dual_uniform_prefill_pallas",
    "csa_state_step_fused_pallas",
]
