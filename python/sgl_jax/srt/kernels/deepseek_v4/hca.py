"""V4 ABI/numerical adapter to the original HCA Pallas kernels.

The framework still owns physical token pages, private FP32 scratch and prefix
snapshots. A physical token page owns ONE HCA record, not a page of 128 records.
Weights and cache dtypes are unchanged. There is no automatic backend fallback.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.deepseek_v4.numerics import rope_angles
from sgl_jax.srt.kernels.hca.attention import _streaming_attention
from sgl_jax.srt.kernels.hca.compressor import (
    _hca_emit_selected_pallas,
    hca_project_fused_pallas,
)
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule


def validate_backend(backend):
    if backend not in ("pallas", "reference"):
        raise ValueError("V4 HCA backend must be pallas or reference; no automatic fallback")
    return backend


def schedule_for(config):
    if (
        config.ratio != 128
        or config.head_dim != 512
        or config.rope_dim != 64
        or config.window != 128
    ):
        raise ValueError("original HCA V4 adapter requires r128/d512/rope64/window128")
    schedule = get_hca_kernel_schedule(
        jax.devices()[0].device_kind,
        page_size=1,
        max_compressed_entries=max(1, config.max_context // 128),
        local_heads=config.heads,
        head_dim=config.head_dim,
    )
    # Full-K projection preserves the accepted eight-row prefill arithmetic.
    # 64-key reductions retain the V4 online-softmax rounding boundaries.
    return replace(
        schedule,
        projection_k_tile=config.hidden,
        projection_batch_tile_max=8,
        compressed_tile=64,
        boundary_large_tile=4 if schedule.platform == "TPU v5p" else schedule.boundary_large_tile,
    )


def project(x, wkv, wgate, metadata, config):
    rows = metadata.router_rows
    blocks = jnp.where((rows >= 0)[..., None], x[jnp.maximum(rows, 0)], 0)
    schedule = schedule_for(config)
    fused_weight = jnp.concatenate((wkv.T, wgate.T), axis=1)
    # The existing V4 state adapter applies absolute-position APE when merging
    # old and current rows. Do not add it twice in this projection primitive.
    ape = jnp.zeros((128, 512), jnp.float32)

    def project_block(block):
        return hca_project_fused_pallas(
            block,
            fused_weight,
            ape,
            jnp.zeros((8,), jnp.int32),
            schedule=schedule,
        )

    # Use the identical eight-row MXU arithmetic for both prefill and decode.
    # The old XLA single-row GEMV has a different FP32 reduction order; this
    # cross-backend difference is measured, not hidden by BF16 scratch storage.
    projected = jax.lax.map(project_block, blocks)
    result = projected.reshape(-1, 2, 512)[metadata.router_output_rows]
    return result[:, 0], result[:, 1]


def emit(values, scores, norm_weight, starts, valid, config):
    phase = rope_angles(jnp.maximum(starts, 0), config)
    return _hca_emit_selected_pallas(
        jnp.stack((values, scores), axis=2),
        valid,
        norm_weight,
        jnp.concatenate((jnp.cos(phase), jnp.sin(phase)), axis=-1),
        schedule=schedule_for(config),
        norm_eps=config.eps,
        numerical_mode="v4",
    )


def attend(q, window_cache, compressed_cache, window_indices, positions, sink, metadata, config):
    window_valid = window_indices >= 0
    window_rows = jnp.where(
        window_valid[..., None], window_cache[jnp.maximum(window_indices, 0)], 0
    )
    # Original paged streaming attention accepts different physical compressed
    # page sizes. Here page_size=1 maps the existing token page directly to its
    # compressed record; no request-major compressed KV copy is constructed.
    records = compressed_cache.reshape(-1, 1, 1, config.head_dim)
    result = _streaming_attention(
        q,
        window_rows,
        jnp.sum(window_valid, axis=1),
        records,
        metadata.page_table.reshape(-1),
        metadata.token_requests * metadata.page_table.shape[1],
        jnp.where(metadata.token_valid, (positions + 1) // 128, 0),
        sink,
        schedule=schedule_for(config),
        softmax_scale=config.head_dim**-0.5,
        numerical_mode="v4",
        interpret=False,
    )
    return jnp.where(metadata.token_valid[:, None, None], result, 0)
