"""Correctness-first, single-request V4 checkpoint execution on four TPU chips.

This is an explicit bring-up runner, not an SGLang serving registration. Routed
experts use expert parallelism; smaller weights and attention are replicated.
Checkpoint FP4/FP8 bytes/scales stay packed in HBM. Only tiled matmuls unpack
weights in VMEM. Cache storage is BF16 with the official activation QAT applied.
Full-position KV storage makes the reference implementation easy to inspect;
the attention mask still enforces the official sliding window.
"""

from __future__ import annotations

import functools
import gc
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.low_bit.formats import (
    activation_fp4_roundtrip,
    activation_fp8_roundtrip,
    round_bf16,
)
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul
from sgl_jax.srt.kernels.low_bit.moe import routed_fp4_experts
from sgl_jax.srt.kernels.mhc import mhc_post_fused, mhc_pre_fused
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "model_loader/deepseek_v4_checkpoint.py"]
    for directory in ("kernels/low_bit", "kernels/mhc"):
        paths.extend((root / directory).glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceConfig:
    hidden: int = 4096
    heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    groups: int = 8
    o_rank: int = 1024
    index_heads: int = 64
    index_dim: int = 128
    index_topk: int = 512
    window: int = 128
    hc: int = 4
    sinkhorn_iters: int = 20
    eps: float = 1e-6
    hc_eps: float = 1e-6
    active_experts: int = 6
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    max_context: int = 256
    ratio: int = 0
    hash_routing: bool = False
    rope_base: float = 10000.0
    original_seq_len: int = 0
    rope_factor: float = 16.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0


def _fixed_tree_sum_last(x):
    """Deterministic last-axis reduction independent of leading token shape."""
    value = x
    while value.shape[-1] > 1:
        if value.shape[-1] % 2:
            value = jnp.pad(value, ((0, 0),) * (value.ndim - 1) + ((0, 1),))
        value = value[..., 0::2] + value[..., 1::2]
    return value[..., 0]


def _fixed_tree_mean_last(x):
    return _fixed_tree_sum_last(x) / jnp.float32(x.shape[-1])


def rms_norm(x, weight, eps):
    xf = x.astype(jnp.float32)
    variance = _fixed_tree_mean_last(xf * xf)[..., None]
    normalized = xf * jax.lax.rsqrt(variance + eps)
    return round_bf16(normalized * weight.astype(jnp.float32))


def rope(x, positions, config, *, inverse=False):
    rd = config.rope_dim
    frequency = 1.0 / (config.rope_base ** (jnp.arange(0, rd, 2, dtype=jnp.float32) / rd))
    if config.original_seq_len:

        def correction(rotations):
            return (
                rd
                * math.log(config.original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(config.rope_base))
            )

        low = max(math.floor(correction(config.beta_fast)), 0)
        high = min(math.ceil(correction(config.beta_slow)), rd - 1)
        ramp = jnp.clip((jnp.arange(rd // 2) - low) / (high - low if high != low else 0.001), 0, 1)
        frequency = frequency / config.rope_factor * ramp + frequency * (1 - ramp)
    phase = positions.astype(jnp.float32)[:, None] * frequency[None, :]
    phase = phase.reshape((positions.shape[0],) + (1,) * (x.ndim - 2) + (rd // 2,))
    cosine, sine = jnp.cos(phase), jnp.sin(phase) * (-1 if inverse else 1)
    paired = x[..., -rd:].astype(jnp.float32).reshape(*x.shape[:-1], rd // 2, 2)
    a, b = paired[..., 0], paired[..., 1]
    rotated = jnp.stack((a * cosine - b * sine, a * sine + b * cosine), axis=-1)
    return jnp.concatenate((x[..., :-rd], round_bf16(rotated.reshape(*x.shape[:-1], rd))), axis=-1)


def hadamard_rotate(x):
    # F32 butterfly and one final BF16 rounding, matching the CUDA reference.
    width = x.shape[-1]
    if width & (width - 1):
        raise ValueError("Hadamard width must be a power of two")
    value = x.astype(jnp.float32)
    stride = 1
    while stride < width:
        grouped = value.reshape(*x.shape[:-1], -1, 2, stride)
        a, b = grouped[..., 0, :], grouped[..., 1, :]
        value = jnp.stack((a + b, a - b), axis=-2).reshape(x.shape)
        stride *= 2
    return round_bf16(value * width**-0.5)


def linear(x, weights, prefix, *, quantize=True):
    weight = weights[prefix + ".weight"]
    scales = weights.get(prefix + ".scale")
    fmt = "bf16" if scales is None else "fp8"
    return low_bit_matmul(
        x, weight, scales, weight_format=fmt, quantize_activation=quantize and scales is not None
    )


def empty_cache(config):
    cache = {"window": jnp.zeros((config.max_context, config.head_dim), jnp.bfloat16)}
    if config.ratio:
        for prefix, dim in (("main", config.head_dim), ("index", config.index_dim)):
            if prefix == "index" and config.ratio != 4:
                continue
            coff = 2 if config.ratio == 4 else 1
            cache[prefix + ".kv"] = jnp.zeros((coff * config.ratio, coff * dim), jnp.float32)
            cache[prefix + ".score"] = jnp.full(
                (coff * config.ratio, coff * dim), -jnp.inf, jnp.float32
            )
            cache[prefix + ".compressed"] = jnp.zeros(
                (config.max_context // config.ratio, dim), jnp.bfloat16
            )
    return cache


def compress(x, positions, weights, cache, config, *, index=False, trace=None):
    ratio = config.ratio
    dim = config.index_dim if index else config.head_dim
    prefix = "attn.indexer.compressor" if index else "attn.compressor"
    state_prefix = "index" if index else "main"
    overlap = ratio == 4
    # BF16 checkpoint values are exact inputs; FP32 accumulation/output avoids
    # the extra rounding that would occur with the BF16 linear helper.
    kv = jnp.matmul(x, weights[prefix + ".wkv.weight"].T, preferred_element_type=jnp.float32)
    score = jnp.matmul(x, weights[prefix + ".wgate.weight"].T, preferred_element_type=jnp.float32)
    initial = tuple(cache[state_prefix + suffix] for suffix in (".kv", ".score", ".compressed"))

    def step(state, item):
        values, scores, compressed = state
        value, score_value, pos = item
        slot = (ratio if overlap else 0) + pos % ratio
        values = values.at[slot].set(value)
        scores = scores.at[slot].set(score_value + weights[prefix + ".ape"][pos % ratio])

        def emit(state):
            values, scores, compressed = state
            if overlap:
                # Explicit channel selection avoids v5p's incorrect lowering
                # of sliced concatenation after a dynamic scratch update.
                columns = jnp.arange(dim)[None] + (jnp.arange(2 * ratio) >= ratio)[:, None] * dim
                pool_values = jnp.take_along_axis(values, columns, axis=1)
                pool_scores = jnp.take_along_axis(scores, columns, axis=1)
            else:
                pool_values, pool_scores = values, scores
            # Keep the official [batch, rows, channels] vectorized pooling
            # structure in both phases. The old hand-written serial reduction
            # rounded a real 7939 fixture differently from prefill at a BF16
            # midpoint. Use explicit channel gathers above, not the legacy
            # sliced concatenation that motivated the serial workaround.
            pooled = round_bf16(
                jnp.sum(pool_values[None] * jax.nn.softmax(pool_scores[None], axis=1), axis=1)
            )
            pooled = rms_norm(pooled, weights[prefix + ".norm.weight"], config.eps)
            pooled = rope(pooled, (pos + 1 - ratio)[None], config)
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
            compressed = compressed.at[(pos + 1) // ratio - 1].set(pooled[0])
            if overlap:
                values = values.at[:ratio].set(values[ratio:])
                scores = scores.at[:ratio].set(scores[ratio:])
            return values, scores, compressed

        return (
            jax.lax.cond(
                (pos + 1) % ratio == 0, emit, lambda state: state, (values, scores, compressed)
            ),
            None,
        )

    if x.shape[0] == 1:
        # Keep decode outside a scan. The v5p scan -> dynamic state update ->
        # pooling lowering failed the real-weight CPU oracle; a barrier alone
        # does not repair it. Prefill follows the official vectorized path.
        final, _ = step(initial, (kv[0], score[0], positions[0]))
    else:
        values, scores, compressed = initial
        complete, remainder = divmod(x.shape[0], ratio)
        cutoff = complete * ratio
        # Multi-token serving chunks start on a shared 128-token boundary, so
        # they are aligned for both compressor ratios (4 and 128). `positions`
        # remains dynamic: every full 128-token chunk reuses the same executable.
        start = positions[0]
        compressed_start = start // ratio
        if complete:
            grouped_kv = kv[:cutoff].reshape(complete, ratio, -1)
            grouped_score = (
                score[:cutoff].reshape(complete, ratio, -1) + weights[prefix + ".ape"][None]
            )
            if overlap:
                previous_kv = jnp.concatenate(
                    (values[None, :ratio, :dim], grouped_kv[:-1, :, :dim]), axis=0
                )
                previous_score = jnp.concatenate(
                    (scores[None, :ratio, :dim], grouped_score[:-1, :, :dim]),
                    axis=0,
                )
                pool_values = jnp.concatenate((previous_kv, grouped_kv[:, :, dim:]), axis=1)
                pool_scores = jnp.concatenate((previous_score, grouped_score[:, :, dim:]), axis=1)
            else:
                pool_values, pool_scores = grouped_kv, grouped_score
            pooled = round_bf16(jnp.sum(pool_values * jax.nn.softmax(pool_scores, axis=1), axis=1))
            pooled = rms_norm(pooled, weights[prefix + ".norm.weight"], config.eps)
            pooled = rope(
                pooled,
                start + jnp.arange(complete, dtype=jnp.int32) * ratio,
                config,
            )
            if index:
                pooled = hadamard_rotate(pooled)
                if trace is not None:
                    trace["index_compressed_before_quant"] = pooled
                pooled = activation_fp4_roundtrip(pooled)
            else:
                pooled = jnp.concatenate(
                    (
                        activation_fp8_roundtrip(pooled[:, : -config.rope_dim], 64),
                        pooled[:, -config.rope_dim :],
                    ),
                    axis=-1,
                )
            compressed = jax.lax.dynamic_update_slice(compressed, pooled, (compressed_start, 0))
            # Match the single-token state machine after the last complete
            # group. These scratch rows are what the next chunk/decode call
            # consumes when it reaches another compression boundary.
            last_kv, last_score = grouped_kv[-1], grouped_score[-1]
            if overlap:
                values = values.at[:ratio].set(last_kv)
                values = values.at[ratio:].set(last_kv)
                scores = scores.at[:ratio].set(last_score)
                scores = scores.at[ratio:].set(last_score)
            else:
                values, scores = last_kv, last_score
        if remainder:
            offset = ratio if overlap else 0
            values = values.at[offset : offset + remainder].set(kv[cutoff:])
            scores = scores.at[offset : offset + remainder].set(
                score[cutoff:] + weights[prefix + ".ape"][:remainder]
            )
        final = values, scores, compressed
    for suffix, value in zip((".kv", ".score", ".compressed"), final):
        cache[state_prefix + suffix] = value
    return cache


def _single_query_attention(q, kv, indices, sink, scale):
    """Fixed matrix/reduction shapes for both prefill and cached decode."""
    count = (indices.shape[-1] + 63) // 64 * 64
    indices = jnp.pad(indices, ((0, count - indices.shape[-1]),), constant_values=-1)
    initial = (
        jnp.full(q.shape[:-1], -1e30, jnp.float32),
        jnp.zeros(q.shape[:-1], jnp.float32),
        jnp.zeros(q.shape, jnp.float32),
    )

    def block(i, state):
        maximum, denominator, numerator = state
        ids = jax.lax.dynamic_slice_in_dim(indices, i * 64, 64, axis=0)
        keys = kv[jnp.maximum(ids, 0)]
        score = jnp.matmul(q, keys.T, preferred_element_type=jnp.float32) * scale
        valid = ids[None, :] >= 0
        score = jnp.where(valid, score, -1e30)
        next_max = jnp.maximum(maximum, jnp.max(score, axis=-1))
        alpha = jnp.exp(maximum - next_max)
        probability = jnp.where(valid, jnp.exp(score - next_max[..., None]), 0.0)
        numerator = numerator * alpha[..., None] + jnp.matmul(
            round_bf16(probability),
            keys,
            preferred_element_type=jnp.float32,
        )
        denominator = denominator * alpha + jnp.sum(probability, axis=-1)
        return next_max, denominator, numerator

    maximum, denominator, numerator = jax.lax.fori_loop(0, count // 64, block, initial)
    # Include the learned sink only in the denominator, with stable scaling.
    final_max = jnp.maximum(maximum, sink)
    rescale = jnp.exp(maximum - final_max)
    denom = denominator * rescale + jnp.exp(sink - final_max)
    return round_bf16(numerator * rescale[..., None] / denom[..., None])


def sparse_attention(q, kv, indices, sink, scale):
    """64-key streaming softmax with a batch-size-invariant query program.

    A sequential query map is deliberate for correctness bring-up. Batched
    dot lowering changed BF16 softmax rounding between prefill and decode.
    """
    return jax.lax.map(
        lambda row: _single_query_attention(row[0], kv, row[1], sink, scale), (q, indices)
    )


def official_head_collapse(streams, fn, scale, base, *, eps=1e-6, hc_eps=1e-6):
    """Official F32 projection -> RMS scalar -> gates, with no BF16 pre-round."""
    flat = streams.astype(jnp.float32).reshape(streams.shape[0], -1)
    inverse_rms = jax.lax.rsqrt(_fixed_tree_mean_last(flat * flat)[:, None] + eps)
    mixes = jnp.matmul(flat, fn.T, precision=jax.lax.Precision.HIGHEST) * inverse_rms
    pre = jax.nn.sigmoid(mixes * scale + base) + hc_eps
    values = [pre[:, i, None] * streams[:, i].astype(jnp.float32) for i in range(streams.shape[1])]
    return round_bf16(functools.reduce(jnp.add, values))


@functools.lru_cache(maxsize=8)
def compiled_head(config, mesh):
    def head(value, shared):
        collapsed = official_head_collapse(
            value,
            shared["hc_head_fn"],
            shared["hc_head_scale"],
            shared["hc_head_base"],
            eps=config.eps,
            hc_eps=config.hc_eps,
        )
        normalized = rms_norm(collapsed, shared["norm.weight"], config.eps)
        return jnp.matmul(normalized, shared["head.weight"].T, preferred_element_type=jnp.float32)

    return jax.jit(
        jax.shard_map(head, mesh=mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False)
    )


def attention(x, positions, weights, cache, config):
    cache = dict(cache)
    qr = rms_norm(linear(x, weights, "attn.wq_a"), weights["attn.q_norm.weight"], config.eps)
    q = linear(qr, weights, "attn.wq_b").reshape(x.shape[0], config.heads, config.head_dim)
    # The official final Q normalization operates in BF16, including square/mean.
    square = round_bf16(q.astype(jnp.float32) ** 2)
    variance = round_bf16(_fixed_tree_mean_last(square.astype(jnp.float32))[..., None])
    variance = round_bf16(variance.astype(jnp.float32) + config.eps)
    inverse_rms = round_bf16(jax.lax.rsqrt(variance.astype(jnp.float32)))
    q = rope(
        round_bf16(q.astype(jnp.float32) * inverse_rms.astype(jnp.float32)),
        positions,
        config,
    )
    kv = rms_norm(linear(x, weights, "attn.wkv"), weights["attn.kv_norm.weight"], config.eps)
    kv = rope(kv, positions, config)
    kv = jnp.concatenate(
        (activation_fp8_roundtrip(kv[:, : -config.rope_dim], 64), kv[:, -config.rope_dim :]),
        axis=-1,
    )
    cache["window"] = cache["window"].at[positions].set(kv)
    # Always expose the complete sliding window. A continuation/final chunk can
    # be shorter than the window while still needing keys from earlier chunks.
    # Invalid leading/future slots remain -1, preserving the same 64-key blocks
    # for whole prefill, chunked prefill, and decode.
    window_width = config.window
    window_ids = (
        jnp.maximum(positions[:, None] - config.window + 1, 0) + jnp.arange(window_width)[None, :]
    )
    indices = jnp.where(window_ids <= positions[:, None], window_ids, -1)
    all_kv = cache["window"]
    trace = {"qr": qr, "q": q, "kv": kv}
    if config.ratio:
        cache = compress(x, positions, weights, cache, config)
        compressed = cache["main.compressed"]
        available = (positions[:, None] + 1) // config.ratio
        if config.ratio == 4:
            cache = compress(x, positions, weights, cache, config, index=True, trace=trace)
            iq = linear(qr, weights, "attn.indexer.wq_b").reshape(
                x.shape[0], config.index_heads, config.index_dim
            )
            iq = hadamard_rotate(rope(iq, positions, config))
            trace["index_q_before_quant"] = iq
            iq = activation_fp4_roundtrip(iq)
            # Keep the official BF16 boundaries observable under fused JIT.
            # A plain BF16 -> FP32 cast pair can retain excess precision and
            # change tied index scores/top-k order (including on CPU).
            iw = round_bf16(
                linear(x, weights, "attn.indexer.weights_proj").astype(jnp.float32)
                * (config.index_dim**-0.5 * config.index_heads**-0.5)
            )
            score = round_bf16(
                jnp.einsum(
                    "mhd,td->mht", iq, cache["index.compressed"], preferred_element_type=jnp.float32
                )
            )
            score = round_bf16(
                jnp.maximum(score.astype(jnp.float32), 0) * iw.astype(jnp.float32)[..., None]
            )
            score = round_bf16(jnp.sum(score.astype(jnp.float32), axis=1)).astype(jnp.float32)
            score = jnp.where(jnp.arange(compressed.shape[0])[None, :] < available, score, -jnp.inf)
            _, selected = jax.lax.top_k(score, min(config.index_topk, compressed.shape[0]))
            trace.update(index_q=iq, index_score=score, index_selected=selected)
        else:
            selected = jnp.broadcast_to(
                jnp.arange(compressed.shape[0]), (x.shape[0], compressed.shape[0])
            )
        selected = jnp.where(selected < available, selected + config.max_context, -1)
        indices = jnp.concatenate((indices, selected), axis=-1)
        all_kv = jnp.concatenate((all_kv, compressed), axis=0)
    value = sparse_attention(q, all_kv, indices, weights["attn.attn_sink"], config.head_dim**-0.5)
    trace["attention_value"] = value
    value = rope(value, positions, config, inverse=True).reshape(x.shape[0], config.groups, -1)
    wo = weights["attn.wo_a.weight"]
    scale = weights["attn.wo_a.scale"]
    # Official wo_a uses BF16 math without activation quantization. Preserve that
    # path while retaining the original FP8 checkpoint between invocations.
    grouped = [
        low_bit_matmul(
            value[:, group],
            wo[group * config.o_rank : (group + 1) * config.o_rank],
            scale[group * config.o_rank // 128 : (group + 1) * config.o_rank // 128],
            weight_format="fp8",
            quantize_activation=False,
        )
        for group in range(config.groups)
    ]
    output = linear(jnp.concatenate(grouped, axis=-1), weights, "attn.wo_b")
    return output, cache, trace


def _position_aligned_router_logits(x, positions, weight, block_size=8):
    """Use the same router dot shape/lane for whole, chunked, and decode.

    TPU's native F32 dot can differ by an ulp when M changes. That tiny change
    can cross a routing/activation quantizer boundary and amplify across many
    layers. Each token is therefore placed in a fixed-size block according to
    its absolute position. Chunk boundaries are multiples of 128, while the
    dynamic offset also makes a one-token decode match its full-prefill lane.
    """
    if x.ndim != 2 or positions.shape != (x.shape[0],):
        raise ValueError("router requires one absolute position per token")
    blocks = (x.shape[0] + 2 * block_size - 2) // block_size
    padded_rows = blocks * block_size
    offset = positions[0] % block_size
    padded = jnp.zeros((padded_rows, x.shape[1]), x.dtype)
    padded = jax.lax.dynamic_update_slice(padded, x, (offset, 0))
    matrix = weight.astype(jnp.float32).T
    scores = jax.lax.map(
        lambda rows: jnp.matmul(
            rows.astype(jnp.float32),
            matrix,
            precision=jax.lax.Precision.HIGHEST,
        ),
        padded.reshape(blocks, block_size, x.shape[1]),
    ).reshape(padded_rows, weight.shape[0])
    return jax.lax.dynamic_slice(scores, (offset, 0), (x.shape[0], weight.shape[0]))


def route(x, positions, token_ids, weights, config):
    score = _position_aligned_router_logits(x, positions, weights["ffn.gate.weight"])
    score = jnp.sqrt(jax.nn.softplus(score))
    if config.hash_routing:
        indices = weights["ffn.gate.tid2eid"][token_ids]
    else:
        _, indices = jax.lax.top_k(score + weights["ffn.gate.bias"], config.active_experts)
    selected = jnp.take_along_axis(score, indices, axis=-1)
    # Keep the six-value normalization as an explicit, fixed reduction tree.
    # v5p otherwise chose a different vector reduction for M=15 vs M=271,
    # changing a routing weight by one FP32 ulp at a chunk boundary.
    denominator = functools.reduce(jnp.add, (selected[:, i] for i in range(config.active_experts)))[
        :, None
    ]
    return indices, selected / denominator * config.route_scale


def moe(x, positions, token_ids, weights, config):
    indices, routing = route(x, positions, token_ids, weights, config)
    routed = routed_fp4_experts(
        x,
        *(weights["experts." + key] for key in ("w1", "w3", "w2", "s1", "s3", "s2")),
        indices,
        routing,
        swiglu_limit=config.swiglu_limit,
    )
    gate = jnp.minimum(
        linear(x, weights, "ffn.shared_experts.w1").astype(jnp.float32), config.swiglu_limit
    )
    up = jnp.clip(
        linear(x, weights, "ffn.shared_experts.w3").astype(jnp.float32),
        -config.swiglu_limit,
        config.swiglu_limit,
    )
    hidden = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    shared = linear(hidden, weights, "ffn.shared_experts.w2")
    return (routed + shared.astype(jnp.float32)).astype(jnp.bfloat16), indices, routing


def layer_step(streams, positions, token_ids, weights, cache, *, config):
    trace = {}
    for sublayer in ("attn", "ffn"):
        residual = streams
        x, post, comb = mhc_pre_fused(
            streams,
            weights["hc_" + sublayer + "_fn"],
            weights["hc_" + sublayer + "_scale"],
            weights["hc_" + sublayer + "_base"],
            hc_mult=config.hc,
            sinkhorn_iters=config.sinkhorn_iters,
            norm_eps=config.eps,
            hc_eps=config.hc_eps,
            dot_precision=jax.lax.Precision.HIGHEST,
        )
        trace[sublayer + ".pre"] = x
        x = rms_norm(x, weights[sublayer + "_norm.weight"], config.eps)
        trace[sublayer + ".norm"] = x
        if sublayer == "attn":
            x, cache, attention_trace = attention(x, positions, weights, cache, config)
            trace.update({"attn." + key: value for key, value in attention_trace.items()})
        else:
            x, indices, routing = moe(x, positions, token_ids, weights, config)
            trace.update(expert_ids=indices, routing_weights=routing)
        trace[sublayer + ".operator"] = x
        streams = mhc_post_fused(x, residual, post, comb, precision=jax.lax.Precision.HIGHEST)
        trace[sublayer + ".post"] = streams
    return streams, cache, trace


def config_for_layer(source, layer_id, max_context):
    ratio = source["compress_ratios"][layer_id]
    yarn = source["rope_scaling"]
    return ReferenceConfig(
        hidden=source["hidden_size"],
        heads=source["num_attention_heads"],
        head_dim=source["head_dim"],
        rope_dim=source["qk_rope_head_dim"],
        groups=source["o_groups"],
        o_rank=source["o_lora_rank"],
        index_heads=source["index_n_heads"],
        index_dim=source["index_head_dim"],
        index_topk=source["index_topk"],
        window=source["sliding_window"],
        hc=source["hc_mult"],
        sinkhorn_iters=source["hc_sinkhorn_iters"],
        eps=source["rms_norm_eps"],
        hc_eps=source["hc_eps"],
        active_experts=source["num_experts_per_tok"],
        route_scale=source["routed_scaling_factor"],
        swiglu_limit=source["swiglu_limit"],
        max_context=max_context,
        ratio=ratio,
        hash_routing=layer_id < source["num_hash_layers"],
        rope_base=source["compress_rope_theta"] if ratio else source["rope_theta"],
        original_seq_len=yarn["original_max_position_embeddings"] if ratio else 0,
        rope_factor=yarn["factor"],
        beta_fast=yarn["beta_fast"],
        beta_slow=yarn["beta_slow"],
    )


def weight_specs(weights):
    return {key: P("tensor", None, None) if key.startswith("experts.") else P() for key in weights}


def load_layer(checkpoint, layer_id, mesh, *, include_experts=True):
    """Bounded host staging; never allocate a full BF16 layer/expert collection."""
    prefix = f"layers.{layer_id}."
    weights = {}
    for name in checkpoint.weight_map:
        if not name.startswith(prefix) or ".ffn.experts." in name:
            continue
        key = name[len(prefix) :]
        info = checkpoint.tensor_info(name)
        value = checkpoint.read_tensor(name)
        if info["dtype"] == "I64":
            if (
                not key.endswith("tid2eid")
                or np.any(value < 0)
                or np.any(value >= checkpoint.config["n_routed_experts"])
            ):
                raise ValueError(f"unexpected/out-of-range integer checkpoint field: {name}")
            value = value.astype(np.int32)  # Lossless routing-index conversion, not weights.
        weights[key] = jax.device_put(value, NamedSharding(mesh, P()))
    if include_experts:
        total = checkpoint.config["n_routed_experts"]
        devices = list(mesh.devices.flat)
        if total % len(devices):
            raise ValueError("expert count must divide the device mesh")
        local_count = total // len(devices)
        for projection in (1, 3, 2):
            arrays, scale_arrays = [], []
            for shard, device in enumerate(devices):
                experts = [
                    checkpoint.load_linear(f"layers.{layer_id}.ffn.experts.{expert}.w{projection}")
                    for expert in range(shard * local_count, (shard + 1) * local_count)
                ]
                data = np.stack([expert.data for expert in experts])
                scales = np.stack([expert.scales for expert in experts])
                arrays.append(jax.device_put(data, device).block_until_ready())
                scale_arrays.append(jax.device_put(scales, device).block_until_ready())
                del experts, data, scales
            sharding = NamedSharding(mesh, P("tensor", None, None))
            for key, parts in ((f"w{projection}", arrays), (f"s{projection}", scale_arrays)):
                shape = (total, *parts[0].shape[1:])
                weights["experts." + key] = jax.make_array_from_single_device_arrays(
                    shape, sharding, parts
                )
    jax.block_until_ready(weights)
    gc.collect()
    return weights


@functools.lru_cache(maxsize=16)
def compiled_layer(config, mesh):
    # Config, not weight arrays, is captured. Equivalent layer types share the
    # executable even though every layer has its own checkpoint tensors.
    def run(streams, positions, token_ids, weights, cache):
        return jax.shard_map(
            functools.partial(layer_step, config=config),
            mesh=mesh,
            in_specs=(P(), P(), P(), weight_specs(weights), P()),
            out_specs=(P(), P(), P()),
            check_vma=False,
        )(streams, positions, token_ids, weights, cache)

    return jax.jit(run)


class DeepSeekV4Reference:
    def __init__(self, directory, *, max_context=256, progress=print):
        if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
            raise RuntimeError("the full checkpoint milestone requires exactly four TPU devices")
        self.checkpoint = DeepSeekV4Checkpoint(directory)
        c = self.checkpoint.config
        if (
            c["expert_dtype"] != "fp4"
            or c["scoring_func"] != "sqrtsoftplus"
            or c["n_shared_experts"] != 1
        ):
            raise ValueError("unsupported checkpoint architecture/quantization")
        if max_context < 128 or max_context % 128:
            raise ValueError("reference max_context must be a positive multiple of 128")
        self.mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
        self.configs = [config_for_layer(c, i, max_context) for i in range(c["num_hidden_layers"])]
        self.progress = progress
        self.layers = []
        self.shared = {}
        self.position = 0
        self.cache = []

    def load(self):
        for name in self.checkpoint.weight_map:
            if name.startswith(("layers.", "mtp.")):
                continue
            self.shared[name] = jax.device_put(
                self.checkpoint.read_tensor(name), NamedSharding(self.mesh, P())
            )
        for layer_id, config in enumerate(self.configs):
            start = time.perf_counter()
            weights = load_layer(self.checkpoint, layer_id, self.mesh)
            self.layers.append(weights)
            self.progress(
                {
                    "event": "layer_loaded",
                    "layer": layer_id,
                    "ratio": config.ratio,
                    "seconds": time.perf_counter() - start,
                    "global_array_bytes": sum(x.nbytes for x in weights.values()),
                    "hbm": [device.memory_stats() for device in jax.devices()],
                }
            )
        self.reset()

    def reset(self):
        self.position = 0
        self.cache = [
            jax.device_put(empty_cache(config), NamedSharding(self.mesh, P()))
            for config in self.configs
        ]

    def step(self, token_ids, *, trace_layers=()):
        ids = np.asarray(token_ids, np.int32)
        if (
            ids.ndim != 1
            or not ids.size
            or np.any(ids < 0)
            or np.any(ids >= self.checkpoint.config["vocab_size"])
        ):
            raise ValueError("expected a nonempty single-request vector of valid token IDs")
        if self.position + ids.size > self.configs[0].max_context:
            raise ValueError("reference cache capacity exceeded")
        if self.position and ids.size != 1:
            raise ValueError(
                "reference supports one prefill followed by single-token cached decode"
            )
        ids = jax.device_put(ids, NamedSharding(self.mesh, P()))
        positions = jax.device_put(
            np.arange(self.position, self.position + ids.size, dtype=np.int32),
            NamedSharding(self.mesh, P()),
        )
        embedded = self.shared["embed.weight"][ids]
        streams = jnp.repeat(embedded[:, None, :], self.configs[0].hc, axis=1)
        traces = {}
        for layer_id, (config, weights) in enumerate(zip(self.configs, self.layers)):
            start = time.perf_counter()
            streams, self.cache[layer_id], trace = compiled_layer(config, self.mesh)(
                streams, positions, ids, weights, self.cache[layer_id]
            )
            jax.block_until_ready(streams)
            if layer_id in trace_layers:
                traces[layer_id] = jax.device_get(trace)
            self.progress(
                {
                    "event": "layer_executed",
                    "position": self.position,
                    "tokens": ids.size,
                    "layer": layer_id,
                    "seconds": time.perf_counter() - start,
                }
            )
        logits = compiled_head(self.configs[0], self.mesh)(streams[-1:], self.shared)
        jax.block_until_ready(logits)
        self.position += ids.size
        return logits, traces
