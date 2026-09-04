"""Small DeepSeek-V4 vertical slice using the production TPU kernels.

This is intentionally not a serving model registration.  It keeps the kernel-
critical V4 dimensions while shrinking ranks, experts, layers, and context so
the model topology can be validated without a checkpoint.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, fields
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.kernels.csa.csa_memory import CSAKVPool, CSARecurrentStatePool
from sgl_jax.srt.kernels.csa.tune import (
    CSA_ATTENTION_DIM,
    CSA_COMPRESSION_RATIO,
    CSA_DEFAULT_PAGE_SIZE,
    CSA_DUAL_PROJECTION_DIM,
    CSA_HIDDEN_DIM,
    CSA_INDEX_DIM,
    CSA_INDEX_HEADS,
    CSA_INDEX_PROJECTED_DIM,
    CSA_MAIN_PROJECTED_DIM,
    CSA_ROPE_FREQUENCY_DIM,
    CSA_WINDOW_SIZE,
)
from sgl_jax.srt.kernels.mhc.mhc import (
    mhc_head_collapse_fused,
    mhc_post_fused,
    mhc_pre_fused,
    mix_hc_width,
)
from sgl_jax.srt.layers.attention.csa_backend import CSABackend
from sgl_jax.srt.layers.attention.hca_backend import HCABackend
from sgl_jax.srt.mem_cache.hca_allocator import HCAKVPoolAllocator
from sgl_jax.srt.mem_cache.hca_pool import HCAKVPool, HCARecurrentStatePool
from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

from . import deepseek_v4_oracle

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires TPU")

_SWA_COMPRESSION_RATIO = 0
_HCA_COMPRESSION_RATIO = 128
_STACK_NRMSE_LIMIT = 3e-2
_ORACLE_NRMSE_LIMIT = 2e-2
_LOCAL_NRMSE_LIMIT = 1e-2


@dataclass(frozen=True)
class _Config:
    compression_ratios: tuple[int, ...] = (
        _SWA_COMPRESSION_RATIO,
        _SWA_COMPRESSION_RATIO,
        CSA_COMPRESSION_RATIO,
        _HCA_COMPRESSION_RATIO,
        CSA_COMPRESSION_RATIO,
    )
    hidden: int = CSA_HIDDEN_DIM
    heads: int = CSA_INDEX_HEADS
    head_dim: int = CSA_ATTENTION_DIM
    hc_mult: int = 4
    q_rank: int = 128
    o_groups: int = 8
    o_rank: int = 64
    experts: int = 4
    top_k: int = 2
    expert_intermediate: int = 32
    vocab: int = 256
    window: int = CSA_WINDOW_SIZE
    rope_theta: float = 10_000.0
    swiglu_limit: float = 10.0
    residual_output_gain: float = 0.1
    hc_gate_scale: float = 0.1
    router_bias_span: float = 1.5
    norm_eps: float = 1e-6
    hc_eps: float = 1e-6
    sinkhorn_iters: int = 20

    def __post_init__(self):
        if (
            self.hidden != CSA_HIDDEN_DIM
            or self.heads != CSA_INDEX_HEADS
            or self.head_dim != CSA_ATTENTION_DIM
        ):
            raise ValueError(
                "the production HCA/CSA kernels require hidden=4096 and 64x512"
            )
        if set(self.compression_ratios) - {
            _SWA_COMPRESSION_RATIO,
            CSA_COMPRESSION_RATIO,
            _HCA_COMPRESSION_RATIO,
        }:
            raise ValueError("V4 attention layers must be SWA, CSA, or HCA")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Weights:
    embedding: jax.Array
    q_a: jax.Array
    q_b: jax.Array
    kv: jax.Array
    o_a: jax.Array
    o_b: jax.Array
    index_q: jax.Array
    index_weight: jax.Array
    csa_dual: jax.Array
    csa_main_ape: jax.Array
    csa_index_ape: jax.Array
    hca_kv: jax.Array
    hca_gate: jax.Array
    hca_ape: jax.Array
    route: jax.Array
    route_bias: jax.Array
    expert_gate: jax.Array
    expert_up: jax.Array
    expert_down: jax.Array
    shared_gate: jax.Array
    shared_up: jax.Array
    shared_down: jax.Array
    hc_attn_fn: jax.Array
    hc_attn_scale: jax.Array
    hc_attn_base: jax.Array
    hc_ffn_fn: jax.Array
    hc_ffn_scale: jax.Array
    hc_ffn_base: jax.Array
    hc_head_fn: jax.Array
    hc_head_scale: jax.Array
    hc_head_base: jax.Array
    lm_head: jax.Array
    cos: jax.Array
    sin: jax.Array


def _make_weights(config: _Config, max_context: int, seed: int = 20260904) -> _Weights:
    rng = np.random.default_rng(seed)

    def bf16(shape, fan_in, gain=1.0):
        values = gain * rng.standard_normal(shape, dtype=np.float32) / math.sqrt(fan_in)
        return jnp.asarray(values, jnp.bfloat16)

    def fp32(shape, fan_in):
        values = rng.standard_normal(shape, dtype=np.float32) / math.sqrt(fan_in)
        return jnp.asarray(values)

    d, h, hd = config.hidden, config.heads, config.head_dim
    group_width = h * hd // config.o_groups
    mix = mix_hc_width(config.hc_mult)
    rope_frequency = jnp.arange(CSA_ROPE_FREQUENCY_DIM, dtype=jnp.float32)
    rope_dim = 2 * CSA_ROPE_FREQUENCY_DIM
    inv_frequency = 1.0 / (config.rope_theta ** (2.0 * rope_frequency / rope_dim))
    angles = jnp.arange(max_context, dtype=jnp.float32)[:, None] * inv_frequency[None]
    return _Weights(
        embedding=bf16((config.vocab, d), d),
        q_a=bf16((d, config.q_rank), d),
        q_b=bf16((config.q_rank, h * hd), config.q_rank),
        kv=bf16((d, hd), d),
        o_a=bf16((config.o_groups, config.o_rank, group_width), group_width),
        o_b=bf16(
            (config.o_groups * config.o_rank, d),
            config.o_groups * config.o_rank,
            gain=config.residual_output_gain,
        ),
        index_q=bf16((config.q_rank, CSA_INDEX_HEADS * CSA_INDEX_DIM), config.q_rank),
        index_weight=fp32((d, CSA_INDEX_HEADS), d),
        csa_dual=bf16((d, CSA_DUAL_PROJECTION_DIM), d),
        csa_main_ape=fp32((CSA_COMPRESSION_RATIO, CSA_MAIN_PROJECTED_DIM), d),
        csa_index_ape=fp32((CSA_COMPRESSION_RATIO, CSA_INDEX_PROJECTED_DIM), d),
        hca_kv=bf16((hd, d), d),
        hca_gate=bf16((hd, d), d),
        hca_ape=fp32((_HCA_COMPRESSION_RATIO, hd), d),
        route=fp32((d, config.experts), d),
        route_bias=jnp.linspace(
            config.router_bias_span,
            -config.router_bias_span,
            config.experts,
            dtype=jnp.float32,
        ),
        expert_gate=bf16((config.experts, d, config.expert_intermediate), d),
        expert_up=bf16((config.experts, d, config.expert_intermediate), d),
        expert_down=bf16(
            (config.experts, config.expert_intermediate, d),
            config.expert_intermediate,
            gain=config.residual_output_gain,
        ),
        shared_gate=bf16((d, config.expert_intermediate), d),
        shared_up=bf16((d, config.expert_intermediate), d),
        shared_down=bf16(
            (config.expert_intermediate, d),
            config.expert_intermediate,
            gain=config.residual_output_gain,
        ),
        hc_attn_fn=fp32((mix, config.hc_mult * d), config.hc_mult * d),
        hc_attn_scale=jnp.full((3,), config.hc_gate_scale, jnp.float32),
        hc_attn_base=jnp.zeros((mix,), jnp.float32),
        hc_ffn_fn=fp32((mix, config.hc_mult * d), config.hc_mult * d),
        hc_ffn_scale=jnp.full((3,), config.hc_gate_scale, jnp.float32),
        hc_ffn_base=jnp.zeros((mix,), jnp.float32),
        hc_head_fn=fp32((config.hc_mult, config.hc_mult * d), config.hc_mult * d),
        hc_head_scale=jnp.full((1,), config.hc_gate_scale, jnp.float32),
        hc_head_base=jnp.zeros((config.hc_mult,), jnp.float32),
        lm_head=bf16((d, config.vocab), d),
        cos=jnp.cos(angles),
        sin=jnp.sin(angles),
    )


def _rms_norm(x, weight, eps):
    normalized = x.astype(jnp.float32) * jax.lax.rsqrt(
        jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + eps
    )
    return (normalized * weight).astype(x.dtype)


def _rope(x, positions, cos, sin, *, inverse=False):
    rope_dim = 2 * CSA_ROPE_FREQUENCY_DIM
    prefix, tail = x[..., :-rope_dim], x[..., -rope_dim:].astype(jnp.float32)
    pairs = tail.reshape(*tail.shape[:-1], CSA_ROPE_FREQUENCY_DIM, 2)
    table_shape = (
        (positions.shape[0],) + (1,) * (pairs.ndim - 3) + (CSA_ROPE_FREQUENCY_DIM,)
    )
    c = cos[positions].reshape(table_shape)
    s = sin[positions].reshape(table_shape)
    if inverse:
        s = -s
    real = pairs[..., 0] * c - pairs[..., 1] * s
    imag = pairs[..., 0] * s + pairs[..., 1] * c
    rotated = jnp.stack((real, imag), axis=-1).reshape(tail.shape).astype(x.dtype)
    return jnp.concatenate((prefix, rotated), axis=-1)


@functools.partial(jax.jit, static_argnames=("config",))
def _attention_inputs(x, positions, weights: _Weights, config: _Config):
    qr = _rms_norm(
        x @ weights.q_a, jnp.ones((config.q_rank,), jnp.float32), config.norm_eps
    )
    q = (qr @ weights.q_b).reshape(-1, config.heads, config.head_dim)
    q = _rms_norm(q, jnp.ones((config.head_dim,), jnp.float32), config.norm_eps)
    q = _rope(q, positions, weights.cos, weights.sin)
    kv = _rms_norm(
        x @ weights.kv, jnp.ones((config.head_dim,), jnp.float32), config.norm_eps
    )
    kv = _rope(kv, positions, weights.cos, weights.sin)
    index_q = (qr @ weights.index_q).reshape(-1, CSA_INDEX_HEADS, CSA_INDEX_DIM)
    index_q = _rope(index_q, positions, weights.cos, weights.sin)
    index_weights = x.astype(jnp.float32) @ weights.index_weight
    index_weights *= CSA_INDEX_DIM**-0.5 * CSA_INDEX_HEADS**-0.5
    return q, kv, index_q, index_weights


@functools.partial(jax.jit, static_argnames=("config",))
def _attention_output(x, positions, weights: _Weights, config: _Config):
    x = _rope(x, positions, weights.cos, weights.sin, inverse=True)
    grouped = jax.lax.reshape(
        x,
        (
            x.shape[0],
            config.o_groups,
            config.heads * config.head_dim // config.o_groups,
        ),
        out_sharding=P("data", None, None),
    )
    low_rank = jnp.einsum("tgd,grd->tgr", grouped, weights.o_a)
    low_rank = jax.lax.reshape(
        low_rank,
        (x.shape[0], config.o_groups * config.o_rank),
        out_sharding=P("data", None),
    )
    return (low_rank @ weights.o_b).astype(jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=("config", "layer_id"))
def _moe(x, token_ids, weights: _Weights, config: _Config, layer_id: int):
    scores = jnp.sqrt(jax.nn.softplus(x.astype(jnp.float32) @ weights.route))
    if layer_id < 3:
        first = jnp.mod(token_ids, config.experts)
        second = jnp.mod(
            first + 1 + jnp.mod(token_ids, config.experts - 1), config.experts
        )
        selected = jnp.stack((first, second), axis=-1)
    else:
        _, selected = jax.lax.top_k(scores + weights.route_bias, config.top_k)
    selected_scores = jnp.take_along_axis(scores, selected, axis=-1)
    selected_scores /= jnp.sum(selected_scores, axis=-1, keepdims=True)
    dispatch = (
        jnp.zeros_like(scores)
        .at[jnp.arange(x.shape[0])[:, None], selected]
        .set(selected_scores)
    )
    gate = jnp.einsum("td,edi->tei", x, weights.expert_gate).astype(jnp.float32)
    up = jnp.einsum("td,edi->tei", x, weights.expert_up).astype(jnp.float32)
    gate = jnp.minimum(gate, config.swiglu_limit)
    up = jnp.clip(up, -config.swiglu_limit, config.swiglu_limit)
    expert_hidden = jax.nn.silu(gate) * up
    expert_output = jnp.einsum(
        "tei,eid->ted", expert_hidden.astype(jnp.bfloat16), weights.expert_down
    )
    routed = jnp.sum(dispatch[..., None] * expert_output.astype(jnp.float32), axis=1)
    shared_gate = (x @ weights.shared_gate).astype(jnp.float32)
    shared_up = (x @ weights.shared_up).astype(jnp.float32)
    shared_hidden = jax.nn.silu(
        jnp.minimum(shared_gate, config.swiglu_limit)
    ) * jnp.clip(
        shared_up,
        -config.swiglu_limit,
        config.swiglu_limit,
    )
    shared = shared_hidden.astype(jnp.bfloat16) @ weights.shared_down
    return (routed + shared.astype(jnp.float32)).astype(jnp.bfloat16)


def _swa_attention(q, kv, positions, query_lengths, seq_lens, cache, sink):
    outputs = []
    token_start = 0
    updated = cache
    scale = q.shape[-1] ** -0.5
    for request, (query_length, sequence_length) in enumerate(
        zip(query_lengths, seq_lens, strict=True)
    ):
        token_stop = token_start + query_length
        prefix_length = sequence_length - query_length
        history_positions = jnp.arange(
            max(0, prefix_length - CSA_WINDOW_SIZE), prefix_length
        )
        history = cache[request, jnp.mod(history_positions, CSA_WINDOW_SIZE)]
        request_kv = kv[token_start:token_stop]
        keys = jnp.concatenate((history, request_kv), axis=0)
        key_positions = jnp.concatenate(
            (history_positions, positions[token_start:token_stop])
        )
        scores = (
            jnp.einsum(
                "thd,kd->thk",
                q[token_start:token_stop].astype(jnp.float32),
                keys.astype(jnp.float32),
            )
            * scale
        )
        query_positions = positions[token_start:token_stop]
        valid = (key_positions[None] <= query_positions[:, None]) & (
            key_positions[None] > query_positions[:, None] - CSA_WINDOW_SIZE
        )
        scores = jnp.where(valid[:, None], scores, -jnp.inf)
        maximum = jnp.maximum(jnp.max(scores, axis=-1), sink[None])
        probability = jnp.exp(scores - maximum[..., None])
        denominator = jnp.sum(probability, axis=-1) + jnp.exp(sink[None] - maximum)
        output = jnp.einsum("thk,kd->thd", probability, keys.astype(jnp.float32))
        outputs.append((output / denominator[..., None]).astype(jnp.bfloat16))
        retained = min(query_length, CSA_WINDOW_SIZE)
        retained_positions = query_positions[-retained:]
        updated = updated.at[request, jnp.mod(retained_positions, CSA_WINDOW_SIZE)].set(
            request_kv[-retained:]
        )
        token_start = token_stop
    return jnp.concatenate(outputs), updated


class _CSAPageTables:
    def __init__(self, requests, max_context):
        compressed_pages = math.ceil(
            max_context / CSA_COMPRESSION_RATIO / CSA_DEFAULT_PAGE_SIZE
        )
        self.compressed = (
            1 + np.arange(requests, dtype=np.int32)[:, None] * compressed_pages
        )
        self.compressed = (
            self.compressed + np.arange(compressed_pages, dtype=np.int32)[None]
        )
        window_pages = math.ceil(max_context / CSA_DEFAULT_PAGE_SIZE)
        self.window = np.broadcast_to(
            1 + np.arange(requests, dtype=np.int32)[:, None], (requests, window_pages)
        ).copy()

    def page_tables(self, request_ids, _seq_lens):
        return self.compressed[request_ids], self.window[request_ids]


class _Runtime:
    def __init__(self, config: _Config, requests: int, max_context: int, mesh):
        self.mesh = mesh
        hca_layers = tuple(
            layer
            for layer, ratio in enumerate(config.compression_ratios)
            if ratio == _HCA_COMPRESSION_RATIO
        )
        csa_layers = tuple(
            layer
            for layer, ratio in enumerate(config.compression_ratios)
            if ratio == CSA_COMPRESSION_RATIO
        )
        with jax.set_mesh(mesh):
            self.hca_state = HCARecurrentStatePool(hca_layers, requests, mesh)
            self.hca_pool = HCAKVPool(
                max(requests * max_context, 512),
                CSA_DEFAULT_PAGE_SIZE,
                jnp.bfloat16,
                len(hca_layers),
                mesh,
                max_num_requests=requests,
                max_context_len=max_context,
                layer_ids=hca_layers,
            )
            self.request_pool = HybridReqToTokenPool(
                requests, max_context, np.int32, self.hca_state, dp_size=1
            )
            self.hca_allocator = HCAKVPoolAllocator(self.hca_pool, self.request_pool)
            request_objects = [
                SimpleNamespace(
                    req_pool_idx=None,
                    recurrent_pool_idx=None,
                    is_chunked=0,
                    kv_committed_len=0,
                    dp_rank=0,
                )
                for _ in range(requests)
            ]
            self.request_ids = np.asarray(
                self.hca_allocator.alloc(request_objects), np.int32
            )
            self.csa_state = CSARecurrentStatePool(csa_layers, requests, mesh)
            self.csa_pool = CSAKVPool(
                max(requests * max_context, 512),
                CSA_DEFAULT_PAGE_SIZE,
                jnp.bfloat16,
                len(csa_layers),
                mesh,
                max_num_requests=requests,
                max_context_len=max_context,
                layer_ids=csa_layers,
            )
        self.hca = HCABackend(mesh=mesh)
        self.hca.allocator = self.hca_allocator
        self.csa = CSABackend(mesh=mesh)
        self.csa.page_table_provider = _CSAPageTables(requests, max_context)
        self.swa = {
            layer: jnp.zeros((requests, config.window, config.head_dim), jnp.bfloat16)
            for layer, ratio in enumerate(config.compression_ratios)
            if ratio == _SWA_COMPRESSION_RATIO
        }

    def prepare(self, mode, positions, query_lengths, prefix_lengths):
        seq_lens = np.asarray(query_lengths, np.int32) + np.asarray(
            prefix_lengths, np.int32
        )
        self.hca_allocator.ensure_compressed_capacity(self.request_ids, seq_lens)
        worker_batch = SimpleNamespace(
            forward_mode=mode,
            req_pool_indices=self.request_ids,
            seq_lens=seq_lens,
            positions=np.asarray(positions, np.int32),
            extend_seq_lens=(
                None
                if mode == ForwardMode.DECODE
                else np.asarray(query_lengths, np.int32)
            ),
            extend_prefix_lens=(
                None
                if mode == ForwardMode.DECODE
                else np.asarray(prefix_lengths, np.int32)
            ),
            recurrent_indices=self.request_pool.get_linear_recurrent_indices(
                self.request_ids
            ),
        )
        self.hca.forward_metadata = self.hca.get_forward_metadata(worker_batch)
        self.csa.forward_metadata = self.csa.get_forward_metadata(worker_batch)
        return seq_lens


class _MiniDeepseekV4:
    def __init__(self, config: _Config, weights: _Weights, runtime: _Runtime):
        self.config = config
        self.weights = weights
        self.runtime = runtime
        self.norm = jnp.ones((config.hidden,), jnp.float32)
        self.main_norm = jnp.ones((CSA_ATTENTION_DIM,), jnp.float32)
        self.index_norm = jnp.ones((CSA_INDEX_DIM,), jnp.float32)
        self.sink = jnp.zeros((config.heads,), jnp.float32)

    def _attention(self, layer_id, ratio, x, positions, query_lengths, seq_lens, mode):
        config, weights = self.config, self.weights
        q, kv, index_q, index_weights = _attention_inputs(x, positions, weights, config)
        layer = SimpleNamespace(layer_id=layer_id, scaling=config.head_dim**-0.5)
        forward_batch = SimpleNamespace(forward_mode=mode, positions=positions)
        if ratio == _SWA_COMPRESSION_RATIO:
            output, cache = _swa_attention(
                q,
                kv,
                positions,
                query_lengths,
                seq_lens,
                self.runtime.swa[layer_id],
                self.sink,
            )
            self.runtime.swa[layer_id] = cache
        elif ratio == CSA_COMPRESSION_RATIO:
            output, update = self.runtime.csa(
                q,
                kv,
                kv,
                layer,
                forward_batch,
                self.runtime.csa_pool,
                recurrent_state_pool=self.runtime.csa_state,
                compressor_input=x,
                dual_weight=weights.csa_dual,
                main_ape=weights.csa_main_ape,
                index_ape=weights.csa_index_ape,
                main_norm=self.main_norm,
                index_norm=self.index_norm,
                cos=weights.cos,
                sin=weights.sin,
                index_query=index_q,
                index_weights=index_weights,
                attention_sink=self.sink,
            )
            main, index, window, main_nope, main_rope, index_cache = update
            self.runtime.csa_state.replace_csa_states(layer_id, main, index)
            self.runtime.csa_pool.replace_compressor_buffers(
                layer_id, main_nope, main_rope, index_cache
            )
            self.runtime.csa_pool.window_buffer[
                self.runtime.csa_pool._layer_index(layer_id)
            ] = window
            output = output.reshape(-1, config.heads, config.head_dim)
        else:

            def put(value, spec):
                return jax.device_put(value, NamedSharding(self.runtime.mesh, spec))

            output, update = self.runtime.hca(
                put(q, P("data", "tensor", None)),
                put(kv, P("data", None)),
                put(kv, P("data", None)),
                layer,
                SimpleNamespace(
                    forward_mode=mode,
                    positions=put(positions, P("data")),
                ),
                self.runtime.hca_pool,
                recurrent_state_pool=self.runtime.hca_state,
                compressor_input=put(x, P("data", None)),
                wkv=put(weights.hca_kv, P(None, None)),
                wgate=put(weights.hca_gate, P(None, None)),
                ape=put(weights.hca_ape, P(None, None)),
                norm_weight=put(jnp.ones((config.head_dim,), jnp.bfloat16), P(None)),
                cos=put(weights.cos, P(None, None)),
                sin=put(weights.sin, P(None, None)),
                attention_sink=put(self.sink, P("tensor")),
            )
            state, window, compressed = update
            layer_index = self.runtime.hca_pool._layer_index(layer_id)
            self.runtime.hca_state.state_buffers[layer_index] = state
            self.runtime.hca_pool.window_buffer[layer_index] = window
            self.runtime.hca_pool.compressed_buffer[layer_index] = compressed
            output = output.reshape(-1, config.heads, config.head_dim)
        output = jax.sharding.reshard(output, P("data", None, None))
        return _attention_output(output, positions, weights, config)

    def _sublayer(self, streams, fn, scale, base, operation, *, trace, label):
        config = self.config
        streams = jax.sharding.reshard(streams, P(None, None, None))
        residual = streams
        x, post, comb = mhc_pre_fused(
            streams,
            fn,
            scale,
            base,
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.sinkhorn_iters,
            norm_eps=config.norm_eps,
            hc_eps=config.hc_eps,
        )
        normalized = _rms_norm(x, self.norm, config.norm_eps)
        block_output = operation(normalized)
        output = mhc_post_fused(block_output, residual, post, comb)
        if trace is not None:
            trace[f"{label}.pre"] = x
            trace[f"{label}.norm"] = normalized
            trace[f"{label}.operator"] = block_output
            trace[f"{label}.post"] = output
        return output

    def step(
        self, token_ids, query_lengths, prefix_lengths, mode, *, return_trace=False
    ):
        positions_np = np.concatenate(
            [
                np.arange(prefix, prefix + query, dtype=np.int32)
                for prefix, query in zip(prefix_lengths, query_lengths, strict=True)
            ]
        )
        positions = jnp.asarray(positions_np)
        token_ids = jnp.asarray(token_ids)
        seq_lens = self.runtime.prepare(
            mode, positions_np, query_lengths, prefix_lengths
        )
        streams = jnp.repeat(
            self.weights.embedding[token_ids][:, None],
            self.config.hc_mult,
            axis=1,
        )
        trace = {} if return_trace else None
        for layer_id, ratio in enumerate(self.config.compression_ratios):
            streams = self._sublayer(
                streams,
                self.weights.hc_attn_fn,
                self.weights.hc_attn_scale,
                self.weights.hc_attn_base,
                lambda x, layer_id=layer_id, ratio=ratio: self._attention(
                    layer_id, ratio, x, positions, query_lengths, seq_lens, mode
                ),
                trace=trace,
                label=f"layer{layer_id}.attention",
            )
            streams = self._sublayer(
                streams,
                self.weights.hc_ffn_fn,
                self.weights.hc_ffn_scale,
                self.weights.hc_ffn_base,
                lambda x, layer_id=layer_id: _moe(
                    x, token_ids, self.weights, self.config, layer_id
                ),
                trace=trace,
                label=f"layer{layer_id}.ffn",
            )
        streams = jax.sharding.reshard(streams, P(None, None, None))
        hidden = mhc_head_collapse_fused(
            streams,
            self.weights.hc_head_fn,
            self.weights.hc_head_scale,
            self.weights.hc_head_base,
            hc_mult=self.config.hc_mult,
            norm_eps=self.config.norm_eps,
            hc_eps=self.config.hc_eps,
        )
        if trace is not None:
            trace["head.collapse"] = hidden
        hidden = _rms_norm(hidden, self.norm, self.config.norm_eps)
        if trace is not None:
            trace["head.norm"] = hidden
        logits = hidden.astype(jnp.float32) @ self.weights.lm_head.astype(jnp.float32)
        if trace is not None:
            trace["logits"] = logits
            return logits, trace
        return logits


def _mesh():
    return jax.sharding.Mesh(
        np.asarray(jax.devices()[:1], object).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _model(config, weights, requests, max_context, mesh):
    return _MiniDeepseekV4(
        config, weights, _Runtime(config, requests, max_context, mesh)
    )


def _reference_inputs(config, weights):
    reference_config = deepseek_v4_oracle.Config(
        **{
            field.name: getattr(config, field.name)
            for field in fields(deepseek_v4_oracle.Config)
        }
    )
    reference_weights = {
        field.name: np.asarray(getattr(weights, field.name), np.float32)
        for field in fields(weights)
    }
    return reference_weights, reference_config


def _reference(token_ids, config, weights):
    reference_weights, reference_config = _reference_inputs(config, weights)
    return deepseek_v4_oracle.run(reference_weights, token_ids, reference_config)


def _local_reference(token_ids, config, weights, actual):
    reference_weights, reference_config = _reference_inputs(config, weights)
    return deepseek_v4_oracle.local_references(
        reference_weights, token_ids, reference_config, actual
    )


def _nrmse(actual, expected):
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    rmse = np.sqrt(np.mean(np.square(actual - expected)))
    reference_rms = np.sqrt(np.mean(np.square(expected)))
    return float(rmse / reference_rms)


def _assert_nrmse(actual, expected, limit=_STACK_NRMSE_LIMIT):
    error = _nrmse(actual, expected)
    assert error < limit, error


def _assert_reference_trace(actual, expected):
    for name, value in actual.items():
        if name.endswith(".post") or name.startswith("head.") or name == "logits":
            _assert_nrmse(value, expected[name], _ORACLE_NRMSE_LIMIT)


def _assert_local_trace(actual, expected):
    for name, value in actual.items():
        _assert_nrmse(value, expected[name], _LOCAL_NRMSE_LIMIT)


def test_deepseek_v4_prefill_and_decode_are_state_equivalent():
    config = _Config()
    max_context = 256
    mesh = _mesh()
    weights = _make_weights(config, max_context)
    token_ids = np.arange(133, dtype=np.int32) % config.vocab
    with jax.set_mesh(mesh):
        full_model = _model(config, weights, 1, max_context, mesh)
        full, full_trace = full_model.step(
            token_ids[:132], (132,), (0,), ForwardMode.EXTEND, return_trace=True
        )
        full_next, full_next_trace = full_model.step(
            token_ids[132:],
            (1,),
            (132,),
            ForwardMode.DECODE,
            return_trace=True,
        )
        split = _model(config, weights, 1, max_context, mesh)
        _, split_prefix_trace = split.step(
            token_ids[:128],
            (128,),
            (0,),
            ForwardMode.EXTEND,
            return_trace=True,
        )
        decoded = []
        decoded_trace = {name: [] for name in full_trace}
        for position in range(128, 133):
            logits, trace = split.step(
                token_ids[position : position + 1],
                (1,),
                (position,),
                ForwardMode.DECODE,
                return_trace=True,
            )
            decoded.append(logits)
            for name, value in trace.items():
                decoded_trace[name].append(value)
        decoded = jnp.concatenate(decoded)
        decoded_trace = {
            name: jnp.concatenate(values) for name, values in decoded_trace.items()
        }
        jax.block_until_ready((full, full_next, decoded))
    expected, expected_trace = _reference(token_ids, config, weights)
    full_path_trace = {
        name: np.concatenate((np.asarray(value), np.asarray(full_next_trace[name])))
        for name, value in full_trace.items()
    }
    split_path_trace = {
        name: np.concatenate((np.asarray(value), np.asarray(decoded_trace[name])))
        for name, value in split_prefix_trace.items()
    }
    _assert_reference_trace(full_path_trace, expected_trace)
    _assert_reference_trace(
        {name: np.asarray(value) for name, value in decoded_trace.items()},
        {name: value[-5:] for name, value in expected_trace.items()},
    )
    _assert_nrmse(
        np.concatenate((np.asarray(full), np.asarray(full_next))),
        expected,
        _ORACLE_NRMSE_LIMIT,
    )
    _assert_nrmse(decoded, expected[-5:], _ORACLE_NRMSE_LIMIT)
    _assert_local_trace(
        full_path_trace,
        _local_reference(token_ids, config, weights, full_path_trace),
    )
    _assert_local_trace(
        split_path_trace,
        _local_reference(token_ids, config, weights, split_path_trace),
    )
    for name, actual in decoded_trace.items():
        if name.endswith(".post") or name.startswith("head."):
            _assert_nrmse(actual[:4], full_trace[name][-4:])
    _assert_nrmse(decoded[:4], full[-4:])
    _assert_nrmse(decoded[4:], full_next)


def test_deepseek_v4_ragged_batch_matches_independent_requests():
    config = _Config()
    max_context = 256
    mesh = _mesh()
    weights = _make_weights(config, max_context, seed=20260905)
    prefixes = (
        np.arange(128, dtype=np.int32) % config.vocab,
        (np.arange(128, dtype=np.int32) + 17) % config.vocab,
    )
    suffixes = (
        np.asarray([211], np.int32),
        (np.arange(8, dtype=np.int32) + 93) % config.vocab,
    )
    with jax.set_mesh(mesh):
        batched = _model(config, weights, 2, max_context, mesh)
        batched.step(np.concatenate(prefixes), (128, 128), (0, 0), ForwardMode.EXTEND)
        combined, combined_trace = batched.step(
            np.concatenate(suffixes),
            (1, 8),
            (128, 128),
            ForwardMode.EXTEND,
            return_trace=True,
        )
        independent = []
        independent_trace = {name: [] for name in combined_trace}
        for prefix, suffix in zip(prefixes, suffixes, strict=True):
            model = _model(config, weights, 1, max_context, mesh)
            model.step(prefix, (128,), (0,), ForwardMode.EXTEND)
            logits, trace = model.step(
                suffix,
                (len(suffix),),
                (128,),
                ForwardMode.EXTEND,
                return_trace=True,
            )
            independent.append(logits)
            for name, value in trace.items():
                independent_trace[name].append(value)
        independent = jnp.concatenate(independent)
        independent_trace = {
            name: jnp.concatenate(values) for name, values in independent_trace.items()
        }
        jax.block_until_ready((combined, independent))
    expected = []
    expected_trace = {name: [] for name in combined_trace}
    for prefix, suffix in zip(prefixes, suffixes, strict=True):
        logits, trace = _reference(np.concatenate((prefix, suffix)), config, weights)
        expected.append(logits[-len(suffix) :])
        for name, value in trace.items():
            expected_trace[name].append(value[-len(suffix) :])
    expected = np.concatenate(expected)
    expected_trace = {
        name: np.concatenate(values) for name, values in expected_trace.items()
    }
    _assert_reference_trace(
        {name: np.asarray(value) for name, value in combined_trace.items()},
        expected_trace,
    )
    _assert_reference_trace(
        {name: np.asarray(value) for name, value in independent_trace.items()},
        expected_trace,
    )
    _assert_nrmse(combined, expected, _ORACLE_NRMSE_LIMIT)
    _assert_nrmse(independent, expected, _ORACLE_NRMSE_LIMIT)
    for name, actual in combined_trace.items():
        if name.endswith(".post") or name.startswith("head."):
            _assert_nrmse(actual, independent_trace[name])
    _assert_nrmse(combined, independent)
