"""Batch-invariance and independent CPU checks for the V4 reference path."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    ReferenceConfig,
    _position_aligned_router_logits,
    compress,
    empty_cache,
    rms_norm,
    route,
    sparse_attention,
)
from sgl_jax.test.deepseek_v4_cpu_oracle import _sparse_attn


def test_attention_prefill_decode_are_bitwise_batch_invariant():
    rng = np.random.default_rng(1127)
    q = rng.normal(size=(17, 64, 512)).astype(ml_dtypes.bfloat16)
    kv = rng.normal(size=(256, 512)).astype(ml_dtypes.bfloat16)
    ids = np.arange(128, dtype=np.int32)[None] + np.arange(17, dtype=np.int32)[:, None]
    sink = rng.normal(size=(64,)).astype(np.float32)
    compiled = jax.jit(sparse_attention, static_argnums=(4,))
    batch = np.asarray(compiled(q, kv, ids, sink, 512**-0.5), np.float32)
    single = np.concatenate(
        [
            np.asarray(compiled(q[i : i + 1], kv, ids[i : i + 1], sink, 512**-0.5), np.float32)
            for i in range(len(q))
        ]
    )
    np.testing.assert_array_equal(batch, single)
    torch.set_num_threads(8)
    with torch.inference_mode():
        expected = (
            _sparse_attn(
                torch.from_numpy(q.astype(np.float32)).to(torch.bfloat16)[None],
                torch.from_numpy(kv.astype(np.float32)).to(torch.bfloat16)[None],
                torch.from_numpy(sink),
                torch.from_numpy(ids)[None],
                512**-0.5,
            )[0]
            .float()
            .numpy()
        )
    error = np.linalg.norm(batch - expected) / np.linalg.norm(expected)
    assert error < 0.005, f"CPU sparse attention NRMSE {error}"


def _compressor_weights(rng, config, *, index=False):
    dim = config.index_dim if index else config.head_dim
    prefix = "attn.indexer.compressor" if index else "attn.compressor"
    width = 2 * dim if config.ratio == 4 else dim
    return {
        prefix + ".wkv.weight": jnp.asarray(
            rng.normal(scale=0.05, size=(width, config.hidden)), jnp.bfloat16
        ),
        prefix + ".wgate.weight": jnp.asarray(
            rng.normal(scale=0.05, size=(width, config.hidden)), jnp.bfloat16
        ),
        prefix + ".ape": jnp.asarray(
            rng.normal(scale=0.05, size=(config.ratio, width)), jnp.float32
        ),
        prefix + ".norm.weight": jnp.asarray(
            rng.normal(loc=1.0, scale=0.05, size=(dim,)), jnp.bfloat16
        ),
    }


def test_compressor_state_is_bitwise_inherited_across_128_token_chunks():
    rng = np.random.default_rng(20260907)
    inputs = jnp.asarray(rng.normal(size=(271, 128)), jnp.bfloat16)
    for ratio, index in ((4, False), (4, True), (128, False)):
        config = ReferenceConfig(
            hidden=128,
            head_dim=128,
            rope_dim=64,
            index_dim=128,
            max_context=384,
            ratio=ratio,
        )
        weights = _compressor_weights(rng, config, index=index)
        whole = compress(
            inputs,
            jnp.arange(len(inputs), dtype=jnp.int32),
            weights,
            empty_cache(config),
            config,
            index=index,
        )
        chunked = empty_cache(config)
        for start, end in ((0, 128), (128, 256), (256, len(inputs))):
            chunked = compress(
                inputs[start:end],
                jnp.arange(start, end, dtype=jnp.int32),
                weights,
                chunked,
                config,
                index=index,
            )
        state_prefix = "index" if index else "main"
        for suffix in (".kv", ".score", ".compressed"):
            np.testing.assert_array_equal(
                np.asarray(chunked[state_prefix + suffix]),
                np.asarray(whole[state_prefix + suffix]),
            )


def test_router_dot_is_bitwise_invariant_across_chunk_and_decode_shapes():
    rng = np.random.default_rng(20260907)
    x = jnp.asarray(rng.normal(size=(272, 128)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(size=(256, 128)), jnp.bfloat16)
    positions = jnp.arange(len(x), dtype=jnp.int32)
    whole = _position_aligned_router_logits(x, positions, weight)
    chunked = jnp.concatenate(
        [
            _position_aligned_router_logits(x[:128], positions[:128], weight),
            _position_aligned_router_logits(x[128:256], positions[128:256], weight),
            _position_aligned_router_logits(x[256:271], positions[256:271], weight),
            _position_aligned_router_logits(x[271:], positions[271:], weight),
        ]
    )
    np.testing.assert_array_equal(np.asarray(chunked), np.asarray(whole))

    config = ReferenceConfig(hidden=128, active_experts=6, hash_routing=False)
    weights = {
        "ffn.gate.weight": weight,
        "ffn.gate.bias": jnp.asarray(rng.normal(size=(256,)), jnp.float32),
    }
    token_ids = jnp.arange(len(x), dtype=jnp.int32)
    whole_ids, whole_weights = route(x, positions, token_ids, weights, config)
    pieces = [
        route(x[a:b], positions[a:b], token_ids[a:b], weights, config)
        for a, b in ((0, 128), (128, 256), (256, 271), (271, 272))
    ]
    chunk_ids = jnp.concatenate([piece[0] for piece in pieces])
    chunk_weights = jnp.concatenate([piece[1] for piece in pieces])
    np.testing.assert_array_equal(np.asarray(chunk_ids), np.asarray(whole_ids))
    np.testing.assert_array_equal(np.asarray(chunk_weights), np.asarray(whole_weights))


def test_rms_norm_is_bitwise_invariant_across_token_shapes():
    rng = np.random.default_rng(20260907)
    x = jnp.asarray(rng.normal(size=(272, 512)), jnp.bfloat16)
    weight = jnp.asarray(rng.normal(loc=1.0, scale=0.1, size=(512,)), jnp.bfloat16)
    whole = rms_norm(x, weight, 1e-6)
    chunked = jnp.concatenate(
        [
            rms_norm(x[:128], weight, 1e-6),
            rms_norm(x[128:256], weight, 1e-6),
            rms_norm(x[256:271], weight, 1e-6),
            rms_norm(x[271:], weight, 1e-6),
        ]
    )
    np.testing.assert_array_equal(np.asarray(chunked), np.asarray(whole))
