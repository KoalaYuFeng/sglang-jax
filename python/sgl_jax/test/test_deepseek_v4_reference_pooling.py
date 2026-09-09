"""The oracle must not invent a phase-dependent BF16 pooling result."""

import jax
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.model_executor import deepseek_v4_reference as ref


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires the TPU reduction lowering")
def test_reference_prefill_decode_pooling_matches_high_precision(monkeypatch):
    # One real channel from layer 2 / position 7939, serialized as FP32 bits.
    values = np.asarray(
        [
            0xBEAEE190,
            0xBE8D77CD,
            0x3D7CA8F8,
            0x3E56D2CF,
            0x3D4C6B62,
            0xBDF4D456,
            0x3DF316E9,
            0x3B950D66,
        ],
        np.uint32,
    ).view(np.float32)
    scores = np.asarray(
        [
            0x3EC8432F,
            0x3F5B3614,
            0x3C2D6AF0,
            0xBDA1F6A0,
            0xBF4D4CDA,
            0xBFA0F32E,
            0xBF757BE2,
            0xBF6421F2,
        ],
        np.uint32,
    ).view(np.float32)
    config = ref.ReferenceConfig(hidden=4096, head_dim=512, ratio=4, max_context=128)
    # Isolate pooling at its public compressor boundary. Identity downstream
    # transforms expose the BF16 pool without hiding it behind norm/RoPE/QAT.
    monkeypatch.setattr(ref, "rms_norm", lambda x, weight, eps: x)
    monkeypatch.setattr(ref, "rope", lambda x, positions, config: x)
    monkeypatch.setattr(ref, "activation_fp8_roundtrip", lambda x, block_size: x)
    x = np.zeros((8, config.hidden), ml_dtypes.bfloat16)
    weights = {}
    for suffix, data in (("wkv.weight", values), ("wgate.weight", scores)):
        weight = np.zeros((1024, config.hidden), ml_dtypes.bfloat16)
        for token in range(8):
            # Three BF16 terms exactly reconstruct each captured FP32 value;
            # this keeps real BF16-input/weight projection, not a mocked dot.
            remaining = np.float32(data[token])
            for part in range(3):
                value = np.asarray(remaining).astype(ml_dtypes.bfloat16)
                feature = 3 * token + part
                weight[(0 if token < 4 else 512) + 39, feature] = value
                x[token, feature] = 1
                remaining = np.float32(remaining - np.float32(value))
            assert remaining == 0
        weights["attn.compressor." + suffix] = weight
    weights["attn.compressor.ape"] = np.zeros((4, 1024), np.float32)
    weights["attn.compressor.norm.weight"] = np.ones(512, ml_dtypes.bfloat16)

    @jax.jit
    def run(x, pos, weights, cache):
        return ref.compress(x, pos, weights, dict(cache), config)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1], object), ("tensor",))
    with jax.set_mesh(mesh):
        whole = run(x, np.arange(8, dtype=np.int32), weights, ref.empty_cache(config))
        decoded = run(x[:4], np.arange(4, dtype=np.int32), weights, ref.empty_cache(config))
        for token in range(4, 8):
            decoded = run(x[token : token + 1], np.asarray([token], np.int32), weights, decoded)
        whole, decoded = jax.device_get((whole, decoded))
    for suffix in ("kv", "score"):
        np.testing.assert_array_equal(whole["main." + suffix], decoded["main." + suffix])
    probability = np.exp(scores.astype(np.float64) - np.max(scores))
    probability /= probability.sum()
    expected = np.asarray(np.sum(values.astype(np.float64) * probability)).astype(
        ml_dtypes.bfloat16
    )
    assert whole["main.compressed"][1, 39] == expected
    assert decoded["main.compressed"][1, 39] == expected
    np.testing.assert_array_equal(whole["main.compressed"], decoded["main.compressed"])


@pytest.mark.parametrize("ratio,index", [(4, False), (4, True), (128, False)])
def test_reference_compressor_prefill_decode_state_agrees(ratio, index):
    rng = np.random.default_rng(20260907)
    config = ref.ReferenceConfig(
        hidden=128, head_dim=128, index_dim=128, ratio=ratio, max_context=256
    )
    dim = config.index_dim if index else config.head_dim
    width = 2 * dim if ratio == 4 else dim
    prefix = "attn.indexer.compressor" if index else "attn.compressor"
    # Exact dyadic dot products isolate pooling/state inheritance from the
    # separately tested GEMM-versus-GEMV FP32 accumulation-order difference.
    weights = {
        prefix + suffix: (rng.integers(-8, 9, size=(width, config.hidden)) / 32).astype(
            ml_dtypes.bfloat16
        )
        for suffix in (".wkv.weight", ".wgate.weight")
    }
    weights[prefix + ".ape"] = rng.normal(scale=0.05, size=(ratio, width)).astype(np.float32)
    weights[prefix + ".norm.weight"] = rng.normal(loc=1.0, scale=0.05, size=dim).astype(
        ml_dtypes.bfloat16
    )
    x = (rng.integers(-8, 9, size=(128 + ratio, config.hidden)) / 8).astype(ml_dtypes.bfloat16)

    @jax.jit
    def run(x, pos, weights, cache):
        return ref.compress(x, pos, weights, dict(cache), config, index=index)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1], object), ("tensor",))
    with jax.set_mesh(mesh):
        whole = run(x, np.arange(len(x), dtype=np.int32), weights, ref.empty_cache(config))
        decoded = run(x[:128], np.arange(128, dtype=np.int32), weights, ref.empty_cache(config))
        for token in range(128, len(x)):
            decoded = run(x[token : token + 1], np.asarray([token], np.int32), weights, decoded)
        whole, decoded = jax.device_get((whole, decoded))
    state = "index" if index else "main"
    for suffix in (".kv", ".score", ".compressed"):
        np.testing.assert_array_equal(whole[state + suffix], decoded[state + suffix])
