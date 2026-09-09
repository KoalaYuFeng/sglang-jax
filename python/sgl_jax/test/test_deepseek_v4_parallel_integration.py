"""V4-only head TP and pure-decode projection adapters; no scheduler changes."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from sgl_jax.srt.kernels.deepseek_v4 import csa
from sgl_jax.srt.kernels.deepseek_v4.compressor import compress
from sgl_jax.srt.kernels.deepseek_v4.numerics import V4LayerConfig
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.model_loader.deepseek_v4_native import (
    HEAD_SHARDED_WEIGHTS,
    load_layer,
    weight_specs,
)
from sgl_jax.srt.models.deepseek_v4 import (
    _boolean_option,
    attention_uses_tp,
    local_attention_config,
    use_batched_csa_decode,
)
from sgl_jax.test.kernels.test_deepseek_v4_reference import _compressor_weights
from sgl_jax.test.test_deepseek_v4_paged import make_batch, make_cache

TPU = pytest.mark.skipif(jax.default_backend() != "tpu", reason="real Mosaic lowering required")


def assert_bits(expected, actual):
    a, b = np.asarray(expected), np.asarray(actual)
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8)
    )


def test_attention_tp_partitions_only_complete_head_weights_and_their_scales():
    weights = {
        key: np.zeros((64,) if key.endswith("sink") else (256, 128), np.uint8)
        for key in HEAD_SHARDED_WEIGHTS
    }
    weights.update(
        {
            "experts.w1": np.zeros((8, 4, 4), np.uint8),
            "attn.wo_b.weight": np.zeros((256, 128), np.uint8),
            "attn.indexer.wq_b.weight": np.zeros((256, 128), np.uint8),
            "attn.compressor.wkv.weight": np.zeros((256, 128), np.uint8),
        }
    )
    for enabled in (False, True):
        specs = weight_specs(weights, attention_tp=enabled)
        for name, value in weights.items():
            expected = (
                P("tensor", None, None)
                if name.startswith("experts.")
                else P("tensor", *([None] * (value.ndim - 1)))
                if enabled and name in HEAD_SHARDED_WEIGHTS
                else P()
            )
            assert specs[name] == expected


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_only_csa_hca_localize_heads_without_changing_index_or_state_config(ratio):
    config = V4LayerConfig(ratio=ratio)
    assert local_attention_config(config, False) is config
    local = local_attention_config(config, True)
    assert attention_uses_tp(config, True) == (ratio != 0)
    assert local == (replace(config, heads=16, groups=2) if ratio else config)
    assert local.index_heads == 64 and local.hidden == 4096


def test_head_tp_and_boolean_config_reject_ambiguous_inputs():
    with pytest.raises(ValueError, match="complete heads"):
        local_attention_config(V4LayerConfig(ratio=4, groups=3), True)
    assert _boolean_option(SimpleNamespace(), "v4_attention_tp") is False
    for value in ("true", "false", 1, None):
        with pytest.raises(ValueError, match="boolean"):
            _boolean_option(SimpleNamespace(v4_attention_tp=value), "v4_attention_tp")


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.DECODE])
def test_batched_projection_is_selected_only_by_pure_decode_mode(mode):
    assert use_batched_csa_decode(True, mode) == (mode == ForwardMode.DECODE)
    assert use_batched_csa_decode(False, mode) is False


def test_native_loader_shards_raw_weight_bytes_and_scales_together():
    tensors = {
        "attn.wq_b.weight": np.arange(512, dtype=np.uint8).reshape(128, 4),
        "attn.wq_b.scale": np.arange(128, dtype=np.uint8).reshape(4, 32),
        "attn.attn_sink": np.arange(64, dtype=np.float32),
        "attn.wo_b.weight": np.arange(128, dtype=np.uint8).reshape(32, 4),
    }
    checkpoint = SimpleNamespace(
        weight_map={"layers.2." + key: "unused" for key in tensors},
        tensor_info=lambda _: {"dtype": "U8"},
        read_tensor=lambda name: tensors[name.removeprefix("layers.2.")],
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",))
    with jax.set_mesh(mesh):
        result = load_layer(checkpoint, 2, mesh, include_experts=False, attention_tp=True)
    for name, value in result.items():
        assert_bits(tensors[name], value)
        assert value.sharding.spec == weight_specs(tensors, attention_tp=True)[name]


def decode_fixture(batch_size, offset):
    pages = [list(range(1 + 64 * i, 65 + 64 * i)) for i in range(batch_size)]
    prefixes = [7976 + offset] * batch_size
    counts = [1] * batch_size
    if batch_size == 4:
        prefixes, counts = [7976 + offset, 0, 8104 + offset, 8191], [1, 0, 1, 1]
    batch = make_batch(prefixes, counts, pages=pages, padding=batch_size - sum(counts), decode=True)
    if batch_size == 4:
        for name in ("positions", "out_cache_loc"):
            packed = getattr(batch, name).copy()
            packed[2:4] = packed[1:3]
            packed[1] = -1 if name == "out_cache_loc" else 0
            setattr(batch, name, packed)
    return V4PagedBackend(max_context=8192).get_forward_metadata(batch)


@TPU
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batched_decode_adapter_matches_original_gemv_for_every_position_lane(dim, batch_size):
    rng = np.random.default_rng(7979)
    config = V4LayerConfig(ratio=4, max_context=8192)
    weights = [
        jnp.asarray(rng.normal(scale=0.02, size=(2 * dim, 4096)), jnp.bfloat16) for _ in range(2)
    ]
    x = jnp.asarray(rng.normal(size=(batch_size, 4096)), jnp.bfloat16)
    if batch_size == 4:
        x = x.at[1].set(jnp.nan)  # Invalid rows must not poison state/projection.
    baseline = jax.jit(lambda x, a, b, m: csa.project(x, a, b, m, config))
    candidate = jax.jit(lambda x, a, b, m: csa.project(x, a, b, m, config, decode_batch=True))
    for offset in range(8):
        metadata = decode_fixture(batch_size, offset)
        expected, actual = baseline(x, *weights, metadata), candidate(x, *weights, metadata)
        live = metadata.token_valid
        for a, b in zip(expected, actual, strict=True):
            assert_bits(np.asarray(a)[live], np.asarray(b)[live])
            np.testing.assert_array_equal(np.asarray(b)[~live], 0)
    hlo = candidate.lower(x, *weights, metadata).compile().as_text()
    calls = [
        line
        for line in hlo.splitlines()
        if "custom-call(" in line and "csa-compressor-project" in line
    ]
    assert len(calls) == 1 and "decode-batched" in calls[0]


@TPU
@pytest.mark.parametrize("dim", [128, 512])
def test_batched_decode_projection_matches_every_physical_chip(dim):
    if jax.device_count() != 4:
        pytest.skip("four physical devices required")
    rng = np.random.default_rng(dim)
    config = V4LayerConfig(ratio=4, max_context=8192)
    x = rng.normal(size=(4, 4096)).astype(jnp.bfloat16)
    weights = [rng.normal(scale=0.02, size=(2 * dim, 4096)).astype(jnp.bfloat16) for _ in range(2)]
    metadata = decode_fixture(4, 3)
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    with jax.set_mesh(mesh):
        inputs = jax.tree.map(
            lambda value: jax.device_put(value, NamedSharding(mesh, P())), (x, *weights, metadata)
        )
        run = jax.jit(
            jax.shard_map(
                lambda x, a, b, m: csa.project(x, a, b, m, config, decode_batch=True),
                mesh=mesh,
                in_specs=(P(), P(), P(), P()),
                out_specs=(P(), P()),
                check_vma=False,
            )
        )
        actual = run(*inputs)
        for value in actual:
            assert len(value.addressable_shards) == 4
            for shard in value.addressable_shards:
                assert_bits(value, shard.data)


@TPU
@pytest.mark.parametrize("index", [False, True])
def test_decode_projection_preserves_forked_compressor_state_through_r4_boundary(index):
    rng = np.random.default_rng(8023)
    config = V4LayerConfig(ratio=4, max_context=384)
    weights = _compressor_weights(rng, config, index=index)
    backend = V4PagedBackend(max_context=384)
    prefill = jax.jit(
        lambda x, c, w, m: compress(x, w, c, config, m, index=index, csa_backend="pallas")
    )
    prefix = prefill(
        jnp.asarray(rng.normal(size=(128, 4096)), jnp.bfloat16),
        make_cache(config),
        weights,
        backend.get_forward_metadata(make_batch([0], [128])),
    )
    baseline = jax.jit(
        lambda x, c, w, m: compress(x, w, c, config, m, index=index, csa_backend="pallas")
    )
    candidate = jax.jit(
        lambda x, c, w, m: compress(
            x, w, c, config, m, index=index, csa_backend="pallas", csa_decode_batch=True
        )
    )
    expected, actual = prefix, prefix
    for position in range(128, 133):
        metadata = backend.get_forward_metadata(
            make_batch(
                [position, position],
                [1, 1],
                slots=[0, 3],
                pages=[[1, 2, 3], [1, 4, 5]],
                decode=True,
            )
        )
        x = jnp.asarray(rng.normal(size=(2, 4096)), jnp.bfloat16)
        expected = baseline(x, expected, weights, metadata)
        actual = candidate(x, actual, weights, metadata)
        for name in expected:
            assert_bits(expected[name], actual[name])
