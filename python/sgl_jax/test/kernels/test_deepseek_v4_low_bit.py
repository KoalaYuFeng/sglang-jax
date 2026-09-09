"""Independent byte/NumPy oracles and actual four-chip low-bit TPU execution."""

import functools
import json
import struct

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.low_bit.formats import (
    activation_fp4_roundtrip,
    activation_fp8_roundtrip,
    decode_e8m0,
    decode_fp4_codes,
    decode_fp8,
    dequantize_fp4,
    dequantize_fp8,
    encode_fp4_codes,
    encode_fp8,
    quantize_activation_fp8,
    round_bf16,
    unpack_fp4,
)
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul
from sgl_jax.srt.kernels.low_bit.moe import routed_fp4_experts
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint

_FP4_LUT = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


@pytest.fixture(autouse=True)
def isolated_mesh():
    with jax.set_mesh(Mesh(np.asarray(jax.devices()[:1]), ("lowbit_test",))):
        yield


def _bf16(x):
    return np.asarray(x, dtype=ml_dtypes.bfloat16).astype(np.float32)


def _scale(raw):
    # Arithmetic CPU oracle, independent from the device bit implementation.
    return np.exp2(np.asarray(raw, np.float64) - 127).astype(np.float32)


def _reference_weight(raw, scales, fmt):
    if fmt == "bf16":
        return raw.astype(np.float32)
    if fmt == "fp4":
        values = np.empty((raw.shape[0], raw.shape[1] * 2), np.float32)
        values[:, 0::2] = _FP4_LUT[raw & 15]
        values[:, 1::2] = _FP4_LUT[raw >> 4]
        return _bf16(values * np.repeat(_scale(scales), 32, axis=1))
    values = raw.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    expanded = np.repeat(np.repeat(_scale(scales), 128, axis=0), 128, axis=1)
    return _bf16(values * expanded[: raw.shape[0], : raw.shape[1]])


def _reference_activation(x):
    grouped = x.astype(np.float32).reshape(x.shape[0], -1, 128)
    amax = np.maximum(np.max(np.abs(grouped), axis=-1), np.float32(1e-4))
    scale = np.exp2(np.ceil(np.log2(amax.astype(np.float64) / 448))).astype(np.float32)
    q = np.clip(grouped / scale[..., None], -448, 448).astype(ml_dtypes.float8_e4m3fn)
    return _bf16(q.astype(np.float32) * scale[..., None]).reshape(x.shape)


def _fixture_weight(fmt, n=512, k=512, seed=41):
    rng = np.random.default_rng(seed)
    if fmt == "fp4":
        raw = rng.integers(0, 256, (n, k // 2), np.uint8)
        scales = rng.integers(119, 125, (n, k // 32), np.uint8)
    elif fmt == "fp8":
        raw = (rng.normal(size=(n, k)) * 5).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
        scales = rng.integers(119, 125, ((n + 127) // 128, k // 128), np.uint8)
    else:
        raw = (rng.normal(size=(n, k)) * 0.05).astype(ml_dtypes.bfloat16)
        scales = None
    return raw, scales


def _write_checkpoint(directory, fmt, raw, scales):
    """Minimal spec-compatible safetensors fixture; no production writer reused."""
    prefix = "layers.0.ffn.experts.0.w1" if fmt == "fp4" else "layers.0.attn.wq_a"
    tensors = [(prefix + ".weight", {"fp4": "I8", "fp8": "F8_E4M3", "bf16": "BF16"}[fmt], raw)]
    if scales is not None:
        tensors.append((prefix + ".scale", "F8_E8M0", scales))
    offset, header, payloads = 0, {}, []
    for name, dtype, data in tensors:
        payload = data.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(data.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        payloads.append(payload)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    filename = "model-00001-of-00001.safetensors"
    (directory / filename).write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads)
    )
    (directory / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "expert_dtype": "fp4"})
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: filename for name in header}})
    )
    return prefix


def _assert_close(actual, expected):
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    assert np.all(np.isfinite(actual))
    error = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
    assert error < 5e-3, f"NRMSE={error}"
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=2e-2)


def test_fp4_all_codes_and_all_packed_bytes():
    codes = jnp.arange(16, dtype=jnp.uint8)
    actual = np.asarray(jax.jit(decode_fp4_codes)(codes))
    np.testing.assert_array_equal(
        actual.view(np.uint16), _FP4_LUT.astype(ml_dtypes.bfloat16).view(np.uint16)
    )
    packed = np.arange(256, dtype=np.uint8).reshape(8, 32)
    expected = np.stack((_FP4_LUT[packed & 15], _FP4_LUT[packed >> 4]), axis=-1).reshape(8, 64)
    np.testing.assert_array_equal(np.asarray(jax.jit(unpack_fp4)(packed)), expected)


def test_e8m0_all_codes_and_fp8_all_finite_codes():
    raw = np.arange(256, dtype=np.uint8)
    actual = np.asarray(jax.jit(decode_e8m0)(raw))
    expected = raw.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    actual = np.asarray(jax.jit(decode_fp8)(raw))
    expected = raw.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(actual[np.isfinite(expected)], expected[np.isfinite(expected)])


def test_dynamic_activation_fp8_matches_cpu():
    rng = np.random.default_rng(52)
    x = _bf16(rng.normal(size=(9, 512)).astype(np.float32))
    x[0] = 0
    actual = jax.jit(activation_fp8_roundtrip)(jnp.asarray(x, jnp.bfloat16))
    np.testing.assert_array_equal(np.asarray(actual, np.float32), _reference_activation(x))
    raw, scales = jax.jit(quantize_activation_fp8)(jnp.asarray(x, jnp.bfloat16))
    assert raw.dtype == jnp.uint8 and scales.dtype == jnp.uint8
    assert raw.shape == (9, 512) and scales.shape == (9, 4)


def test_bf16_rounding_boundary_survives_fusion():
    bits = np.arange(0x3B00, 0x4400, dtype=np.uint32) << 16
    values = bits.view(np.float32)
    midpoints = (values[1:] + values[:-1]) / 2
    values = np.concatenate(
        (values, midpoints, np.nextafter(midpoints, -np.inf), np.nextafter(midpoints, np.inf))
    )
    values = np.concatenate((values, -values))
    actual = np.asarray(jax.jit(lambda x: round_bf16(x).astype(jnp.float32))(values))
    np.testing.assert_array_equal(actual, _bf16(values))


def test_fp8_encoding_rounding_ties_and_saturation():
    positive = np.arange(127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    midpoints = (positive[:-1] + positive[1:]) / 2
    values = np.concatenate(
        (
            positive,
            midpoints,
            np.nextafter(midpoints, -np.inf),
            np.nextafter(midpoints, np.inf),
            [449, 1000],
        )
    ).astype(np.float32)
    values = np.concatenate((values, -values))
    expected = np.clip(values, -448, 448).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    actual = np.asarray(jax.jit(encode_fp8)(values))
    np.testing.assert_array_equal(actual, expected)


def test_fp4_activation_rounding_and_indexer_quantization():
    positive = _FP4_LUT[:8]
    boundaries = (positive[:-1] + positive[1:]) / 2
    values = np.concatenate(
        (
            positive,
            boundaries,
            np.nextafter(boundaries, -np.inf),
            np.nextafter(boundaries, np.inf),
            [7, 100],
        )
    ).astype(np.float32)
    values = np.concatenate((values, -values))
    distance = np.abs(np.abs(values[:, None]) - positive)
    # Reverse even/odd priority at exact ties, independently of threshold encoding.
    priority = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    expected = priority[np.argmin(distance[:, priority], axis=-1)].astype(np.uint8)
    expected |= np.signbit(values).astype(np.uint8) * 8
    np.testing.assert_array_equal(np.asarray(jax.jit(encode_fp4_codes)(values)), expected)
    x = _bf16(np.random.default_rng(113).normal(size=(9, 128)).astype(np.float32))
    x[0] = 0
    grouped = x.reshape(9, 4, 32)
    amax = np.maximum(np.abs(grouped).max(axis=-1, keepdims=True), np.float32(6 * 2.0**-126))
    scale = np.exp2(np.ceil(np.log2(amax.astype(np.float64) / 6))).astype(np.float32)
    normalized = grouped / scale
    distance = np.abs(np.abs(normalized[..., None]) - positive)
    codes = priority[np.argmin(distance[..., priority], axis=-1)].astype(np.uint8)
    codes |= np.signbit(normalized).astype(np.uint8) * 8
    expected = _bf16(_FP4_LUT[codes] * scale).reshape(x.shape)
    actual = np.asarray(jax.jit(activation_fp4_roundtrip)(jnp.asarray(x, jnp.bfloat16)), np.float32)
    np.testing.assert_array_equal(actual, expected)


def test_fp8_kv_quantization_64_column_blocks():
    x = _bf16(np.random.default_rng(117).normal(size=(9, 448)).astype(np.float32))
    grouped = x.reshape(9, 7, 64)
    amax = np.maximum(np.abs(grouped).max(axis=-1, keepdims=True), np.float32(1e-4))
    scale = np.exp2(np.ceil(np.log2(amax.astype(np.float64) / 448))).astype(np.float32)
    expected = _bf16((grouped / scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale)
    actual = jax.jit(functools.partial(activation_fp8_roundtrip, block_size=64))(
        jnp.asarray(x, jnp.bfloat16)
    )
    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected.reshape(x.shape))


@pytest.mark.parametrize("fmt", ("fp4", "fp8", "bf16"))
def test_checkpoint_load_preserves_bytes_and_shards(tmp_path, fmt):
    raw, scale = _fixture_weight(fmt)
    prefix = _write_checkpoint(tmp_path, fmt, raw, scale)
    reader = DeepSeekV4Checkpoint(tmp_path)
    loaded = reader.load_linear(prefix)
    assert loaded.data.tobytes() == raw.tobytes()
    assert loaded.logical_shape == (512, 512)
    assert loaded.weight_format == fmt
    assert loaded.nbytes == raw.nbytes + (0 if scale is None else scale.nbytes)
    parts = [reader.load_linear(prefix, shard_index=i, shard_count=4) for i in range(4)]
    assert np.concatenate([part.data for part in parts]).tobytes() == raw.tobytes()
    if scale is not None:
        assert loaded.scales.tobytes() == scale.tobytes()
        assert np.concatenate([part.scales for part in parts]).tobytes() == scale.tobytes()
        fn = dequantize_fp4 if fmt == "fp4" else dequantize_fp8
        actual = jax.jit(fn)(loaded.data, loaded.scales)
        np.testing.assert_array_equal(
            np.asarray(actual, np.float32), _reference_weight(raw, scale, fmt)
        )


@pytest.mark.parametrize("tp_size", (1, 4), ids=("tp1", "tp4"))
@pytest.mark.parametrize("scenario", ("prefill_decode", "ragged"))
@pytest.mark.parametrize("fmt", ("fp4", "fp8", "bf16"))
def test_low_bit_four_templates(tmp_path, tp_size, scenario, fmt):
    if jax.default_backend() != "tpu":
        pytest.skip("requires real TPU compilation")
    assert len(jax.devices()) >= tp_size, "four-chip gate requires all four devices"
    raw, scale = _fixture_weight(fmt)
    prefix = _write_checkpoint(tmp_path, fmt, raw, scale)
    loaded = DeepSeekV4Checkpoint(tmp_path).load_linear(prefix)
    mesh = Mesh(np.asarray(jax.devices()[:tp_size]), ("tensor",))
    w = jax.device_put(loaded.data, NamedSharding(mesh, P("tensor", None)))
    s = (
        None
        if scale is None
        else jax.device_put(loaded.scales, NamedSharding(mesh, P("tensor", None)))
    )
    for shard in w.addressable_shards:
        assert shard.data.shape == (512 // tp_size, raw.shape[1])
    local = functools.partial(low_bit_matmul, weight_format=fmt, quantize_activation=True)
    fn = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P(), P("tensor", None), None if s is None else P("tensor", None)),
            out_specs=P(None, "tensor"),
            check_vma=False,
        )
    )
    rng = np.random.default_rng(71)
    x = _bf16(rng.normal(size=(17, 512)).astype(np.float32))
    reference_w = _reference_weight(raw, scale, fmt)
    expected = _bf16(_reference_activation(x) @ reference_w.T)

    def execute(values):
        return fn(jax.device_put(jnp.asarray(values, jnp.bfloat16), NamedSharding(mesh, P())), w, s)

    with jax.set_mesh(mesh):
        whole = execute(x)
        _assert_close(whole, expected)
        if scenario == "prefill_decode":
            split = jnp.concatenate((execute(x[:16]), execute(x[16:])))
        else:
            split = jnp.concatenate((execute(x[:1]), execute(x[1:9]), execute(x[9:])))
    np.testing.assert_array_equal(np.asarray(whole), np.asarray(split))
    assert whole.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "tensor")), ndim=2)
    assert all(shard.data.shape == (17, 512 // tp_size) for shard in whole.addressable_shards)


@pytest.mark.parametrize(
    "fmt,n,k", (("fp4", 2048, 4096), ("fp4", 4096, 2048), ("fp8", 4096, 8192), ("bf16", 1024, 4096))
)
def test_model_sized_online_matmul_on_four_chips(fmt, n, k, record_property):
    if jax.default_backend() != "tpu":
        pytest.skip("requires real TPU compilation")
    assert len(jax.devices()) == 4
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    raw, scales = _fixture_weight(fmt, n, k)
    rng = np.random.default_rng(119)
    x = _bf16(rng.normal(size=(1, k)).astype(np.float32))
    reference = _bf16(x @ _reference_weight(raw, scales, fmt).T)
    with jax.set_mesh(mesh):
        w = jax.device_put(raw, NamedSharding(mesh, P("tensor", None)))
        s = (
            None
            if scales is None
            else jax.device_put(scales, NamedSharding(mesh, P("tensor", None)))
        )
        a = jax.device_put(x.astype(ml_dtypes.bfloat16), NamedSharding(mesh, P()))
        fn = jax.jit(
            jax.shard_map(
                functools.partial(low_bit_matmul, weight_format=fmt),
                mesh=mesh,
                in_specs=(P(), P("tensor", None), None if s is None else P("tensor", None)),
                out_specs=P(None, "tensor"),
                check_vma=False,
            )
        )
        compiled = fn.lower(a, w, s).compile()
        output = compiled(a, w, s)
        _assert_close(output, reference)
        stats = compiled.memory_analysis()
        record_property("hbm_argument_bytes_per_device", stats.argument_size_in_bytes)
        record_property("hbm_temporary_bytes_per_device", stats.temp_size_in_bytes)
        record_property(
            "checkpoint_weight_bytes_global", raw.nbytes + (0 if scales is None else scales.nbytes)
        )
        # A full BF16 weight shard is not a temporary produced by online dequant.
        assert stats.temp_size_in_bytes < n * k * 2 // 4


def test_loader_rejects_wrong_scale_shape_and_unsafe_index(tmp_path):
    raw, scale = _fixture_weight("fp4")
    prefix = _write_checkpoint(tmp_path, "fp4", raw, scale[:, :-1])
    with pytest.raises(ValueError, match="scale metadata"):
        DeepSeekV4Checkpoint(tmp_path).load_linear(prefix)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {prefix + ".weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="escapes"):
        DeepSeekV4Checkpoint(tmp_path).load_linear(prefix)


@pytest.mark.parametrize("tp_size", (1, 4))
def test_routed_fp4_swiglu_scale_before_down_projection(tp_size):
    if jax.default_backend() != "tpu":
        pytest.skip("requires real TPU compilation")
    assert len(jax.devices()) >= tp_size
    e, h, intermediate = 8, 512, 256
    matrices = []
    scale_arrays = []
    for index, (n, k) in enumerate(((intermediate, h), (intermediate, h), (h, intermediate))):
        pairs = [_fixture_weight("fp4", n, k, seed=300 + index * e + expert) for expert in range(e)]
        matrices.append(np.stack([pair[0] for pair in pairs]))
        scale_arrays.append(np.stack([pair[1] for pair in pairs]))
    x = _bf16(np.random.default_rng(311).normal(size=(9, h)).astype(np.float32))
    ids = np.stack((np.arange(9) % e, (np.arange(9) + 3) % e), axis=-1).astype(np.int32)
    routes = np.tile(np.array([0.25, 1.25], np.float32), (9, 1))
    expected = np.zeros(x.shape, np.float32)
    for expert in range(e):
        a = _reference_activation(x)
        gate, up = [
            _bf16(a @ _reference_weight(matrices[i][expert], scale_arrays[i][expert], "fp4").T)
            for i in range(2)
        ]
        gate = np.minimum(gate, 10.0)
        up = np.clip(up, -10.0, 10.0)
        route = np.sum(np.where(ids == expert, routes, 0), axis=-1)
        hidden = _bf16(route[:, None] * (gate / (1 + np.exp(-gate))) * up)
        down = _reference_weight(matrices[2][expert], scale_arrays[2][expert], "fp4")
        expected += _bf16(_reference_activation(hidden) @ down.T)
    mesh = Mesh(np.asarray(jax.devices()[:tp_size]), ("tensor",))
    with jax.set_mesh(mesh):
        sharded = [
            jax.device_put(value, NamedSharding(mesh, P("tensor", None, None)))
            for value in matrices + scale_arrays
        ]
        replicated = [
            jax.device_put(value, NamedSharding(mesh, P()))
            for value in (x.astype(ml_dtypes.bfloat16), ids, routes)
        ]
        fn = jax.jit(
            jax.shard_map(
                routed_fp4_experts,
                mesh=mesh,
                in_specs=(P(), *(P("tensor", None, None) for _ in range(6)), P(), P()),
                out_specs=P(),
                check_vma=False,
            )
        )
        result = fn(replicated[0], *sharded, replicated[1], replicated[2])
        _assert_close(result, expected)
