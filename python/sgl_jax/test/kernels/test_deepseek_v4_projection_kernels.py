"""Standalone gates: unchanged numerical baseline plus independent NumPy GEMM."""

import functools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.deepseek_v4 import normalization, numerics, projections
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip, round_bf16
from sgl_jax.srt.kernels.deepseek_v4.fp8 import fp8_linear
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul


@pytest.fixture(autouse=True)
def isolated_mesh():
    with jax.set_mesh(Mesh(np.asarray(jax.devices()[:1]), ("projection_test",))):
        yield


def inputs(m, k, n, seed=71):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(m, k)).astype(ml_dtypes.bfloat16)
    x[0] = 0
    w = (rng.normal(size=(n, k)) * 3).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    s = rng.integers(119, 126, ((n + 127) // 128, k // 128), dtype=np.uint8)
    return x, w, s


def exact(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.isfinite(actual.astype(np.float32)).all()
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize("m,width", [(1, 512), (4, 4096), (9, 1024), (128, 4096), (512, 512)])
def test_exact_rms_norm(m, width):
    x = jnp.asarray(inputs(m, width, 128)[0])
    weight = jnp.asarray(np.random.default_rng(42).normal(size=width), jnp.bfloat16)
    actual = normalization.rms_norm(x, weight)
    expected = jax.jit(numerics.rms_norm)(x, weight, 1e-6)
    exact(actual, expected)


def baseline_q(q, positions, config):
    square = round_bf16(q.astype(jnp.float32) ** 2)
    variance = round_bf16(numerics._fixed_tree_mean_last(square.astype(jnp.float32))[..., None])
    variance = round_bf16(variance.astype(jnp.float32) + config.eps)
    inverse = round_bf16(jax.lax.rsqrt(variance.astype(jnp.float32)))
    return numerics.rope(
        round_bf16(q.astype(jnp.float32) * inverse.astype(jnp.float32)), positions, config
    )


@pytest.mark.parametrize("m,heads", [(1, 16), (4, 64), (9, 3), (128, 16)])
@pytest.mark.skipif(
    jax.default_backend() != "tpu",
    reason="full YaRN/RoPE bit-exact gate requires TPU lowering; CPU FMA contraction differs",
)
def test_exact_qnorm_rope(m, heads):
    config = numerics.V4LayerConfig(original_seq_len=65536)
    q = jnp.asarray(inputs(m * heads, 512, 128)[0]).reshape(m, heads, 512)
    positions = jnp.arange(m, dtype=jnp.int32) + 8023

    def candidate(q, positions):
        # Keep phase construction in the same compiled program as production.
        # Eager YaRN evaluation adds FP32 boundaries absent in the reference JIT.
        phase = numerics.rope_angles(positions, config)
        return normalization.qnorm_rope(q, jnp.cos(phase), jnp.sin(phase))

    actual = jax.jit(candidate)(q, positions)
    expected = jax.jit(functools.partial(baseline_q, config=config))(q, positions)
    exact(actual, expected)


def test_qnorm_rounding_tree_with_identity_rotation():
    # CPU interpretation still checks every norm rounding boundary; the
    # composite rotary arithmetic is checked bitwise on real TPU above.
    q = jnp.asarray(inputs(128 * 16, 512, 128)[0]).reshape(128, 16, 512)
    config = numerics.V4LayerConfig()
    actual = normalization.qnorm_rope(q, jnp.ones((128, 32)), jnp.zeros((128, 32)))
    expected = jax.jit(lambda q: baseline_q(q, jnp.zeros(128, jnp.int32), config))(q)
    exact(actual, expected)


@pytest.mark.parametrize("m", [1, 4, 9, 128])
def test_exact_kv_norm_rope_quantization(m):
    config = numerics.V4LayerConfig()
    x = jnp.asarray(inputs(m, 512, 128)[0])
    w = jnp.linspace(0.5, 1.5, 512, dtype=jnp.bfloat16)
    positions = jnp.arange(m, dtype=jnp.int32) + 8023

    def candidate(x, w, positions):
        phase = numerics.rope_angles(positions, config)
        return normalization.rms_norm(
            x, w, cosine=jnp.cos(phase), sine=jnp.sin(phase), quantize_nope=True
        )

    actual = jax.jit(candidate)(x, w, positions)

    def baseline(x, w, positions):
        kv = numerics.rope(numerics.rms_norm(x, w, config.eps), positions, config)
        return jnp.concatenate((activation_fp8_roundtrip(kv[:, :-64], 64), kv[:, -64:]), axis=-1)

    exact(actual, jax.jit(baseline)(x, w, positions))


@pytest.mark.parametrize("block_m", [8, 32, 64, 128])
@pytest.mark.parametrize(
    "m,k,n,quantize", [(4, 512, 257, True), (129, 512, 256, True), (9, 1024, 128, False)]
)
def test_fp8_gmm_matches_baseline_and_numpy(block_m, m, k, n, quantize):
    host_x, host_w, host_s = inputs(m, k, n)
    x, w, s = map(jnp.asarray, (host_x, host_w, host_s))
    actual = fp8_linear(x, w, s, block_m=block_m, quantize_activation=quantize)
    expected = low_bit_matmul(x, w, s, weight_format="fp8", quantize_activation=quantize)
    exact(actual, expected)
    # Independent NumPy E4M3 + exponent scales and FP32 GEMM, not the baseline.
    lhs = host_x.astype(np.float32)
    if quantize:
        blocks = lhs.reshape(m, -1, 128)
        scale = np.exp2(
            np.ceil(
                np.log2(
                    np.maximum(np.abs(blocks).max(-1, keepdims=True), 1e-4).astype(np.float64) / 448
                )
            )
        ).astype(np.float32)
        lhs = (
            ((blocks / scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale)
            .astype(ml_dtypes.bfloat16)
            .astype(np.float32)
            .reshape(m, k)
        )
    expanded = np.repeat(
        np.repeat(np.exp2(host_s.astype(np.float32) - 127), 128, axis=0), 128, axis=1
    )
    rhs = (
        (host_w.view(ml_dtypes.float8_e4m3fn).astype(np.float32) * expanded[:n])
        .astype(ml_dtypes.bfloat16)
        .astype(np.float32)
    )
    oracle = (lhs @ rhs.T).astype(ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_allclose(np.asarray(actual, np.float32), oracle, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("m,groups", [(1, 2), (4, 8), (9, 2), (128, 2)])
def test_inverse_rope_grouped_wo_a(m, groups):
    heads, width, rank = groups * 2, 512, 128
    config = numerics.V4LayerConfig(heads=heads, groups=groups, o_rank=rank)
    x, w, s = inputs(m, heads * width, groups * rank)
    _, w, s = inputs(m, 2 * width, groups * rank)
    x, w, s = jnp.asarray(x.reshape(m, heads, width)), jnp.asarray(w), jnp.asarray(s)
    positions = jnp.arange(m, dtype=jnp.int32) + 8023

    def candidate(x, w, s, positions):
        phase = numerics.rope_angles(positions, config)
        return projections.inverse_rope_fp8_wo_a(
            x, w, s, jnp.cos(phase), jnp.sin(phase), groups=groups
        )

    actual = jax.jit(candidate)(x, w, s, positions)

    def baseline(x, w, s, positions):
        value = numerics.rope(x, positions, config, inverse=True).reshape(m, groups, -1)
        return jnp.concatenate(
            [
                low_bit_matmul(
                    value[:, g], w[g * rank : (g + 1) * rank], s[g : g + 1], weight_format="fp8"
                )
                for g in range(groups)
            ],
            axis=-1,
        )

    exact(actual, jax.jit(baseline)(x, w, s, positions))


def test_merged_packing_preserves_bytes_and_projections():
    weights = {}
    for target, sources in projections.MERGED_PROJECTIONS.items():
        for index, source in enumerate(sources):
            n = (index + 1) * 128 if target == "attn.wqkv_a" else 128
            x, w, s = inputs(9, 512, n, seed=41 + index)
            weights[source + ".weight"], weights[source + ".scale"] = w, s
    weights["attn.q_norm.weight"] = np.ones(128, ml_dtypes.bfloat16)
    packed = projections.pack_merged_weights(weights)
    assert sum(v.nbytes for v in packed.values()) == sum(v.nbytes for v in weights.values())
    restored = projections.unpack_merged_weights(packed)
    assert restored.keys() == weights.keys()
    for name, value in weights.items():
        np.testing.assert_array_equal(value, restored[name])
    for target, sources in projections.MERGED_PROJECTIONS.items():
        actual = projections.merged_linear(
            jnp.asarray(x), {k: jnp.asarray(v) for k, v in packed.items()}, target, 128
        )
        for part, source in zip(actual, sources, strict=True):
            expected = low_bit_matmul(
                jnp.asarray(x),
                jnp.asarray(weights[source + ".weight"]),
                jnp.asarray(weights[source + ".scale"]),
                weight_format="fp8",
                quantize_activation=True,
            )
            exact(part, expected)
            assert source + ".weight" not in packed


def test_dense_options_and_merged_loader_contract():
    from types import SimpleNamespace

    from sgl_jax.srt.kernels.deepseek_v4.dense import DenseKernels
    from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer, weight_specs

    for kwargs in ({"fp8_backend": "auto"}, {"merged_projections": True}, {"fused_norm": 1}):
        with pytest.raises(ValueError):
            DenseKernels(**kwargs)
    tensors = {"attn.q_norm.weight": np.ones(128, ml_dtypes.bfloat16)}
    for sources in projections.MERGED_PROJECTIONS.values():
        for name in sources:
            _, w, s = inputs(1, 512, 128)
            tensors[name + ".weight"], tensors[name + ".scale"] = w, s
    checkpoint = SimpleNamespace(
        weight_map={"layers.2." + key: "unused" for key in tensors},
        tensor_info=lambda _: {"dtype": "U8"},
        read_tensor=lambda name: tensors[name.removeprefix("layers.2.")],
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",))
    with jax.set_mesh(mesh):
        loaded = load_layer(
            checkpoint, 2, mesh, include_experts=False, attention_tp=True, merged_projections=True
        )
        for spec in weight_specs(loaded, attention_tp=True).values():
            assert spec == P()  # QA/KV/shared projections stay replicated.
        for key, value in projections.unpack_merged_weights(loaded).items():
            np.testing.assert_array_equal(value, tensors[key])


def test_reject_bad_formats():
    x, w, s = map(jnp.asarray, inputs(1, 512, 128))
    with pytest.raises(ValueError, match="raw uint8"):
        fp8_linear(x, w.astype(jnp.bfloat16), s)
    with pytest.raises(ValueError, match="compact"):
        fp8_linear(x, w, s.astype(jnp.float32))
    with pytest.raises(ValueError, match="power of two"):
        normalization.rms_norm(jnp.ones((1, 384), jnp.bfloat16), jnp.ones(384))


def test_fp8_tp4_output_sharding():
    if len(jax.devices()) != 4:
        pytest.skip("requires the four-chip TPU host")
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    host = inputs(9, 512, 1024)
    with jax.set_mesh(mesh):
        x, w, s = [
            jax.device_put(value, NamedSharding(mesh, spec))
            for value, spec in zip(host, (P(), P("tensor", None), P("tensor", None)), strict=True)
        ]
        sharded = jax.jit(
            jax.shard_map(
                fp8_linear,
                mesh=mesh,
                in_specs=(P(), P("tensor", None), P("tensor", None)),
                out_specs=P(None, "tensor"),
                check_vma=False,
            )
        )
        actual = sharded(x, w, s)
    exact(
        actual,
        low_bit_matmul(*map(jnp.asarray, host), weight_format="fp8", quantize_activation=True),
    )


def test_qnorm_and_grouped_wo_a_tp4():
    if len(jax.devices()) != 4:
        pytest.skip("requires the four-chip TPU host")
    config = numerics.V4LayerConfig(o_rank=128, original_seq_len=65536)
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    host_x = inputs(9 * 64, 512, 128)[0].reshape(9, 64, 512)
    _, host_w, host_s = inputs(9, 4096, 1024)
    host_p = np.arange(9, dtype=np.int32) + 8023

    def fused(x, w, s, p):
        phase = numerics.rope_angles(p, config)
        cos, sin = jnp.cos(phase), jnp.sin(phase)
        return (
            normalization.qnorm_rope(x, cos, sin),
            projections.inverse_rope_fp8_wo_a(x, w, s, cos, sin, groups=2),
        )

    with jax.set_mesh(mesh):
        specs = (P(None, "tensor", None), P("tensor", None), P("tensor", None), P())
        arrays = [
            jax.device_put(value, NamedSharding(mesh, spec))
            for value, spec in zip((host_x, host_w, host_s, host_p), specs, strict=True)
        ]
        actual_q, actual_wo = jax.jit(
            jax.shard_map(
                fused,
                mesh=mesh,
                in_specs=specs,
                out_specs=(P(None, "tensor", None), P(None, "tensor")),
                check_vma=False,
            )
        )(*arrays)

    def reference(x, w, s, p):
        value = numerics.rope(x, p, config, inverse=True).reshape(9, 8, 4096)
        projected = jnp.concatenate(
            [
                low_bit_matmul(
                    value[:, g], w[g * 128 : (g + 1) * 128], s[g : g + 1], weight_format="fp8"
                )
                for g in range(8)
            ],
            axis=-1,
        )
        return baseline_q(x, p, config), projected

    expected_q, expected_wo = jax.jit(reference)(
        *map(jnp.asarray, (host_x, host_w, host_s, host_p))
    )
    exact(actual_q, expected_q)
    exact(actual_wo, expected_wo)
