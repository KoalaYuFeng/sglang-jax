"""Explicit FP4 loader ABI and actual MoE entry-point gates, before whole-model use."""

from types import SimpleNamespace

import jax
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.deepseek_v4.moe_gmm import gmm_fp4_experts
from sgl_jax.srt.kernels.low_bit.fp4_tuning import tuned_fp4_tile_m
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer, original_fp4_scale_view
from sgl_jax.test.test_deepseek_v4_moe_gmm import _weights


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 32, 64, 127, 128, 129, 256, 512])
def test_shape_policy_is_explicit_and_conservative(tokens):
    assert tuned_fp4_tile_m(tokens) == (32 if tokens >= 128 else 8)


@pytest.mark.parametrize("tokens", [0, -1, True, 1.0, None])
def test_shape_policy_rejects_invalid_capacity(tokens):
    with pytest.raises(ValueError, match="positive integer"):
        tuned_fp4_tile_m(tokens)


@pytest.mark.parametrize("chips", [1, 4])
def test_loader_transposes_only_compact_scales_and_retains_expert_ownership(chips):
    if len(jax.devices()) < chips:
        pytest.skip(f"requires {chips} devices")
    original = _weights(16)
    by_projection = {p: (original[i], original[i + 3]) for i, p in enumerate((1, 3, 2))}
    reads = []

    def read_linear(name):
        fields = name.split(".")
        expert, projection = int(fields[-2]), int(fields[-1][1:])
        reads.append((expert, projection))
        weights, scales = by_projection[projection]
        return SimpleNamespace(data=weights[expert], scales=scales[expert])

    checkpoint = SimpleNamespace(
        weight_map={}, config={"n_routed_experts": 16}, load_linear=read_linear
    )
    mesh = Mesh(np.asarray(jax.devices()[:chips]), ("tensor",))
    with jax.set_mesh(mesh):
        baseline = load_layer(checkpoint, 3, mesh)
        candidate = load_layer(checkpoint, 3, mesh, fp4_scale_transposed=True)
        restored = original_fp4_scale_view(candidate, transposed=True)
        for key, expected in baseline.items():
            actual = candidate[key]
            assert actual.dtype == np.uint8 and actual.nbytes == expected.nbytes
            assert actual.sharding.spec == P("tensor", None, None)
            assert len(actual.addressable_shards) == chips
            np.testing.assert_array_equal(restored[key], expected)
            assert actual.shape[0] == 16
            for shard in actual.addressable_shards:
                assert shard.data.shape[0] == 16 // chips
            if key.startswith("experts.s"):
                assert actual.shape == (expected.shape[0], expected.shape[2], expected.shape[1])
            else:
                assert restored[key] is actual
        assert original_fp4_scale_view(candidate)["experts.s1"] is candidate["experts.s1"]
        # Each projection is loaded once for each backend, never from BF16 values.
        assert len(reads) == 16 * 3 * 2
        with pytest.raises(ValueError, match="boolean"):
            load_layer(checkpoint, 3, mesh, fp4_scale_transposed="true")
        with pytest.raises(ValueError, match="boolean"):
            original_fp4_scale_view(candidate, transposed="true")


@pytest.mark.parametrize(
    "chips,tokens,pattern",
    [(1, 1, "normal"), (1, 17, "normal")]
    + [(4, n, "normal") for n in (1, 2, 4, 17, 32, 64, 127, 128, 129)]
    + [(4, n, p) for n in (4, 129) for p in ("duplicate_cancel", "one_shard", "inactive")],
)
def test_actual_tuned_entry_is_bitwise_equal_to_existing_gmm(chips, tokens, pattern):
    if jax.default_backend() != "tpu" or len(jax.devices()) < chips:
        pytest.skip(f"hardware MoE entry gate requires {chips} TPU chips")
    rng = np.random.default_rng(619)
    experts, top_k = 16, 6
    x = (rng.normal(size=(tokens, 128)) * 20).astype(ml_dtypes.bfloat16)
    ids = np.stack([rng.choice(experts, top_k, replace=False) for _ in range(tokens)]).astype(
        np.int32
    )
    mixing = rng.uniform(0.01, 0.5, ids.shape).astype(np.float32)
    if pattern == "duplicate_cancel":
        ids[:] = [1, 1, 2, 2, 3, 3]
        mixing[:] = [0.25, -0.25, 0.5, 0.25, 0.25, 0.125]
    elif pattern == "one_shard":
        ids[:] = [12, 13, 14, 15, 12, 13]
    elif pattern == "inactive":
        mixing[:] = 0
    mesh = Mesh(np.asarray(jax.devices()[:chips]), ("tensor",))
    specs = (P(), *(P("tensor", None, None) for _ in range(6)), P(), P())
    with jax.set_mesh(mesh):
        raw = _weights(experts)
        inputs = (x, *raw, ids, mixing)
        candidate_inputs = (
            x,
            *raw[:3],
            *(np.ascontiguousarray(s.swapaxes(1, 2)) for s in raw[3:]),
            ids,
            mixing,
        )

        def run(values, tuned):
            placed = tuple(
                jax.device_put(value, NamedSharding(mesh, spec))
                for value, spec in zip(values, specs, strict=True)
            )
            function = jax.jit(
                jax.shard_map(
                    lambda *args: gmm_fp4_experts(*args, num_experts=experts, tuned=tuned),
                    mesh=mesh,
                    in_specs=specs,
                    out_specs=P(),
                    check_vma=False,
                )
            )
            return np.asarray(function(*placed))

        expected, actual = run(inputs, False), run(candidate_inputs, True)
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
