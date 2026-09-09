"""Shared-driver regressions and independent/legacy gates for V4 FP4 GMM."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.deepseek_v4.moe import grouped_fp4_experts, validate_backend
from sgl_jax.srt.kernels.deepseek_v4.moe_gmm import gmm_fp4_experts, pack_routes
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm, make_group_metadata
from sgl_jax.srt.kernels.low_bit.gmm import grouped_fp4_matmul
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul
from sgl_jax.test.kernels.test_deepseek_v4_low_bit import (
    _assert_close,
    _bf16,
    _fixture_weight,
    _reference_activation,
    _reference_weight,
)


@pytest.fixture(autouse=True)
def single_device_mesh():
    with jax.set_mesh(Mesh(np.asarray(jax.devices()[:1]), ("gmm_test",))):
        yield


def _weights(experts, hidden=128, intermediate=256):
    result = []
    for projection, n, k in (
        (1, intermediate, hidden),
        (3, intermediate, hidden),
        (2, hidden, intermediate),
    ):
        pairs = [_fixture_weight("fp4", n, k, 1000 * projection + e) for e in range(experts)]
        result.append((np.stack([w for w, _ in pairs]), np.stack([s for _, s in pairs])))
    return tuple(w for w, _ in result) + tuple(s for _, s in result)


def test_route_packing_coalesces_duplicates_and_masks_inactive_slots():
    ids = jnp.array([[3, 1, 3], [0, 2, 1], [-1, 9, 2]], jnp.int32)
    mixing = jnp.array([[0.25, 0.5, 0.75], [0, 0, 0], [5, 5, 0.5]], jnp.float32)
    permutation, sizes, sorted_ids, sorted_weights = jax.jit(lambda i, w: pack_routes(i, w, 4))(
        ids, mixing
    )
    np.testing.assert_array_equal(sizes, [0, 1, 1, 1, 13])
    np.testing.assert_array_equal(sorted_ids[:3], [1, 2, 3])
    np.testing.assert_array_equal(sorted_weights[:3], [0.5, 0.5, 1.0])
    np.testing.assert_array_equal(sorted_weights[3:], 0)
    assert len(set(np.asarray(permutation).tolist())) == 16


def test_shared_epmoe_permutation_is_unchanged():
    from sgl_jax.srt.layers.moe import EPMoE

    ids = jnp.array([[3, 1, 2], [1, 0, 3]], jnp.int32)
    inputs = jnp.arange(8).reshape(2, 4)
    layer = SimpleNamespace(num_experts=4, num_experts_per_tok=3)
    result = EPMoE._permute(layer, inputs, ids)
    expected_order = np.argsort(np.asarray(ids).reshape(-1), kind="stable")
    np.testing.assert_array_equal(result[0], inputs)
    np.testing.assert_array_equal(result[1], expected_order // 3)
    np.testing.assert_array_equal(result[2], expected_order)
    np.testing.assert_array_equal(result[3], [1, 2, 1, 2])


@pytest.mark.parametrize("start", [0, 1, 4])
def test_shared_group_schedule_never_visits_empty_experts(start):
    sizes = jnp.array([0, 3, 0, 9, 4], jnp.int32)
    metadata, count = jax.jit(
        lambda s: make_group_metadata(
            group_sizes=s,
            m=16,
            tm=8,
            start_group=jnp.int32(start),
            num_nonzero_groups=1,
            visit_empty_groups=False,
        )
    )(sizes)
    np.testing.assert_array_equal(np.asarray(metadata.group_ids)[: int(count)], start)
    assert int(count) == [0, 1, 1][[0, 1, 4].index(start)]


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("hidden", [128, 512])
def test_fp4_shared_gmm_matches_numpy_and_existing_tile(hidden, empty):
    rng = np.random.default_rng(847)
    x = rng.normal(size=(24, hidden)).astype(ml_dtypes.bfloat16)
    pairs = [_fixture_weight("fp4", 256, hidden, 91 + e) for e in range(4)]
    weights = np.stack([w for w, _ in pairs])
    scales = np.stack([s for _, s in pairs])
    sizes = [1, 0, 9, 0, 3, 11] if not empty else [1, 0, 0, 0, 0, 23]
    actual = grouped_fp4_matmul(
        jnp.asarray(x),
        jnp.asarray(weights),
        jnp.asarray(scales),
        jnp.array(sizes, jnp.int32),
        first_expert=jnp.int32(1),
        interpret=jax.default_backend() != "tpu",
    )
    expected = np.zeros((24, 256), np.float32)
    legacy = np.zeros_like(expected)
    offset = 0
    for group, count in enumerate(sizes):
        if count and 1 <= group < 5:
            w, s = pairs[group - 1]
            expected[offset : offset + count] = _bf16(
                _reference_activation(x[offset : offset + count]) @ _reference_weight(w, s, "fp4").T
            )
            legacy[offset : offset + count] = np.asarray(
                low_bit_matmul(
                    jnp.asarray(x[offset : offset + count]),
                    jnp.asarray(w),
                    jnp.asarray(s),
                    weight_format="fp4",
                    quantize_activation=True,
                ),
                np.float32,
            )
        offset += count
    _assert_close(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual, np.float32), legacy)


@pytest.mark.parametrize("scaled", [False, True])
def test_ordinary_gmm_without_adapter_still_matches_numpy(scaled):
    rng = np.random.default_rng(36)
    x = rng.normal(size=(128, 128)).astype(ml_dtypes.bfloat16)
    weights = rng.normal(size=(4, 128, 128)).astype(ml_dtypes.bfloat16)
    sizes = jnp.array([1, 0, 126, 1], jnp.int32)
    scales = np.full((4, 1, 1, 128), 0.125, np.float32) if scaled else None
    actual = gmm(
        jnp.asarray(x),
        jnp.asarray(weights),
        sizes,
        rhs_scale=None if scales is None else jnp.asarray(scales),
        tiling=(128, 128, 128),
        preferred_element_type=jnp.bfloat16,
        interpret=jax.default_backend() != "tpu",
    )
    expected, offset = np.empty((128, 128), np.float32), 0
    for expert, count in enumerate(np.asarray(sizes)):
        value = x[offset : offset + count].astype(np.float32) @ weights[expert].astype(np.float32)
        expected[offset : offset + count] = _bf16(value * (0.125 if scaled else 1.0))
        offset += count
    _assert_close(actual, expected)


@pytest.mark.parametrize(
    "chips,tokens,pattern",
    [
        (1, 1, "normal"),
        (1, 17, "normal"),
        (1, 4, "duplicate"),
        (4, 1, "normal"),
        (4, 2, "normal"),
        (4, 4, "normal"),
        (4, 17, "normal"),
        (4, 128, "normal"),
        (4, 4, "duplicate"),
        (4, 4, "one_shard"),
        (4, 4, "inactive"),
    ],
)
def test_moe_shared_gmm_matches_legacy_on_device(chips, tokens, pattern):
    if len(jax.devices()) < chips:
        pytest.skip(f"requires {chips} devices")
    rng = np.random.default_rng(294)
    experts, top_k = 16, 6
    x = (rng.normal(size=(tokens, 128)) * 20).astype(ml_dtypes.bfloat16)
    ids = np.stack([rng.choice(experts, top_k, replace=False) for _ in range(tokens)]).astype(
        np.int32
    )
    if pattern == "duplicate":
        ids[:, 1] = ids[:, 0]
    if pattern == "one_shard":
        ids[:] = [0, 1, 2, 3, 0, 1]
    mixing = rng.uniform(0.01, 0.5, ids.shape).astype(np.float32)
    if pattern == "inactive":
        mixing[:] = 0
    mesh = Mesh(np.asarray(jax.devices()[:chips]), ("tensor",))
    specs = (P(), *(P("tensor", None, None) for _ in range(6)), P(), P())
    with jax.set_mesh(mesh):
        values = (x, *_weights(experts), ids, mixing)
        values = tuple(
            jax.device_put(v, NamedSharding(mesh, spec))
            for v, spec in zip(values, specs, strict=True)
        )
        legacy = jax.jit(
            jax.shard_map(
                grouped_fp4_experts, mesh=mesh, in_specs=specs, out_specs=P(), check_vma=False
            )
        )
        candidate = jax.jit(
            jax.shard_map(
                lambda *args: gmm_fp4_experts(
                    *args, num_experts=experts, interpret=jax.default_backend() != "tpu"
                ),
                mesh=mesh,
                in_specs=specs,
                out_specs=P(),
                check_vma=False,
            )
        )
        expected = legacy(*values)
        actual = candidate(*values)
        assert np.isfinite(np.asarray(actual)).all()
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("backend", ["legacy", "gmm", "gmm_tuned"])
def test_explicit_moe_backend(backend):
    assert validate_backend(backend) == backend


def test_unknown_moe_backend_rejected():
    with pytest.raises(ValueError, match="no automatic fallback"):
        validate_backend("auto")


def test_mismatched_expert_parallel_contract_rejected():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",))
    with jax.set_mesh(mesh):
        values = (
            jnp.ones((1, 128), jnp.bfloat16),
            *(jnp.asarray(w) for w in _weights(4)),
            jnp.zeros((1, 1), jnp.int32),
            jnp.ones((1, 1), jnp.float32),
        )
        run = jax.shard_map(
            lambda *v: gmm_fp4_experts(*v, num_experts=16),
            mesh=mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )
        with pytest.raises(ValueError, match="local expert count times EP size"):
            jax.jit(run)(*values)


@pytest.mark.parametrize("backend", ["gmm", "gmm_tuned"])
def test_selected_gmm_failure_does_not_fall_back(monkeypatch, backend):
    from sgl_jax.srt.kernels.deepseek_v4 import moe as module, moe_gmm

    monkeypatch.setattr(module, "route", lambda *args: (None, None))

    def failed(*args, **kwargs):
        raise RuntimeError("injected GMM failure")

    monkeypatch.setattr(moe_gmm, "gmm_fp4_experts", failed)
    weights = {"experts." + key: None for key in ("w1", "w3", "w2", "s1", "s3", "s2")}
    with pytest.raises(RuntimeError, match="injected GMM failure"):
        module.moe(None, None, weights, SimpleNamespace(swiglu_limit=10), None, backend=backend)
