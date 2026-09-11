"""Independent arithmetic and request-layout gates for V4 EP reduction."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.deepseek_v4.collectives import ordered_ep_sum


def numpy_tree(partials):
    # Explicit left-deep FP32 tree, independent of JAX or either V4 backend.
    total = np.asarray(partials[0], np.float32)
    for rank in range(1, len(partials)):
        total = np.add(total, partials[rank], dtype=np.float32)
    return total


def run(partials, shared):
    ranks = len(partials)
    if jax.device_count() < ranks:
        pytest.skip(f"requires {ranks} devices (CPU or TPU)")
    mesh = Mesh(np.asarray(jax.devices()[:ranks]), ("expert",))
    with jax.set_mesh(mesh):
        p = jax.device_put(partials, NamedSharding(mesh, P("expert", None, None)))
        s = jax.device_put(shared, NamedSharding(mesh, P()))

        def reduce(value, shared):
            result = ordered_ep_sum(value[0], "expert")
            return result, (result + shared.astype(jnp.float32)).astype(jnp.bfloat16)

        compiled = jax.jit(
            jax.shard_map(
                reduce,
                mesh=mesh,
                in_specs=(P("expert", None, None), P()),
                out_specs=(P(), P()),
                check_vma=False,
            )
        )
        values = compiled(p, s)
        for value in values:
            for shard in value.addressable_shards:
                np.testing.assert_array_equal(np.asarray(shard.data), np.asarray(value))
        return tuple(np.asarray(value) for value in values)


@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 128, 257])
def test_real_halfway_counterexample_is_batch_and_row_invariant(tokens):
    # Frozen real layer-28/channel-2847 partials, not a model-generated oracle.
    parts = np.array([-0.1513671875, 2**-27, 0.042236328125, 0], np.float32)
    x = np.broadcast_to(parts[:, None, None], (4, tokens, 4096)).copy()
    shared = np.full((tokens, 4096), 0.1904296875, ml_dtypes.bfloat16)
    actual, rounded = run(x, shared)
    expected = numpy_tree(x)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        rounded, (expected + shared.astype(np.float32)).astype(ml_dtypes.bfloat16)
    )
    np.testing.assert_array_equal(rounded, np.full_like(rounded, 0.0810546875))


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_numpy_tree_cancellation_request_reorder_and_padding(ranks):
    rng = np.random.default_rng(4100)
    partials = rng.normal(size=(ranks, 13, 128)).astype(np.float32)
    partials *= np.exp2(rng.integers(-20, 20, size=partials.shape)).astype(np.float32)
    if ranks >= 2:
        partials[1, :, :32] = -partials[0, :, :32]
    shared = rng.normal(size=(13, 128)).astype(ml_dtypes.bfloat16)
    order = np.array([7, 0, 7, 12, 4, 0, 2, 10, 1, 8, 3, 6, 5, 9, 11])
    for indices in (np.arange(13), order):
        x = np.pad(partials[:, indices], ((0, 0), (0, 3), (0, 0)))
        s = np.pad(shared[indices], ((0, 3), (0, 0)))
        actual, rounded = run(x, s)
        expected = numpy_tree(x)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            rounded, (expected + s.astype(np.float32)).astype(ml_dtypes.bfloat16)
        )


def test_non_fp32_partials_rejected():
    with pytest.raises(ValueError, match="FP32"):
        ordered_ep_sum(jnp.zeros((1, 128), jnp.bfloat16))
