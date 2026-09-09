"""Decode-only batching gate; no model or Engine is constructed."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.kernels.csa.compressor import (
    csa_project_decode_pallas,
    csa_project_pallas,
)
from sgl_jax.test.kernels.csa_compressor_cases import read_arrays
from sgl_jax.test.kernels.csa_projection_cases import (
    numpy_v4_projection,
    synthetic_projection,
)

TPU = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="real Mosaic lowering required"
)


@jax.jit
def sequential(x, weight):
    return jax.lax.map(
        lambda row: csa_project_pallas(row[None], weight, projection_mode="v4_gemv")[0],
        x,
    )


def same_bits(a, b):
    np.testing.assert_array_equal(
        np.asarray(a).view(np.uint32), np.asarray(b).view(np.uint32)
    )


@TPU
@pytest.mark.parametrize("width", [512, 2048])
@pytest.mark.parametrize("tokens", [1, 2, 4, 9])
def test_batched_gemv_matches_sequential_and_independent_cpu(width, tokens):
    host_x, host_weight = synthetic_projection(tokens, width)
    with jax.set_mesh(Mesh(np.asarray(jax.devices()[:1]), ("tensor",))):
        x, weight = jnp.asarray(host_x), jnp.asarray(host_weight)
        actual = csa_project_decode_pallas(x, weight)
        same_bits(actual, sequential(x, weight))
        same_bits(actual, numpy_v4_projection(host_x, host_weight))
        assert actual.dtype == jnp.float32
        hlo = csa_project_decode_pallas.lower(x, weight).compile().as_text()
        calls = [
            line
            for line in hlo.splitlines()
            if "custom-call(" in line
            and "csa-compressor-project-decode-batched" in line
        ]
        assert len(calls) == 1


@TPU
@pytest.mark.parametrize("width", [512, 2048])
def test_batched_gemv_order_partition_and_replicated_four_chips(width):
    x, weight = synthetic_projection(9, width)
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("tensor",))
    assert mesh.size == 4, "this acceptance gate requires four physical chips"
    with jax.set_mesh(mesh):
        spec = NamedSharding(mesh, P())
        run = jax.jit(
            jax.shard_map(
                csa_project_decode_pallas,
                mesh=mesh,
                in_specs=(P(), P()),
                out_specs=P(),
                check_vma=False,
            )
        )
        args = [jax.device_put(a, spec) for a in (x, weight)]
        expected = numpy_v4_projection(x, weight)
        actual = run(*args)
        for shard in actual.addressable_shards:
            same_bits(shard.data, expected)
        order = np.array([8, 3, 0, 5, 2, 7, 1, 4, 6])
        same_bits(run(jax.device_put(x[order], spec), args[1]), expected[order])
        parts = [
            np.asarray(run(jax.device_put(x[a:b], spec), args[1]))
            for a, b in ((0, 1), (1, 6), (6, 9))
        ]
        same_bits(np.concatenate(parts), expected)


@TPU
def test_batched_gemv_frozen_real_7979():
    prefix = os.environ.get("CSA_V4_PROJECTION_FIXTURE")
    if not prefix:
        pytest.skip("set CSA_V4_PROJECTION_FIXTURE to the immutable 7979 fixture")
    names = ["activation"]
    for kind in ("main", "index"):
        names.extend(
            f"{kind}.{suffix}"
            for suffix in (
                "wkv.weight",
                "wgate.weight",
                "reference.kv",
                "reference.scores",
            )
        )
    fixture = read_arrays(prefix, names)
    with jax.set_mesh(Mesh(np.asarray(jax.devices()[:1]), ("tensor",))):
        for kind in ("main", "index"):
            weight = np.concatenate(
                (fixture[f"{kind}.wkv.weight"], fixture[f"{kind}.wgate.weight"]), axis=0
            ).T
            expected = np.concatenate(
                (fixture[f"{kind}.reference.kv"], fixture[f"{kind}.reference.scores"]),
                axis=1,
            )
            actual = csa_project_decode_pallas(
                jnp.asarray(fixture["activation"]), jnp.asarray(weight)
            )
            same_bits(actual, expected)
            same_bits(actual, numpy_v4_projection(fixture["activation"], weight))


@pytest.mark.parametrize(
    "tokens,hidden,width", [(0, 4096, 512), (1, 128, 512), (1, 4096, 128)]
)
def test_batched_gemv_rejects_unvalidated_shapes(tokens, hidden, width):
    with pytest.raises(ValueError, match="V4 batched GEMV"):
        csa_project_decode_pallas(
            jnp.zeros((tokens, hidden), jnp.bfloat16),
            jnp.zeros((hidden, width), jnp.bfloat16),
        )
