"""Standalone Pallas pool precision; does not load a model or scheduler."""

import hashlib
import io
import os
import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental import pallas as pl

from sgl_jax.srt.kernels.csa.numerics import _exp_nonpositive, index_pool_v4


def pool(values, scores, tile=1):
    def kernel(v_ref, s_ref, out_ref):
        out_ref[...] = index_pool_v4(v_ref[...], s_ref[...])[:, None, :]

    groups = values.shape[0]
    padding = (-groups) % tile
    return pl.pallas_call(
        kernel,
        grid=((groups + padding) // tile,),
        in_specs=(pl.BlockSpec((tile, 8, 128), lambda i: (i, 0, 0)),) * 2,
        out_specs=pl.BlockSpec((tile, 1, 128), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((groups + padding, 1, 128), jnp.float32),
        interpret=jax.default_backend() != "tpu",
    )(
        jnp.pad(values, ((0, padding), (0, 0), (0, 0))),
        jnp.pad(scores, ((0, padding), (0, 0), (0, 0))),
    )[:groups, 0]


def oracle(values, scores):
    scores = scores.astype(np.float64)
    probability = np.exp(scores - scores.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    return np.sum(values.astype(np.float64) * probability, axis=1), probability


@pytest.mark.parametrize("seed", [219, 6575, 7715])
def test_pool_error_against_independent_fp64(seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(32, 8, 128)).astype(np.float32) * 4
    scores = rng.normal(size=values.shape).astype(np.float32) * 4
    expected, probability = oracle(values, scores)
    actual = np.asarray(jax.jit(pool)(jnp.asarray(values), jnp.asarray(scores)))
    condition = np.sum(np.abs(values) * probability, axis=1)
    assert np.all(np.abs(actual - expected) <= 3e-6 * condition + 1e-7)


def test_empty_and_single_live_windows():
    values = jnp.arange(8 * 128, dtype=jnp.float32).reshape(1, 8, 128)
    scores = jnp.full_like(values, -jnp.inf)
    np.testing.assert_array_equal(jax.jit(pool)(values, scores), np.zeros((1, 128)))
    scores = scores.at[:, 3, :].set(0)
    np.testing.assert_array_equal(jax.jit(pool)(values, scores), values[:, 3])


@pytest.mark.parametrize("tile", [1, 4])
def test_cancellation_midpoint_and_partial_tile(tile):
    # Expanded emitter fixture exposed group 34/channel 41. Preserve the
    # entire independently generated fixture, not just a hand-picked scalar.
    rng = np.random.default_rng(4)
    values, scores = [rng.normal(size=(65, 8, 128)).astype(np.float32) for _ in range(2)]
    scores[0, :4] = -np.inf
    expected, _ = oracle(values, scores)
    actual = np.asarray(jax.jit(lambda v, s: pool(v, s, tile))(values, scores))
    np.testing.assert_array_equal(
        actual.astype(ml_dtypes.bfloat16), expected.astype(ml_dtypes.bfloat16)
    )


def test_group642_reference_sensitive_midpoint():
    # Captured layer-2/request-0/channel-6 window, terminal position 2571.
    # The CPU FP32 reference rounds down; independent FP64 on BOTH captured
    # projection inputs, and FP64 projection itself, support rounding up.
    values = np.array(
        [
            -1.6556360721588135,
            3.9646973609924316,
            3.3482189178466797,
            4.171531677246094,
            2.1508309841156006,
            -0.33818182349205017,
            -3.258816957473755,
            -0.2623070478439331,
        ],
        np.float32,
    )
    scores = np.array(
        [
            3.1230273246765137,
            6.716819763183594,
            0.24085715413093567,
            4.31021785736084,
            0.33632636070251465,
            -0.9633171558380127,
            1.6258368492126465,
            0.8685742020606995,
        ],
        np.float32,
    )
    values, scores = [np.broadcast_to(a[None, :, None], (4, 8, 128)) for a in (values, scores)]
    expected, _ = oracle(values, scores)
    assert np.all(expected.astype(ml_dtypes.bfloat16) == 3.796875)
    for tile in (1, 4):
        actual = np.asarray(jax.jit(lambda v, s, tile=tile: pool(v, s, tile))(values, scores))
        np.testing.assert_array_equal(
            actual.astype(ml_dtypes.bfloat16), expected.astype(ml_dtypes.bfloat16)
        )


def test_exp_range_reduction():
    host = np.linspace(-87, 0, 16384, dtype=np.float32).reshape(128, 128)

    def kernel(x_ref, out_ref):
        out_ref[...] = _exp_nonpositive(x_ref[...])

    run = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(host.shape, jnp.float32),
        interpret=jax.default_backend() != "tpu",
    )
    actual = np.asarray(jax.jit(run)(jnp.asarray(host)))
    expected = np.exp(host.astype(np.float64))
    assert np.max(np.abs(actual - expected) / expected) < 1e-6


@pytest.mark.parametrize("layer,group", [(2, 856), (2, 1643), (2, 1928), (22, 272)])
def test_frozen_midpoint_windows(layer, group):
    path = os.environ.get("SGL_V4_NUMERICAL_EVIDENCE")
    if not path:
        pytest.skip("set SGL_V4_NUMERICAL_EVIDENCE to the frozen diagnostic archive")
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "521582e7815418122b1aa78abc888b79d41fc72c29d8e9f42a20cd37103deac3"
    )
    with tarfile.open(path, "r:gz") as archive:
        name = f"v4-index-residual-20260911-01/l{layer:02d}-g{group}-native.npz"
        with np.load(io.BytesIO(archive.extractfile(name).read())) as data:
            selected = data["starts"] == group * 4
            order = np.argsort(3 - (data["slots"][data["requests"][selected]] - 1))
            values, scores = [
                data[key][selected][order] for key in ("raw_group_values", "raw_group_scores")
            ]
    values, scores = [
        np.concatenate((a[:, :4, :128], a[:, 4:, 128:]), axis=1) for a in (values, scores)
    ]
    expected, _ = oracle(values, scores)
    actual = np.asarray(jax.jit(pool)(jnp.asarray(values), jnp.asarray(scores)))
    np.testing.assert_array_equal(
        actual.astype(ml_dtypes.bfloat16), expected.astype(ml_dtypes.bfloat16)
    )
