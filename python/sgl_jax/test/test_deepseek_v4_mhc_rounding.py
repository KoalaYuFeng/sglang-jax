"""An integrated FFN -> mHC call must observe its BF16 activation boundary."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.mhc import mhc_post_fused


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real TPU lowering")
@pytest.mark.parametrize("backend", ["xla", "pallas"])
@pytest.mark.parametrize("rounded_operand", ["operator", "residual", "both"])
def test_fused_post_preserves_bf16_inputs(backend, rounded_operand):
    rng = np.random.default_rng(8023)
    n, hc, hidden = 128, 4, 4096
    raw_x = rng.normal(size=(n, hidden)).astype(np.float32)
    raw_res = rng.normal(size=(n, hc, hidden)).astype(np.float32)
    x = raw_x.astype(ml_dtypes.bfloat16)
    res = raw_res.astype(ml_dtypes.bfloat16)
    post = rng.uniform(0.1, 1.9, (n, hc)).astype(np.float32)
    comb = rng.uniform(0.01, 1.0, (n, hc, hc)).astype(np.float32)
    comb /= comb.sum(axis=1, keepdims=True)
    expected = (
        (
            post.astype(np.float64)[..., None] * x.astype(np.float64)[:, None, :]
            + np.einsum("nij,nid->njd", comb.astype(np.float64), res.astype(np.float64))
        )
        .astype(ml_dtypes.bfloat16)
        .astype(np.float32)
    )

    @jax.jit
    def integrated(operator, residual, p, c):
        operator = operator.astype(jnp.bfloat16)
        residual = residual.astype(jnp.bfloat16)
        return (
            operator,
            residual,
            mhc_post_fused(
                operator, residual, p, c, backend=backend, precision=jax.lax.Precision.HIGHEST
            ),
        )

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1], object), ("tensor",))
    with jax.set_mesh(mesh):
        actual_x, actual_res, output = integrated(
            raw_x if rounded_operand in ("operator", "both") else x,
            raw_res if rounded_operand in ("residual", "both") else res,
            post,
            comb,
        )
        np.testing.assert_array_equal(np.asarray(actual_x), x)
        np.testing.assert_array_equal(np.asarray(actual_res), res)
        actual = np.asarray(output, np.float32)
    # FP32 accumulation may straddle an occasional final BF16 midpoint, but
    # cannot replace either BF16 input with its unrounded FP32 producer.
    relative_rmse = np.sqrt(np.mean((actual - expected) ** 2)) / np.sqrt(np.mean(expected**2))
    assert relative_rmse < 2e-5, relative_rmse
    assert np.mean(actual != expected) < 1e-4
