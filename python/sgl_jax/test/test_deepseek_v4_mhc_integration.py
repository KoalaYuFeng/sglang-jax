"""Existing Pallas mHC kernels must implement V4 semantics without fallback."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.deepseek_v4.mhc import head_collapse, post, validate_backend
from sgl_jax.srt.kernels.mhc import mhc_head_collapse_fused


def _bf16(value):
    return np.asarray(value).astype(ml_dtypes.bfloat16)


def _head_oracle(streams, fn, scale, base):
    """Independent FP64 oracle, with no normalized-input BF16 boundary."""
    x = np.asarray(streams, np.float64)
    flat = x.reshape(x.shape[0], -1)
    inverse = 1 / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + 1e-6)
    mixes = (flat @ np.asarray(fn, np.float64).T) * inverse
    gate = 1 / (1 + np.exp(-(mixes * scale + base))) + 1e-6
    return _bf16(np.sum(x * gate[..., None], axis=1))


def _inputs(tokens):
    rng = np.random.default_rng(8023)
    return (
        _bf16(rng.normal(0, 0.1, (tokens, 4, 4096))),
        rng.normal(0, 0.01, (4, 4 * 4096)).astype(np.float32),
        np.asarray([0.8], np.float32),
        rng.normal(0, 0.05, (4,)).astype(np.float32),
    )


def _close(expected, actual):
    a, b = np.asarray(expected, np.float32), np.asarray(actual, np.float32)
    assert a.shape == b.shape
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    nrmse = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)
    # FP32 reduction can straddle isolated BF16 midpoints. This operator gate
    # is 25x tighter than the unchanged 0.5% full-logit acceptance threshold.
    assert nrmse < 2e-4, nrmse


@contextmanager
def _single_device():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1], object), ("tensor",))
    with jax.set_mesh(mesh):
        yield


@pytest.mark.parametrize("backend", ["auto", "xla", "unknown", None])
def test_no_implicit_backend_fallback(backend):
    with pytest.raises(ValueError, match="no automatic fallback"):
        validate_backend(backend)


@pytest.mark.parametrize("backend", ["pallas", "reference"])
def test_model_layer_selects_explicit_post_backend(monkeypatch, backend):
    from types import SimpleNamespace

    import sgl_jax.srt.models.deepseek_v4 as model

    selected = []
    monkeypatch.setattr(model, "mhc_pre_fused", lambda x, *a, **k: (x[:, 0], None, None))
    monkeypatch.setattr(model.DenseKernels, "norm", lambda self, x, *a: x)
    monkeypatch.setattr(model, "attention", lambda x, p, w, c, *a, **k: (x, c))
    monkeypatch.setattr(model, "moe", lambda x, *a, **k: x)

    def capture_post(x, residual, *args, backend):
        selected.append(backend)
        return residual

    monkeypatch.setattr(model, "mhc_post", capture_post)
    weights = {
        **{f"hc_{s}_{k}": None for s in ("attn", "ffn") for k in ("fn", "scale", "base")},
        "attn_norm.weight": None,
        "ffn_norm.weight": None,
    }
    config = SimpleNamespace(hc=4, sinkhorn_iters=20, eps=1e-6, hc_eps=1e-6)
    layer = model.DeepseekV4DecoderLayer(config, mhc_backend=backend)
    streams = jnp.ones((1, 4, 8), jnp.bfloat16)
    metadata = SimpleNamespace(token_valid=jnp.asarray([True]))
    result, _ = layer(streams, None, None, weights, {}, metadata, None)
    np.testing.assert_array_equal(result, streams)
    assert selected == [backend, backend]


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real Mosaic lowering")
@pytest.mark.parametrize("tokens", [1, 2, 4, 17, 128])
def test_pallas_head_matches_fp64_and_retained_reference(tokens):
    inputs = _inputs(tokens)
    expected = _head_oracle(*inputs)
    with _single_device():
        values = tuple(jnp.asarray(x) for x in inputs)
        actual = head_collapse(*values, backend="pallas")
        reference = head_collapse(*values, backend="reference")
        _close(expected, actual)
        _close(reference, actual)


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real Mosaic lowering")
def test_pallas_head_is_batch_and_order_invariant():
    inputs = _inputs(128)
    order = np.arange(127, -1, -1)
    with _single_device():
        streams, fn, scale, base = (jnp.asarray(x) for x in inputs)
        packed = np.asarray(head_collapse(streams, fn, scale, base))
        reordered = np.asarray(head_collapse(streams[order], fn, scale, base))
        np.testing.assert_array_equal(reordered, packed[order])
        for count in (1, 2, 4):
            small = head_collapse(streams[:count], fn, scale, base)
            np.testing.assert_array_equal(np.asarray(small), packed[:count])


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real Mosaic lowering")
def test_pallas_post_is_batch_and_order_invariant():
    residual, _, _, _ = _inputs(128)
    rng = np.random.default_rng(8023)
    x = _bf16(rng.normal(0, 0.1, (128, 4096)))
    gates = rng.uniform(0.1, 1.9, (128, 4)).astype(np.float32)
    comb = rng.uniform(0.01, 1, (128, 4, 4)).astype(np.float32)
    comb /= comb.sum(axis=1, keepdims=True)
    order = np.arange(127, -1, -1)
    with _single_device():
        values = tuple(jnp.asarray(a) for a in (x, residual, gates, comb))
        packed = np.asarray(post(*values))
        reordered = np.asarray(post(*(a[order] for a in values)))
        np.testing.assert_array_equal(reordered, packed[order])
        for count in (1, 2, 4):
            small = np.asarray(post(*(a[:count] for a in values)))
            np.testing.assert_array_equal(small, packed[:count])


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real Mosaic lowering")
@pytest.mark.parametrize("tokens", [1, 2, 4, 128])
def test_integrated_post_head_contains_executed_pallas_calls(tokens):
    residual, fn, scale, base = _inputs(tokens)
    rng = np.random.default_rng(7939)
    raw_x = rng.normal(0, 0.1, (tokens, 4096)).astype(np.float32)
    gates = rng.uniform(0.1, 1.9, (tokens, 4)).astype(np.float32)
    comb = rng.uniform(0.01, 1, (tokens, 4, 4)).astype(np.float32)
    comb /= comb.sum(axis=1, keepdims=True)
    expected_post = _bf16(
        gates.astype(np.float64)[..., None] * _bf16(raw_x).astype(np.float64)[:, None]
        + np.einsum("nij,nid->njd", comb.astype(np.float64), residual.astype(np.float64))
    )

    @jax.jit
    def run(x, r, g, c, f, s, b):
        mixed = post(x.astype(jnp.bfloat16), r, g, c, backend="pallas")
        return mixed, head_collapse(mixed, f, s, b, backend="pallas")

    with _single_device():
        args = tuple(jnp.asarray(x) for x in (raw_x, residual, gates, comb, fn, scale, base))
        compiled = run.lower(*args).compile()
        mixed, output = compiled(*args)
        _close(expected_post, mixed)
        # Test the head on its actual materialized BF16 input, so a post
        # midpoint cannot be mistaken for an error in the head's contract.
        _close(_head_oracle(np.asarray(mixed), fn, scale, base), output)
        hlo = compiled.as_text()
        assert "mhc-post" in hlo
        assert "mhc-collapse-head-fp32-post" in hlo
        assert "custom-call" in hlo


@pytest.mark.parametrize(
    "mode,precision,match",
    [
        ("other", jax.lax.Precision.HIGHEST, "head_rms_mode"),
        ("fp32_post", jax.lax.Precision.DEFAULT, "HIGHEST"),
    ],
)
def test_invalid_head_contract_rejected(mode, precision, match):
    # The contract is rejected before any Pallas lowering or device schedule.
    with pytest.raises(ValueError, match=match):
        mhc_head_collapse_fused(
            jnp.zeros((1, 4, 8), jnp.bfloat16),
            jnp.zeros((4, 32), jnp.float32),
            jnp.ones((1,), jnp.float32),
            jnp.zeros((4,), jnp.float32),
            hc_mult=4,
            norm_eps=1e-6,
            hc_eps=1e-6,
            dot_precision=precision,
            head_rms_mode=mode,
        )
