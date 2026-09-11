"""Original HCA kernels: explicit V4 numerics and existing physical-page ABI."""

from dataclasses import replace

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.deepseek_v4 import hca
from sgl_jax.srt.kernels.deepseek_v4.compressor import _project, compress, physical_locations
from sgl_jax.srt.kernels.deepseek_v4.numerics import (
    V4LayerConfig,
    _single_query_attention,
    rms_norm,
    rope,
)
from sgl_jax.srt.kernels.low_bit.formats import round_bf16
from sgl_jax.srt.kernels.hca.compressor import _hca_emit_selected_pallas
from sgl_jax.srt.kernels.hca.attention import _v4_probability_sum_64
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.test.kernels.test_deepseek_v4_reference import _compressor_weights
from sgl_jax.test.test_deepseek_v4_paged import make_batch, make_cache

CONFIG = V4LayerConfig(ratio=128, max_context=384)
TPU = pytest.mark.skipif(jax.default_backend() != "tpu", reason="real Mosaic lowering required")


def close(a, b, tolerance=2e-4, *, label=""):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    assert a.shape == b.shape
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    error = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)
    assert error < tolerance, (label, error)


@pytest.mark.parametrize("mode", ["auto", "xla", "unknown", None])
def test_hca_has_no_automatic_fallback(mode):
    with pytest.raises(ValueError, match="no automatic fallback"):
        hca.validate_backend(mode)


@TPU
def test_probability_reduction_preserves_retained_fp32_bits():
    # Rare FP32 denominator differences cross BF16 midpoints; an aggregate
    # operator tolerance missed the real position-39 failure. Require exact
    # sums over varied masked and unmasked probability inputs instead.
    rng = np.random.default_rng(39)

    @jax.jit
    def probability_and_sum(q, kv, length):
        score = jnp.matmul(q, kv.T, preferred_element_type=jnp.float32) * 512**-0.5
        valid = jnp.arange(64)[None] < length
        score = jnp.where(valid, score, -1e30)
        probability = jnp.where(valid, jnp.exp(score - jnp.max(score, axis=-1, keepdims=True)), 0)
        return probability, jnp.sum(probability, axis=-1)

    values, reference = [], []
    # The retained reducer consumes an MXU-produced matrix. Reducing a plain
    # [64,64] HBM input is a different XLA layout, not this numerical contract.
    for scale in (0.1, 1.0, 4.0):
        for length in (0, 1, 7, 8, 31, 32, 39, 40, 63, 64):
            q, kv = [
                jnp.asarray(rng.normal(scale=scale, size=(64, 512)), jnp.bfloat16) for _ in range(2)
            ]
            probability, denominator = probability_and_sum(q, kv, jnp.int32(length))
            values.append(np.asarray(probability))
            reference.append(np.asarray(denominator))
    values, reference = np.stack(values), np.stack(reference)

    def kernel(x_ref, y_ref):
        y_ref[0] = jnp.broadcast_to(_v4_probability_sum_64(x_ref[0]), (64, 128))

    call = jax.jit(
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((len(values), 64, 128), jnp.float32),
            grid=(len(values),),
            in_specs=[pl.BlockSpec((1, 64, 64), lambda i: (i, 0, 0))],
            out_specs=pl.BlockSpec((1, 64, 128), lambda i: (i, 0, 0)),
            name="v4-hca-probability-sum-regression",
        )
    )
    actual = np.asarray(call(jnp.asarray(values)))[:, :, 0]
    stripes = values.reshape(len(values), 64, 2, 4, 8)
    expected = ((stripes[..., 0, :] + stripes[..., 1, :]) + stripes[..., 2, :]) + stripes[..., 3, :]
    for width in (4, 2, 1):
        expected = expected[..., :width] + expected[..., width : 2 * width]
    expected = expected[:, :, 0, 0] + expected[:, :, 1, 0]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, reference)


@TPU
@pytest.mark.parametrize("tokens", [1, 8, 128])
def test_original_projection_matches_retained_arithmetic(tokens):
    rng = np.random.default_rng(8023)
    x = jnp.asarray(rng.normal(size=(tokens, 4096)), jnp.bfloat16)
    a, b = [jnp.asarray(rng.normal(scale=0.01, size=(512, 4096)), jnp.bfloat16) for _ in range(2)]
    metadata = V4PagedBackend(max_context=384).get_forward_metadata(
        make_batch([0], [tokens], decode=tokens == 1)
    )
    run = jax.jit(lambda x, a, b, m: hca.project(x, a, b, m, CONFIG))
    actual = run(x, a, b, metadata)
    reference = jax.jit(lambda x, w, m: _project(x, w, m))
    for weight, result in zip((a, b), actual, strict=True):
        expected = reference(x, weight, metadata)
        if tokens == 1:
            # The original MXU projection and retained XLA GEMV have different
            # F32 reduction trees. This gate is 10,000x tighter than full-logit
            # tolerance; batch/chunk state invariance is separately bitwise.
            close(expected, result, tolerance=5e-7)
        else:
            np.testing.assert_array_equal(expected, result)
        close(np.asarray(x, np.float64) @ np.asarray(weight, np.float64).T, result, tolerance=5e-7)
    calls = [
        line
        for line in run.lower(x, a, b, metadata).compile().as_text().splitlines()
        if "custom-call(" in line
    ]
    assert any("hca-state-project" in line for line in calls)


@TPU
@pytest.mark.parametrize("entries", [1, 4, 9, 65])
def test_original_emitter_matches_retained_pool_norm_rope(entries):
    rng = np.random.default_rng(128)
    values, scores = [
        jnp.asarray(rng.normal(size=(entries, 128, 512)), jnp.float32) for _ in range(2)
    ]
    norm = jnp.asarray(rng.normal(1, 0.1, (512,)), jnp.bfloat16)
    starts = jnp.arange(entries, dtype=jnp.int32) * 128
    valid = jnp.arange(entries) != entries - 1 if entries > 1 else jnp.ones((1,), jnp.bool_)
    run = jax.jit(lambda v, s, n, p, live: hca.emit(v, s, n, p, live, CONFIG))
    actual = run(values, scores, norm, starts, valid)
    expected = jax.jit(
        lambda v, s, n, p: rope(
            rms_norm(round_bf16(jnp.sum(v * jax.nn.softmax(s, axis=1), axis=1)), n, CONFIG.eps),
            p,
            CONFIG,
        )
    )(values, scores, norm, starts)
    close(
        np.asarray(expected)[np.asarray(valid)],
        np.asarray(actual)[np.asarray(valid)],
        label="retained pool/norm/rope",
    )
    np.testing.assert_array_equal(
        np.asarray(expected)[np.asarray(valid)].view(np.uint16),
        np.asarray(actual)[np.asarray(valid)].view(np.uint16),
    )
    np.testing.assert_array_equal(np.asarray(actual)[~np.asarray(valid)], 0)
    # Independent FP64 pooling and RMS, with BF16 checkpoint boundaries.
    v, s = np.asarray(values, np.float64), np.asarray(scores, np.float64)
    prob = np.exp(s - s.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    pooled = np.sum(v * prob, axis=1).astype(ml_dtypes.bfloat16).astype(np.float64)
    normed = (
        (
            pooled
            / np.sqrt(np.mean(pooled**2, axis=-1, keepdims=True) + CONFIG.eps)
            * np.asarray(norm, np.float64)
        )
        .astype(ml_dtypes.bfloat16)
        .astype(np.float64)
    )
    # The original emitter ABI accepts cos/sin tables, not absolute positions.
    # Supply identical CPU-generated FP32 tables to that kernel and its FP64
    # oracle. The adapter's TPU-generated tables were checked above against
    # the retained path. Mixing separately computed tables is not an identical-
    # input kernel test: TPU/NumPy pow differ measurably at long positions.
    frequency = np.float32(1) / np.power(
        np.float32(CONFIG.rope_base), np.arange(0, 64, 2, dtype=np.float32) / np.float32(64)
    )
    angles = (np.asarray(starts, np.float32)[:, None] * frequency[None]).astype(np.float64)
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)
    oracle_actual = _hca_emit_selected_pallas(
        jnp.stack((values, scores), axis=2),
        valid,
        norm,
        jnp.asarray(np.concatenate((cosine, sine), axis=-1)),
        schedule=hca.schedule_for(CONFIG),
        norm_eps=CONFIG.eps,
        numerical_mode="v4",
    )
    cosine, sine = cosine.astype(np.float64), sine.astype(np.float64)
    a, b = normed[:, -64::2].copy(), normed[:, -63::2].copy()
    normed[:, -64::2] = a * cosine - b * sine
    normed[:, -63::2] = a * sine + b * cosine
    close(
        normed.astype(ml_dtypes.bfloat16)[np.asarray(valid)],
        np.asarray(oracle_actual)[np.asarray(valid)],
        label="FP64 pool/norm/rotation with identical input tables",
    )
    calls = [
        line
        for line in run.lower(values, scores, norm, starts, valid).compile().as_text().splitlines()
        if "custom-call(" in line
    ]
    assert any("hca-boundary-snapshot" in line and "-v4" in line for line in calls)


@TPU
@pytest.mark.parametrize(
    "prefixes,counts",
    [
        ([0], [1]),
        ([39, 65], [1, 1]),
        ([127, 254], [1, 1]),
        ([121, 250], [8, 9]),
        ([8023, 8191], [1, 1]),
        ([8191, 8192], [1, 1]),
        ([8223, 8319], [1, 1]),
    ],
)
def test_original_attention_noncontiguous_token_pages(prefixes, counts):
    rng = np.random.default_rng(8023)
    context = max(384, (max(p + n for p, n in zip(prefixes, counts)) + 127) // 128 * 128)
    config = replace(CONFIG, max_context=context)
    pages = (
        None if context == 384
        else rng.permutation(np.arange(1, 1 + 2 * (context // 128))).reshape(2, -1).tolist()
    )
    capacity = 12 if pages is None else 1 + 2 * (context // 128)
    batch = make_batch(prefixes, counts, pages=pages, padding=3)
    meta = V4PagedBackend(max_context=context).get_forward_metadata(batch)
    positions = jnp.asarray(batch.positions)
    tokens = len(positions)
    q = jnp.asarray(rng.normal(size=(tokens, 64, 512)), jnp.bfloat16)
    window = jnp.asarray(rng.normal(size=(capacity * 128, 512)), jnp.bfloat16).at[:128].set(0)
    compressed = jnp.asarray(rng.normal(size=(capacity, 512)), jnp.bfloat16).at[0].set(0)
    sink = jnp.asarray(rng.normal(size=(64,)), jnp.float32).at[0].set(1000)
    logical = jnp.maximum(positions[:, None] - 127, 0) + jnp.arange(128)[None]
    indices = physical_locations(meta, meta.token_requests, logical)
    indices = jnp.where((logical <= positions[:, None]) & meta.token_valid[:, None], indices, -1)
    physical = meta.page_table[meta.token_requests]
    cids = jnp.where(
        (jnp.arange(context // 128)[None] < (positions[:, None] + 1) // 128)
        & meta.token_valid[:, None],
        physical + len(window),
        -1,
    )
    all_indices = jnp.concatenate((indices, cids), axis=1)
    expected = jax.jit(
        lambda q, kv, ids, sink: jax.lax.map(
            lambda item: _single_query_attention(item[0], kv, item[1], sink, 512**-0.5), (q, ids)
        )
    )(q, jnp.concatenate((window, compressed)), all_indices, sink)
    run = jax.jit(lambda q, w, c, ids, p, s, m: hca.attend(q, w, c, ids, p, s, m, config))
    actual = run(q, window, compressed, indices, positions, sink, meta)
    close(expected, actual)
    # Do not let aggregate tolerance hide BF16 midpoint flips again.
    np.testing.assert_array_equal(
        np.asarray(expected).view(np.uint16), np.asarray(actual).view(np.uint16)
    )
    np.testing.assert_array_equal(np.asarray(actual)[~np.asarray(meta.token_valid)], 0)
    calls = [
        line
        for line in run.lower(q, window, compressed, indices, positions, sink, meta)
        .compile()
        .as_text()
        .splitlines()
        if "custom-call(" in line
    ]
    assert any("hca-paged-stream" in line and "-v4" in line for line in calls)
    order = jnp.arange(tokens - 1, -1, -1)
    reordered = replace(
        meta, token_requests=meta.token_requests[order], token_valid=meta.token_valid[order]
    )
    np.testing.assert_array_equal(
        actual[order],
        run(q[order], window, compressed, indices[order], positions[order], sink, reordered),
    )
    singleton = replace(
        meta, token_requests=meta.token_requests[:1], token_valid=meta.token_valid[:1]
    )
    np.testing.assert_array_equal(
        actual[:1], run(q[:1], window, compressed, indices[:1], positions[:1], sink, singleton)
    )


@TPU
def test_hca_state_emission_and_prefix_fork_are_bitwise_chunk_batch_invariant():
    rng = np.random.default_rng(8023)
    weights = _compressor_weights(rng, CONFIG)
    inputs = [jnp.asarray(rng.normal(size=(271, 4096)), jnp.bfloat16) for _ in range(2)]
    backend = V4PagedBackend(max_context=384)

    @jax.jit
    def run(x, w, c, m):
        return compress(x, w, dict(c), CONFIG, m, hca_backend="pallas")

    expected = [
        run(x, weights, make_cache(CONFIG), backend.get_forward_metadata(make_batch([0], [271])))
        for x in inputs
    ]
    cache = make_cache(CONFIG)
    prior = [0, 0]
    for lengths, order in [
        ([3, 7], [0, 1]),
        ([127, 128], [1, 0]),
        ([128, 129], [0, 1]),
        ([259, 259], [1, 0]),
        ([260, 260], [1, 0]),
        ([271, 271], [0, 1]),
    ]:
        counts = [lengths[i] - prior[i] for i in order]
        batch = make_batch(
            [prior[i] for i in order],
            counts,
            slots=order,
            pages=[[1 + i * 3, 3 + i * 3, 2 + i * 3] for i in order],
            padding=5,
        )
        x = jnp.pad(
            jnp.concatenate([inputs[i][prior[i] : lengths[i]] for i in order]), ((0, 5), (0, 0))
        )
        cache = run(x, weights, cache, backend.get_forward_metadata(batch))
        prior = lengths
    for request in range(2):
        for suffix in ("kv", "score"):
            np.testing.assert_array_equal(
                cache["main." + suffix][request + 1], expected[request]["main." + suffix][1]
            )
            np.testing.assert_array_equal(
                np.asarray(cache["main.snapshot_" + suffix])[[1 + 3 * request, 3 + 3 * request]],
                np.asarray(expected[request]["main.snapshot_" + suffix])[[1, 3]],
            )
        np.testing.assert_array_equal(
            np.asarray(cache["main.compressed"])[[1 + 3 * request, 3 + 3 * request]],
            np.asarray(expected[request]["main.compressed"])[[1, 3]],
        )
    # Restore a shared prefix into an unrelated dirty slot; the other request
    # must not change. No scheduler/allocator-specific state ownership is added.
    for suffix in ("kv", "score"):
        cache["main." + suffix] = cache["main." + suffix].at[4].set(999)
    other = [cache["main." + suffix][2] for suffix in ("kv", "score")]
    fork = make_batch([256], [15], slots=[3], pages=[[1, 3, 8]], padding=1)
    cache = run(
        jnp.pad(inputs[0][256:], ((0, 1), (0, 0))),
        weights,
        cache,
        backend.get_forward_metadata(fork),
    )
    for suffix, untouched in zip(("kv", "score"), other, strict=True):
        np.testing.assert_array_equal(cache["main." + suffix][4], expected[0]["main." + suffix][1])
        np.testing.assert_array_equal(cache["main." + suffix][2], untouched)
    fresh = make_batch([0, 0], [3, 0], slots=[3, 1], pages=[[8, 9, 10], [4, 6, 5]], padding=5)
    cache = run(
        jnp.pad(inputs[1][:3], ((0, 5), (0, 0))),
        weights,
        cache,
        backend.get_forward_metadata(fresh),
    )
    empty = run(
        inputs[1][:3],
        weights,
        make_cache(CONFIG),
        backend.get_forward_metadata(make_batch([0], [3])),
    )
    for suffix, untouched in zip(("kv", "score"), other, strict=True):
        np.testing.assert_array_equal(cache["main." + suffix][4], empty["main." + suffix][1])
        np.testing.assert_array_equal(cache["main." + suffix][2], untouched)
