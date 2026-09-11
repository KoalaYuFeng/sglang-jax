"""Original CSA programs: V4 numerical boundaries and paged/ragged ABI."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.kernels.csa.compressor import csa_emit_selected_pallas
from sgl_jax.srt.kernels.deepseek_v4 import csa
from sgl_jax.srt.kernels.deepseek_v4.compressor import _project, compress, physical_locations
from sgl_jax.srt.kernels.deepseek_v4.numerics import (
    V4LayerConfig,
    _single_query_attention,
    rms_norm,
    rope,
)
from sgl_jax.srt.kernels.low_bit.formats import activation_fp4_roundtrip, round_bf16
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.test.kernels.csa_compressor_cases import make_case, select_channels
from sgl_jax.test.kernels.test_deepseek_v4_reference import _compressor_weights
from sgl_jax.test.test_deepseek_v4_paged import make_batch, make_cache

CONFIG = V4LayerConfig(ratio=4, max_context=8192)
TPU = pytest.mark.skipif(jax.default_backend() != "tpu", reason="real Mosaic lowering required")


def close(expected, actual, tolerance):
    expected, actual = np.asarray(expected, np.float64), np.asarray(actual, np.float64)
    assert expected.shape == actual.shape
    assert np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))
    error = np.linalg.norm(expected - actual) / max(np.linalg.norm(expected), 1e-12)
    assert error < tolerance, error


@pytest.mark.parametrize("mode", ["auto", "xla", "unknown", None])
def test_no_automatic_csa_fallback(mode):
    with pytest.raises(ValueError, match="no automatic fallback"):
        csa.validate_backend(mode)


@TPU
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("tokens", [1, 8, 128])
def test_original_csa_projection_fp32_and_mxu_dispatch(dim, tokens):
    rng = np.random.default_rng(8023)
    x = jnp.asarray(rng.normal(size=(tokens, 4096)), jnp.bfloat16)
    weights = [
        jnp.asarray(rng.normal(scale=0.01, size=(2 * dim, 4096)), jnp.bfloat16) for _ in range(2)
    ]
    meta = V4PagedBackend(max_context=8192).get_forward_metadata(
        make_batch([0], [tokens], decode=tokens == 1)
    )
    run = jax.jit(lambda x, a, b, m: csa.project(x, a, b, m, CONFIG))
    actual = run(x, *weights, meta)
    for weight, value in zip(weights, actual, strict=True):
        assert value.dtype == jnp.float32
        expected = jax.jit(_project)(x, weight, meta)
        np.testing.assert_array_equal(expected, value)
        close(np.asarray(x, np.float64) @ np.asarray(weight, np.float64).T, value, 5e-7)
    hlo = run.lower(x, *weights, meta).compile().as_text()
    assert any(
        "custom-call(" in line and "csa-compressor-project" in line for line in hlo.splitlines()
    )


@TPU
@pytest.mark.parametrize("dim", [128, 512])
def test_original_csa_decode_projection_exact_across_lanes_batches_and_mixed_queries(dim):
    rng = np.random.default_rng(7979)
    weights = [
        jnp.asarray(rng.normal(scale=0.03, size=(2 * dim, 4096)), jnp.bfloat16) for _ in range(2)
    ]
    backend = V4PagedBackend(max_context=8192)
    run = jax.jit(lambda x, a, b, m: csa.project(x, a, b, m, CONFIG))
    reference = jax.jit(_project)
    pages = [list(range(1 + i * 64, 65 + i * 64)) for i in range(3)]
    for offset in range(8):
        x = jnp.asarray(rng.normal(size=(2, 4096)), jnp.bfloat16)
        for batch, activation in (
            (make_batch([7976 + offset], [1], pages=pages[:1], decode=True), x[:1]),
            (
                make_batch(
                    [7976 + offset, 0, 8104 + offset],
                    [1, 0, 1],
                    pages=pages,
                    padding=1,
                    decode=True,
                ),
                jnp.stack((x[0], jnp.zeros((4096,), jnp.bfloat16), x[1])),
            ),
        ):
            if len(batch.seq_lens) == 3:
                for field in ("positions", "out_cache_loc"):
                    value = getattr(batch, field).copy()
                    value[2], value[1] = value[1], 0 if field == "positions" else -1
                    setattr(batch, field, value)
            meta = backend.get_forward_metadata(batch)
            actual = run(activation, *weights, meta)
            live = np.asarray(meta.token_valid)
            for w, value in zip(weights, actual, strict=True):
                expected = reference(activation, w, meta)
                np.testing.assert_array_equal(np.asarray(expected)[live], np.asarray(value)[live])
        mixed = jnp.asarray(rng.normal(size=(10, 4096)), jnp.bfloat16)
        meta = backend.get_forward_metadata(
            make_batch([7968, 7976 + offset], [9, 1], pages=pages[:2])
        )
        for w, value in zip(weights, run(mixed, *weights, meta), strict=True):
            np.testing.assert_array_equal(reference(mixed, w, meta), value)


@TPU
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("groups", [1, 4, 9, 65])
def test_original_csa_selected_emitter_matches_precision_contract_and_fp64(dim, groups):
    rng = np.random.default_rng(4)
    v, s = [jnp.asarray(rng.normal(size=(groups, 8, dim)), jnp.float32) for _ in range(2)]
    # First group has no previous overlap; padding is an independent mask.
    s = s.at[0, :4].set(-jnp.inf)
    norm = jnp.asarray(rng.normal(1, 0.1, (dim,)), jnp.bfloat16)
    starts = jnp.arange(groups, dtype=jnp.int32) * 4 + 7932
    valid = jnp.ones((groups,), jnp.bool_).at[-1].set(groups == 1)
    run = jax.jit(lambda v, s, n, p, live: csa.emit(v, s, n, p, live, CONFIG))
    actual = run(v, s, norm, starts, valid)
    # Independent FP64 pooling, never the candidate's polynomial/helper.
    value, score = np.asarray(v, np.float64), np.asarray(s, np.float64)
    probability = np.exp(score - score.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    pooled = np.sum(value * probability, axis=1).astype(ml_dtypes.bfloat16).astype(np.float64)
    if dim == 128:
        # Index pooling deliberately corrects the old exp/softmax approximation.
        # Keep a bitwise gate, but against independently rounded FP64 pooling,
        # not the obsolete implementation's rounding error. The independent
        # official-Python matrix is separate and retains its original limits.
        expected = jax.jit(lambda v, n, p: rope(rms_norm(v, n, CONFIG.eps), p, CONFIG))(
            jnp.asarray(pooled, jnp.bfloat16), norm, starts
        )
    else:
        expected = jax.jit(
            lambda v, s, n, p: rope(
                rms_norm(round_bf16(jnp.sum(v * jax.nn.softmax(s, axis=1), axis=1)), n, CONFIG.eps),
                p,
                CONFIG,
            )
        )(v, s, norm, starts)
    live = np.asarray(valid)
    np.testing.assert_array_equal(
        np.asarray(actual)[live].view(np.uint16), np.asarray(expected)[live].view(np.uint16)
    )
    np.testing.assert_array_equal(np.asarray(actual)[~live], 0)
    # Independent FP64 pooling/RMS/RoPE with identical FP32 input tables.
    normalized = (
        (
            pooled
            / np.sqrt(np.mean(pooled**2, axis=-1, keepdims=True) + CONFIG.eps)
            * np.asarray(norm, np.float64)
        )
        .astype(ml_dtypes.bfloat16)
        .astype(np.float64)
    )
    frequency = np.float32(1) / np.power(
        np.float32(CONFIG.rope_base), np.arange(0, 64, 2, dtype=np.float32) / np.float32(64)
    )
    phase = (np.asarray(starts, np.float32)[:, None] * frequency).astype(np.float64)
    cosine, sine = np.cos(phase).astype(np.float32), np.sin(phase).astype(np.float32)
    oracle_actual = csa_emit_selected_pallas(
        v, s, norm, jnp.asarray(np.concatenate((cosine, sine), axis=-1)), valid, norm_eps=CONFIG.eps
    )
    a, b = normalized[:, -64::2].copy(), normalized[:, -63::2].copy()
    normalized[:, -64::2] = a * cosine.astype(np.float64) - b * sine.astype(np.float64)
    normalized[:, -63::2] = a * sine.astype(np.float64) + b * cosine.astype(np.float64)
    close(normalized.astype(ml_dtypes.bfloat16)[live], np.asarray(oracle_actual)[live], 2e-4)
    hlo = run.lower(v, s, norm, starts, valid).compile().as_text()
    assert any(
        "custom-call(" in line and "csa-compressor-snapshot" in line for line in hlo.splitlines()
    )


@TPU
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("groups", [2, 5, 33])
def test_csa_raw_window_adapter_preserves_selected_abi_without_outer_gather(dim, groups):
    case = make_case(dim, groups)
    starts = jnp.arange(groups, dtype=jnp.int32) * 4 + 7976
    args = (
        jnp.asarray(case.values),
        jnp.asarray(case.scores),
        jnp.asarray(case.norm),
        starts,
        jnp.asarray(case.valid),
    )
    run = jax.jit(lambda v, s, n, p, live: csa.emit(v, s, n, p, live, CONFIG, raw_overlap=True))
    # Both production adapters generate RoPE inside the encompassing JIT.
    # An eager primitive-by-primitive table has different TPU transcendental
    # lowering, and is not the previous production compilation boundary.
    selected_run = jax.jit(lambda v, s, n, p, live: csa.emit(v, s, n, p, live, CONFIG))
    expected = selected_run(
        jnp.asarray(select_channels(case.values)),
        jnp.asarray(np.where(case.valid[:, None, None], select_channels(case.scores), 0)),
        *args[2:],
    )
    np.testing.assert_array_equal(
        np.asarray(run(*args)).view(np.uint16), np.asarray(expected).view(np.uint16)
    )
    instructions = [
        line.partition("backend_config=")[0]
        for line in run.lower(*args).compile().as_text().splitlines()
    ]
    assert any("custom-call(" in line and "v4-overlap-t1" in line for line in instructions)
    assert not any(" gather(" in line for line in instructions)


@TPU
@pytest.mark.parametrize("index", [False, True])
def test_csa_raw_window_integration_trace_views_are_not_kernel_intermediates(index):
    rng = np.random.default_rng(7979)
    cfg = replace(CONFIG, max_context=384)
    weights = _compressor_weights(rng, cfg, index=index)
    x = jnp.asarray(rng.normal(size=(16, 4096)), jnp.bfloat16)
    meta = V4PagedBackend(max_context=384).get_forward_metadata(make_batch([0], [16]))
    cache = make_cache(cfg)

    def call(x, w, c, m, *, traced):
        trace = {} if traced else None
        result = compress(x, w, dict(c), cfg, m, index=index, csa_backend="pallas", trace=trace)
        return (result, trace) if traced else result

    run = jax.jit(lambda x, w, c, m: call(x, w, c, m, traced=False))
    observed, trace = jax.jit(lambda x, w, c, m: call(x, w, c, m, traced=True))(
        x, weights, cache, meta
    )
    for name, expected in run(x, weights, cache, meta).items():
        np.testing.assert_array_equal(expected, observed[name], err_msg=name)
    assert {"raw_group_values", "raw_group_scores", "csa_emitted"} <= trace.keys()
    assert {"group_values", "group_scores", "pooled", "normalized"}.isdisjoint(trace)
    np.testing.assert_array_equal(
        trace["selected_view_values"], select_channels(np.asarray(trace["raw_group_values"]))
    )
    np.testing.assert_array_equal(
        trace["selected_view_scores"],
        np.where(
            np.asarray(meta.group4_starts >= 0)[:, None, None],
            select_channels(np.asarray(trace["raw_group_scores"])),
            0,
        ),
    )
    hlo = run.lower(x, weights, cache, meta).compile().as_text()
    assert "v4-overlap-t1" in hlo
    # Other gathers own request/page lookup; only the old channel-half
    # take_along_axis operations should disappear from normal execution.
    assert "take_along_axis" not in hlo


@TPU
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("case", ["short", "8k", "decode_inert", "8320"])
def test_original_csa_indexer_scores_topk_and_dynamic_ragged_pages(case, ties):
    rng = np.random.default_rng(8023)
    context = 384 if case == "short" else (8320 if case == "8320" else 8192)
    cfg = replace(CONFIG, max_context=context)
    pages = rng.permutation(np.arange(1, 1 + 3 * (context // 128))).reshape(3, -1).tolist()
    prefixes, counts = (
        ([0, 124, 127], [9, 3, 1]) if case == "short" else ([7932, 8022, 8186], [8, 2, 6])
    )
    if case == "decode_inert":
        prefixes, counts = [8023, 0, 8191], [1, 0, 1]
    if case == "8320":
        prefixes, counts = [8190, 8218, 8318], [4, 6, 2]
    batch = make_batch(
        prefixes, counts, pages=pages, padding=3, decode=case == "decode_inert", slots=[3, 1, 0]
    )
    if case == "decode_inert":
        for field in ("positions", "out_cache_loc"):
            value = getattr(batch, field).copy()
            value[2], value[1] = value[1], 0 if field == "positions" else -1
            setattr(batch, field, value)
    meta = V4PagedBackend(max_context=context).get_forward_metadata(batch)
    tokens = len(batch.input_ids)
    query = activation_fp4_roundtrip(jnp.asarray(rng.normal(size=(tokens, 64, 128)), jnp.bfloat16))
    if ties:
        query = jnp.zeros_like(query)
    mixing = jnp.asarray(rng.normal(scale=0.01, size=(tokens, 64)), jnp.bfloat16)
    cache = activation_fp4_roundtrip(
        jnp.asarray(rng.normal(size=((1 + 3 * context // 128) * 32, 128)), jnp.bfloat16)
    )

    @jax.jit
    def reference(q, w, cache, metadata):
        candidates = jnp.arange(context // 4)

        def one(item):
            q, w, request, position = item
            physical = physical_locations(metadata, request[None], ((candidates + 1) * 4 - 1)[None])
            keys = cache[physical[0] // 4]
            score = round_bf16(jnp.einsum("hd,td->ht", q, keys, preferred_element_type=jnp.float32))
            score = round_bf16(
                jnp.maximum(score.astype(jnp.float32), 0) * w.astype(jnp.float32)[:, None]
            )
            score = round_bf16(jnp.sum(score.astype(jnp.float32)[None], axis=1)[0]).astype(
                jnp.float32
            )
            score = jnp.where(candidates < (position + 1) // 4, score, -jnp.inf)
            return score, jax.lax.top_k(score, min(512, context // 4))[1]

        requests = metadata.token_requests
        positions = (
            metadata.prefix_lens[requests] + jnp.arange(tokens) - metadata.query_starts[requests]
        )
        return jax.lax.map(one, (q, w, requests, positions))

    traces = []

    @jax.jit
    def run(q, w, cache, m):
        traces.append(1)
        return csa.select(q, w, cache, m, cfg)

    reordered = make_batch(
        prefixes[::-1],
        counts[::-1],
        pages=pages[::-1],
        slots=[0, 1, 3],
        padding=3,
        decode=case == "decode_inert",
    )
    if case == "decode_inert":
        for field in ("positions", "out_cache_loc"):
            value = getattr(reordered, field).copy()
            value[2], value[1] = value[1], 0 if field == "positions" else -1
            setattr(reordered, field, value)
    changed = V4PagedBackend(max_context=context).get_forward_metadata(reordered)
    for metadata in (meta, changed):
        actual = run(query, mixing, cache, metadata)
        expected = reference(query, mixing, cache, metadata)
        live = np.asarray(metadata.token_valid)
        for a, b in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(np.asarray(a)[live], np.asarray(b)[live])
        assert np.all(np.isneginf(np.asarray(actual[0])[~live]))
    assert len(traces) == 1, "request lengths/order must not specialize the compiled indexer"
    hlo = run.lower(query, mixing, cache, meta).compile().as_text()
    assert any(
        "custom-call(" in line and "StreamIdxTC" in line and "v4" in line
        for line in hlo.splitlines()
    )


@TPU
@pytest.mark.parametrize("tokens,selected", [(1, 96), (3, 512), (8, 512), (17, 96)])
@pytest.mark.parametrize("extreme_sink", [False, True])
def test_original_csa_joint_attention_exact_blocks_masks_and_sink(tokens, selected, extreme_sink):
    rng = np.random.default_rng(64)
    q = jnp.asarray(rng.normal(size=(tokens, 64, 512)), jnp.bfloat16)
    window = jnp.asarray(rng.normal(size=(tokens, 128, 512)), jnp.bfloat16)
    keys = jnp.asarray(rng.normal(size=(tokens, selected, 512)), jnp.bfloat16)
    window_lengths = jnp.asarray(np.resize([128, 3, 0, 127], tokens), jnp.int32)
    lengths = jnp.asarray(np.resize([selected, 1, 0, 65], tokens), jnp.int32)
    valid = jnp.arange(128)[None] < window_lengths[:, None]
    sink = jnp.asarray(np.full(64, 1000) if extreme_sink else rng.normal(size=64), jnp.float32)
    run = jax.jit(lambda q, w, v, k, n, s: csa.attend(q, w, v, k, n, s, CONFIG))
    actual = run(q, window, valid, keys, lengths, sink)

    @jax.jit
    def reference(q, window, valid, keys, lengths, sink):
        all_kv = jnp.concatenate((window, keys), axis=1)
        masks = jnp.concatenate((valid, jnp.arange(selected)[None] < lengths[:, None]), axis=1)
        indices = jnp.where(masks, jnp.arange(128 + selected)[None], -1)
        return jax.lax.map(
            lambda row: _single_query_attention(*row, sink, 512**-0.5), (q, all_kv, indices)
        )

    expected = reference(q, window, valid, keys, lengths, sink)
    np.testing.assert_array_equal(
        np.asarray(expected).view(np.uint16), np.asarray(actual).view(np.uint16)
    )
    assert np.all(np.isfinite(np.asarray(actual, np.float32)))
    hlo = run.lower(q, window, valid, keys, lengths, sink).compile().as_text()
    assert any(
        "custom-call(" in line and "csa-joint-attention" in line and "v4" in line
        for line in hlo.splitlines()
    )


@TPU
@pytest.mark.parametrize("index", [False, True])
def test_csa_overlap_state_matches_retained_chunks_batch_fork_and_reuse_bitwise(index):
    rng = np.random.default_rng(8023)
    cfg = replace(CONFIG, max_context=384)
    weights = _compressor_weights(rng, cfg, index=index)
    prefix = "index" if index else "main"
    inputs = [jnp.asarray(rng.normal(size=(271, 4096)), jnp.bfloat16) for _ in range(2)]
    backend = V4PagedBackend(max_context=384)

    @jax.jit
    def run(x, w, cache, meta):
        return compress(x, w, dict(cache), cfg, meta, index=index, csa_backend="pallas")

    single_meta = backend.get_forward_metadata(make_batch([0], [271]))
    expected = [run(x, weights, make_cache(cfg), single_meta) for x in inputs]
    reference = jax.jit(lambda x, w, c, m: compress(x, w, dict(c), cfg, m, index=index))
    retained = reference(inputs[0], weights, make_cache(cfg), single_meta)
    for name in retained:
        np.testing.assert_array_equal(retained[name], expected[0][name], err_msg=name)

    # Retained V4 intentionally uses GEMV for a one-token query and MXU for
    # longer queries. Compare identical chunk schedules, not the FP32 scratch
    # of a whole-prefill MXU against a singleton GEMV at a page boundary.
    # Each chunk below is checked bitwise against independent B1 reference
    # histories, including page snapshots and already-written compressed KV.
    expected = [make_cache(cfg), make_cache(cfg)]

    def assert_request(cache, request, oracle):
        for suffix in ("kv", "score"):
            np.testing.assert_array_equal(
                cache[f"{prefix}.{suffix}"][request + 1], oracle[f"{prefix}.{suffix}"][1]
            )
            np.testing.assert_array_equal(
                np.asarray(cache[f"{prefix}.snapshot_{suffix}"])[
                    [1 + 3 * request, 3 + 3 * request, 2 + 3 * request]
                ],
                np.asarray(oracle[f"{prefix}.snapshot_{suffix}"])[[1, 3, 2]],
            )
        physical = np.asarray([1, 3, 2])[:, None] * 32 + np.arange(32)
        np.testing.assert_array_equal(
            np.asarray(cache[f"{prefix}.compressed"])[physical + 96 * request],
            np.asarray(oracle[f"{prefix}.compressed"])[physical],
        )

    cache, prior = make_cache(cfg), [0, 0]
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
        for request in range(2):
            one = make_batch([prior[request]], [lengths[request] - prior[request]])
            expected[request] = reference(
                inputs[request][prior[request] : lengths[request]],
                weights,
                expected[request],
                backend.get_forward_metadata(one),
            )
            assert_request(cache, request, expected[request])
        prior = lengths

    for suffix in ("kv", "score"):
        cache[f"{prefix}.{suffix}"] = cache[f"{prefix}.{suffix}"].at[4].set(999)
    untouched = [cache[f"{prefix}.{suffix}"][2] for suffix in ("kv", "score")]
    fork = make_batch([256], [15], slots=[3], pages=[[1, 3, 8]], padding=1)
    cache = run(
        jnp.pad(inputs[0][256:], ((0, 1), (0, 0))),
        weights,
        cache,
        backend.get_forward_metadata(fork),
    )
    for suffix, other in zip(("kv", "score"), untouched, strict=True):
        np.testing.assert_array_equal(
            cache[f"{prefix}.{suffix}"][4], expected[0][f"{prefix}.{suffix}"][1]
        )
        np.testing.assert_array_equal(cache[f"{prefix}.{suffix}"][2], other)

    fresh = make_batch([0, 0], [3, 0], slots=[3, 1], pages=[[8, 9, 10], [4, 6, 5]], padding=5)
    cache = run(
        jnp.pad(inputs[1][:3], ((0, 5), (0, 0))),
        weights,
        cache,
        backend.get_forward_metadata(fresh),
    )
    empty = run(
        inputs[1][:3], weights, make_cache(cfg), backend.get_forward_metadata(make_batch([0], [3]))
    )
    for suffix, other in zip(("kv", "score"), untouched, strict=True):
        np.testing.assert_array_equal(
            cache[f"{prefix}.{suffix}"][4], empty[f"{prefix}.{suffix}"][1]
        )
        np.testing.assert_array_equal(cache[f"{prefix}.{suffix}"][2], other)
