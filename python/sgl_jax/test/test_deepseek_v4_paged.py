"""Native V4 paging/packing contracts, independent of checkpoint allocation."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.deepseek_v4.compressor import compress
from sgl_jax.srt.kernels.deepseek_v4.numerics import V4LayerConfig
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.mem_cache.deepseek_v4_paged_pool import layer_buffer_specs, pool_size_bytes
from sgl_jax.srt.model_executor.deepseek_v4_reference import compress as reference_compress
from sgl_jax.srt.model_executor.deepseek_v4_reference import empty_cache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.test.kernels.test_deepseek_v4_reference import _compressor_weights


def make_batch(prefixes, counts, *, slots=None, pages=None, padding=0, decode=False):
    prefixes, counts = np.asarray(prefixes, np.int32), np.asarray(counts, np.int32)
    bs = len(counts)
    pages = pages or [[1 + i * 3, 3 + i * 3, 2 + i * 3] for i in range(bs)]
    positions, locations, cache_loc = [], [], []
    for prefix, count, physical in zip(prefixes, counts, pages, strict=True):
        seq = np.arange(prefix, prefix + count, dtype=np.int32)
        positions.extend(seq)
        locations.extend(np.asarray(physical)[seq // 128] * 128 + seq % 128)
        for page in physical[: (int(prefix + count) + 127) // 128]:
            cache_loc.extend(page * 128 + np.arange(128))
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
        dp_size=1,
        real_bs=int(np.count_nonzero(counts)),
        real_input_ids_len=int(sum(counts)),
        req_pool_indices=np.asarray(slots if slots is not None else np.arange(bs), np.int32),
        seq_lens=prefixes + counts,
        extend_prefix_lens=prefixes,
        extend_seq_lens=counts,
        input_ids=np.ones(sum(counts) + padding, np.int32),
        positions=np.pad(np.asarray(positions, np.int32), (0, padding)),
        out_cache_loc=np.pad(np.asarray(locations, np.int32), (0, padding), constant_values=-1),
        cache_loc=np.asarray(cache_loc, np.int32),
    )


def make_cache(config, capacity=1536, requests=4):
    return {
        name: jnp.full(shape, fill, dtype)
        for name, (shape, dtype, fill) in layer_buffer_specs(config, capacity, requests).items()
    }


def test_native_serving_contract_allows_batching_radix_and_partial_chunks():
    from sgl_jax.srt.configs.deepseek_v4 import validate_v4_serving_args
    from sgl_jax.test.test_deepseek_v4_framework import _args

    args = _args()
    args.page_size, args.max_running_requests, args.max_total_tokens = 128, 4, 32768
    args.disable_radix_cache = args.disable_overlap_schedule = False
    args.chunked_prefill_size = 64
    validate_v4_serving_args(args, SimpleNamespace(context_len=8192))
    args.max_running_requests = None
    validate_v4_serving_args(args, SimpleNamespace(context_len=8192))
    assert args.max_running_requests == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("page_size", 1),
        ("max_running_requests", 0),
        ("max_total_tokens", 255),
        ("hicache_storage", "none"),
        ("tp_size", 2),
        ("ep_size", 1),
        ("enable_lora", True),
        ("disaggregation_mode", "decode"),
    ],
)
def test_native_serving_contract_rejects_unvalidated_paths(field, value):
    from sgl_jax.srt.configs.deepseek_v4 import validate_v4_serving_args
    from sgl_jax.test.test_deepseek_v4_framework import _args

    args = _args()
    args.page_size, args.max_running_requests = 128, 4
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_v4_serving_args(args, SimpleNamespace(context_len=256))


def test_metadata_lengths_are_dynamic_and_pages_are_physical():
    backend = V4PagedBackend(max_context=384)
    a = backend.get_forward_metadata(make_batch([3, 129], [4, 2], padding=2))
    b = backend.get_forward_metadata(make_batch([5, 127], [1, 4], padding=3, slots=[3, 1]))
    assert jax.tree.structure(a) == jax.tree.structure(b)
    assert [x.shape for x in jax.tree.leaves(a)] == [x.shape for x in jax.tree.leaves(b)]
    calls = []

    @jax.jit
    def run(meta):
        calls.append(1)
        return meta.query_lens.sum()

    assert int(run(a)) == 6
    assert int(run(b)) == 5
    assert len(calls) == 1
    np.testing.assert_array_equal(a.page_table, [[1, 0, 0], [4, 6, 0]])
    np.testing.assert_array_equal(b.req_slots, [4, 2])


@pytest.fixture
def official_rope_reference(monkeypatch):
    """Align only reference coefficients; retain all bitwise paging assertions.

    The native RoPE fix uses host FP32 coefficients, whereas the frozen bring-up
    reference still traces pow into XLA. An isolated legacy-phase control made
    all eight affected CPU cases pass. Use independently constructed official
    PyTorch coefficients here, not the native implementation as its own oracle.
    This patch is scoped to one test; the historical runtime reference is intact.
    """
    pytest.importorskip("torch")
    from sgl_jax.srt.model_executor import deepseek_v4_reference
    from sgl_jax.test.deepseek_v4_reference_rope import official_recipe_rope

    monkeypatch.setattr(deepseek_v4_reference, "rope", official_recipe_rope)


@pytest.mark.parametrize("ratio,index", [(4, False), (4, True), (128, False)])
def test_ragged_compressor_matches_reference_and_reuses_prefix(ratio, index, official_rope_reference):
    rng = np.random.default_rng(20260907)
    config = V4LayerConfig(hidden=128, head_dim=128, index_dim=128, max_context=384, ratio=ratio)
    weights = _compressor_weights(rng, config, index=index)
    inputs = [jnp.asarray(rng.normal(size=(271, 128)), jnp.bfloat16) for _ in range(2)]
    backend = V4PagedBackend(max_context=384)
    prefix = "index" if index else "main"
    reference = jax.jit(
        lambda x, cache, weights: reference_compress(
            x, jnp.arange(len(x)), weights, dict(cache), config, index=index
        )
    )
    expected = [reference(x, empty_cache(config), weights) for x in inputs]

    @jax.jit
    def run(x, cache, meta, weights):
        return compress(x, weights, dict(cache), config, meta, index=index)

    cache = make_cache(config)
    # Deliberately cross r=4 and r=128 boundaries mid-chunk; request order changes.
    previous = [0, 0]
    for lengths, order in [
        ([3, 7], [0, 1]),
        ([129, 128], [1, 0]),
        ([130, 259], [0, 1]),
        ([271, 271], [1, 0]),
    ]:
        counts = [lengths[i] - previous[i] for i in order]
        batch = make_batch(
            [previous[i] for i in order],
            counts,
            slots=order,
            pages=[[1 + i * 3, 3 + i * 3, 2 + i * 3] for i in order],
            padding=5,
        )
        x = jnp.concatenate([inputs[i][previous[i] : lengths[i]] for i in order])
        x = jnp.pad(x, ((0, 5), (0, 0)))
        cache = run(x, cache, backend.get_forward_metadata(batch), weights)
        previous = lengths
    for request in range(2):
        for suffix in (".kv", ".score"):
            np.testing.assert_array_equal(
                cache[prefix + suffix][request + 1], expected[request][prefix + suffix]
            )
        groups = np.arange(271 // ratio)
        physical = np.asarray([1 + request * 3, 3 + request * 3, 2 + request * 3])
        ends = (groups + 1) * ratio - 1
        stored = (physical[ends // 128] * 128 + ends % 128) // ratio
        np.testing.assert_array_equal(
            cache[prefix + ".compressed"][stored],
            expected[request][prefix + ".compressed"][: len(groups)],
        )

    # Fork request 0's two shared pages into a dirty, unrelated slot and new page.
    fork = make_batch([256], [15], slots=[3], pages=[[1, 3, 8]], padding=1)
    cache[prefix + ".kv"] = cache[prefix + ".kv"].at[4].set(999)
    cache[prefix + ".score"] = cache[prefix + ".score"].at[4].set(999)
    old_other = cache[prefix + ".kv"][2]
    cache = run(
        jnp.pad(inputs[0][256:], ((0, 1), (0, 0))),
        cache,
        backend.get_forward_metadata(fork),
        weights,
    )
    for suffix in (".kv", ".score"):
        np.testing.assert_array_equal(cache[prefix + suffix][4], expected[0][prefix + suffix])
    np.testing.assert_array_equal(cache[prefix + ".kv"][2], old_other)
    # Fresh prefix=0 resets only the reused slot, not other live requests.
    fresh = make_batch([0, 0], [3, 0], slots=[3, 1], pages=[[8, 9, 10], [4, 6, 5]], padding=5)
    cache = run(
        jnp.pad(inputs[1][:3], ((0, 5), (0, 0))),
        cache,
        backend.get_forward_metadata(fresh),
        weights,
    )
    initial = reference(inputs[1][:3], empty_cache(config), weights)
    for suffix in (".kv", ".score"):
        np.testing.assert_array_equal(cache[prefix + suffix][4], initial[prefix + suffix])
    np.testing.assert_array_equal(cache[prefix + ".kv"][2], old_other)


def test_pool_accounting_includes_snapshots_and_private_scratch():
    configs = [replace(V4LayerConfig(), ratio=r) for r in (0, 4, 128)]
    actual = sum(a.size * a.dtype.itemsize for c in configs for a in make_cache(c).values())
    assert pool_size_bytes(configs, 1536, 4) == actual


def test_standard_allocator_radix_fork_eviction_and_pool_roundtrip():
    from jax.sharding import Mesh
    from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
    from sgl_jax.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams, MatchPrefixParams
    from sgl_jax.srt.mem_cache.deepseek_v4_paged_pool import V4PagedKVPool
    from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, ReqToTokenPool
    from sgl_jax.srt.mem_cache.radix_cache import RadixCache, RadixKey

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",))
    config = V4LayerConfig(hidden=128, head_dim=128, index_dim=128, ratio=4)
    with jax.set_mesh(mesh):
        pool = V4PagedKVPool([config], mesh, capacity=1024, requests=4)
        pools = MemoryPools(token_to_kv_pool=pool)
        restored = jax.tree.unflatten(jax.tree.structure(pool), jax.tree.leaves(pool))
        assert restored.requests == 4 and restored.size == 1024
        pools.replace_all({"token_to_kv_pool": restored.layers})
        allocator = PagedTokenToKVPoolAllocator(1024, 128, pool, debug_mode=True)
        req_pool = ReqToTokenPool(4, 260, np.int32)
        radix = RadixCache(req_pool, allocator, page_size=128)
        allocated = allocator.alloc(384)
        # Two shared pages in deliberately non-monotonic physical order.
        shared = np.concatenate((allocated[256:], allocated[:128]))
        allocator.free(allocated[128:256])
        key = RadixKey(list(range(256)), None, 0)
        assert radix.insert(InsertParams(key=key, value=shared)) == 0
        match = radix.match_prefix(MatchPrefixParams(key=key))
        np.testing.assert_array_equal(match.device_indices, shared)
        radix.inc_lock_ref(match.last_device_node)
        # Standard allocator allocates private continuation pages for each fork.
        tails = allocator.alloc_extend([256, 256], [259, 260], [int(shared[-1])] * 2, 7)
        assert tails[0] // 128 != tails[3] // 128
        assert radix.evict(EvictParams(num_tokens=256, dp_rank=0)).num_tokens_evicted == 0
        allocator.free(tails)
        radix.dec_lock_ref(match.last_device_node)
        assert radix.evict(EvictParams(num_tokens=256, dp_rank=0)).num_tokens_evicted == 256
        assert allocator.available_size() == 1024
        # Cache-copy order follows logical pages, not sorted physical IDs.
        layer = dict(pool.layers[0])
        layer["window"] = layer["window"].at[384:512].set(3).at[128:256].set(1)
        layer["main.snapshot_kv"] = layer["main.snapshot_kv"].at[3].set(30).at[1].set(10)
        pool.replace_buffer([layer])
        saved = pool.get_cpu_copy(shared)
        destinations = np.concatenate((np.arange(640, 768), np.arange(512, 640)))
        pool.load_cpu_copy(saved, destinations)
        np.testing.assert_array_equal(pool.layers[0]["window"][640:768], 3)
        np.testing.assert_array_equal(pool.layers[0]["window"][512:640], 1)
        np.testing.assert_array_equal(pool.layers[0]["main.snapshot_kv"][5], 30)
        np.testing.assert_array_equal(pool.layers[0]["main.snapshot_kv"][4], 10)


@pytest.mark.parametrize("capture", [0, 1, 2])
@pytest.mark.parametrize("logprobs", [False, True])
def test_standard_logits_logprobs_and_hidden_capture(capture, logprobs):
    from flax import nnx
    from jax.sharding import NamedSharding, PartitionSpec as P
    from sgl_jax.srt.layers.embeddings import ParallelLMHead
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
    from sgl_jax.srt.models.deepseek_v4 import V4LogitsProcessor
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    if len(jax.devices()) != 4:
        pytest.skip("four-device explicit mesh contract")
    mesh = create_device_mesh([1, 4], [1, 1])
    rng = np.random.default_rng(68)
    with jax.set_mesh(mesh):
        hidden = jax.device_put(
            rng.normal(size=(8, 128)).astype(jnp.bfloat16), NamedSharding(mesh, P("data", None))
        )
        head = ParallelLMHead(32, 128, mesh=mesh, dtype=jnp.bfloat16)
        weight = rng.normal(size=(32, 128)).astype(jnp.bfloat16)
        head.embedding = nnx.Param(jax.device_put(weight, NamedSharding(mesh, P("tensor", None))))
        array = lambda a: jax.device_put(np.asarray(a, np.int32), NamedSharding(mesh, P("data")))
        meta = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode(capture),
            logits_indices=array([2, 6]),
            extend_return_logprob=logprobs,
            input_logprob_indices_device=array([0, 1, 3, 4, 5]) if logprobs else None,
            extend_input_logprob_token_ids_device=array([1, 2, 4, 5, 6]) if logprobs else None,
        )
        processor = V4LogitsProcessor(32, mesh)
        output = nnx.jit(lambda processor, head, hidden, meta: processor(hidden, head, meta))(
            processor, head, hidden, meta
        )
    expected = np.asarray(hidden, np.float32) @ weight.astype(np.float32).T
    np.testing.assert_allclose(output.next_token_logits, expected[[2, 6]], rtol=1e-5, atol=1e-5)
    if capture:
        np.testing.assert_array_equal(
            output.hidden_states, np.asarray(hidden) if capture == 2 else np.asarray(hidden)[[2, 6]]
        )
    else:
        assert output.hidden_states is None
    if logprobs:
        rows = expected[[0, 1, 3, 4, 5]]
        shifted = rows - rows.max(axis=-1, keepdims=True)
        log_probability = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        np.testing.assert_allclose(
            output.input_token_logprobs,
            log_probability[np.arange(5), [1, 2, 4, 5, 6]],
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("req_pool_indices", np.asarray([0, 0])),
        ("out_cache_loc", np.asarray([129, 512])),
        ("positions", np.asarray([1, 0])),
        ("real_input_ids_len", 3),
    ],
)
def test_invalid_ownership_fails_before_dispatch(field, value):
    batch = make_batch([0, 0], [1, 1])
    setattr(batch, field, value)
    with pytest.raises(ValueError):
        V4PagedBackend(max_context=384).get_forward_metadata(batch)


@pytest.mark.parametrize("hash_routing", [False, True])
def test_packed_router_preserves_each_requests_absolute_lane(hash_routing):
    from sgl_jax.srt.kernels.deepseek_v4.moe import route
    from sgl_jax.srt.model_executor.deepseek_v4_reference import route as reference_route

    rng = np.random.default_rng(43)
    config = V4LayerConfig(hidden=128, hash_routing=hash_routing)
    x = jnp.asarray(rng.normal(size=(19, 128)), jnp.bfloat16)
    ids = jnp.arange(19, dtype=jnp.int32)
    weights = {
        "ffn.gate.weight": jnp.asarray(rng.normal(size=(256, 128)), jnp.bfloat16),
        "ffn.gate.bias": jnp.asarray(rng.normal(size=(256,)), jnp.float32),
        "ffn.gate.tid2eid": jnp.asarray(rng.integers(0, 256, (32, 6)), jnp.int32),
    }
    batch = make_batch([3, 127], [11, 5], padding=3)
    meta = V4PagedBackend(max_context=384).get_forward_metadata(batch)
    actual_ids, actual_weights = jax.jit(lambda x: route(x, ids, weights, config, meta))(x)
    for start, end, prefix in ((0, 11, 3), (11, 16, 127)):
        expected_ids, expected_weights = reference_route(
            x[start:end], jnp.arange(prefix, prefix + end - start), ids[start:end], weights, config
        )
        np.testing.assert_array_equal(actual_ids[start:end], expected_ids)
        np.testing.assert_array_equal(actual_weights[start:end], expected_weights)
    np.testing.assert_array_equal(actual_weights[16:], 0)


def test_grouped_fp4_dispatch_matches_unpacked_token_order():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from sgl_jax.srt.kernels.deepseek_v4.moe import grouped_fp4_experts
    from sgl_jax.srt.kernels.low_bit.moe import routed_fp4_experts
    from sgl_jax.test.kernels.test_deepseek_v4_low_bit import _fixture_weight

    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    experts = len(jax.devices()) * 2
    rng = np.random.default_rng(4)
    weights, scales = [], []
    for projection in range(3):
        pairs = [
            _fixture_weight("fp4", 128, 128, seed=100 * projection + i) for i in range(experts)
        ]
        weights.append(
            jax.device_put(
                np.stack([w for w, s in pairs]), NamedSharding(mesh, P("tensor", None, None))
            )
        )
        scales.append(
            jax.device_put(
                np.stack([s for w, s in pairs]), NamedSharding(mesh, P("tensor", None, None))
            )
        )
    x = jax.device_put(rng.normal(size=(19, 128)).astype(jnp.bfloat16), NamedSharding(mesh, P()))
    ids = jax.device_put(
        rng.integers(0, experts, (19, 2), dtype=np.int32), NamedSharding(mesh, P())
    )
    routes = jax.device_put(rng.uniform(size=(19, 2)).astype(np.float32), NamedSharding(mesh, P()))
    routes = routes.at[-3:].set(0)

    def run(function):
        return jax.jit(
            jax.shard_map(
                function,
                mesh=mesh,
                in_specs=(P(), *(P("tensor", None, None) for _ in range(6)), P(), P()),
                out_specs=P(),
                check_vma=False,
            )
        )(x, *weights, *scales, ids, routes)

    with jax.set_mesh(mesh):
        expected, actual = run(routed_fp4_experts), run(grouped_fp4_experts)
    np.testing.assert_array_equal(actual, expected)


def attention_weights(config):
    from sgl_jax.test.kernels.test_deepseek_v4_low_bit import _fixture_weight

    rng = np.random.default_rng(456)
    pairs = {
        "attn.wq_a": (128, config.hidden),
        "attn.wq_b": (config.heads * config.head_dim, 128),
        "attn.wkv": (config.head_dim, config.hidden),
        "attn.wo_b": (config.hidden, config.groups * config.o_rank),
        "attn.indexer.wq_b": (config.index_heads * config.index_dim, 128),
        "attn.indexer.weights_proj": (config.index_heads, config.hidden),
    }
    weights = {
        name + ".weight": jnp.asarray(rng.normal(scale=0.05, size=shape), jnp.bfloat16)
        for name, shape in pairs.items()
    }
    w, s = _fixture_weight(
        "fp8", config.groups * config.o_rank, config.heads * config.head_dim // config.groups
    )
    weights["attn.wo_a.weight"], weights["attn.wo_a.scale"] = jnp.asarray(w), jnp.asarray(s)
    for name, dim in (("q_norm", 128), ("kv_norm", config.head_dim)):
        weights["attn." + name + ".weight"] = jnp.ones(dim, jnp.bfloat16)
    weights["attn.attn_sink"] = jnp.zeros(config.heads, jnp.float32)
    if config.ratio:
        weights.update(_compressor_weights(rng, config))
        if config.ratio == 4:
            weights.update(_compressor_weights(rng, config, index=True))
    return weights


def test_index_score_jit_cannot_elide_official_bf16_rounding():
    from sgl_jax.srt.model_executor.deepseek_v4_reference import attention as reference_attention

    config = V4LayerConfig(
        hidden=128,
        heads=2,
        head_dim=128,
        groups=1,
        o_rank=128,
        index_heads=2,
        index_dim=128,
        index_topk=8,
        max_context=384,
        ratio=4,
    )
    weights = attention_weights(config)
    x = jnp.asarray(np.random.default_rng(12).normal(size=(129, 128)), jnp.bfloat16)
    ref = jax.jit(
        lambda x, w: reference_attention(x, jnp.arange(len(x)), w, empty_cache(config), config)[2]
    )
    trace = ref(x, weights)
    score = np.asarray(trace["index_score"], np.float32)
    finite = score[np.isfinite(score)]
    # Independent NumPy cast, not the production round_bf16 helper. The old
    # fused reference produced -0.26549375 here, not the required -0.265625.
    np.testing.assert_array_equal(finite, finite.astype(jnp.bfloat16).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(trace["index_selected"])[0], np.arange(8))


@pytest.mark.parametrize("context", [8192, 8320])
def test_8k_page_index_topk_and_compressor_boundary(context, official_rope_reference):
    """8K address space with >512 candidates and two independent terminal pages."""
    from sgl_jax.srt.kernels.deepseek_v4.attention import attention
    from sgl_jax.srt.model_executor.deepseek_v4_reference import attention as reference_attention

    config = V4LayerConfig(
        hidden=128,
        heads=2,
        head_dim=128,
        groups=1,
        o_rank=128,
        index_heads=2,
        index_dim=128,
        index_topk=512,
        max_context=context,
        ratio=4,
    )
    rng = np.random.default_rng(8192)
    weights = attention_weights(config)
    page_count = context // 128
    pages = [list(range(1, 2 * page_count + 1, 2)), list(range(2, 2 * page_count + 1, 2))]
    batch = make_batch([context - 1, context - 129], [1, 1], pages=pages, decode=True)
    metadata = V4PagedBackend(max_context=context, capacity=2 * context).get_forward_metadata(batch)
    cache = make_cache(config, capacity=2 * context)
    references = []
    for request in range(2):
        ref = empty_cache(config)
        for key, value in ref.items():
            ref[key] = jnp.asarray(rng.normal(scale=0.1, size=value.shape), value.dtype)
            if key == "window" or key.endswith(".compressed"):
                stride = 128 if key == "window" else 32
                locations = (
                    np.asarray(pages[request])[:, None] * stride + np.arange(stride)
                ).ravel()
                cache[key] = cache[key].at[locations].set(ref[key])
            else:
                cache[key] = cache[key].at[request + 1].set(ref[key])
        references.append(ref)
    x = jnp.asarray(rng.normal(size=(2, 128)), jnp.bfloat16)
    run = jax.jit(
        lambda x, pos, cache, meta, loc, weights: attention(
            x, pos, weights, cache, config, meta, loc
        )
    )
    actual, updated = run(x, batch.positions, cache, metadata, batch.out_cache_loc, weights)
    ref_run = jax.jit(
        lambda x, pos, cache, weights: reference_attention(x, pos, weights, cache, config)
    )
    for request in range(2):
        expected, ref_cache, trace = ref_run(
            x[request : request + 1],
            batch.positions[request : request + 1],
            references[request],
            weights,
        )
        assert trace["index_selected"].shape == (1, 512)
        assert int(np.max(np.asarray(trace["index_selected"]))) > 512
        np.testing.assert_array_equal(actual[request], expected[0])
        for prefix in ("main", "index"):
            for suffix in ("kv", "score"):
                # A few FP32 ulps can cross a later BF16 pooling boundary
                # (real position 8023). Require the original single-request
                # arithmetic, not merely a self-consistent batched answer.
                np.testing.assert_array_equal(
                    updated[f"{prefix}.{suffix}"][request + 1],
                    ref_cache[f"{prefix}.{suffix}"],
                )


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_paged_attention_matches_independent_requests_and_batched_decode(ratio, official_rope_reference):
    from sgl_jax.srt.kernels.deepseek_v4.attention import attention
    from sgl_jax.srt.model_executor.deepseek_v4_reference import attention as reference_attention

    config = V4LayerConfig(
        hidden=128,
        heads=2,
        head_dim=128,
        groups=1,
        o_rank=128,
        index_heads=2,
        index_dim=128,
        index_topk=8,
        max_context=384,
        ratio=ratio,
    )
    weights = attention_weights(config)
    rng = np.random.default_rng(12)
    inputs = [jnp.asarray(rng.normal(size=(length, 128)), jnp.bfloat16) for length in (130, 133)]
    backend = V4PagedBackend(max_context=384)
    expected_caches = []
    outputs = []
    ref = jax.jit(
        lambda x, positions, cache, weights: reference_attention(
            x, positions, weights, cache, config
        )[:2]
    )
    for x in inputs:
        output, cache = ref(x[:-1], jnp.arange(len(x) - 1), empty_cache(config), weights)
        outputs.append(output)
        expected_caches.append(cache)

    run = jax.jit(
        lambda x, pos, cache, meta, loc, weights: attention(
            x, pos, weights, cache, config, meta, loc
        )
    )
    batch = make_batch([0, 0], [129, 132], padding=3)
    x = jnp.pad(jnp.concatenate([x[:-1] for x in inputs]), ((0, 3), (0, 0)))
    actual, cache = run(
        x,
        batch.positions,
        make_cache(config),
        backend.get_forward_metadata(batch),
        batch.out_cache_loc,
        weights,
    )
    for request, length in enumerate((129, 132)):
        locations = np.asarray([[1, 3, 2], [4, 6, 5]][request])
        rows = np.arange(length)
        np.testing.assert_array_equal(
            cache["window"][locations[rows // 128] * 128 + rows % 128],
            expected_caches[request]["window"][:length],
        )
        if ratio:
            terminal = (np.arange(length // ratio) + 1) * ratio - 1
            physical = (locations[terminal // 128] * 128 + terminal % 128) // ratio
            for key in ("main.compressed", "index.compressed"):
                if key in cache:
                    np.testing.assert_array_equal(
                        cache[key][physical],
                        expected_caches[request][key][: len(terminal)],
                        err_msg=key,
                    )
    np.testing.assert_array_equal(actual[:261], jnp.concatenate(outputs))
    np.testing.assert_array_equal(actual[261:], 0)
    batch = make_batch([132, 129], [1, 1], slots=[1, 0], pages=[[4, 6, 5], [1, 3, 2]], decode=True)
    actual, _ = run(
        jnp.stack([inputs[1][-1], inputs[0][-1]]),
        batch.positions,
        cache,
        backend.get_forward_metadata(batch),
        batch.out_cache_loc,
        weights,
    )
    for row, request in enumerate((1, 0)):
        expected, _ = ref(
            inputs[request][-1:],
            jnp.asarray([len(inputs[request]) - 1]),
            expected_caches[request],
            weights,
        )
        np.testing.assert_array_equal(actual[row], expected[0])
