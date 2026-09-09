"""CPU contract tests; real-checkpoint execution is a separate four-TPU gate."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.deepseek_v4 import validate_v4_server_args
from sgl_jax.srt.layers.attention.deepseek_v4_backend import DeepseekV4Backend, DeepseekV4Metadata
from sgl_jax.srt.mem_cache.deepseek_v4_pool import DeepseekV4TokenToKVPool
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.deepseek_v4_reference import ReferenceConfig
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


def _batch(count=4, mode=ForwardMode.EXTEND, length=None, prefix=0):
    length = prefix + count if length is None else length
    return SimpleNamespace(
        real_bs=1,
        dp_size=1,
        real_input_ids_len=count,
        req_pool_indices=np.asarray([0]),
        seq_lens=np.asarray([length]),
        input_ids=np.arange(count, dtype=np.int32),
        positions=np.arange(length - count, length, dtype=np.int32),
        out_cache_loc=np.arange(length - count + 1, length + 1),
        extend_prefix_lens=np.asarray([prefix]),
        extend_seq_lens=np.asarray([count]),
        forward_mode=mode,
        return_logprob=False,
    )


def _args():
    return SimpleNamespace(
        tp_size=4,
        dp_size=1,
        ep_size=4,
        moe_dp_size=1,
        max_running_requests=1,
        page_size=1,
        attention_backend="deepseek_v4",
        disable_radix_cache=True,
        disable_hybrid_swa_memory=True,
        disable_overlap_schedule=True,
        max_total_tokens=256,
        max_prefill_tokens=128,
        chunked_prefill_size=-1,
        speculative_algorithm=None,
        enable_lora=False,
        enable_sequence_parallel=False,
        pd_disaggregation="",
        ep_dispatch_algorithm=None,
        enable_return_routed_experts=False,
        enable_expert_balance_debug=False,
        kv_cache_dtype="auto",
        load_format="auto",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 1),
        ("ep_size", 1),
        ("max_running_requests", 2),
        ("disable_radix_cache", False),
        ("chunked_prefill_size", 64),
        ("disable_overlap_schedule", False),
        ("kv_cache_dtype", "float8_e4m3fn"),
        ("load_format", "dummy"),
        ("max_total_tokens", 4096),
    ],
)
def test_unsupported_server_modes_fail_before_loading(field, value):
    args = _args()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        validate_v4_server_args(args, SimpleNamespace(context_len=256))


def test_supported_server_contract():
    validate_v4_server_args(_args(), SimpleNamespace(context_len=256))


def test_supported_8k_chunked_server_contract():
    args = _args()
    args.max_total_tokens = 8192
    args.chunked_prefill_size = 128
    validate_v4_server_args(args, SimpleNamespace(context_len=8192))


def test_8k_requires_fixed_chunking():
    args = _args()
    args.max_total_tokens = 8192
    with pytest.raises(ValueError, match="chunked_prefill_size=128"):
        validate_v4_server_args(args, SimpleNamespace(context_len=8192))


@pytest.mark.parametrize(
    "field,value",
    [
        ("real_bs", 2),
        ("dp_size", 2),
        ("req_pool_indices", np.asarray([1])),
        ("seq_lens", np.asarray([257])),
        ("extend_prefix_lens", np.asarray([1])),
        ("positions", np.asarray([1, 2, 3, 4])),
        ("out_cache_loc", np.asarray([0, 2, 3, 4])),
        ("return_logprob", True),
        ("forward_mode", ForwardMode.MIXED),
    ],
)
def test_invalid_batch_fails_closed(field, value):
    batch = _batch()
    setattr(batch, field, value)
    with pytest.raises(ValueError):
        DeepseekV4Backend(max_context=256).get_forward_metadata(batch)


def test_padding_does_not_change_real_compressor_length():
    batch = _batch()
    batch.input_ids = np.pad(batch.input_ids, (0, 124))
    batch.positions = np.pad(batch.positions, (0, 124))
    batch.out_cache_loc = np.pad(batch.out_cache_loc, (0, 124), constant_values=-1)
    metadata = DeepseekV4Backend(max_context=256).get_forward_metadata(batch)
    assert metadata.valid_tokens == 4


def test_decode_metadata_is_independent_of_position():
    backend = DeepseekV4Backend(max_context=256)
    a = backend.get_forward_metadata(_batch(1, ForwardMode.DECODE, 132))
    b = backend.get_forward_metadata(_batch(1, ForwardMode.DECODE, 136))
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    assert a == b == DeepseekV4Metadata(1)


@pytest.mark.parametrize("prefix,count", [(128, 128), (256, 17), (8064, 127)])
def test_chunked_prefill_metadata_uses_absolute_prefix(prefix, count):
    metadata = DeepseekV4Backend(max_context=8192).get_forward_metadata(
        _batch(count, prefix=prefix)
    )
    assert metadata == DeepseekV4Metadata(count)


def test_chunked_prefill_rejects_misaligned_continuation():
    with pytest.raises(ValueError, match="128-token boundary"):
        DeepseekV4Backend(max_context=8192).get_forward_metadata(_batch(4, prefix=129))


def test_precompile_uses_complete_prefill_rows():
    backend = DeepseekV4Backend(max_context=256)
    batch = _batch(1)
    batch.input_ids = np.ones(132, np.int32)
    backend.prepare_precompile_batch(batch)
    assert backend.get_forward_metadata(batch).valid_tokens == 132
    assert batch.logits_indices.tolist() == [131]


def test_cache_roundtrip_update_and_reset():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",))
    configs = tuple(replace(ReferenceConfig(), ratio=ratio) for ratio in (0, 4, 128))
    pool = DeepseekV4TokenToKVPool(configs, mesh)
    assert "index.kv" not in pool.layers[2]
    assert "index.kv" in pool.layers[1]
    assert pool.get_kv_size_bytes() > 0
    restored = jax.tree_util.tree_unflatten(*reversed(jax.tree_util.tree_flatten(pool)))
    assert restored.configs == pool.configs
    pools = MemoryPools(token_to_kv_pool=pool)

    @jax.jit
    def update(pools):
        return {
            "token_to_kv_pool": tuple(
                {key: value + 1 for key, value in layer.items()}
                for layer in pools.token_to_kv_pool.layers
            )
        }

    # Other collected TPU tests may leave a four-device Explicit mesh active.
    # This contract test owns a one-device pool and must scope its own mesh.
    with jax.set_mesh(mesh):
        pools.replace_all(update(pools))
    np.testing.assert_array_equal(pool.layers[0]["window"], 1)
    pool.reset()
    np.testing.assert_array_equal(pool.layers[0]["window"], 0)
    assert np.all(np.isneginf(np.asarray(pool.layers[1]["main.score"])))
    with pytest.raises(ValueError):
        pool.replace_buffer(pool.layers[:1])
    with pytest.raises(NotImplementedError):
        pool.get_fused_kv_buffer(0)


def test_nested_raw_weights_remain_dynamic_under_nnx_split():
    model = nnx.Dict(weights=nnx.Param({"fp4": jnp.asarray([0x12, 0xFE], jnp.uint8)}))
    graph, state = nnx.split(model)

    @jax.jit
    def apply(state):
        return nnx.merge(graph, state).weights.get_value()["fp4"]

    np.testing.assert_array_equal(apply(state), [0x12, 0xFE])
    model.weights.set_value({"fp4": jnp.asarray([0x34, 0xAB], jnp.uint8)})
    _, changed = nnx.split(model)
    np.testing.assert_array_equal(apply(changed), [0x34, 0xAB])
    assert apply._cache_size() == 1


def test_shape_only_construction_does_not_create_empty_param_shardings():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]), ("tensor",), axis_types=(jax.sharding.AxisType.Explicit,)
    )
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: nnx.Dict(weights=nnx.data(None)))
        model.weights = nnx.Param({"fp4": jnp.asarray([0x12, 0xFE], jnp.uint8)})
        graph, state = nnx.split(model)
        restored = nnx.merge(graph, state)
        np.testing.assert_array_equal(restored.weights.get_value()["fp4"], [0x12, 0xFE])


def test_whole_forward_executes_every_distinct_layer_without_reference_dispatch(monkeypatch):
    import sgl_jax.srt.models.deepseek_v4 as native

    configs = tuple(replace(ReferenceConfig(), hidden=8, hc=4, head_dim=8) for _ in range(43))
    shared = {
        "embed.weight": jnp.ones((8, 8), jnp.bfloat16),
        "hc_head_fn": jnp.zeros((4, 32), jnp.float32),
        "hc_head_scale": jnp.ones((1,), jnp.float32),
        "hc_head_base": jnp.zeros((4,), jnp.float32),
        "norm.weight": jnp.ones((8,), jnp.bfloat16),
        "head.weight": jnp.ones((8, 8), jnp.bfloat16),
    }
    layers = tuple({"value": jnp.asarray(i + 1, jnp.float32)} for i in range(43))
    caches = tuple({"window": jnp.zeros((256, 8), jnp.bfloat16)} for _ in configs)

    def layer_step(self, streams, positions, ids, weights, cache, metadata, locations):
        value = streams + weights["value"].astype(jnp.bfloat16)
        return (
            value,
            {"window": cache["window"].at[positions].set(weights["value"].astype(jnp.bfloat16))},
        )

    monkeypatch.setattr(native.DeepseekV4DecoderLayer, "__call__", layer_step)

    @jax.jit
    def forward(ids, positions, shared, layers, caches):
        return native.whole_model_forward(
            shared["embed.weight"][ids],
            positions,
            ids,
            shared,
            layers,
            caches,
            None,
            positions,
            configs=configs,
            mhc_backend="reference",  # 8-wide arithmetic tests framework state, not Pallas tiling
        )

    output, updates = forward(jnp.asarray([1]), jnp.asarray([0]), shared, layers, caches)
    assert output.shape == (1, 8)
    assert len(updates) == 43
    for i, cache in enumerate(updates):
        np.testing.assert_array_equal(cache["window"][0], i + 1)
    _, updates = forward(jnp.asarray([2]), jnp.asarray([1]), shared, layers, updates)
    assert forward._cache_size() == 1
    assert np.all(np.isfinite(np.asarray(output)))
    jaxpr = jax.make_jaxpr(
        lambda ids, positions, shared, layers, caches: native.whole_model_forward(
            shared["embed.weight"][ids],
            positions,
            ids,
            shared,
            layers,
            caches,
            None,
            positions,
            configs=configs,
            mhc_backend="reference",
        )
    )(jnp.asarray([1]), jnp.asarray([0]), shared, layers, caches)
    assert sum(eqn.primitive.name == "optimization_barrier" for eqn in jaxpr.jaxpr.eqns) == 43


def test_whole_forward_does_not_clear_unrelated_page_rows(monkeypatch):
    import sgl_jax.srt.models.deepseek_v4 as native

    configs = tuple(replace(ReferenceConfig(), hidden=8, hc=4, head_dim=8) for _ in range(43))
    shared = {
        "embed.weight": jnp.ones((8, 8), jnp.bfloat16),
        "hc_head_fn": jnp.zeros((4, 32), jnp.float32),
        "hc_head_scale": jnp.ones((1,), jnp.float32),
        "hc_head_base": jnp.zeros((4,), jnp.float32),
        "norm.weight": jnp.ones((8,), jnp.bfloat16),
        "head.weight": jnp.ones((8, 8), jnp.bfloat16),
    }
    layers = tuple({"value": jnp.asarray(i + 1, jnp.float32)} for i in range(43))
    caches = tuple({"window": jnp.full((256, 8), 99, jnp.bfloat16)} for _ in configs)

    def layer_step(self, streams, positions, ids, weights, cache, metadata, locations):
        return (
            streams,
            {"window": cache["window"].at[positions].set(weights["value"].astype(jnp.bfloat16))},
        )

    monkeypatch.setattr(native.DeepseekV4DecoderLayer, "__call__", layer_step)

    @jax.jit
    def forward(ids, positions, caches):
        return native.whole_model_forward(
            shared["embed.weight"][ids],
            positions,
            ids,
            shared,
            layers,
            caches,
            None,
            positions,
            configs=configs,
            mhc_backend="reference",
        )[1]

    first = forward(jnp.asarray([1]), jnp.asarray([0]), caches)
    second = forward(jnp.asarray([2]), jnp.asarray([128]), first)
    for layer, cache in enumerate(second):
        np.testing.assert_array_equal(cache["window"][0], layer + 1)
        np.testing.assert_array_equal(cache["window"][128], layer + 1)
        np.testing.assert_array_equal(cache["window"][1], 99)
    assert forward._cache_size() == 1


def test_four_device_model_runner_and_sampler_contract(monkeypatch):
    """Small arithmetic, real model/JIT/pool/sampler interfaces on four devices."""
    if len(jax.devices()) != 4:
        pytest.skip("requires four CPU devices or four TPU devices")
    import sgl_jax.srt.models.deepseek_v4 as native
    from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
    from sgl_jax.srt.mem_cache.deepseek_v4_paged_pool import V4PagedKVPool
    from sgl_jax.srt.layers.sampler import Sampler
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerSamplingInfo
    from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.model_executor.model_runner import ModelRunner
    from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
    from sgl_jax.srt.server_args import ServerArgs
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    mesh = create_device_mesh([1, 4], [1, 1])
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        hidden_size=8,
        num_attention_heads=64,
        head_dim=8,
        qk_rope_head_dim=4,
        o_groups=8,
        o_lora_rank=8,
        index_n_heads=64,
        index_head_dim=8,
        index_topk=8,
        sliding_window=128,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        num_experts_per_tok=6,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        compress_ratios=[0] * 43,
        num_hash_layers=3,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_scaling={
            "factor": 16.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 65536,
        },
        v4_max_context=256,
        expert_dtype="fp4",
        scoring_func="sqrtsoftplus",
        n_shared_experts=1,
        n_routed_experts=256,
        num_hidden_layers=43,
        vocab_size=32,
    )
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(lambda: native.DeepseekV4ForCausalLM(config, mesh))
        value = jax.device_put(np.asarray([1], np.uint8), NamedSharding(mesh, P()))
        model.shared = nnx.Param({"value": value})
        model.layers = nnx.Param(tuple({"value": value} for _ in range(43)))
        model.embed_tokens.embedding = nnx.Param(
            jax.device_put(
                np.ones((32, 8), np.float32).astype(jnp.bfloat16), NamedSharding(mesh, P())
            )
        )
        model.lm_head.embedding = nnx.Param(
            jax.device_put(
                np.broadcast_to(np.arange(1, 33)[:, None], (32, 8)).astype(jnp.bfloat16),
                NamedSharding(mesh, P("tensor", None)),
            )
        )
        pool = V4PagedKVPool(model.configs, mesh, capacity=256, requests=4)

    traced_shapes = []

    def core(embedded, positions, ids, shared, layers, caches, metadata, locations, **kwargs):
        traced_shapes.append(ids.shape)
        hidden = jnp.full((ids.shape[0], 8), 0.125, jnp.bfloat16)
        updates = tuple({"window": layer["window"] + jnp.bfloat16(1)} for layer in caches)
        return hidden, updates

    monkeypatch.setattr(native, "whole_model_forward", core)
    sa = ServerArgs(model_path="dummy", device=jax.default_backend(), moe_backend="epmoe")
    runner = object.__new__(ModelRunner)
    runner.model, runner.mesh, runner.server_args = model, mesh, sa
    runner.sampler = Sampler(nnx.Rngs(42), mesh)
    runner._sampler_base_rng, runner._sampler_step = jax.random.PRNGKey(42), 0
    runner.use_sort_for_toppk_minp = sa.use_sort_for_toppk_minp
    runner.token_to_kv_pool = pool
    runner.memory_pools = MemoryPools(token_to_kv_pool=pool)
    runner.tp_size, runner.forward_pass_id = 4, 0
    runner.model_config = SimpleNamespace(hf_config=config, is_embedding=False)
    runner.attn_backend = V4PagedBackend(max_context=256, mesh=mesh)
    monkeypatch.setattr(
        "sgl_jax.srt.model_executor.model_runner.aot_dispatch_requested", lambda: True
    )
    runner.initialize_jit()
    assert not hasattr(runner, "_run_model_dispatcher")
    cm = CompilationManager(sa, 1, 8, 1, 4, 1, 255, 32)
    cache_sizes = []
    for position in range(4):
        batch = cm._make_dummy_batch(1, 1, ForwardMode.DECODE, 256, dp_size=1, per_dp_bs_size=1)
        batch.positions = np.asarray([position], np.int32)
        batch.seq_lens = np.asarray([position + 1], np.int32)
        batch.out_cache_loc = np.asarray([128 + position], np.int32)
        batch.cache_loc[:128] = np.arange(128, 256, dtype=np.int32)
        batch.sampling_info = ModelWorkerSamplingInfo.generate_for_precompile_all_greedy(1, 32)
        batch.sampling_info.vocab_mask = None
        forward_batch = ForwardBatch.init_new(batch, runner)
        output, _, _ = runner.forward(
            forward_batch, LogitsMetadata.from_model_worker_batch(batch, mesh)
        )
        np.testing.assert_array_equal(np.asarray(output.next_token_logits), np.arange(32)[None] + 1)
        sample_metadata = SamplingMetadata.from_model_worker_batch(batch, 0, mesh, 32)
        # Several legacy TPU test modules install a process-global Explicit
        # mesh at import time. Production samples outside an Explicit mesh;
        # reproduce that contract instead of inheriting unrelated test state.
        with jax.set_mesh(None):
            tokens, _, _ = runner.sample(output, sample_metadata)
        np.testing.assert_array_equal(tokens, [31])
        np.testing.assert_array_equal(pool.layers[0]["window"], position + 1)
        cache_sizes.append(runner._jitted_run_model._cache_size())
    # Initial device_put buffers and returned/donated buffers can produce
    # distinct C++ dispatch-cache entries without retracing or compiling.
    assert traced_shapes == [(1,)]
    assert cache_sizes[1:] == [cache_sizes[1]] * 3
