"""Historical B1 contracts retained only as independent regression fixtures."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.configs.deepseek_v4 import V4_CHUNK_SIZE, V4_MAX_CONTEXT
from sgl_jax.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    AttentionBackendMetadata,
)
from sgl_jax.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools, ReqToTokenPool
from sgl_jax.srt.model_executor.deepseek_v4_reference import empty_cache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

V4_COMPRESSOR_BOUNDARY = 128


@register_pytree_node_class
@dataclass
class DeepseekV4Metadata(AttentionBackendMetadata):
    # Prefill specializes to the true length: do not execute padding through
    # the stateful compressor. Decode always has valid_tokens=1, independent
    # of position. General dynamically padded prefill is a later milestone.
    valid_tokens: int = 1

    def tree_flatten(self):
        return (), self.valid_tokens

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data)


class DeepseekV4Backend(AttentionBackend):
    def __init__(self, *, max_context):
        self.max_context = max_context
        self.forward_metadata = nnx.data(DeepseekV4Metadata())

    @staticmethod
    def get_max_running_reqests(max_context_len, page_size):
        return 1

    def prepare_precompile_batch(self, batch):
        """Use real token rows, not one token followed by compressor padding."""
        count = len(batch.input_ids) if batch.forward_mode == ForwardMode.EXTEND else 1
        batch.real_input_ids_len = count
        batch.input_ids = np.ones(count, np.int32)
        batch.positions = np.arange(count, dtype=np.int32)
        batch.out_cache_loc = np.arange(1, count + 1, dtype=np.int32)
        batch.seq_lens = np.asarray([count], np.int32)
        if batch.forward_mode == ForwardMode.EXTEND:
            batch.extend_seq_lens = np.asarray([count], np.int32)
            batch.logits_indices = np.asarray([count - 1], np.int32)

    def get_forward_metadata(self, batch):
        lengths = np.asarray(batch.seq_lens, np.int32)
        if batch.real_bs != 1 or lengths.shape != (1,) or batch.dp_size != 1:
            raise ValueError("V4 currently supports one unpadded request, DP=1")
        if np.asarray(batch.req_pool_indices).tolist() != [0]:
            raise ValueError("V4 requires the single allocated request slot 0")
        length = int(lengths[0])
        if not 1 <= length <= self.max_context:
            raise ValueError("V4 sequence exceeds the bounded cache capacity")
        if batch.forward_mode == ForwardMode.EXTEND:
            prefix_lens = np.asarray(batch.extend_prefix_lens, np.int32)
            extend_lens = np.asarray(batch.extend_seq_lens, np.int32)
            if prefix_lens.shape != (1,) or extend_lens.shape != (1,):
                raise ValueError("V4 requires exactly one prefix/extend length")
            prefix, count = int(prefix_lens[0]), int(extend_lens[0])
            if prefix < 0 or count < 1 or prefix + count != length:
                raise ValueError("V4 prefix + extend length must equal sequence length")
            # The vectorized ratio-4 and ratio-128 compressor paths inherit
            # partial state at a common 128-token boundary. The scheduler's
            # fixed 128-token chunks guarantee this for every continuation;
            # the final chunk may have any positive length.
            if prefix and prefix % V4_COMPRESSOR_BOUNDARY:
                raise ValueError(
                    "V4 continuation chunks must start on a 128-token boundary"
                )
        elif batch.forward_mode == ForwardMode.DECODE:
            count = 1
        else:
            raise ValueError(f"V4 does not support forward mode {batch.forward_mode}")
        if batch.real_input_ids_len != count or len(batch.input_ids) < count:
            raise ValueError("V4 input length does not match the declared request")
        positions = np.asarray(batch.positions, np.int32)[:count]
        if not np.array_equal(
            positions, np.arange(length - count, length, dtype=np.int32)
        ):
            raise ValueError("V4 requires contiguous absolute token positions")
        locations = np.asarray(batch.out_cache_loc)[:count]
        if np.any(locations <= 0) or np.any(locations > self.max_context):
            raise ValueError(
                "V4 forward references unallocated/out-of-range token slots"
            )
        if batch.return_logprob:
            raise ValueError("V4 input-token logprobs are not yet supported")
        return DeepseekV4Metadata(count)


@register_pytree_node_class
class DeepseekV4TokenToKVPool(KVCache):
    def __init__(self, configs, mesh):
        self.configs = tuple(configs)
        super().__init__(configs[0].max_context, 1, jnp.bfloat16, len(configs), mesh)
        self.kv_sharding = NamedSharding(mesh, P())
        self.reset()

    def reset(self):
        self.layers = tuple(
            jax.device_put(empty_cache(config), self.kv_sharding)
            for config in self.configs
        )
        self.mem_usage = self.get_kv_size_bytes() / 2**30

    def tree_flatten(self):
        return (self.layers,), (self.configs, self.mesh)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        configs, mesh = aux_data
        obj = object.__new__(cls)
        KVCache.__init__(
            obj, configs[0].max_context, 1, jnp.bfloat16, len(configs), mesh
        )
        obj.configs = configs
        obj.kv_sharding = NamedSharding(mesh, P())
        obj.layers = children[0]
        return obj

    def replace_buffer(self, layers):
        if len(layers) != self.layer_num:
            raise ValueError("V4 cache update must include every layer")
        self.layers = tuple(layers)

    def get_kv_size_bytes(self):
        return sum(
            x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(self.layers)
        )

    def get_fused_kv_buffer(self, layer_id):
        raise NotImplementedError(
            "V4 owns window/compressed/state arrays, not fused MHA KV"
        )

    def get_kv_buffer(self, layer_id):
        raise NotImplementedError(
            "V4 cache must be accessed through its per-layer state"
        )

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError("V4 updates all cache state inside the model forward")


def init_deepseek_v4_pools(runner):
    if (
        runner.req_to_token_pool is not None
        or runner.token_to_kv_pool_allocator is not None
    ):
        raise ValueError("V4 cannot reuse an external request/KV allocator")
    capacity = runner.model_config.context_len
    runner.kv_cache_dtype = jnp.bfloat16
    runner.max_total_num_tokens = capacity
    runner.req_to_token_pool = ReqToTokenPool(1, capacity + 4, np.int32)
    runner.token_to_kv_pool = DeepseekV4TokenToKVPool(runner.model.configs, runner.mesh)
    runner.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        size=capacity, kvcache=runner.token_to_kv_pool, dp_size=1
    )
    runner.memory_pools = MemoryPools(token_to_kv_pool=runner.token_to_kv_pool)


def validate_v4_server_args(args, model_config):
    """Legacy B1 bring-up contract, retained for its independent regression fixtures."""
    required = {
        "tp_size": 4,
        "dp_size": 1,
        "ep_size": 4,
        "moe_dp_size": 1,
        "max_running_requests": 1,
        "page_size": 1,
        "attention_backend": "deepseek_v4",
        "disable_radix_cache": True,
        "disable_hybrid_swa_memory": True,
        "disable_overlap_schedule": True,
    }
    for name, value in required.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"V4 integration requires {name}={value!r}")
    context_len = model_config.context_len
    if (
        context_len < V4_CHUNK_SIZE
        or context_len > V4_MAX_CONTEXT
        or context_len % V4_CHUNK_SIZE
    ):
        raise ValueError(
            f"V4 context_length must be a multiple of {V4_CHUNK_SIZE} "
            f"in [{V4_CHUNK_SIZE}, {V4_MAX_CONTEXT}]"
        )
    if args.max_total_tokens not in (None, model_config.context_len):
        raise ValueError("V4 max_total_tokens must equal context_length")
    chunk_size = args.chunked_prefill_size
    if chunk_size is not None and chunk_size > 0 and chunk_size != V4_CHUNK_SIZE:
        raise ValueError(
            f"V4 chunked prefill requires chunked_prefill_size={V4_CHUNK_SIZE}"
        )
    if context_len > 256 and chunk_size != V4_CHUNK_SIZE:
        raise ValueError(
            f"V4 contexts above 256 require chunked_prefill_size={V4_CHUNK_SIZE}"
        )
    if chunk_size == V4_CHUNK_SIZE and args.max_prefill_tokens < V4_CHUNK_SIZE:
        raise ValueError(
            f"V4 chunked prefill requires max_prefill_tokens >= {V4_CHUNK_SIZE}"
        )
    if args.speculative_algorithm or args.enable_lora or args.enable_sequence_parallel:
        raise ValueError(
            "V4 does not yet support speculative decoding, LoRA, or sequence parallel"
        )
    if args.pd_disaggregation or args.ep_dispatch_algorithm:
        raise ValueError(
            "V4 does not yet support PD disaggregation or dynamic expert placement"
        )
    if args.enable_return_routed_experts or args.enable_expert_balance_debug:
        raise ValueError(
            "V4 routed-expert capture/balance debugging is not yet integrated"
        )
    if args.kv_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError(
            "V4 currently stores cache in BF16 with the original activation QAT"
        )
    if str(args.load_format) not in ("auto", "safetensors"):
        raise ValueError(
            "V4 requires the original safetensors checkpoint, not dummy weights"
        )
