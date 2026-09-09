"""Request-owned V4 cache, integrated with the framework pool update protocol.

This bounded pool owns one complete request. Token allocator locations account
for admission, while the numerical kernels address cache by absolute position.
Radix sharing, offload and multiple request slots must remain disabled.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools, ReqToTokenPool
from sgl_jax.srt.model_executor.deepseek_v4_reference import empty_cache


@register_pytree_node_class
class DeepseekV4TokenToKVPool(KVCache):
    def __init__(self, configs, mesh):
        self.configs = tuple(configs)
        super().__init__(configs[0].max_context, 1, jnp.bfloat16, len(configs), mesh)
        self.kv_sharding = NamedSharding(mesh, P())
        self.reset()

    def reset(self):
        self.layers = tuple(
            jax.device_put(empty_cache(config), self.kv_sharding) for config in self.configs
        )
        self.mem_usage = self.get_kv_size_bytes() / 2**30

    def tree_flatten(self):
        return (self.layers,), (self.configs, self.mesh)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        configs, mesh = aux_data
        obj = object.__new__(cls)
        KVCache.__init__(obj, configs[0].max_context, 1, jnp.bfloat16, len(configs), mesh)
        obj.configs = configs
        obj.kv_sharding = NamedSharding(mesh, P())
        obj.layers = children[0]
        return obj

    def replace_buffer(self, layers):
        if len(layers) != self.layer_num:
            raise ValueError("V4 cache update must include every layer")
        self.layers = tuple(layers)

    def get_kv_size_bytes(self):
        return sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(self.layers))

    def get_fused_kv_buffer(self, layer_id):
        raise NotImplementedError("V4 owns window/compressed/state arrays, not fused MHA KV")

    def get_kv_buffer(self, layer_id):
        raise NotImplementedError("V4 cache must be accessed through its per-layer state")

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError("V4 updates all cache state inside the model forward")


def init_deepseek_v4_pools(runner):
    if runner.req_to_token_pool is not None or runner.token_to_kv_pool_allocator is not None:
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
