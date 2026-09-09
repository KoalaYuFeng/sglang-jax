"""V4 physical pages, page-end compressor snapshots and active request scratch.

The ordinary PagedTokenToKVPoolAllocator/RadixCache own every physical page.
Snapshots share the same lifetime, so prefix forks need no scheduler-specific
state ownership. Partial compressor state is private to the active request.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools, ReqToTokenPool

PAGE_SIZE = 128


def _ordered_pages(indices):
    pages = np.asarray(indices, np.int32) // PAGE_SIZE
    _, first = np.unique(pages, return_index=True)
    pages = pages[np.sort(first)]
    return pages[pages > 0]


def layer_buffer_specs(config, capacity, requests):
    rows = capacity + PAGE_SIZE  # physical page zero is never allocated
    pages = rows // PAGE_SIZE
    specs = {"window": ((rows, config.head_dim), jnp.bfloat16, 0)}
    if config.ratio:
        overlap = 2 if config.ratio == 4 else 1
        for prefix, dim in (("main", config.head_dim), ("index", config.index_dim)):
            if prefix == "index" and config.ratio != 4:
                continue
            state = (overlap * config.ratio, overlap * dim)
            specs[prefix + ".compressed"] = ((rows // config.ratio, dim), jnp.bfloat16, 0)
            for suffix, fill in (("kv", 0), ("score", -np.inf)):
                specs[f"{prefix}.{suffix}"] = ((requests + 1, *state), jnp.float32, fill)
                specs[f"{prefix}.snapshot_{suffix}"] = ((pages, *state), jnp.float32, fill)
    return specs


def pool_size_bytes(configs, capacity, requests):
    return sum(
        int(np.prod(shape)) * np.dtype(dtype).itemsize
        for config in configs
        for shape, dtype, _ in layer_buffer_specs(config, capacity, requests).values()
    )


@lru_cache(maxsize=128)
def _allocator(shape, dtype, fill, sharding):
    return jax.jit(lambda: jnp.full(shape, fill, dtype), out_shardings=sharding)


@register_pytree_node_class
class V4PagedKVPool(KVCache):
    def __init__(self, configs, mesh, *, capacity, requests):
        if capacity <= 0 or capacity % PAGE_SIZE or requests <= 0:
            raise ValueError("V4 requires positive page-aligned capacity and request slots")
        self.configs, self.requests = tuple(configs), requests
        super().__init__(capacity, PAGE_SIZE, jnp.bfloat16, len(configs), mesh)
        self.kv_sharding = NamedSharding(mesh, P())
        self.layers = tuple(
            {
                name: _allocator(shape, jnp.dtype(dtype), fill, self.kv_sharding)()
                for name, (shape, dtype, fill) in layer_buffer_specs(c, capacity, requests).items()
            }
            for c in self.configs
        )
        self.mem_usage = self.get_kv_size_bytes() / 2**30

    def tree_flatten(self):
        return (self.layers,), (self.configs, self.mesh, self.size, self.requests)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        configs, mesh, capacity, requests = aux_data
        obj = object.__new__(cls)
        KVCache.__init__(obj, capacity, PAGE_SIZE, jnp.bfloat16, len(configs), mesh)
        obj.configs, obj.requests = configs, requests
        obj.kv_sharding = NamedSharding(mesh, P())
        obj.layers = children[0]
        return obj

    def replace_buffer(self, layers):
        if len(layers) != len(self.configs):
            raise ValueError("V4 cache updates must contain every layer")
        self.layers = tuple(layers)

    def get_kv_size_bytes(self):
        return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(self.layers))

    def get_fused_kv_buffer(self, layer_id):
        return self.layers[layer_id]

    def get_kv_buffer(self, layer_id):
        window = self.layers[layer_id]["window"]
        return window, window

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError("V4 attention returns donated page/state updates from forward")

    def get_cpu_copy(self, indices):
        pages = _ordered_pages(indices)
        result = []
        for config, layer in zip(self.configs, self.layers, strict=True):
            copy = {}
            for key, value in layer.items():
                if ".snapshot_" in key:
                    copy[key] = np.asarray(jax.device_get(value[pages]))
                elif key == "window" or key.endswith(".compressed"):
                    stride = PAGE_SIZE if key == "window" else PAGE_SIZE // config.ratio
                    rows = (pages[:, None] * stride + np.arange(stride)).ravel()
                    copy[key] = np.asarray(jax.device_get(value[rows]))
            result.append(copy)
        return {"page_count": len(pages), "layers": result}

    def load_cpu_copy(self, copy, indices):
        pages = _ordered_pages(indices)
        if len(pages) != copy["page_count"]:
            raise ValueError("V4 cache restore must preserve the number of pages")
        updated = []
        for config, layer, saved in zip(self.configs, self.layers, copy["layers"], strict=True):
            layer = dict(layer)
            for key, values in saved.items():
                stride = PAGE_SIZE if key == "window" else PAGE_SIZE // max(config.ratio, 1)
                rows = (
                    pages
                    if ".snapshot_" in key
                    else (pages[:, None] * stride + np.arange(stride)).ravel()
                )
                layer[key] = layer[key].at[rows].set(jax.device_put(values, self.kv_sharding))
            updated.append(layer)
        self.layers = tuple(updated)


def init_v4_paged_pools(runner):
    args = runner.server_args
    requests = args.max_running_requests or 4
    capacity = args.max_total_tokens or runner.model_config.context_len * requests
    if capacity % PAGE_SIZE or capacity < runner.model_config.context_len:
        raise ValueError("V4 total token capacity must be page-aligned and fit one full context")
    if runner.req_to_token_pool is not None or runner.token_to_kv_pool_allocator is not None:
        raise ValueError("V4 cannot reuse an incompatible external allocator")
    required = pool_size_bytes(runner.model.configs, capacity, requests)
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        if "bytes_limit" in stats:
            budget = int(stats["bytes_limit"] * args.mem_fraction_static) - stats.get(
                "bytes_in_use", 0
            )
            if required > budget:
                raise ValueError(
                    f"V4 pages/snapshots need {required / 2**30:.2f} GiB per device; "
                    f"available static budget is {budget / 2**30:.2f} GiB"
                )
    runner.kv_cache_dtype = jnp.bfloat16
    runner.max_total_num_tokens = capacity
    runner.req_to_token_pool = ReqToTokenPool(
        requests, runner.model_config.context_len + 4, np.int32
    )
    runner.token_to_kv_pool = V4PagedKVPool(
        runner.model.configs, runner.mesh, capacity=capacity, requests=requests
    )
    runner.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
        capacity, PAGE_SIZE, runner.token_to_kv_pool
    )
    runner.memory_pools = MemoryPools(token_to_kv_pool=runner.token_to_kv_pool)
