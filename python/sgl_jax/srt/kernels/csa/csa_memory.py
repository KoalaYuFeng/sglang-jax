"""Physical cache and compressor-state storage for CSA."""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.layout import Format, Layout
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.csa.tune import (
    CSA_ATTENTION_DIM,
    CSA_CACHE_PACKING,
    CSA_COMPRESSION_RATIO,
    CSA_DEFAULT_PAGE_SIZE,
    CSA_INDEX_PROJECTED_DIM,
    CSA_INDEX_RECORD_BYTES,
    CSA_MAIN_PROJECTED_DIM,
    CSA_STATE_SLOTS,
    CSA_WINDOW_SIZE,
    TPU_V6E,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import get_dtype_packing
from sgl_jax.srt.mem_cache.memory_pool import KVCache

_TPU_UINT8_CACHE_LAYOUT = Layout(
    (0, 1, 2, 3),
    (
        (CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
        (CSA_CACHE_PACKING, 1),
    ),
)


def _initial_state(rows: int, width: int):
    shape = (rows, CSA_STATE_SLOTS, width)
    return jnp.stack(
        (jnp.zeros(shape, jnp.float32), jnp.full(shape, -jnp.inf, jnp.float32)),
        axis=2,
    )


@register_pytree_node_class
class CSARecurrentStatePool:
    """Own main-KV and Lightning-Indexer compressor states for each CSA layer."""

    _STATE_NAMES = ("main", "index")
    _STATE_WIDTHS = (CSA_MAIN_PROJECTED_DIM, CSA_INDEX_PROJECTED_DIM)

    def __init__(
        self,
        layer_ids: list[int] | tuple[int, ...],
        size: int,
        mesh: jax.sharding.Mesh,
    ):
        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("CSA layer ids must be nonempty and unique")
        if size <= 0:
            raise ValueError("CSA recurrent size must be positive")
        if mesh.size != 1:
            raise ValueError("CSA memory currently supports one TPU device")

        self.linear_recurrent_layer_ids = layer_ids
        self.layers_mapping = {layer_id: index for index, layer_id in enumerate(layer_ids)}
        self.size = size
        self.dp_size = 1
        self.slots_per_rank = size
        self.total_slots = size + 1
        self.mesh = mesh
        self.state_sharding = NamedSharding(mesh, P(None, None, None, None))
        self.state_buffers = self._create_buffers()

    def _create_buffers(self):
        initializers = {
            width: jax.jit(
                partial(_initial_state, self.total_slots, width),
                out_shardings=self.state_sharding,
            )
            for width in self._STATE_WIDTHS
        }
        with jax.set_mesh(self.mesh):
            return [
                initializers[width]()
                for _ in self.linear_recurrent_layer_ids
                for width in self._STATE_WIDTHS
            ]

    def _layer_index(self, layer_id: int) -> int:
        try:
            return self.layers_mapping[int(layer_id)]
        except KeyError as exc:
            raise ValueError(f"layer_id={layer_id} is not a CSA layer") from exc

    def _buffer_index(self, layer_id: int, state_name: str) -> int:
        try:
            state_index = self._STATE_NAMES.index(state_name)
        except ValueError as exc:
            raise ValueError(f"unknown CSA state {state_name!r}") from exc
        return self._layer_index(layer_id) * len(self._STATE_NAMES) + state_index

    def get_state(self, layer_id: int, state_name: str):
        return self.state_buffers[self._buffer_index(layer_id, state_name)]

    def get_csa_states(self, layer_id: int):
        return self.get_state(layer_id, "main"), self.get_state(layer_id, "index")

    def get_linear_recurrent_layer_cache(self, layer_id: int):
        return self.get_csa_states(layer_id), []

    def replace_csa_states(self, layer_id: int, main, index) -> None:
        for name, value in zip(self._STATE_NAMES, (main, index), strict=True):
            buffer_index = self._buffer_index(layer_id, name)
            current = self.state_buffers[buffer_index]
            if value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError(f"replacement state {name!r} has the wrong shape or dtype")
            self.state_buffers[buffer_index] = value

    def reset_slots(self, global_slots) -> None:
        global_slots = jnp.asarray(global_slots, jnp.int32)
        if not global_slots.size:
            return
        with jax.set_mesh(self.mesh):
            for index, buffer in enumerate(self.state_buffers):
                width = self._STATE_WIDTHS[index % len(self._STATE_WIDTHS)]
                reset = _initial_state(global_slots.shape[0], width)
                self.state_buffers[index] = buffer.at[global_slots].set(
                    reset,
                    mode="promise_in_bounds",
                    unique_indices=True,
                    out_sharding=self.state_sharding,
                )

    def clear(self) -> None:
        self.state_buffers = self._create_buffers()

    def copy_slots(self, src_indices, dst_indices):
        slot_sharding = NamedSharding(self.mesh, P(None))
        src_indices = jax.sharding.reshard(src_indices, slot_sharding)
        dst_indices = jax.sharding.reshard(dst_indices, slot_sharding)

        def _copy(buffer):
            buffer = jax.lax.optimization_barrier(buffer)
            values = jnp.where(
                (src_indices == 0).reshape(-1, 1, 1, 1),
                buffer[dst_indices],
                buffer[src_indices],
            )
            buffer = buffer.at[dst_indices].set(
                values,
                out_sharding=self.state_sharding,
            )
            return jax.lax.optimization_barrier(buffer)

        return [_copy(buffer) for buffer in self.state_buffers], []

    def replace_buffer(self, buffers) -> None:
        if isinstance(buffers, dict):
            states = buffers["state_buffers"]
            conv = ()
        else:
            states, conv = buffers
        if conv or len(states) != len(self.state_buffers):
            raise ValueError("CSA requires one main/index state pair per layer")
        self.state_buffers = list(states)

    def get_size_bytes(self) -> int:
        return sum(buffer.size * buffer.dtype.itemsize for buffer in self.state_buffers)

    def tree_flatten(self):
        children = (tuple(self.state_buffers), tuple())
        aux = (
            self.linear_recurrent_layer_ids,
            self.size,
            self.total_slots,
            self.mesh,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        layer_ids, size, total_slots, mesh = aux
        obj = object.__new__(cls)
        obj.linear_recurrent_layer_ids = layer_ids
        obj.layers_mapping = {layer_id: index for index, layer_id in enumerate(layer_ids)}
        obj.size = size
        obj.dp_size = 1
        obj.slots_per_rank = size
        obj.total_slots = total_slots
        obj.mesh = mesh
        obj.state_sharding = NamedSharding(mesh, P(None, None, None, None))
        obj.state_buffers = list(children[0])
        return obj


@register_pytree_node_class
class CSAKVPool(KVCache):
    """Own CSA's SWA and packed compressed-cache buffers."""

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: jnp.dtype,
        layer_num: int,
        mesh: jax.sharding.Mesh,
        *,
        max_num_requests: int,
        max_context_len: int,
        layer_ids: list[int] | tuple[int, ...] | None = None,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ):
        super().__init__(size, page_size, dtype, layer_num, mesh, start_layer, end_layer)
        if page_size != CSA_DEFAULT_PAGE_SIZE:
            raise ValueError(f"CSA cache requires page_size={CSA_DEFAULT_PAGE_SIZE}")
        if jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
            raise ValueError("CSA sliding-window cache must use BF16")
        if size <= 0:
            raise ValueError("CSA token capacity must be positive")
        if max_num_requests <= 0:
            raise ValueError("request capacity must be positive")
        if mesh.size != 1:
            raise ValueError("CSA memory currently supports one TPU device")

        self.max_num_requests = max_num_requests
        self.max_context_len = max_context_len
        self.dp_size = 1
        if layer_ids is None:
            layer_ids = range(self.start_layer, self.start_layer + layer_num)
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if len(self.layer_ids) != layer_num or len(set(self.layer_ids)) != layer_num:
            raise ValueError("layer_ids must contain one unique id per CSA layer")
        self.layers_mapping = {layer_id: index for index, layer_id in enumerate(self.layer_ids)}
        self.packing = get_dtype_packing(dtype)
        if page_size % self.packing:
            raise ValueError("CSA page_size must be divisible by dtype packing")
        self.cache_sharding = NamedSharding(mesh, P(None, None, None, None))
        self.packed_sharding = NamedSharding(mesh, P(None, None, None, None))
        self.packed_format = Format(_TPU_UINT8_CACHE_LAYOUT, self.packed_sharding)
        (
            self.window_buffer,
            self.main_nope_buffer,
            self.main_rope_buffer,
            self.index_buffer,
        ) = self._create_buffers()
        self.mem_usage = self.get_kv_size_bytes() / 2**30

    @property
    def window_pages_per_rank(self) -> int:
        return self.max_num_requests * (CSA_WINDOW_SIZE // self.page_size)

    @property
    def compressed_pages_per_rank(self) -> int:
        entries = self.size // CSA_COMPRESSION_RATIO
        pages = math.ceil(entries / self.page_size)
        return pages + self.max_num_requests

    def _physical_pages(self, pages_per_rank: int) -> int:
        return pages_per_rank + 1

    def _create_buffers(self):
        window_shape = (
            self._physical_pages(self.window_pages_per_rank),
            self.page_size // self.packing,
            self.packing,
            CSA_ATTENTION_DIM,
        )
        compressed_pages = self._physical_pages(self.compressed_pages_per_rank)
        main_nope_shape = (
            compressed_pages,
            self.page_size,
            CSA_CACHE_PACKING,
            TPU_V6E.vector_lanes,
        )
        main_rope_shape = (
            compressed_pages,
            self.page_size // CSA_CACHE_PACKING,
            CSA_CACHE_PACKING,
            TPU_V6E.vector_lanes,
        )
        index_shape = (
            compressed_pages,
            self.page_size // CSA_CACHE_PACKING,
            CSA_CACHE_PACKING,
            CSA_INDEX_RECORD_BYTES,
        )
        init_window = jax.jit(
            partial(jnp.zeros, shape=window_shape, dtype=self.dtype),
            out_shardings=self.cache_sharding,
        )
        init_main_nope = jax.jit(
            partial(jnp.zeros, shape=main_nope_shape, dtype=jnp.uint8),
            out_shardings=self.packed_format,
        )
        init_main_rope = jax.jit(
            partial(jnp.zeros, shape=main_rope_shape, dtype=jnp.uint8),
            out_shardings=self.packed_format,
        )
        init_index = jax.jit(
            partial(jnp.zeros, shape=index_shape, dtype=jnp.uint8),
            out_shardings=self.packed_format,
        )
        with jax.set_mesh(self.mesh):
            return (
                [init_window() for _ in range(self.layer_num)],
                [init_main_nope() for _ in range(self.layer_num)],
                [init_main_rope() for _ in range(self.layer_num)],
                [init_index() for _ in range(self.layer_num)],
            )

    def _layer_index(self, layer_id: int) -> int:
        try:
            return self.layers_mapping[int(layer_id)]
        except KeyError as exc:
            raise ValueError(f"layer_id={layer_id} is not a CSA layer") from exc

    def get_fused_kv_buffer(self, layer_id: int):
        return self.window_buffer[self._layer_index(layer_id)]

    def get_kv_buffer(self, layer_id: int):
        buffer = self.get_fused_kv_buffer(layer_id)
        return buffer, buffer

    def set_kv_buffer(self, layer_id, loc, cache_k, cache_v=None, is_decode=False):
        del cache_v, is_decode
        index = self._layer_index(layer_id)
        cache = self.window_buffer[index]
        loc = jax.sharding.reshard(loc, NamedSharding(self.mesh, P(None)))
        cache_k = jax.sharding.reshard(
            cache_k,
            NamedSharding(self.mesh, P(None, None)),
        )
        flat = cache.reshape(-1, cache.shape[-1])
        safe = jnp.where(loc >= 0, loc, flat.shape[0])
        self.window_buffer[index] = (
            flat.at[safe]
            .set(
                cache_k.astype(cache.dtype),
                mode="drop",
                out_sharding=self.cache_sharding,
            )
            .reshape(cache.shape)
        )

    def get_compressor_buffers(self, layer_id: int):
        index = self._layer_index(layer_id)
        return (
            self.main_nope_buffer[index],
            self.main_rope_buffer[index],
            self.index_buffer[index],
        )

    def replace_compressor_buffers(self, layer_id: int, main_nope, main_rope, index) -> None:
        layer_index = self._layer_index(layer_id)
        replacements = (main_nope, main_rope, index)
        current = (
            self.main_nope_buffer[layer_index],
            self.main_rope_buffer[layer_index],
            self.index_buffer[layer_index],
        )
        for name, old, new in zip(
            ("main_nope", "main_rope", "index"), current, replacements, strict=True
        ):
            if new.shape != old.shape or new.dtype != old.dtype:
                raise ValueError(f"replacement buffer {name!r} has the wrong shape or dtype")
        self.main_nope_buffer[layer_index] = main_nope
        self.main_rope_buffer[layer_index] = main_rope
        self.index_buffer[layer_index] = index

    def replace_buffer(self, buffers) -> None:
        if isinstance(buffers, dict):
            window = buffers["window_buffer"]
            main_nope = buffers["main_nope_buffer"]
            main_rope = buffers["main_rope_buffer"]
            index = buffers["index_buffer"]
        else:
            window, main_nope, main_rope, index = buffers
        groups = (window, main_nope, main_rope, index)
        if any(len(group) != self.layer_num for group in groups):
            raise ValueError("one CSA cache update is required per layer")
        self.window_buffer = list(window)
        self.main_nope_buffer = list(main_nope)
        self.main_rope_buffer = list(main_rope)
        self.index_buffer = list(index)

    def get_kv_size_bytes(self) -> int:
        groups = (
            self.window_buffer,
            self.main_nope_buffer,
            self.main_rope_buffer,
            self.index_buffer,
        )
        return sum(buffer.size * buffer.dtype.itemsize for group in groups for buffer in group)

    def get_cpu_copy(self, indices):
        del indices
        return jax.device_get(
            (
                self.window_buffer,
                self.main_nope_buffer,
                self.main_rope_buffer,
                self.index_buffer,
            )
        )

    def load_cpu_copy(self, kv_cache_cpu, indices):
        del indices
        window, main_nope, main_rope, index = kv_cache_cpu
        self.window_buffer = [jax.device_put(value, self.cache_sharding) for value in window]
        self.main_nope_buffer = [jax.device_put(value, self.packed_format) for value in main_nope]
        self.main_rope_buffer = [jax.device_put(value, self.packed_format) for value in main_rope]
        self.index_buffer = [jax.device_put(value, self.packed_format) for value in index]

    def tree_flatten(self):
        children = (
            tuple(self.window_buffer),
            tuple(self.main_nope_buffer),
            tuple(self.main_rope_buffer),
            tuple(self.index_buffer),
        )
        aux = {
            "size": self.size,
            "page_size": self.page_size,
            "dtype": self.dtype,
            "layer_num": self.layer_num,
            "mesh": self.mesh,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "max_num_requests": self.max_num_requests,
            "max_context_len": self.max_context_len,
            "dp_size": self.dp_size,
            "layer_ids": self.layer_ids,
            "mem_usage": self.mem_usage,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        for name, value in aux.items():
            setattr(obj, name, value)
        obj.layers_mapping = {layer_id: index for index, layer_id in enumerate(obj.layer_ids)}
        obj.packing = get_dtype_packing(obj.dtype)
        obj.cache_sharding = NamedSharding(obj.mesh, P(None, None, None, None))
        obj.packed_sharding = NamedSharding(obj.mesh, P(None, None, None, None))
        obj.packed_format = Format(_TPU_UINT8_CACHE_LAYOUT, obj.packed_sharding)
        (
            obj.window_buffer,
            obj.main_nope_buffer,
            obj.main_rope_buffer,
            obj.index_buffer,
        ) = (list(group) for group in children)
        return obj


__all__ = ["CSAKVPool", "CSARecurrentStatePool"]
