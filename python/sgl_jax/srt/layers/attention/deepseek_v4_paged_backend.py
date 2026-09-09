"""V4 adapter for ordinary packed batches and the framework's paged allocator."""

from dataclasses import dataclass, fields

import jax
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    AttentionBackendMetadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

V4_PAGE_SIZE = 128


@register_pytree_node_class
@dataclass
class V4PagedMetadata(AttentionBackendMetadata):
    req_slots: object = None
    prefix_lens: object = None
    seq_lens: object = None
    query_lens: object = None
    query_starts: object = None
    token_requests: object = None
    token_valid: object = None
    page_table: object = None
    router_rows: object = None
    router_output_rows: object = None
    group4_requests: object = None
    group4_starts: object = None
    group128_requests: object = None
    group128_starts: object = None

    def tree_flatten(self):
        # Request lengths and boundaries are dynamic; only bucket shapes compile.
        return tuple(getattr(self, f.name) for f in fields(self)), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class V4PagedBackend(AttentionBackend):
    def __init__(self, *, max_context, mesh=None, max_requests=4, capacity=None):
        self.max_context = max_context
        self.max_requests = max_requests
        self.capacity = capacity
        self.mesh = mesh
        self.forward_metadata = nnx.data(V4PagedMetadata())

    @staticmethod
    def get_max_running_reqests(max_context_len, page_size):
        if page_size != V4_PAGE_SIZE:
            raise ValueError("V4 requires 128-token pages for compressor snapshots")
        return max(1, 131072 // ((max_context_len + 127) // 128))

    def prepare_precompile_batch(self, batch):
        """Construct valid synthetic page ownership without changing the allocator."""
        tokens, requests = len(batch.input_ids), len(batch.seq_lens)
        active = min(tokens, requests)
        qlens = np.zeros(requests, np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            qlens[:active] = 1
        else:
            qlens[:active] = tokens // active
            qlens[: tokens % active] += 1
        if qlens.max(initial=0) > self.max_context:
            raise ValueError("V4 precompile token bucket exceeds per-request context")
        positions = np.zeros(tokens, np.int32)
        locations = np.full(tokens, -1, np.int32)
        needed = int(np.sum((qlens + 127) // 128)) * 128
        cache_loc = np.zeros(max(len(batch.cache_loc), needed), np.int32)
        row, physical, offset = 0, 128, 0
        for length in qlens:
            if not length:
                continue
            aligned = (int(length) + 127) // 128 * 128
            positions[row : row + length] = np.arange(length, dtype=np.int32)
            locations[row : row + length] = physical + np.arange(length, dtype=np.int32)
            cache_loc[offset : offset + aligned] = physical + np.arange(aligned)
            row, physical, offset = row + int(length), physical + aligned, offset + aligned
        batch.input_ids = np.ones(tokens, np.int32)
        batch.positions, batch.out_cache_loc, batch.cache_loc = positions, locations, cache_loc
        batch.req_pool_indices = np.arange(requests, dtype=np.int32)
        batch.seq_lens = qlens.copy()
        batch.real_bs, batch.real_input_ids_len = active, int(qlens.sum())
        if batch.forward_mode == ForwardMode.EXTEND:
            batch.extend_prefix_lens = np.zeros(requests, np.int32)
            batch.extend_seq_lens = qlens.copy()
            batch.logits_indices = np.maximum(np.cumsum(qlens) - 1, 0).astype(np.int32)

    def get_forward_metadata(self, batch):
        if batch.dp_size != 1:
            raise ValueError("V4 paged serving uses DP=1; batch size is configurable")
        lengths = np.asarray(batch.seq_lens, np.int32)
        reqs = np.asarray(batch.req_pool_indices, np.int32)
        if lengths.ndim != 1 or reqs.shape != lengths.shape:
            raise ValueError("V4 requires per-request lengths and slots")
        if np.any(lengths < 0) or np.any(lengths > self.max_context):
            raise ValueError("V4 sequence exceeds cache context")
        active = lengths > 0
        if np.any(reqs[active] < 0) or np.any(reqs[active] >= self.max_requests):
            raise ValueError("V4 request references an unallocated slot")
        if len(np.unique(reqs[active])) != np.count_nonzero(active):
            raise ValueError("V4 batch repeats a live request slot")
        if batch.forward_mode == ForwardMode.DECODE:
            qlens = active.astype(np.int32)
            prefixes = lengths - qlens
        elif batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            qlens = np.asarray(batch.extend_seq_lens, np.int32)
            prefixes = np.asarray(batch.extend_prefix_lens, np.int32)
            if qlens.shape != lengths.shape or prefixes.shape != lengths.shape:
                raise ValueError("V4 requires one prefix/query length per request")
            if np.any(prefixes < 0) or np.any(qlens < 0) or np.any(prefixes + qlens != lengths):
                raise ValueError("V4 prefix + query lengths must equal sequence lengths")
            if np.any(active & (qlens == 0)):
                raise ValueError("V4 live requests require at least one query token")
        else:
            raise ValueError(f"V4 does not support {batch.forward_mode}")
        tokens, bs = len(batch.input_ids), lengths.size
        real_tokens = int(qlens.sum())
        if real_tokens > tokens or batch.real_input_ids_len != real_tokens:
            raise ValueError("V4 token count does not match ragged query lengths")
        starts = np.concatenate((np.zeros(1, np.int32), np.cumsum(qlens)[:-1])).astype(np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            starts = np.arange(bs, dtype=np.int32)
        token_reqs, valid = np.zeros(tokens, np.int32), np.zeros(tokens, np.bool_)
        positions, locations = np.asarray(batch.positions), np.asarray(batch.out_cache_loc)
        if positions.shape != (tokens,) or locations.shape != (tokens,):
            raise ValueError("V4 positions/cache locations must align with padded token rows")
        pages = np.zeros((bs, (self.max_context + 127) // 128), np.int32)
        cache_loc, cache_offset = np.asarray(batch.cache_loc, np.int32), 0
        router_rows = np.full(((tokens + 7) // 8 + bs, 8), -1, np.int32)
        router_output, router_group = np.zeros(tokens, np.int32), 0
        groups = {
            r: (
                np.zeros((tokens + r - 1) // r + bs, np.int32),
                np.full((tokens + r - 1) // r + bs, -1, np.int32),
            )
            for r in (4, 128)
        }
        group_counts = {4: 0, 128: 0}
        for i, (prefix, count, length) in enumerate(zip(prefixes, qlens, lengths, strict=True)):
            if not count:
                continue
            begin, end = int(starts[i]), int(starts[i] + count)
            expected = np.arange(prefix, length, dtype=np.int32)
            if not np.array_equal(positions[begin:end], expected):
                raise ValueError("V4 positions must be contiguous within each request")
            token_reqs[begin:end], valid[begin:end] = i, True
            num_pages = (int(length) + 127) // 128
            physical = cache_loc[cache_offset : cache_offset + num_pages * 128 : 128]
            if physical.size != num_pages or np.any(physical <= 0) or np.any(physical % 128):
                raise ValueError("V4 requires allocated, aligned physical pages")
            if self.capacity is not None and np.any(physical >= self.capacity + 128):
                raise ValueError("V4 physical page exceeds pool capacity")
            pages[i, :num_pages] = physical // 128
            if not np.array_equal(locations[begin:end], physical[expected // 128] + expected % 128):
                raise ValueError("V4 writes do not match the scheduler page table")
            cache_offset += num_pages * 128
            block_count = (int(prefix) % 8 + int(count) + 7) // 8
            for row, position in zip(range(begin, end), expected, strict=True):
                block = router_group + (int(position) // 8 - int(prefix) // 8)
                router_rows[block, position % 8] = row
                router_output[row] = block * 8 + position % 8
            router_group += block_count
            for ratio, (requests, group_starts) in groups.items():
                for end_pos in range((int(prefix) // ratio + 1) * ratio, int(length) + 1, ratio):
                    index = group_counts[ratio]
                    requests[index], group_starts[index] = i, end_pos - ratio
                    group_counts[ratio] += 1
        metadata = V4PagedMetadata(
            np.where(active, reqs + 1, 0),
            prefixes,
            lengths,
            qlens,
            starts,
            token_reqs,
            valid,
            pages,
            router_rows,
            router_output,
            *groups[4],
            *groups[128],
        )
        if self.mesh is not None:
            metadata = jax.tree.map(
                lambda a: jax.device_put(a, NamedSharding(self.mesh, P())), metadata
            )
        return metadata

    def __call__(self, x, positions, weights, cache, config, locations):
        from sgl_jax.srt.kernels.deepseek_v4.attention import attention

        return attention(x, positions, weights, cache, config, self.forward_metadata, locations)
