"""SGLang-JAX attention backend for DeepSeek-V4 CSA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.csa.csa import build_csa_step
from sgl_jax.srt.kernels.csa.tune import (
    CSA_ATTENTION_DIM,
    CSA_ATTENTION_HEADS,
    CSA_COMPRESSION_RATIO,
    CSA_DEFAULT_PAGE_SIZE,
    CSA_DUAL_PROJECTION_DIM,
    CSA_HIDDEN_DIM,
    CSA_INDEX_DIM,
    CSA_INDEX_HEADS,
    CSA_INDEX_PROJECTED_DIM,
    CSA_MAIN_PROJECTED_DIM,
    CSA_ROPE_FREQUENCY_DIM,
    get_csa_max_running_requests,
)
from sgl_jax.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    AttentionBackendMetadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


@register_pytree_node_class
@dataclass
class CSABackendMetadata(AttentionBackendMetadata):
    """Per-forward request mapping and static CSA specialization."""

    state_slots: jax.Array | None = None
    query_seq_ids: jax.Array | None = None
    cu_q_lens: jax.Array | None = None
    compressed_page_indices: jax.Array | None = None
    window_page_indices: jax.Array | None = None
    seq_lens: jax.Array | None = None
    distribution: jax.Array | None = None
    query_lengths: tuple[int, ...] = ()
    query_start_slots: tuple[int, ...] = ()
    uniform_prefill: bool = False

    def tree_flatten(self):
        children = (
            self.state_slots,
            self.query_seq_ids,
            self.cu_q_lens,
            self.compressed_page_indices,
            self.window_page_indices,
            self.seq_lens,
            self.distribution,
        )
        return children, (
            self.query_lengths,
            self.query_start_slots,
            self.uniform_prefill,
        )

    @classmethod
    def tree_unflatten(cls, static, children):
        query_lengths, query_start_slots, uniform_prefill = static
        return cls(*children, query_lengths, query_start_slots, uniform_prefill)


@dataclass
class CSABackend(AttentionBackend):
    """Run the complete single-device CSA operator family."""

    def __init__(
        self,
        *,
        num_attn_heads: int = CSA_ATTENTION_HEADS,
        head_dim: int = CSA_ATTENTION_DIM,
        compressor_hidden_size: int = CSA_HIDDEN_DIM,
        page_size: int = CSA_DEFAULT_PAGE_SIZE,
        compress_ratio: int = CSA_COMPRESSION_RATIO,
        mesh: jax.sharding.Mesh,
    ):
        if mesh is None or mesh.size != 1:
            raise ValueError("CSABackend currently requires a single-device mesh")
        if (
            num_attn_heads != CSA_ATTENTION_HEADS
            or head_dim != CSA_ATTENTION_DIM
            or compressor_hidden_size != CSA_HIDDEN_DIM
            or page_size != CSA_DEFAULT_PAGE_SIZE
            or compress_ratio != CSA_COMPRESSION_RATIO
        ):
            raise ValueError("production CSA requires H=64, D=512, hidden=4096, page=128, ratio=4")
        self.num_heads = num_attn_heads
        self.head_dim = head_dim
        self.compressor_hidden_size = compressor_hidden_size
        self.page_size = page_size
        self.compress_ratio = compress_ratio
        self.mesh = mesh
        self.forward_metadata = nnx.data(CSABackendMetadata())
        # The scheduler-owned cache manager exposes page_tables(req_ids, seq_lens)
        # and returns (compressed_pages, window_pages), both [batch, pages].
        self.page_table_provider = None

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        if self.page_table_provider is None:
            raise RuntimeError("model_runner must attach a CSA page-table provider first")
        req_pool_indices = np.asarray(batch.req_pool_indices, np.int32)
        seq_lens = np.asarray(batch.seq_lens, np.int32)
        positions = np.asarray(batch.positions, np.int32).reshape(-1)
        if req_pool_indices.shape != seq_lens.shape or not seq_lens.size:
            raise ValueError("req_pool_indices and seq_lens must be nonempty and aligned")

        if batch.forward_mode == ForwardMode.DECODE:
            query_lengths = np.ones_like(seq_lens)
            prefix_lens = seq_lens - query_lengths
            uniform_prefill = False
            distribution = np.full((3,), seq_lens.size, np.int32)
        elif batch.forward_mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            query_lengths = np.asarray(batch.extend_seq_lens, np.int32)
            if query_lengths.shape != seq_lens.shape:
                raise ValueError("extend_seq_lens must have one value per CSA request")
            prefix_lens = (
                seq_lens - query_lengths
                if batch.extend_prefix_lens is None
                else np.asarray(batch.extend_prefix_lens, np.int32)
            )
            if prefix_lens.shape != seq_lens.shape:
                raise ValueError("extend_prefix_lens must have one value per CSA request")
            uniform_prefill = bool(
                batch.forward_mode == ForwardMode.EXTEND
                and np.all(prefix_lens == 0)
                and np.all(query_lengths == query_lengths[0])
            )
            distribution = np.asarray((0, 0, seq_lens.size), np.int32)
        else:
            raise ValueError(f"CSA does not support {batch.forward_mode}")
        if np.any(query_lengths <= 0):
            raise ValueError("every CSA request must contain at least one query token")
        if np.any(prefix_lens < 0) or np.any(prefix_lens + query_lengths != seq_lens):
            raise ValueError("CSA prefix and query lengths must sum to seq_lens")
        if int(query_lengths.sum()) != positions.size:
            raise ValueError("CSA positions must contain exactly one row per query token")
        expected_positions = np.concatenate(
            [
                np.arange(prefix, sequence, dtype=np.int32)
                for prefix, sequence in zip(prefix_lens, seq_lens, strict=True)
            ]
        )
        if not np.array_equal(positions, expected_positions):
            raise ValueError("CSA positions must be contiguous within every request")

        if batch.recurrent_indices is None:
            raise ValueError("CSA requires batch.recurrent_indices")
        state_slots = np.asarray(batch.recurrent_indices, np.int32)
        if state_slots.shape != seq_lens.shape or np.any(state_slots == 0):
            raise ValueError("CSA forward references an unallocated recurrent slot")

        compressed_pages, window_pages = self.page_table_provider.page_tables(
            req_pool_indices,
            seq_lens,
        )
        compressed_pages = np.asarray(compressed_pages, np.int32)
        window_pages = np.asarray(window_pages, np.int32)
        batch_size = seq_lens.size
        if (
            compressed_pages.ndim != 2
            or window_pages.ndim != 2
            or compressed_pages.shape[0] != batch_size
            or window_pages.shape[0] != batch_size
            or not compressed_pages.shape[1]
            or not window_pages.shape[1]
        ):
            raise ValueError("CSA page tables must be nonempty [batch,pages] arrays")
        compressed_entries = np.floor_divide(seq_lens, self.compress_ratio)
        compressed_needed = np.maximum(
            1,
            (compressed_entries + self.page_size - 1) // self.page_size,
        )
        window_needed = np.maximum(1, (seq_lens + self.page_size - 1) // self.page_size)
        if np.any(compressed_needed > compressed_pages.shape[1]):
            raise ValueError("compressed page table does not cover every sequence")
        if np.any(window_needed > window_pages.shape[1]):
            raise ValueError("window page table does not cover every sequence")

        cu_q_lens = np.concatenate(
            (np.zeros((1,), np.int32), np.cumsum(query_lengths, dtype=np.int32))
        )
        query_seq_ids = np.repeat(
            np.arange(batch_size, dtype=np.int32),
            query_lengths,
        )
        arrays = device_array(
            (
                state_slots,
                query_seq_ids,
                cu_q_lens,
                compressed_pages,
                window_pages,
                seq_lens,
                distribution,
            ),
            sharding=NamedSharding(self.mesh, P()),
        )
        return CSABackendMetadata(
            *arrays,
            query_lengths=tuple(int(value) for value in query_lengths),
            query_start_slots=tuple(
                int(value) for value in np.mod(prefix_lens, self.compress_ratio)
            ),
            uniform_prefill=uniform_prefill,
        )

    def tree_flatten(self):
        return (self.forward_metadata,), {
            "num_attn_heads": self.num_heads,
            "head_dim": self.head_dim,
            "compressor_hidden_size": self.compressor_hidden_size,
            "page_size": self.page_size,
            "compress_ratio": self.compress_ratio,
            "mesh": self.mesh,
        }

    @classmethod
    def tree_unflatten(cls, static, children):
        backend = cls(**static)
        backend.forward_metadata = children[0]
        return backend

    def _check_inputs(
        self,
        q,
        new_kv,
        compressor_input,
        dual_weight,
        main_ape,
        index_ape,
        main_norm,
        index_norm,
        cos,
        sin,
        index_query,
        index_weights,
        attention_sink,
        max_context_len,
    ):
        tokens = q.shape[0]
        if q.shape != (tokens, self.num_heads, self.head_dim) or q.dtype != jnp.bfloat16:
            raise ValueError("CSA q must be BF16 [T,64,512]")
        if new_kv.shape != (tokens, self.head_dim) or new_kv.dtype != jnp.bfloat16:
            raise ValueError("CSA k/v must be BF16 [T,512]")
        if (
            compressor_input.shape != (tokens, self.compressor_hidden_size)
            or compressor_input.dtype != jnp.bfloat16
        ):
            raise ValueError("compressor_input must be BF16 [T,4096]")
        if (
            dual_weight.shape != (CSA_HIDDEN_DIM, CSA_DUAL_PROJECTION_DIM)
            or dual_weight.dtype != jnp.bfloat16
        ):
            raise ValueError("dual_weight must be BF16 [4096,2560]")
        if (
            main_ape.shape != (self.compress_ratio, CSA_MAIN_PROJECTED_DIM)
            or index_ape.shape != (self.compress_ratio, CSA_INDEX_PROJECTED_DIM)
            or main_ape.dtype != jnp.float32
            or index_ape.dtype != jnp.float32
        ):
            raise ValueError("CSA APE tensors must be FP32 [4,1024] and [4,256]")
        if (
            main_norm.shape != (self.head_dim,)
            or index_norm.shape != (CSA_INDEX_DIM,)
            or main_norm.dtype != jnp.float32
            or index_norm.dtype != jnp.float32
        ):
            raise ValueError("CSA norm weights must be FP32 [512] and [128]")
        if cos.ndim != 2 or cos.shape[1] != CSA_ROPE_FREQUENCY_DIM or sin.shape != cos.shape:
            raise ValueError("CSA RoPE tables must both be [positions,32]")
        if cos.dtype != jnp.float32 or sin.dtype != jnp.float32:
            raise ValueError("CSA RoPE tables must use FP32")
        if cos.shape[0] < max_context_len:
            raise ValueError("CSA RoPE tables do not cover max_context_len")
        if index_query.shape != (tokens, CSA_INDEX_HEADS, CSA_INDEX_DIM):
            raise ValueError("index_query must be [T,64,128]")
        if (
            index_query.dtype != jnp.bfloat16
            or index_weights.shape != (tokens, CSA_INDEX_HEADS)
            or index_weights.dtype != jnp.float32
        ):
            raise ValueError("CSA index queries must be BF16 and weights FP32 [T,64]")
        if attention_sink.shape != (self.num_heads,) or attention_sink.dtype != jnp.float32:
            raise ValueError("attention_sink must be FP32 [64]")

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer,
        forward_batch: ForwardBatch,
        token_to_kv_pool,
        *,
        recurrent_state_pool,
        compressor_input: jax.Array,
        dual_weight: jax.Array,
        main_ape: jax.Array,
        index_ape: jax.Array,
        main_norm: jax.Array,
        index_norm: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        index_query: jax.Array,
        index_weights: jax.Array,
        attention_sink: jax.Array,
        **_kwargs,
    ):
        metadata = self.forward_metadata
        if not metadata.query_lengths:
            raise RuntimeError("CSABackend.forward_metadata has not been prepared")
        if k.ndim == 3 and k.shape[1] == 1:
            new_kv = k[:, 0]
        elif k.ndim == 2:
            new_kv = k
        else:
            raise ValueError("CSA k/v must be [T,512] or [T,1,512]")
        if v.shape != k.shape:
            raise ValueError("CSA k and v must share the same shape")
        self._check_inputs(
            q,
            new_kv,
            compressor_input,
            dual_weight,
            main_ape,
            index_ape,
            main_norm,
            index_norm,
            cos,
            sin,
            index_query,
            index_weights,
            attention_sink,
            token_to_kv_pool.max_context_len,
        )

        layer_id = int(layer.layer_id)
        layer_index = token_to_kv_pool._layer_index(layer_id)
        main_state_pool, index_state_pool = recurrent_state_pool.get_csa_states(layer_id)
        main_nope, main_rope, index_cache = token_to_kv_pool.get_compressor_buffers(layer_id)
        state_slots = metadata.state_slots
        scale = (
            self.head_dim**-0.5 if getattr(layer, "scaling", None) is None else float(layer.scaling)
        )
        step = build_csa_step(
            metadata.query_lengths,
            query_start_slots=metadata.query_start_slots,
            uniform_prefill=metadata.uniform_prefill,
            softmax_scale=scale,
        )
        (
            output,
            _,
            main_state,
            index_state,
            main_nope,
            main_rope,
            index_cache,
            window_cache,
        ) = step(
            compressor_input,
            dual_weight,
            main_ape,
            index_ape,
            main_norm,
            index_norm,
            cos,
            sin,
            forward_batch.positions,
            metadata.cu_q_lens,
            metadata.query_seq_ids,
            metadata.compressed_page_indices,
            metadata.window_page_indices,
            metadata.seq_lens,
            metadata.distribution,
            index_query,
            index_weights,
            q,
            new_kv,
            attention_sink,
            main_state_pool[state_slots],
            index_state_pool[state_slots],
            main_nope,
            main_rope,
            index_cache,
            token_to_kv_pool.window_buffer[layer_index],
        )
        main_state_pool = main_state_pool.at[state_slots].set(main_state)
        index_state_pool = index_state_pool.at[state_slots].set(index_state)
        return output.reshape(output.shape[0], -1).astype(q.dtype), (
            main_state_pool,
            index_state_pool,
            window_cache,
            main_nope,
            main_rope,
            index_cache,
        )

    @staticmethod
    def pack_pool_updates(layer_updates) -> dict:
        main, index, window, main_nope, main_rope, index_cache = zip(
            *layer_updates,
            strict=True,
        )
        states = [state for pair in zip(main, index, strict=True) for state in pair]
        return {
            "token_to_kv_pool": {
                "window_buffer": list(window),
                "main_nope_buffer": list(main_nope),
                "main_rope_buffer": list(main_rope),
                "index_buffer": list(index_cache),
            },
            "recurrent_state_pool": {"state_buffers": states},
        }

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        return get_csa_max_running_requests(max_context_len, page_size)


__all__ = ["CSABackend", "CSABackendMetadata"]
