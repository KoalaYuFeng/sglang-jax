"""Host batch contract for checkpoint-native, single-request V4 execution."""

from dataclasses import dataclass

import numpy as np
from flax import nnx
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import (
    AttentionBackend,
    AttentionBackendMetadata,
)
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
                raise ValueError("V4 continuation chunks must start on a 128-token boundary")
        elif batch.forward_mode == ForwardMode.DECODE:
            count = 1
        else:
            raise ValueError(f"V4 does not support forward mode {batch.forward_mode}")
        if batch.real_input_ids_len != count or len(batch.input_ids) < count:
            raise ValueError("V4 input length does not match the declared request")
        positions = np.asarray(batch.positions, np.int32)[:count]
        if not np.array_equal(positions, np.arange(length - count, length, dtype=np.int32)):
            raise ValueError("V4 requires contiguous absolute token positions")
        locations = np.asarray(batch.out_cache_loc)[:count]
        if np.any(locations <= 0) or np.any(locations > self.max_context):
            raise ValueError("V4 forward references unallocated/out-of-range token slots")
        if batch.return_logprob:
            raise ValueError("V4 input-token logprobs are not yet supported")
        return DeepseekV4Metadata(count)
