"""Independent NumPy block-attention adjudicator, not an acceptance oracle.

Retains FP32 score boundaries, 64-key blocks, BF16 probability/value output
boundaries and the learned sink. Uses FP64 dots, exp and accumulators to inspect
rounding error; this is explicitly not official CPU/GPU execution arithmetic.
"""

import ml_dtypes
import numpy as np


def block_attention_fp64(q, kv, indices, sink, scale):
    q, kv, sink = (np.asarray(v, np.float64) for v in (q, kv, sink))
    indices = np.asarray(indices)
    if q.ndim != 3 or kv.ndim != 2 or indices.ndim != 2 or sink.shape != (q.shape[1],):
        raise ValueError("invalid attention shapes")
    if (
        not q.size
        or not kv.size
        or not indices.size
        or q.shape[2] != kv.shape[1]
        or indices.shape[0] != q.shape[0]
    ):
        raise ValueError("empty or incompatible attention shapes")
    if not all(np.isfinite(v).all() for v in (q, kv, sink, indices)) or not np.isfinite(
        scale
    ):
        raise ValueError("nonfinite attention input")
    if (
        np.any(indices != np.floor(indices))
        or np.any(indices < -1)
        or np.any(indices >= kv.shape[0])
    ):
        raise ValueError("invalid key index")
    count = (indices.shape[1] + 63) // 64 * 64
    indices = np.pad(
        indices.astype(np.int64),
        ((0, 0), (0, count - indices.shape[1])),
        constant_values=-1,
    )
    output = np.empty(q.shape, np.float32)

    def bf16(x):
        return x.astype(ml_dtypes.bfloat16).astype(np.float64)

    def difference(a, b):
        return (a - b).astype(np.float32).astype(np.float64)

    for t, query in enumerate(q):
        maximum = np.full(q.shape[1], -1e30, np.float32).astype(np.float64)
        denominator = np.zeros(q.shape[1], np.float64)
        numerator = np.zeros(query.shape, np.float64)
        for begin in range(0, count, 64):
            ids = indices[t, begin : begin + 64]
            keys = kv[np.maximum(ids, 0)]
            score = ((query @ keys.T).astype(np.float32) * np.float32(scale)).astype(
                np.float64
            )
            valid = ids[None, :] >= 0
            score = np.where(valid, score, float(np.float32(-1e30)))
            new_max = np.maximum(maximum, score.max(-1))
            alpha = np.exp(difference(maximum, new_max))
            probability = np.where(
                valid, np.exp(difference(score, new_max[:, None])), 0.0
            )
            numerator = numerator * alpha[:, None] + bf16(probability) @ keys
            denominator = denominator * alpha + probability.sum(-1)
            maximum = new_max
        final_max = np.maximum(maximum, sink)
        rescale = np.exp(difference(maximum, final_max))
        denominator = denominator * rescale + np.exp(difference(sink, final_max))
        output[t] = bf16(numerator * rescale[:, None] / denominator[:, None])
    return output
