"""Independent CPU inputs/reduction contract for V4 decode projection."""

import ml_dtypes
import numpy as np


def synthetic_projection(tokens, width, seed=8023):
    rng = np.random.default_rng(seed + width + tokens)
    x = rng.normal(size=(tokens, 4096)).astype(ml_dtypes.bfloat16)
    weight = rng.normal(scale=0.03, size=(4096, width)).astype(ml_dtypes.bfloat16)
    if tokens > 1:
        x[-1] = 0
    return x, weight


def numpy_v4_projection(x, weight):
    """Execute the declared FP32 GEMV tree with NumPy, not JAX/Pallas.

    Each wkv/wgate half retains the original main/index chunk size. Work one
    token at a time to avoid allocating [batch,width,4096] on the host.
    """
    width = weight.shape[1]
    if weight.shape[0] != 4096 or width not in (512, 2048):
        raise ValueError("expected the fused Flash main/index projection")
    chunk = 1024 if width == 2048 else 2048
    rhs = weight.astype(np.float32).T
    outputs = []
    for token in x.astype(np.float32):
        product = rhs * token
        partials = []
        for start in range(0, 4096, chunk):
            vectors = product[:, start : start + chunk].reshape(width, chunk // 128, 128)
            total = vectors[:, 0].copy()
            for i in range(1, chunk // 128):
                total = np.add(total, vectors[:, i], dtype=np.float32)
            stripes = total.reshape(width, 16, 8)
            subtotal = stripes[:, 0].copy()
            for i in range(1, 16):
                subtotal = np.add(subtotal, stripes[:, i], dtype=np.float32)
            for half in (4, 2, 1):
                subtotal = np.add(
                    subtotal[:, :half], subtotal[:, half : 2 * half], dtype=np.float32
                )
            partials.append(subtotal[:, 0])
        while len(partials) > 1:
            partials = [
                np.add(partials[i], partials[i + 1], dtype=np.float32)
                for i in range(0, len(partials), 2)
            ]
        outputs.append(partials[0])
    return np.stack(outputs)
