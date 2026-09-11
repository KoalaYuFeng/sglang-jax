"""Diagnostic finite BF16 rounding without an intermediate FP32 conversion."""

import numpy as np


def round_bf16_direct(value):
    """Round finite, BF16-range FP64 values to nearest, ties to even.

    Power-of-two scaling is exact here; np.rint rounds the significand in FP64.
    This deliberately avoids torch/ml_dtypes conversion through FP32.
    """
    x = np.asarray(value, dtype=np.float64)
    if not np.isfinite(x).all() or np.any(np.abs(x) > (2 - 2**-7) * 2**127):
        raise ValueError("expected finite values within BF16 range")
    _, exponent = np.frexp(x)
    step = np.maximum(exponent - 8, -133)
    return np.ldexp(np.rint(np.ldexp(x, -step)), step).astype(np.float32)
