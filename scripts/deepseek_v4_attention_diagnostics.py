"""Pure CPU structural checks for independent attention diagnostics."""

import numpy as np


def fp8_roundtrip_cpu(value, block_size=128):
    """Independent finite E4M3FN / power-of-two-scale nearest-even quantizer.

    Diagnostic oracle only; does not import torch, JAX, or serving helpers.
    Returns dequantized FP32 values and broadcast FP64 block scales.
    """
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("invalid block size")
    x = np.asarray(value, np.float64)
    if x.ndim < 1 or not x.size or x.shape[-1] % block_size or not np.isfinite(x).all():
        raise ValueError("invalid FP8 input")
    grouped = x.reshape(*x.shape[:-1], -1, block_size)
    amax = np.maximum(np.abs(grouped).max(-1, keepdims=True), 1e-4)
    scales = np.exp2(np.ceil(np.log2(amax / 448)))
    raw = np.arange(127)
    exp, mantissa = raw >> 3, raw & 7
    levels = np.where(
        exp == 0, mantissa * 2.0**-9, (1 + mantissa / 8) * np.exp2(exp - 7)
    )
    absolute = np.minimum(np.abs(grouped / scales), 448)
    hi = np.clip(np.searchsorted(levels, absolute), 0, 126)
    lo = np.maximum(hi - 1, 0)
    lower, upper = absolute - levels[lo], levels[hi] - absolute
    choose_hi = (upper < lower) | ((upper == lower) & ((hi & 1) == 0))
    codes = np.where(choose_hi, hi, lo)
    result = np.copysign(levels[codes], grouped) * scales
    return result.reshape(x.shape).astype(np.float32), np.broadcast_to(
        scales, grouped.shape
    ).reshape(x.shape)


def align_masked_indices(actual, reference, block_size=64):
    """Append only masked slots; preserve existing keys and block boundaries.

    Returned arrays must still be compared strictly. No live key is dropped,
    sorted, or repositioned, even if both rows attend to the same key set.
    """
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    arrays = [np.asarray(value) for value in (actual, reference)]
    for value in arrays:
        if value.ndim != 2 or not value.size or not np.isfinite(value).all():
            raise ValueError("expected finite nonempty two-dimensional indices")
        if (
            np.any(value != np.floor(value))
            or np.any(value < -1)
            or np.any(value > np.iinfo(np.int32).max)
        ):
            raise ValueError("indices must be int32-compatible keys or -1 masks")
    if arrays[0].shape[0] != arrays[1].shape[0]:
        raise ValueError("query counts differ")
    width = (
        (max(value.shape[1] for value in arrays) + block_size - 1)
        // block_size
        * block_size
    )
    return tuple(
        np.pad(
            value.astype(np.int32),
            ((0, 0), (0, width - value.shape[1])),
            constant_values=-1,
        )
        for value in arrays
    )
