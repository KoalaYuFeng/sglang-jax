"""Test-only RoPE oracle with the independent official PyTorch FP32 recipe.

The historical single-request JAX reference intentionally remains frozen.
Paging/layout comparisons use this oracle to align the coefficient contract,
not the production NumPy coefficient builder. Runtime imports must not use it.
"""

import functools
import math

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.low_bit.formats import round_bf16


@functools.lru_cache(maxsize=32)
def official_frequencies(config):
    import torch

    rd = config.rope_dim
    frequency = 1 / (config.rope_base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    if config.original_seq_len:

        def correction(rotations):
            return (
                rd
                * math.log(config.original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(config.rope_base))
            )

        low = max(math.floor(correction(config.beta_fast)), 0)
        high = min(math.ceil(correction(config.beta_slow)), rd - 1)
        ramp = (
            (torch.arange(rd // 2, dtype=torch.float32) - low)
            / (high - low if high != low else 0.001)
        ).clamp(0, 1)
        smooth = 1 - ramp
        frequency = frequency / config.rope_factor * (1 - smooth) + frequency * smooth
    result = np.asarray(frequency.numpy()).copy()
    result.flags.writeable = False
    return result


def official_recipe_rope(x, positions, config, *, inverse=False):
    rd = config.rope_dim
    phase = positions.astype(jnp.float32)[:, None] * jnp.asarray(official_frequencies(config))
    phase = phase.reshape((positions.shape[0],) + (1,) * (x.ndim - 2) + (rd // 2,))
    cosine, sine = jnp.cos(phase), jnp.sin(phase) * (-1 if inverse else 1)
    paired = x[..., -rd:].astype(jnp.float32).reshape(*x.shape[:-1], rd // 2, 2)
    a, b = paired[..., 0], paired[..., 1]
    rotated = jnp.stack((a * cosine - b * sine, a * sine + b * cosine), axis=-1)
    return jnp.concatenate((x[..., :-rd], round_bf16(rotated.reshape(*x.shape[:-1], rd))), axis=-1)
