"""CPU controls for diagnostic-only two-component BF16 rounding."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


candidate = load("deepseek_v4_fp8_residual_candidate")
reference = load("deepseek_v4_bf16_reference")


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_normal_midpoints_both_directions(sign, delta):
    # Normal BF16 intervals in the projection-relevant exponent range.
    bits = np.arange(0x3000, 0x4800, dtype=np.uint32)
    midpoint = ((bits << 16) | 32768).view(np.float32) * sign
    low = np.spacing(np.abs(midpoint)) * np.float32(delta / 4)
    value = jax.jit(candidate.bf16_rounding_carrier)(jnp.asarray(midpoint), jnp.asarray(low))
    actual = np.asarray(value.astype(jnp.bfloat16), np.float32)
    expected = reference.round_bf16_direct(midpoint.astype(np.float64) + low.astype(np.float64))
    np.testing.assert_array_equal(actual, expected)


def test_random_finite_pairs():
    rng = np.random.default_rng(20260911)
    high = rng.normal(size=16384).astype(np.float32)
    low = (rng.normal(size=high.shape) * np.spacing(np.abs(high))).astype(np.float32)
    output = jax.jit(candidate.bf16_rounding_carrier)(jnp.asarray(high), jnp.asarray(low))
    expected = reference.round_bf16_direct(high.astype(np.float64) + low.astype(np.float64))
    np.testing.assert_array_equal(np.asarray(output.astype(jnp.bfloat16), np.float32), expected)
