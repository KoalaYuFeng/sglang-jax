"""Exact midpoint checks for the diagnostic BF16 reference."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[3] / "scripts/deepseek_v4_bf16_reference.py"
SPEC = importlib.util.spec_from_file_location("bf16_reference", PATH)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


@pytest.mark.parametrize("sign", [-1, 1])
def test_every_positive_finite_midpoint(sign):
    bits = np.arange(0x7F7F, dtype=np.uint32)
    low = (bits << 16).view(np.float32).astype(np.float64)
    high = ((bits + 1) << 16).view(np.float32).astype(np.float64)
    midpoint = (low + high) / 2
    for values, expected in (
        (np.nextafter(midpoint, low), low),
        (midpoint, np.where(bits & 1, high, low)),
        (np.nextafter(midpoint, high), high),
    ):
        np.testing.assert_array_equal(reference.round_bf16_direct(sign * values), sign * expected)


def test_zero_sign_and_representable_values():
    bits = np.arange(0x7F80, dtype=np.uint32)
    values = (bits << 16).view(np.float32)
    for sign in (-1, 1):
        actual = reference.round_bf16_direct(sign * values)
        np.testing.assert_array_equal(actual.view(np.uint32), (sign * values).view(np.uint32))


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan, 2.0**128])
def test_invalid_input(value):
    with pytest.raises(ValueError):
        reference.round_bf16_direct(value)
