"""Analytical checks for the independent high-precision diagnostic."""

import importlib.util
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[3] / "scripts/deepseek_v4_attention_fp64.py"
SPEC = importlib.util.spec_from_file_location("attention_fp64", PATH)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


@pytest.mark.parametrize("slots", [1, 63, 64, 65, 127, 128, 129, 256])
def test_uniform_attention_and_sink(slots):
    q = np.zeros((2, 3, 128))
    kv = np.ones((slots, 128))
    ids = np.broadcast_to(np.arange(slots), (2, slots))
    result = reference.block_attention_fp64(q, kv, ids, np.zeros(3), 128**-0.5)
    expected = np.asarray(slots / (slots + 1), ml_dtypes.bfloat16).astype(np.float32)
    np.testing.assert_array_equal(result, np.full(q.shape, expected, np.float32))


def test_all_masked_exact_zero():
    result = reference.block_attention_fp64(
        np.ones((1, 2, 128)), np.ones((1, 128)), np.full((1, 129), -1), np.array([-80, 80]), 1.0
    )
    np.testing.assert_array_equal(result, np.zeros_like(result))


@pytest.mark.parametrize("key", [-2, 1, 1.5, np.nan])
def test_bad_key_rejected(key):
    with pytest.raises(ValueError):
        reference.block_attention_fp64(
            np.ones((1, 2, 128)), np.ones((1, 128)), [[key]], np.zeros(2), 1.0
        )
