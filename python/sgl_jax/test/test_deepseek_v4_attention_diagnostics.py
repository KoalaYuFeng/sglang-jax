"""Padding normalization cannot hide live-key, mask, or block-order errors."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[3] / "scripts/deepseek_v4_attention_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("attention_diagnostics", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_trailing_masked_slot_is_the_only_difference_allowed():
    a = np.arange(127)[None]
    b = np.pad(a, ((0, 0), (0, 1)), constant_values=-1)
    aa, bb = module.align_masked_indices(a, b)
    np.testing.assert_array_equal(aa, bb)
    assert aa.shape == (1, 128)


@pytest.mark.parametrize("case", ["live_tail", "order", "mask"])
def test_real_key_differences_stay_visible(case):
    a = np.arange(127)[None]
    b = np.pad(a, ((0, 0), (0, 1)), constant_values=-1)
    if case == "live_tail":
        b[0, -1] = 127
    elif case == "order":
        b[0, 63], b[0, 64] = b[0, 64], b[0, 63]
    else:
        b[0, 5] = -1
    aa, bb = module.align_masked_indices(a, b)
    assert not np.array_equal(aa, bb)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -2, 1.5, 2**32])
def test_invalid_keys_rejected(bad):
    with pytest.raises(ValueError):
        module.align_masked_indices([[bad]], [[0]])


def test_different_query_counts_rejected():
    with pytest.raises(ValueError):
        module.align_masked_indices([[0]], [[0], [0]])


def test_empty_rejected():
    with pytest.raises(ValueError):
        module.align_masked_indices([[]], [[]])


def test_fp8_even_ties_and_signed_zero():
    values = np.zeros((1, 128), np.float32)
    values[0, :6] = [448, 1.0625, 1.1875, -1.0625, -0.0, 0.0009765625]
    actual, scales = module.fp8_roundtrip_cpu(values)
    np.testing.assert_array_equal(actual[0, :6], [448, 1, 1.25, -1, -0.0, 0.0])
    assert np.signbit(actual[0, 4])
    np.testing.assert_array_equal(scales, np.ones_like(scales))


def test_fp8_against_independent_torch_conversion():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(933).normal(size=(8, 128)).astype(np.float32)
    result, scales = module.fp8_roundtrip_cpu(x)
    reference = (
        torch.from_numpy((x / scales).astype(np.float32)).to(torch.float8_e4m3fn).float().numpy()
        * scales
    )
    np.testing.assert_array_equal(result, reference.astype(np.float32))


@pytest.mark.parametrize("value", [np.full((1, 128), np.nan), np.ones((1, 127)), np.ones((0, 128))])
def test_invalid_fp8_input_rejected(value):
    with pytest.raises(ValueError):
        module.fp8_roundtrip_cpu(value)
