"""Frozen arithmetic counterexamples: exact summation cannot undo partial rounding.

These CPU controls test the diagnosed mechanism, not a TPU kernel or model gate.
"""

import importlib.util
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[3] / "scripts/deepseek_v4_bf16_reference.py"
SPEC = importlib.util.spec_from_file_location("bf16_reference", PATH)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


@pytest.mark.parametrize(
    "partials,expected_error",
    [
        (
            [
                -0.01461772620677948,
                0.0016828184016048908,
                -0.0036663711071014404,
                -0.005312085151672363,
                0.016643515788018703,
                0.0018952786922454834,
                0.0034380778670310974,
                -0.0002556741237640381,
            ],
            2.0**-30,
        ),
        (
            [
                0.008263534400612116,
                0.004294350743293762,
                -0.001291029155254364,
                0.02441573143005371,
                -0.028018057346343994,
                0.012808918952941895,
                -0.007136382162570953,
                -0.012625625357031822,
            ],
            -(2.0**-31),
        ),
    ],
)
def test_correct_partial_rounding_can_change_final_bf16(partials, expected_error):
    values = np.asarray(partials, np.float64)
    exact = sum((Fraction.from_float(float(v)) for v in values), Fraction())
    rounded = values.astype(np.float32)
    rounded_exact = sum((Fraction.from_float(float(v)) for v in rounded), Fraction())
    assert rounded_exact - exact == Fraction.from_float(expected_error)
    assert np.count_nonzero(values != rounded.astype(np.float64)) == 1
    assert Fraction.from_float(float(values.sum())) == exact
    assert Fraction.from_float(float(rounded.astype(np.float64).sum())) == rounded_exact
    # At both real coordinates even an exact reduction of correctly rounded
    # partials gives a different final BF16 value from the original products.
    assert reference.round_bf16_direct(float(exact)) != reference.round_bf16_direct(
        float(rounded_exact)
    )
