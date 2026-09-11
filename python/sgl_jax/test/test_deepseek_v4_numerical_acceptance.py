"""Prevent quantization-boundary attribution from hiding numerical failures."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[3] / "scripts/deepseek_v4_numerical_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("v4_acceptance", _PATH)
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)


def fixture():
    reference = np.full((1, 32), 2, np.float32)
    reference[0, 0] = 0.625
    actual = reference.copy()
    actual[0, 0] = 0.62890625
    a, _ = analysis._fp4_cpu(actual, 32)
    b, _ = analysis._fp4_cpu(reference, 32)
    return actual, reference, a, b


def test_known_tie_boundary_is_attributed_not_passed():
    result = analysis.fp4_boundary_report(*fixture())
    assert result["actual_quantizer_mismatches"] == result["reference_quantizer_mismatches"] == 0
    assert result["changed_values"] == result["same_scale_adjacent_boundary_changes"] == 1
    assert result["same_scale_unexplained_changes"] == result["changed_scale_blocks"] == 0
    assert result["classification_only"] and "passed" not in result


def test_wrong_quantized_output_cannot_be_excused_by_a_boundary():
    values = list(fixture())
    values[2][0, 0] = 1.0
    result = analysis.fp4_boundary_report(*values)
    assert result["actual_quantizer_mismatches"] == 1
    assert result["same_scale_unexplained_changes"] == 1


def test_large_upstream_error_stays_visible():
    a, b, _, qb = fixture()
    a[0, 0] = 1.5
    qa, _ = analysis._fp4_cpu(a, 32)
    result = analysis.fp4_boundary_report(a, b, qa, qb)
    assert result["prequant"]["max_abs"] == 0.875
    assert result["same_scale_unexplained_changes"] == 1
    assert "passed" not in result


def test_scale_changes_are_separate():
    a, b, _, qb = fixture()
    a[0, 1] = 7
    qa, _ = analysis._fp4_cpu(a, 32)
    result = analysis.fp4_boundary_report(a, b, qa, qb)
    assert result["changed_scale_blocks"] == 1
    assert result["same_scale_adjacent_boundary_changes"] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_rejected(bad):
    a, b, qa, qb = fixture()
    a[0, 0] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        analysis.fp4_boundary_report(a, b, qa, qb)


@pytest.mark.parametrize("case", ["shape", "empty", "block", "post_shape"])
def test_invalid_shapes_rejected(case):
    a, b, qa, qb = fixture()
    with pytest.raises(ValueError):
        if case == "shape":
            analysis.fp4_boundary_report(a, b[:, :16], qa, qb)
        elif case == "empty":
            analysis.fp4_boundary_report(a[:0], b[:0], qa[:0], qb[:0])
        elif case == "post_shape":
            analysis.fp4_boundary_report(a, b, qa[:, :16], qb[:, :16])
        else:
            analysis.fp4_boundary_report(a, b, qa, qb, block_size=3)


def test_signed_zero_and_ties():
    value = np.zeros((1, 32), np.float32)
    value[0, :5] = [-0.0, 0.625, -0.625, 0.875, 2]
    quantized, _ = analysis._fp4_cpu(value, 32)
    np.testing.assert_array_equal(quantized[0, :5], [-0.0, 0.5, -0.5, 1, 2])
    assert np.signbit(quantized[0, 0])
