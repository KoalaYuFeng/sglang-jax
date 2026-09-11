"""No kernel dependencies: test normalization, alignment and descriptive metrics."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "v4_e2e_metrics",
    Path(__file__).resolve().parents[3] / "scripts/deepseek_v4_endtoend_metrics.py",
)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def test_identical_up_to_logit_offset():
    x = np.array([[1.0, 3.0, 2.0], [5.0, 2.0, -4.0]])
    result = metrics.compare_distributions(x, x + np.array([[100.0], [-200.0]]), [1])
    assert result["top1_equal_count"] == 2
    assert result["max_total_variation"] == 0
    assert result["mean_kl_cpu_to_tpu_nats"] == 0
    assert result["cpu_mean_teacher_nll"] == result["tpu_mean_teacher_nll"]
    assert not result["acceptance_gate_applied"]


def test_known_two_class_difference():
    result = metrics.compare_distributions(np.log([[0.8, 0.2]]), np.log([[0.2, 0.8]]), [0])
    assert result["top1_equal_count"] == 0
    assert result["mean_kl_cpu_to_tpu_nats"] == pytest.approx(0.6 * np.log(4))
    assert result["max_total_variation"] == pytest.approx(0.6)
    assert result["cpu_mean_teacher_nll"] == pytest.approx(-np.log(0.8))


@pytest.mark.parametrize("values", [[], [[np.nan, 1]], [[np.inf, 0]]])
def test_invalid_distribution(values):
    with pytest.raises(ValueError):
        metrics.log_softmax64(values)


@pytest.mark.parametrize("tokens", [[2], [-1], [0, 1], [1.5]])
def test_bad_teacher_alignment(tokens):
    with pytest.raises(ValueError):
        metrics.compare_distributions([[1, 2]], [[1, 2]], tokens)
