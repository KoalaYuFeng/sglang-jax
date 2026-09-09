"""Framework-free V4 compressor inputs and independent NumPy oracle.

This module deliberately does not import a scheduler, model, or Pallas math.
Both implementations consume the same FP32 phase tables; phase generation is
outside the emitter's contract. Historical arrays are read without importing
the model capture/replay drivers.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np


@dataclass(frozen=True)
class CompressorCase:
    name: str
    values: np.ndarray
    scores: np.ndarray
    norm: np.ndarray
    phase: np.ndarray
    valid: np.ndarray

    @property
    def dim(self):
        return self.norm.size

    def inputs(self):
        return self.values, self.scores, self.norm, self.phase, self.valid

    def digest(self):
        digest = hashlib.sha256()
        for value in self.inputs():
            digest.update(str((value.shape, str(value.dtype))).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


def select_channels(value):
    dim = value.shape[-1] // 2
    # NumPy, not the production JAX gather or Pallas selection implementation.
    return np.concatenate((value[:, :4, :dim], value[:, 4:, dim:]), axis=1)


def fp64_oracle(case, eps=1e-6):
    values = select_channels(case.values).astype(np.float64)
    scores = select_channels(case.scores).astype(np.float64)
    values = np.where(case.valid[:, None, None], values, 0)
    scores = np.where(case.valid[:, None, None], scores, 0)
    probability = np.exp(scores - scores.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    pooled = np.sum(values * probability, axis=1).astype(ml_dtypes.bfloat16).astype(np.float64)
    normalized = pooled / np.sqrt(np.mean(pooled * pooled, axis=-1, keepdims=True) + eps)
    normalized = (normalized * case.norm.astype(np.float64)).astype(ml_dtypes.bfloat16)
    result = normalized.astype(np.float64)
    cosine, sine = np.split(case.phase.astype(np.float64), 2, axis=-1)
    real, imag = result[:, -64::2].copy(), result[:, -63::2].copy()
    result[:, -64::2] = real * cosine - imag * sine
    result[:, -63::2] = real * sine + imag * cosine
    return np.where(case.valid[:, None], result, 0).astype(ml_dtypes.bfloat16)


def torch_fp32_oracle(case, eps=1e-6):
    """Official typed primitive sequence, independently executed on the CPU."""
    import torch

    torch.set_num_threads(4)
    values = np.where(case.valid[:, None, None], select_channels(case.values), 0)
    scores = np.where(case.valid[:, None, None], select_channels(case.scores), 0)
    v, s = torch.from_numpy(values), torch.from_numpy(scores)
    pooled = (v * torch.softmax(s, dim=1)).sum(dim=1).to(torch.bfloat16).float()
    variance = pooled.square().mean(-1, keepdim=True)
    norm = torch.from_numpy(case.norm.astype(np.float32))
    normalized = (pooled * torch.rsqrt(variance + eps) * norm).to(torch.bfloat16).float()
    phase = torch.from_numpy(case.phase)
    real, imag = normalized[:, -64::2], normalized[:, -63::2]
    rotated = torch.stack(
        (real * phase[:, :32] - imag * phase[:, 32:], real * phase[:, 32:] + imag * phase[:, :32]),
        dim=-1,
    )
    output = torch.cat((normalized[:, :-64], rotated.flatten(-2)), dim=-1).to(torch.bfloat16)
    return np.where(case.valid[:, None], output.float().numpy(), np.float32(0)).astype(
        ml_dtypes.bfloat16
    )


def oracle_check(case, actual):
    expected = fp64_oracle(case).astype(np.float64)
    error = float(
        np.linalg.norm(actual.astype(np.float64) - expected) / max(np.linalg.norm(expected), 1e-12)
    )
    check = {
        "all_finite": bool(np.all(np.isfinite(actual))),
        "invalid_zero": bool(np.all(actual[~case.valid] == 0)),
        "fp64_nrmse": error,
        "fp64_tolerance": 2e-4,
        "fp64_within_tolerance": error < 2e-4,
        "oracle": "fp64",
        "oracle_passed": error < 2e-4,
    }
    if case.name == "random-d128-g9-s7979":
        # An explicitly adjudicated rounding-boundary regression, NOT an
        # automatic alternate oracle for arbitrary failing cases. CPU FP32
        # and both TPU implementations agree, while FP64 crosses a midpoint.
        if case.digest() != "a3fdc3711a2f56a4ecfd56127501ddffe6cc7247a74da7268ff92e56a5d0fd5f":
            raise ValueError("unreviewed change to frozen FP32 rounding-boundary fixture")
        independent = torch_fp32_oracle(case)
        check["oracle"] = "frozen_boundary_official_cpu_fp32_bitwise"
        check["oracle_passed"] = bool(
            np.array_equal(actual.view(np.uint16), independent.view(np.uint16))
        )
    check["passed"] = check["all_finite"] and check["invalid_zero"] and check["oracle_passed"]
    return check


def make_case(dim, groups, *, seed=7979):
    rng = np.random.default_rng(seed + dim + groups)
    values = rng.normal(size=(groups, 8, 2 * dim)).astype(np.float32)
    scores = rng.normal(size=values.shape).astype(np.float32)
    # Distinct halves expose accidental selection of the same half twice.
    values[:, :, :dim] -= np.float32(0.5)
    values[:, :, dim:] += np.float32(0.75)
    scores[0, :4] = -np.inf  # no preceding window for a new request
    valid = np.ones(groups, np.bool_)
    if groups > 1:
        valid[-1] = False
        # Invalid padded rows must not leak into valid groups or produce NaNs.
        scores[-1] = -np.inf
        values[-1] = np.nan
    norm = rng.normal(1, 0.1, (dim,)).astype(ml_dtypes.bfloat16)
    angles = rng.uniform(-np.pi, np.pi, (groups, 32)).astype(np.float32)
    phase = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
    return CompressorCase(f"random-d{dim}-g{groups}-s{seed}", values, scores, norm, phase, valid)


def read_arrays(prefix, names):
    prefix = Path(prefix)
    schema = json.loads(prefix.with_suffix(".json").read_text())
    result = {}
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
        for name in names:
            value = saved[name]
            if schema[name]["dtype"] == "bfloat16":
                value = value.view(ml_dtypes.bfloat16)
            if (
                list(value.shape) != schema[name]["shape"]
                or str(value.dtype) != schema[name]["dtype"]
            ):
                raise ValueError(f"invalid captured array: {prefix}/{name}")
            result[name] = value
    return result


def real_cases(root):
    root = Path(root)
    replay = root / "v4-original-csa-7979-replay-20260908-01"
    report = json.loads((replay / "report.json").read_text())
    if not report["complete"] or not report["faithful"]:
        raise ValueError("real fixtures require a faithful completed historical replay")
    fixture = read_arrays(
        root / "v4-original-csa-compressor-cross-20260908-01/fixture",
        ("main.norm", "index.norm", "starts", "valid"),
    )
    # Deliberately avoid fixture's *.scores: its old exporter reused those keys
    # for projection scores. Full, correctly shaped windows live in the traces.
    for mode in ("reference", "pallas"):
        trace_path = replay / f"{mode}-7979-layer-10-trace"
        for prefix in ("main", "index"):
            names = [
                f"compressor.{prefix}.{suffix}"
                for suffix in (
                    "raw_group_values",
                    "raw_group_scores",
                    "group_values",
                    "group_scores",
                )
            ]
            trace = read_arrays(trace_path, names)
            values, scores, selected_values, selected_scores = (trace[n] for n in names)
            np.testing.assert_array_equal(select_channels(values), selected_values)
            # Invalid group scores were deliberately zeroed by the old adapter.
            valid = fixture["valid"]
            np.testing.assert_array_equal(select_channels(scores)[valid], selected_scores[valid])
            # Independently construct the pinned Flash checkpoint's FP32 YaRN
            # table from real starts. The phase table is an explicit kernel
            # input; this is not a claim of CPU/GPU transcendental bit equality.
            base, original, factor = 160000, 65536, 16
            low = max(
                math.floor(64 * math.log(original / (32 * 2 * math.pi)) / (2 * math.log(base))),
                0,
            )
            high = min(
                math.ceil(64 * math.log(original / (2 * math.pi)) / (2 * math.log(base))),
                63,
            )
            freq = np.float32(1) / np.power(
                np.float32(base), np.arange(0, 64, 2, dtype=np.float32) / np.float32(64)
            )
            ramp = np.clip((np.arange(32, dtype=np.float32) - low) / np.float32(high - low), 0, 1)
            freq = freq / np.float32(factor) * ramp + freq * (np.float32(1) - ramp)
            angles = np.maximum(fixture["starts"], 0).astype(np.float32)[:, None] * freq[None]
            phase = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
            yield CompressorCase(
                f"real-7979-{mode}-{prefix}",
                values,
                scores,
                fixture[f"{prefix}.norm"],
                phase,
                valid,
            )
