"""Isolate one emitter/oracle discrepancy without model/framework imports."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

from sgl_jax.srt.kernels.csa.compressor import csa_emit_selected_pallas
from sgl_jax.srt.kernels.low_bit.formats import round_bf16
from sgl_jax.test.kernels.csa_compressor_cases import (
    fp64_oracle,
    make_case,
    select_channels,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case = make_case(128, 9)
    values = np.where(case.valid[:, None, None], select_channels(case.values), 0)
    scores = np.where(case.valid[:, None, None], select_channels(case.scores), 0)
    v, s, n, p = (jnp.asarray(a) for a in (values, scores, case.norm, case.phase))

    @jax.jit
    def retained(v, s, n, p):
        probability = jax.nn.softmax(s, axis=1)
        unrounded = jnp.sum(v * probability, axis=1)
        pooled = round_bf16(unrounded).astype(jnp.float32)
        squared = pooled * pooled
        while squared.shape[-1] > 1:
            squared = squared[..., 0::2] + squared[..., 1::2]
        variance = squared / jnp.float32(pooled.shape[-1])
        normalized = round_bf16(pooled * jax.lax.rsqrt(variance + 1e-6) * n.astype(jnp.float32))
        normalized = normalized.astype(jnp.float32)
        real, imag = normalized[:, -64::2], normalized[:, -63::2]
        rotated = jnp.stack(
            (real * p[:, :32] - imag * p[:, 32:], real * p[:, 32:] + imag * p[:, :32]),
            axis=-1,
        )
        result = jnp.concatenate((normalized[:, :-64], rotated.reshape(-1, 64)), axis=-1)
        return probability, unrounded, pooled, variance, normalized, round_bf16(result)

    labels = ("probability", "unrounded", "pooled", "variance", "normalized", "result")
    output = {"jax." + k: np.asarray(a) for k, a in zip(labels, retained(v, s, n, p), strict=True)}
    output["pallas.result"] = np.asarray(
        csa_emit_selected_pallas(v, s, n, p, jnp.asarray(case.valid))
    )
    output["fp64.result"] = fp64_oracle(case)
    # Official CPU primitive sequence, independently executed with PyTorch.
    torch.set_num_threads(4)
    tv, ts = torch.from_numpy(values), torch.from_numpy(scores)
    prob = torch.softmax(ts, dim=1)
    unrounded = (tv * prob).sum(dim=1)
    pooled = unrounded.to(torch.bfloat16).float()
    variance = pooled.square().mean(-1, keepdim=True)
    normalized = (
        (pooled * torch.rsqrt(variance + 1e-6) * torch.from_numpy(case.norm.astype(np.float32)))
        .to(torch.bfloat16)
        .float()
    )
    tp = torch.from_numpy(case.phase)
    a, b = normalized[:, -64::2], normalized[:, -63::2]
    rotated = torch.stack(
        (a * tp[:, :32] - b * tp[:, 32:], a * tp[:, 32:] + b * tp[:, :32]), dim=-1
    )
    result = (
        torch.cat((normalized[:, :-64], rotated.flatten(-2)), dim=-1).to(torch.bfloat16).float()
    )
    for k, a in zip(labels, (prob, unrounded, pooled, variance, normalized, result), strict=True):
        output["torch." + k] = a.numpy()
    dv, ds = values.astype(np.float64), scores.astype(np.float64)
    dp = np.exp(ds - ds.max(axis=1, keepdims=True))
    dp /= dp.sum(axis=1, keepdims=True)
    raw = np.sum(dv * dp, axis=1)
    pool = raw.astype(ml_dtypes.bfloat16).astype(np.float64)
    var = np.mean(pool * pool, axis=-1, keepdims=True)
    norm = (
        (pool / np.sqrt(var + 1e-6) * case.norm.astype(np.float64))
        .astype(ml_dtypes.bfloat16)
        .astype(np.float64)
    )
    for k, a in zip(labels[:-1], (dp, raw, pool, var, norm), strict=True):
        output["fp64." + k] = a
    comparisons = []
    for left, right in (
        ("pallas", "jax"),
        ("pallas", "torch"),
        ("jax", "fp64"),
        ("torch", "fp64"),
    ):
        for stage in labels:
            if left + "." + stage not in output:
                continue
            a, b = (
                output[prefix + "." + stage].astype(np.float64)[case.valid]
                for prefix in (left, right)
            )
            where = np.argwhere(a != b)
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "stage": stage,
                    "different": len(where),
                    "nrmse": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
                    "first": [
                        {
                            "index": row.tolist(),
                            "left": float(a[tuple(row)]),
                            "right": float(b[tuple(row)]),
                        }
                        for row in where[:8]
                    ],
                }
            )
    report = {"input_sha256": case.digest(), "comparisons": comparisons}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.output / "stages.npz", **{k: v.astype(np.float64) for k, v in output.items()})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
