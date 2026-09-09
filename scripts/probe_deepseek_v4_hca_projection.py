"""Audit existing HCA projection against V4's retained arithmetic, not a gate."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.deepseek_v4.compressor import _project
from sgl_jax.srt.kernels.hca.compressor import hca_project_fused_pallas
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.test_deepseek_v4_paged import make_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    layer = checkpoint.config["compress_ratios"].index(128)
    prefix = f"layers.{layer}.attn.compressor."
    a, b = [jnp.asarray(checkpoint.read_tensor(prefix + n + ".weight")) for n in ("wkv", "wgate")]
    ape = jnp.asarray(checkpoint.read_tensor(prefix + "ape"))
    weight = jnp.concatenate((a.T, b.T), axis=1)
    schedule = get_hca_kernel_schedule(
        jax.devices()[0].device_kind,
        page_size=128,
        max_compressed_entries=64,
        local_heads=64,
        head_dim=512,
    )
    rng = np.random.default_rng(8023)
    x = jnp.asarray(rng.normal(size=(128, a.shape[1])), jnp.bfloat16)
    backend = V4PagedBackend(max_context=384)
    report = {"layer": layer, "comparisons": []}
    for tokens in (1, 8, 128):
        meta = backend.get_forward_metadata(make_batch([0], [tokens], decode=tokens == 1))
        expected = jax.jit(lambda x, w, m: _project(x, w, m))(x[:tokens], a, meta)
        for tile_k in (512, a.shape[1]):
            actual = hca_project_fused_pallas(
                x[:tokens],
                weight,
                ape,
                jnp.arange(tokens),
                schedule=replace(schedule, projection_k_tile=tile_k, projection_batch_tile_max=8),
            )[:, 0]
            aa, bb = np.asarray(expected, np.float32), np.asarray(actual, np.float32)
            row = {
                "tokens": tokens,
                "tile_k": tile_k,
                "different": int(np.count_nonzero(aa != bb)),
                "max_abs": float(np.max(np.abs(aa - bb))),
                "nrmse": float(np.linalg.norm(aa - bb) / np.linalg.norm(aa)),
            }
            report["comparisons"].append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
