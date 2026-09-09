"""Find a batch-invariant projection matching frozen single-request arithmetic.

All variants are isolated diagnostics. The captured reference is immutable;
no model/weight format or reference answer is rewritten by this probe.
"""

import argparse
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from replay_deepseek_v4_8023 import difference
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer, _fixed_tree_sum_last
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedMetadata
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint

MODULE = importlib.import_module("sgl_jax.srt.kernels.deepseek_v4.compressor")


def projector(mode):
    def project(x, weight, metadata):
        if mode == "legacy":
            return jnp.matmul(x, weight.T, preferred_element_type=jnp.float32)
        rows = metadata.router_rows
        blocks = jnp.where((rows >= 0)[..., None], x[jnp.maximum(rows, 0)], 0)

        def vector(row):
            if mode in ("gemv8", "hybrid8"):
                return jnp.matmul(row[None], weight.T, preferred_element_type=jnp.float32)[0]
            if mode == "gemv8_barrier":
                a, b = jax.lax.optimization_barrier((row[None], weight.T))
                return jnp.matmul(a, b, preferred_element_type=jnp.float32)[0]
            if mode == "vector8":
                return jax.lax.dot_general(
                    row, weight, (((0,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )
            if mode == "reduce8":
                products = row.astype(jnp.float32)[None] * weight.astype(jnp.float32)
                return jnp.sum(jax.lax.optimization_barrier(products), axis=1)
            if mode == "tree8":
                products = row.astype(jnp.float32)[None] * weight.astype(jnp.float32)
                return _fixed_tree_sum_last(jax.lax.optimization_barrier(products))
            raise ValueError(mode)

        def block(value):
            if mode in ("matrix8", "highest8", "f32_highest8"):
                if mode == "f32_highest8":
                    value, right = value.astype(jnp.float32), weight.T.astype(jnp.float32)
                else:
                    right = weight.T
                return jnp.matmul(
                    value,
                    right,
                    preferred_element_type=jnp.float32,
                    precision=jax.lax.Precision.DEFAULT
                    if mode == "matrix8"
                    else jax.lax.Precision.HIGHEST,
                )
            return jax.lax.map(vector, value)

        if mode == "hybrid8":
            requests = metadata.token_requests[jnp.maximum(rows, 0)]
            single = jnp.any((rows >= 0) & (metadata.query_lens[requests] == 1), axis=1)
            result = jax.lax.map(
                lambda item: jax.lax.cond(
                    item[1],
                    lambda value: jax.lax.map(vector, value),
                    lambda value: jnp.matmul(value, weight.T, preferred_element_type=jnp.float32),
                    item[0],
                ),
                (blocks, single),
            )
        else:
            result = jax.lax.map(block, blocks)
        return result.reshape(-1, weight.shape[0])[metadata.router_output_rows]

    return project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.capture / "report.json").read_text())
    checkpoint = DeepSeekV4Checkpoint(source["checkpoint"])
    config = config_for_layer(checkpoint.config, 6, 8192)
    prefix = "attn.compressor."
    weights = {
        prefix + name: jnp.asarray(checkpoint.read_tensor(f"layers.6.{prefix}{name}"))
        for name in ("wkv.weight", "wgate.weight", "ape", "norm.weight")
    }
    frames = {}
    for name in ("B1-case1", "B2"):
        path = args.capture / name
        trace = load_arrays(args.replay / f"{name}-layer-06-trace")
        frames[name] = {
            "x": trace["attn.norm"],
            "meta": V4PagedMetadata(**load_arrays(path / "metadata")),
            "cache": load_arrays(path / "before" / "layer-06"),
            "expected": trace,
        }
    a, b = frames.values()
    np.testing.assert_array_equal(a["x"][0], b["x"][0])
    for key in ("main.kv", "main.score"):
        b["cache"][key] = np.array(b["cache"][key], copy=True)
        b["cache"][key][b["meta"].req_slots[0]] = a["cache"][key][a["meta"].req_slots[0]]
    report = {}
    for mode in args.modes or (
        "legacy",
        "matrix8",
        "gemv8",
        "gemv8_barrier",
        "vector8",
        "highest8",
        "f32_highest8",
        "reduce8",
        "tree8",
        "hybrid8",
    ):
        print(json.dumps({"event": "mode_start", "mode": mode}), flush=True)

        @jax.jit
        def run(x, weights, cache, meta, _mode=mode):
            trace = {}
            with patch.object(MODULE, "_project", projector(_mode)):
                cache = MODULE.compress(x, weights, dict(cache), config, meta, trace=trace)
            return cache, trace

        outputs = []
        for name, frame in frames.items():
            cache, trace = run(frame["x"], weights, frame["cache"], frame["meta"])
            trace = jax.device_get(trace)
            save_arrays(args.output / f"{mode}-{name}", trace)
            outputs.append(trace)
        report[mode] = {
            "batch_invariance": {
                key: difference(outputs[0][key][0], outputs[1][key][0])
                for key in ("kv", "scores", "pooled", "normalized", "final")
            },
            "original_B1": {
                key: difference(a["expected"]["compressor.main." + key][0], outputs[0][key][0])
                for key in ("kv", "scores", "pooled", "normalized", "final")
            },
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": "mode_complete", "mode": mode, **report[mode]}), flush=True)
        del cache, trace, outputs


if __name__ == "__main__":
    main()
