"""Adjudicate decode softmax/pooling arithmetic with frozen real inputs."""

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from replay_deepseek_v4_8023 import difference, logical_cache
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip, round_bf16
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedMetadata
from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    compress,
    config_for_layer,
    rms_norm,
    rope,
    source_fingerprint,
)
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--cpu-probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check-reference", action="store_true")
    args = p.parse_args()
    r = json.loads((args.replay / "report.json").read_text())
    if not r["faithful"]:
        raise ValueError("requires faithful layer replay")
    args.output.mkdir(parents=True, exist_ok=False)
    layer = r["first_hidden_difference"]
    cp = DeepSeekV4Checkpoint(r["checkpoint"])
    cfg = config_for_layer(cp.config, layer, 8192)
    meta = V4PagedMetadata(**load_arrays(args.replay / "metadata"))
    trace = load_arrays(args.replay / "first-native-trace")
    after = load_arrays(args.replay / "first-reference-after")
    native = logical_cache(
        load_arrays(args.replay / "first-native-after"), meta, 0, cfg, after=True
    )
    cpu = load_arrays(args.cpu_probe / "main-native")
    norm = cp.read_tensor(f"layers.{layer}.attn.compressor.norm.weight")
    values, scores = cpu["group_values"], cpu["group_scores"]
    group = np.flatnonzero(np.asarray(meta.group4_starts) >= 0)[0]
    rows = {}
    for mode in ("legacy", "divide_first", "divide_first_barrier", "vector"):

        @jax.jit
        def calculate(v, s, norm, pos, mode=mode):
            def emit(inputs):
                v, s, norm, pos = inputs
                if mode == "vector":
                    pooled = jnp.sum(v[None] * jax.nn.softmax(s[None], axis=1), axis=1)
                else:
                    maximum = functools.reduce(jnp.maximum, [s[i] for i in range(8)])
                    exponent = [jnp.exp(s[i] - maximum) for i in range(8)]
                    denominator = functools.reduce(jnp.add, exponent)
                    if mode == "legacy":
                        products = [v[i] * exponent[i] / denominator for i in range(8)]
                    else:
                        probs = [e / denominator for e in exponent]
                        if mode == "divide_first_barrier":
                            probs = jax.lax.optimization_barrier(probs)
                        products = [v[i] * probs[i] for i in range(8)]
                    pooled = functools.reduce(jnp.add, products)[None]
                pooled = round_bf16(pooled)
                normalized = rms_norm(pooled, norm, cfg.eps)
                rotated = rope(normalized, (pos + 1 - cfg.ratio)[None], cfg)
                final = jnp.concatenate(
                    (activation_fp8_roundtrip(rotated[:, :-64], 64), rotated[:, -64:]), axis=-1
                )
                return pooled, normalized, final

            return jax.lax.cond(
                pos % 4 == 3,
                emit,
                lambda _: tuple(jnp.zeros((1, 512), jnp.bfloat16) for _ in range(3)),
                (v, s, norm, pos),
            )

        pooled, normalized, final = (
            np.asarray(a)[0] for a in calculate(values, scores, norm, np.int32(r["position"]))
        )
        rows[mode] = {
            "pooled_vs_numpy64": difference(cpu["pooled64"], pooled),
            "pooled_vs_cpu32": difference(cpu["pooled"], pooled),
            "pooled_vs_native": difference(trace["compressor.main.pooled"][group], pooled),
            "normalized_vs_native": difference(
                trace["compressor.main.normalized"][group], normalized
            ),
            "final_vs_native": difference(native["main.compressed"][-1], final),
            "final_vs_reference": difference(after["main.compressed"][r["position"] // 4], final),
        }
        save_arrays(
            args.output / mode, {"pooled": pooled, "normalized": normalized, "final": final}
        )
        print(json.dumps({"mode": mode, **rows[mode]}), flush=True)
    if args.check_reference:
        before = load_arrays(args.replay / "first-reference-before")
        inputs = load_arrays(args.replay / "inputs")
        weights = {
            prefix + suffix: cp.read_tensor(f"layers.{layer}.{prefix}{suffix}")
            for prefix in ("attn.compressor", "attn.indexer.compressor")
            for suffix in (".wkv.weight", ".wgate.weight", ".ape", ".norm.weight")
        }

        @jax.jit
        def current(x, positions, weights, cache):
            cache = compress(x, positions, weights, dict(cache), cfg)
            return compress(x, positions, weights, cache, cfg, index=True)

        mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1], object), ("tensor",))
        with jax.set_mesh(mesh):
            current_cache = jax.device_get(
                current(trace["attn.norm"], inputs["positions"], weights, before)
            )
        checks = {}
        for prefix in ("main", "index"):
            for suffix in (".kv", ".score", ".compressed"):
                name = prefix + suffix
                expected, actual = native[name], current_cache[name]
                if suffix == ".compressed":
                    actual = actual[: len(expected)]
                checks[name] = difference(expected, actual)
        rows["current_reference"] = {
            "reference_source_fingerprint": source_fingerprint(),
            "checks": checks,
            "passed": all(item["bitwise_equal"] for item in checks.values()),
        }
        save_arrays(args.output / "current-reference", current_cache)
        print(json.dumps({"current_reference": rows["current_reference"]}), flush=True)
    (args.output / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
    if args.check_reference and not rows["current_reference"]["passed"]:
        raise AssertionError("current reference must match the frozen real native compressor state")


if __name__ == "__main__":
    main()
