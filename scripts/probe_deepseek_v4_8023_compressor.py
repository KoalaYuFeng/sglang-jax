"""Causal state/projection swaps at the first divergent real CSA layer.

Diagnostic monkeypatches affect only tracing this isolated operation. No
production file, checkpoint, or captured state is modified. The original
isolated result must first match the captured production compressed KV.
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
from replay_deepseek_v4_8023 import difference, logical_cache
from run_deepseek_v4_framework import framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedMetadata
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint

COMPRESSOR = importlib.import_module("sgl_jax.srt.kernels.deepseek_v4.compressor")
MATMUL = jnp.matmul


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--verify-fix", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.capture / "report.json").read_text())
    checkpoint = DeepSeekV4Checkpoint(report["checkpoint"])
    config = config_for_layer(checkpoint.config, args.layer, 8192)
    prefix = "attn.compressor."
    weights = {
        prefix + name: jnp.asarray(checkpoint.read_tensor(f"layers.{args.layer}.{prefix}{name}"))
        for name in ("wkv.weight", "wgate.weight", "ape", "norm.weight")
    }
    frames = {}
    for name in ("B1-case1", "B2"):
        path = args.capture / name
        meta = V4PagedMetadata(**load_arrays(path / "metadata"))
        x = load_arrays(args.replay / f"{name}-layer-{args.layer:02d}-trace")["attn.norm"]
        before = load_arrays(path / "before" / f"layer-{args.layer:02d}")
        after = load_arrays(path / "after" / f"layer-{args.layer:02d}")
        frames[name] = {"x": x, "meta": meta, "before": before, "after": after}
    np.testing.assert_array_equal(frames["B1-case1"]["x"][0], frames["B2"]["x"][0])

    def build(mode="original"):
        @jax.jit
        def run(x, weights, cache, meta, override):
            trace = {}
            calls = 0

            def matmul(a, b, **kw):
                nonlocal calls
                if mode == "row1":
                    result = jax.lax.map(lambda row: MATMUL(row[None], b, **kw)[0], a)
                elif mode == "row8":
                    blocks = jnp.where(
                        (meta.router_rows >= 0)[..., None], a[jnp.maximum(meta.router_rows, 0)], 0
                    )
                    result = jax.lax.map(lambda block: MATMUL(block, b, **kw), blocks)
                    result = result.reshape(-1, b.shape[-1])[meta.router_output_rows]
                else:
                    result = MATMUL(a, b, **kw)
                if mode == "override":
                    result = result.at[0].set(override[calls])
                calls += 1
                return result

            with patch.object(COMPRESSOR.jnp, "matmul", matmul):
                updated = COMPRESSOR.compress(x, weights, dict(cache), config, meta, trace=trace)
            return updated, trace

        return run

    comparisons, outputs = {}, {}
    original = build()
    if args.verify_fix:
        # Old B2 scratch was produced by the unfixed kernel. Start both calls
        # with the same logical historical state; the full-model regression
        # separately recomputes the entire history with the corrected kernel.
        left, right = frames["B1-case1"], frames["B2"]
        for key in ("main.kv", "main.score"):
            right["before"][key] = np.array(right["before"][key], copy=True)
            right["before"][key][right["meta"].req_slots[0]] = left["before"][key][
                left["meta"].req_slots[0]
            ]
        caches = []
        for name, frame in frames.items():
            updated, trace = jax.device_get(
                original(frame["x"], weights, frame["before"], frame["meta"], ())
            )
            caches.append(logical_cache(updated, frame["meta"], 0, config, after=True))
            save_arrays(args.output / f"{name}-trace", trace)
        checks = {
            key: difference(caches[0][key], caches[1][key])
            for key in caches[0]
            if key.startswith("main.")
        }
        fixed_report = {
            "framework_source_fingerprint": framework_fingerprint(),
            "capture": str(args.capture),
            "scope": "same real input and historical state; B1 versus B2 at layer 6/position 8023",
            "checks": checks,
            "complete": all(c["bitwise_equal"] for c in checks.values()),
        }
        (args.output / "report.json").write_text(json.dumps(fixed_report, indent=2) + "\n")
        print(json.dumps(fixed_report), flush=True)
        if not fixed_report["complete"]:
            raise AssertionError("fixed real-state compressor is not batch invariant")
        return

    for name, frame in frames.items():
        updated, trace = jax.device_get(
            original(frame["x"], weights, frame["before"], frame["meta"], ())
        )
        expected = logical_cache(frame["after"], frame["meta"], 0, config, after=True)
        actual = logical_cache(updated, frame["meta"], 0, config, after=True)
        fidelity = difference(expected["main.compressed"], actual["main.compressed"])
        comparisons[name] = {"production_compressed_fidelity": fidelity}
        if not fidelity["bitwise_equal"]:
            raise AssertionError(f"isolated compressor does not reproduce production: {name}")
        outputs[name] = (updated, trace)
        save_arrays(args.output / f"{name}-trace", trace)

    left, right = frames["B1-case1"], frames["B2"]
    override = tuple(outputs["B1-case1"][1][key][0] for key in ("kv", "scores"))
    expected = logical_cache(outputs["B1-case1"][0], left["meta"], 0, config, after=True)
    for swap_state, swap_projection in ((False, False), (True, False), (False, True), (True, True)):
        cache = dict(right["before"])
        if swap_state:
            for key in ("main.kv", "main.score"):
                cache[key] = np.array(cache[key], copy=True)
                cache[key][right["meta"].req_slots[0]] = left["before"][key][
                    left["meta"].req_slots[0]
                ]
        updated, trace = jax.device_get(
            build("override" if swap_projection else "original")(
                right["x"], weights, cache, right["meta"], override
            )
        )
        actual = logical_cache(updated, right["meta"], 0, config, after=True)
        label = f"B2_state{int(swap_state)}_projection{int(swap_projection)}"
        comparisons[label] = difference(expected["main.compressed"], actual["main.compressed"])
        save_arrays(args.output / f"{label}-trace", trace)

    # Determine whether fixed dot shapes eliminate the current-token drift.
    # These do not repair historical captured scratch, so they are not yet
    # a full model fix or an acceptance test.
    for mode in ("row1", "row8"):
        traced = []
        run = build(mode)
        for frame in frames.values():
            _, trace = jax.device_get(run(frame["x"], weights, frame["before"], frame["meta"], ()))
            traced.append(trace)
        comparisons[mode] = {
            key: difference(traced[0][key][0], traced[1][key][0]) for key in ("kv", "scores")
        }

    for key, value in comparisons.items():
        print(json.dumps({"case": key, "comparison": value}), flush=True)
    comparisons["rounding_boundary"] = {}
    for name, (_, trace) in outputs.items():
        values = trace["group_values"][0, :, 468].astype(np.float64)
        scores = trace["group_scores"][0, :, 468].astype(np.float64)
        probabilities = np.exp(scores - np.max(scores))
        probabilities /= np.sum(probabilities)
        comparisons["rounding_boundary"][name] = {
            "pool_from_captured_fp32_inputs_float64": float(np.sum(values * probabilities)),
            "pool_recorded_468": float(trace["pooled"][0, 468]),
            "normalized_recorded_468": float(trace["normalized"][0, 468]),
            "final_recorded_469": float(trace["final"][0, 469]),
        }
    save_arrays(args.output / "B1-input", {"x": left["x"], **weights})
    (args.output / "report.json").write_text(json.dumps(comparisons, indent=2) + "\n")


if __name__ == "__main__":
    main()
