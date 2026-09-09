"""Cross the original/reference attention with captured A/B selected KV.

Uses a faithful layer replay to reconstruct only writes made at the captured
steps. Four small identical-query cases separate historical compressor state
from current joint-attention arithmetic, without changing production code.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from replay_deepseek_v4_8023 import difference
from run_deepseek_v4_framework import framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4 import csa
from sgl_jax.srt.kernels.deepseek_v4.numerics import _single_query_attention, config_for_layer
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--layer", type=int, default=10)
    args = parser.parse_args()
    source = json.loads((args.capture / "report.json").read_text())
    replay = json.loads((args.replay / "report.json").read_text())
    if not replay["complete"] or not replay["faithful"]:
        raise ValueError("requires a faithful completed production A/B replay")
    if source["framework_source_fingerprint"] != framework_fingerprint():
        raise ValueError("requires unchanged failing production source")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = DeepSeekV4Checkpoint(source["checkpoint"])
    traces = args.trace_directory or args.replay
    config = config_for_layer(checkpoint.config, args.layer, 8192)
    sink = checkpoint.read_tensor(f"layers.{args.layer}.attn.attn_sink")
    data, checks = {}, {}
    for mode in ("reference", "pallas"):
        folder = args.capture / mode / f"step-{source['capture_begin']}"
        cache = load_arrays(folder / "before" / f"layer-{args.layer:02d}")
        for position in range(source["capture_begin"], source["position"] + 1):
            inputs = load_arrays(args.capture / mode / f"step-{position}" / "inputs")
            trace = load_arrays(traces / f"{mode}-{position}-layer-{args.layer:02d}-trace")
            cache["window"][inputs["locations"]] = trace["attn.kv"]
            destination = trace["compressor.main.destination"]
            valid = destination < len(cache["main.compressed"])
            cache["main.compressed"][destination[valid]] = trace["compressor.main.final"][valid]
        indices = trace["attn.indices"]
        physical = indices[:, 128:] - len(cache["window"])
        data[mode] = {
            "query": trace["attn.q"],
            "window": cache["window"][indices[:, :128]],
            "window_valid": indices[:, :128] >= 0,
            "selected": cache["main.compressed"][physical],
            "lengths": np.sum(physical >= 0, axis=1, dtype=np.int32),
            "expected": trace["attn.attention_value"],
            "indices": indices,
            "sink": sink,
        }
        save_arrays(args.output / f"{mode}-input", data[mode])
    for key in data["reference"]:
        checks[f"inputs/{key}"] = difference(data["reference"][key], data["pallas"][key])

    @jax.jit
    def retained(query, window, valid, selected, lengths, sink):
        def row(q, w, v, s, length):
            keys = jnp.concatenate((w, s))
            live = jnp.concatenate((v, jnp.arange(s.shape[0]) < length))
            indices = jnp.where(live, jnp.arange(keys.shape[0]), -1)
            return _single_query_attention(q, keys, indices, sink, config.head_dim**-0.5)

        return jax.vmap(row)(query, window, valid, selected, lengths)

    original = jax.jit(lambda *values: csa.attend(*values, config))
    outputs = {}
    for cache_mode, values in data.items():
        call = jax.tree.map(
            jnp.asarray,
            tuple(
                values[key]
                for key in ("query", "window", "window_valid", "selected", "lengths", "sink")
            ),
        )
        for kernel_mode, run in (("reference", retained), ("pallas", original)):
            output = np.asarray(run(*call))
            label = f"kernel_{kernel_mode}/cache_{cache_mode}"
            outputs[label] = output
            for expected_mode, expected in data.items():
                checks[f"{label}/vs_{expected_mode}"] = difference(expected["expected"], output)
    save_arrays(args.output / "outputs", outputs)
    faithful = all(
        checks[f"kernel_{mode}/cache_{mode}/vs_{mode}"]["bitwise_equal"] for mode in data
    )
    report = {
        "complete": faithful,
        "source_fingerprint": framework_fingerprint(),
        "capture": str(args.capture),
        "replay": str(args.replay),
        "trace_directory": str(traces),
        "layer": args.layer,
        "position": source["position"],
        "checks": checks,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    for mode in data:
        if not checks[f"kernel_{mode}/cache_{mode}/vs_{mode}"]["bitwise_equal"]:
            raise AssertionError("isolated attention does not reproduce its captured output")


if __name__ == "__main__":
    main()
