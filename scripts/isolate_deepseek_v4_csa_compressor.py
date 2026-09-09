"""Cross captured reference/Pallas projection windows with both CSA emitters."""

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
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer, rms_norm, rope
from sgl_jax.srt.kernels.low_bit.formats import round_bf16
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--position", type=int, default=7979)
    args = parser.parse_args()
    source = json.loads((args.capture / "report.json").read_text())
    replay = json.loads((args.replay / "report.json").read_text())
    if not replay["complete"] or not replay["faithful"]:
        raise ValueError("requires a faithful completed production A/B replay")
    if source["framework_source_fingerprint"] != framework_fingerprint():
        raise ValueError("requires unchanged failing production source")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = DeepSeekV4Checkpoint(source["checkpoint"])
    config = config_for_layer(checkpoint.config, args.layer, 8192)
    traces = {
        mode: load_arrays(args.replay / f"{mode}-{args.position}-layer-{args.layer:02d}-trace")
        for mode in ("reference", "pallas")
    }
    metadata = load_arrays(args.capture / "reference" / f"step-{args.position}" / "metadata")
    starts = jnp.asarray(metadata["group4_starts"])
    valid = np.asarray(starts) >= 0

    @jax.jit
    def retained(values, scores, norm):
        pooled = round_bf16(jnp.sum(values * jax.nn.softmax(scores, axis=1), axis=1))
        normalized = rms_norm(pooled, norm, config.eps)
        emitted = rope(normalized, jnp.maximum(starts, 0), config)
        return emitted, pooled, normalized

    original = jax.jit(lambda v, s, n: csa.emit(v, s, n, starts, jnp.asarray(valid), config))
    checks, outputs, fixtures = {}, {}, {}
    for prefix in ("main", "index"):
        name = "attn.compressor" if prefix == "main" else "attn.indexer.compressor"
        norm = jnp.asarray(checkpoint.read_tensor(f"layers.{args.layer}.{name}.norm.weight"))
        fixtures[f"{prefix}.norm"] = norm
        expected = {
            "reference": jax.jit(
                lambda normalized: rope(normalized, jnp.maximum(starts, 0), config)
            )(jnp.asarray(traces["reference"][f"compressor.{prefix}.normalized"])),
            "pallas": traces["pallas"][f"compressor.{prefix}.csa_emitted"],
        }
        for input_mode, trace in traces.items():
            values, scores = (
                jnp.asarray(trace[f"compressor.{prefix}.group_{suffix}"])
                for suffix in ("values", "scores")
            )
            fixtures[f"{prefix}.{input_mode}.values"] = values
            fixtures[f"{prefix}.{input_mode}.scores"] = scores
            emitted, pooled, normalized = retained(values, scores, norm)
            outputs[f"{prefix}/{input_mode}/pooled"] = pooled
            outputs[f"{prefix}/{input_mode}/normalized"] = normalized
            for kernel, result in (
                ("reference", emitted),
                ("pallas", original(values, scores, norm)),
            ):
                label = f"{prefix}/kernel_{kernel}/inputs_{input_mode}"
                outputs[label] = result
                for mode in expected:
                    checks[f"{label}/vs_{mode}"] = difference(
                        np.asarray(expected[mode])[valid], np.asarray(result)[valid]
                    )
        checks[f"{prefix}/cross_input_pooled"] = difference(
            np.asarray(outputs[f"{prefix}/reference/pooled"])[valid],
            np.asarray(outputs[f"{prefix}/pallas/pooled"])[valid],
        )
        for suffix in ("wkv", "wgate"):
            fixtures[f"{prefix}.{suffix}.weight"] = checkpoint.read_tensor(
                f"layers.{args.layer}.{name}.{suffix}.weight"
            )
    fixtures["starts"] = starts
    fixtures["valid"] = valid
    fixtures["activation"] = traces["reference"]["attn.norm"]
    for mode, trace in traces.items():
        for prefix in ("main", "index"):
            for suffix in ("kv", "scores"):
                fixtures[f"{prefix}.{mode}.{suffix}"] = trace[f"compressor.{prefix}.{suffix}"]
    save_arrays(args.output / "fixture", fixtures)
    save_arrays(args.output / "outputs", outputs)
    faithful = all(
        checks[f"{prefix}/kernel_{mode}/inputs_{mode}/vs_{mode}"]["bitwise_equal"]
        for prefix in ("main", "index")
        for mode in ("reference", "pallas")
    )
    report = {
        "complete": faithful,
        "source_fingerprint": framework_fingerprint(),
        "capture": str(args.capture),
        "replay": str(args.replay),
        "layer": args.layer,
        "position": args.position,
        "checks": checks,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not faithful:
        raise AssertionError("isolated compressor does not reproduce its captured output")


if __name__ == "__main__":
    main()
