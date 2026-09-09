"""CPU-only reduction-tree investigation of captured BF16-input/F32 GEMV.

This is not a kernel or acceptance gate. It ranks explicit floating-point
addition orders against the actual traced projection to guide a Pallas ABI.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from debug_deepseek_v4_8023 import load_arrays
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def reduce_axis(value, axis, kind):
    value = np.moveaxis(value, axis, -1)
    if kind == "sequential":
        result = value[..., 0].copy()
        for index in range(1, value.shape[-1]):
            result += value[..., index]
        return result
    while value.shape[-1] > 1:
        width = value.shape[-1]
        if kind == "adjacent":
            value = value[..., ::2] + value[..., 1::2]
        else:
            value = value[..., : width // 2] + value[..., width // 2 :]
    return value[..., 0]


def calculate(product, parts, lanes, interleaved, order, kinds):
    width = product.shape[-1]
    if interleaved:
        value = product.reshape(-1, width // parts // lanes, parts, lanes).transpose(0, 2, 1, 3)
    else:
        value = product.reshape(-1, parts, width // parts // lanes, lanes)
    axes = [0, 1, 2]
    for axis, kind in zip(order, kinds, strict=True):
        index = axes.index(axis)
        value = reduce_axis(value, index + 1, kind)
        axes.pop(index)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--position", type=int, default=7983)
    args = parser.parse_args()
    source = json.loads((args.capture / "report.json").read_text())
    checkpoint = DeepSeekV4Checkpoint(source["checkpoint"])
    trace = load_arrays(args.replay / f"reference-{args.position}-layer-{args.layer:02d}-trace")
    activation = trace["attn.norm"][0].astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {}
    for prefix in ("main", "index"):
        name = "attn.compressor" if prefix == "main" else "attn.indexer.compressor"
        weight = checkpoint.read_tensor(f"layers.{args.layer}.{name}.wkv.weight").astype(np.float32)
        product = weight * activation
        expected = trace[f"compressor.{prefix}.kv"][0]
        candidates = []
        for parts, lanes, interleaved, order, stripe_kind in itertools.product(
            (1, 2, 4),
            (8, 16, 32, 64, 128, 256),
            (False, True),
            ((1, 2, 0), (2, 1, 0), (0, 1, 2)),
            ("sequential", "adjacent", "halving"),
        ):
            kinds = tuple(stripe_kind if axis == 1 else "halving" for axis in order)
            options = (parts, lanes, interleaved, order, kinds)
            value = calculate(product[:64], *options)
            candidates.append((int(np.count_nonzero(value == expected[:64])), options))
        best = []
        for matched, options in sorted(candidates, key=lambda item: item[0], reverse=True)[:12]:
            value = calculate(product, *options)
            best.append(
                {
                    "options": options,
                    "sample_matches": matched,
                    "total_matches": int(np.count_nonzero(value == expected)),
                    "total": len(expected),
                    "max_abs": float(np.max(np.abs(value - expected))),
                }
            )
        report[prefix] = best
        print(json.dumps({prefix: best}, indent=2), flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
