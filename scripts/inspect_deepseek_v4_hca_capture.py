"""CPU-only index-level comparison of a saved real HCA layer prefill."""

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np


def arrays(path):
    schema = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
        return {
            key: saved[key].view(ml_dtypes.bfloat16) if info["dtype"] == "bfloat16" else saved[key]
            for key, info in schema.items()
        }


def difference(expected, actual):
    same = np.asarray(expected) == np.asarray(actual)
    first = np.argwhere(~same)
    return {
        "different_elements": len(first),
        "first": first[:16].tolist(),
        "expected": np.asarray(expected[tuple(first[:16].T)], np.float32).tolist()
        if len(first)
        else [],
        "actual": np.asarray(actual[tuple(first[:16].T)], np.float32).tolist()
        if len(first)
        else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    args = parser.parse_args()
    report = json.loads((args.capture / "report.json").read_text())
    tokens = report["tokens"]
    stem = args.capture / f"layer-{args.layer:02d}"
    reference = arrays(Path(str(stem) + "-reference"))
    ref_cache = arrays(Path(str(stem) + "-reference-cache"))
    result = {
        "capture": str(args.capture),
        "source_fingerprint": report["source_fingerprint"],
        "chunks": {},
    }
    for chunk in report["chunks"]:
        trace = arrays(Path(str(stem) + f"-chunk{chunk}-trace"))
        cache = arrays(Path(str(stem) + f"-chunk{chunk}-cache"))
        result["chunks"][str(chunk)] = {
            key: difference(reference[key], trace[key])
            for key in ("attn.norm", "attn.q", "attn.kv", "attn.attention_value", "attn.operator")
        }
        # Long-context diagnostic uses physical pages 1..N; page zero is the
        # reserved sentinel. The independent cache has logical row addressing.
        result["chunks"][str(chunk)].update(
            {
                "window": difference(
                    ref_cache["window"][:tokens], cache["window"][128 : 128 + tokens]
                ),
                "compressed": difference(
                    ref_cache["main.compressed"][: tokens // 128],
                    cache["main.compressed"][1 : 1 + tokens // 128],
                ),
                "final_projection_kv": difference(ref_cache["main.kv"], cache["main.kv"][1]),
                "final_projection_score": difference(
                    ref_cache["main.score"], cache["main.score"][1]
                ),
            }
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
