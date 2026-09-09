"""Identify the recorded XLA FP32 sum tree; host-only, no TPU owner."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def trees(items):
    if len(items) == 1:
        yield items[0]
        return
    # Addition is commutative for these finite nonnegative inputs. Fix the
    # first leaf on the left to avoid equivalent mirrored expressions.
    for mask in range(1 << (len(items) - 1)):
        left = (items[0],) + tuple(x for i, x in enumerate(items[1:]) if mask & (1 << i))
        right = tuple(x for i, x in enumerate(items[1:]) if not mask & (1 << i))
        if right:
            for a in trees(left):
                for b in trees(right):
                    yield (a, b)


def evaluate(tree, values):
    if isinstance(tree, int):
        return values[tree]
    return evaluate(tree[0], values) + evaluate(tree[1], values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.states, allow_pickle=False) as arrays:
        values = arrays["reference_probability"]
        target = arrays["reference"][1, :, 0]
    results = []
    for order in itertools.permutations(range(6)):
        value = values.reshape(64, *([2] * 6))
        axes = list(range(6))
        for bit in order:
            axis = axes.index(bit) + 1
            value = np.take(value, 0, axis=axis) + np.take(value, 1, axis=axis)
            axes.remove(bit)
        results.append({"order": order, "mismatches": int(np.count_nonzero(value != target))})
    results.sort(key=lambda row: row["mismatches"])
    report = {"matches": [r for r in results if not r["mismatches"]], "best": results[:12]}
    report["unbalanced_four"] = []
    for bits in itertools.combinations(range(6), 2):
        axes = [bit for bit in range(6) if bit not in bits]
        partials = np.transpose(
            values.reshape(64, *([2] * 6)),
            (0, *(bit + 1 for bit in bits), *(bit + 1 for bit in axes)),
        )
        partials = partials.reshape(64, 4, *([2] * 4)).swapaxes(0, 1)
        for tree in trees(tuple(range(4))):
            value = evaluate(tree, partials)
            for order in itertools.permutations(axes):
                v, active = value, axes.copy()
                for bit in order:
                    axis = active.index(bit) + 1
                    v = np.take(v, 0, axis=axis) + np.take(v, 1, axis=axis)
                    active.remove(bit)
                mismatch = int(np.count_nonzero(v != target))
                if mismatch <= 2:
                    report["unbalanced_four"].append(
                        {"bits": bits, "tree": tree, "order": order, "mismatches": mismatch}
                    )
    report["fp64_rounded"] = int(
        np.count_nonzero(values.astype(np.float64).sum(axis=1).astype(np.float32) != target)
    )
    # A horizontal reduce can combine four/eight inputs in one wider internal
    # accumulator, rather than being a binary FP32 tree.
    for bits in ((1, 2), (2, 1), (1, 2, 3), (0, 1, 2)):
        axes = list(range(6))
        value = values.reshape(64, *([2] * 6)).astype(np.float64)
        value = value.sum(axis=tuple(bit + 1 for bit in bits)).astype(np.float32)
        for bit in bits:
            axes.remove(bit)
        best = []
        for order in itertools.permutations(axes):
            v, active = value, axes.copy()
            for bit in order:
                axis = active.index(bit) + 1
                v = np.take(v, 0, axis=axis) + np.take(v, 1, axis=axis)
                active.remove(bit)
            best.append({"order": order, "mismatches": int(np.count_nonzero(v != target))})
        report["wide_" + str(bits)] = sorted(best, key=lambda r: r["mismatches"])[:3]
    for tile in (2, 4, 8, 16, 32):
        for direction in ("contiguous", "strided"):
            chunks = values.reshape(64, -1, tile)
            if direction == "strided":
                chunks = values.reshape(64, tile, -1).transpose(0, 2, 1)
            partial = np.zeros(chunks.shape[:2], np.float32)
            for column in range(tile):
                partial = partial + chunks[..., column]
            for final in ("sequential", "numpy"):
                if final == "numpy":
                    result = partial.sum(axis=1)
                else:
                    result = np.zeros((64,), np.float32)
                    for column in range(partial.shape[1]):
                        result = result + partial[:, column]
                report[f"{direction}_{tile}_{final}"] = int(np.count_nonzero(result != target))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
