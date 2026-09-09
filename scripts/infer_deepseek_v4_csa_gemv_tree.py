"""Infer retained GEMV addition clusters with exact BF16 cancellation probes.

For leaves (0,j,k), use (+3*2**24,-3*2**24,+1). The small term survives
exactly when the two large terms cancel before it joins their subtree. This
is diagnostic arithmetic, not model evaluation or a production kernel.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sgl_jax.srt.kernels.deepseek_v4.compressor import _project
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.test.test_deepseek_v4_paged import make_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = V4PagedBackend(max_context=8192).get_forward_metadata(
        make_batch([7979], [1], decode=True, pages=[list(range(1, 65))])
    )
    x = jnp.ones((1, 4096), jnp.bfloat16)
    run = jax.jit(_project)
    report = {}
    for partner in [2**i for i in range(12)]:
        cluster = []
        for begin in range(0, 4096, args.width):
            leaves = np.arange(begin, begin + args.width)
            weight = np.zeros((args.width, 4096), ml_dtypes.bfloat16)
            weight[np.arange(args.width), leaves] = 1
            weight[:, 0] = 3 * 2**24
            weight[:, partner] = -(3 * 2**24)
            output = np.asarray(run(x, jnp.asarray(weight), meta))[0]
            if not np.all((output == 0) | (output == 1)):
                raise AssertionError("unexpected cancellation probe output")
            cluster.extend(leaves[output == 0].tolist())
        report[str(partner)] = cluster
        print(
            json.dumps({"partner": partner, "count": len(cluster), "first": cluster[:80]}),
            flush=True,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
