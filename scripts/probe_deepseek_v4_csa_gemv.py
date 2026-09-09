"""Bounded Pallas VPU projection prototypes on a frozen real GEMV fixture.

No production selection is changed. Exact FP32 agreement is reported; these
prototypes are not an acceptance gate or a performance measurement.
"""

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from replay_deepseek_v4_8023 import difference
from sgl_jax.srt.kernels.deepseek_v4.compressor import _project
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.test.test_deepseek_v4_paged import make_batch


@functools.partial(jax.jit, static_argnames=("tile_n", "mode"))
def project(x, weight, *, tile_n, mode):
    width, hidden = weight.shape

    def kernel(x_ref, w_ref, out_ref):
        product = x_ref[...].astype(jnp.float32) * w_ref[...].astype(jnp.float32)
        if mode == "reduce":
            result = jnp.sum(product, axis=1)
        elif mode == "retained_tree":
            tile_k = 1024 if width == 1024 else 2048
            chunks = []
            for begin in range(0, hidden, tile_k):
                lanes = product[:, begin : begin + 128]
                for offset in range(128, tile_k, 128):
                    lanes = lanes + product[:, begin + offset : begin + offset + 128]
                stripes = lanes.reshape(tile_n, 16, 8)
                reduced = stripes[:, 0]
                for stripe in range(1, 16):
                    reduced = reduced + stripes[:, stripe]
                for half in (4, 2, 1):
                    reduced = reduced[:, :half] + reduced[:, half : 2 * half]
                chunks.append(reduced)
            while len(chunks) > 1:
                chunks = [chunks[i] + chunks[i + 1] for i in range(0, len(chunks), 2)]
            result = chunks[0][:, 0]
        elif mode.startswith("split"):
            parts = int(mode[5:])
            partial = jnp.sum(product.reshape(tile_n, parts, hidden // parts), axis=2)
            result = jnp.sum(partial, axis=1)
        elif mode == "mxu1":
            result = jnp.matmul(x_ref[...], w_ref[...].T, preferred_element_type=jnp.float32)[0]
        else:
            raise ValueError(mode)
        out_ref[...] = result[None]

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, width), jnp.float32),
        grid=(width // tile_n,),
        in_specs=(
            pl.BlockSpec((1, hidden), lambda n: (0, 0)),
            pl.BlockSpec((tile_n, hidden), lambda n: (n, 0)),
        ),
        out_specs=pl.BlockSpec((1, tile_n), lambda n: (0, n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name=f"csa-gemv-probe-{mode}-n{tile_n}",
    )(x, weight)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = load_arrays(args.fixture)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, outputs = [], {}
    for prefix in ("main", "index"):
        x = jnp.asarray(fixture["activation"])
        weight = jnp.asarray(fixture[f"{prefix}.wkv.weight"])
        expected = fixture[f"{prefix}.reference.kv"]
        metadata = V4PagedBackend(max_context=8192).get_forward_metadata(
            make_batch([7979], [1], decode=True, pages=[list(range(1, 65))])
        )
        for name, result in (
            (
                "plain_dot",
                jax.jit(lambda x, w: jnp.matmul(x, w.T, preferred_element_type=jnp.float32))(
                    x, weight
                ),
            ),
            ("retained_project", jax.jit(_project)(x, weight, metadata)),
        ):
            row = {"label": f"{prefix}/{name}", **difference(expected, result)}
            rows.append(row)
            outputs[row["label"]] = result
            print(json.dumps(row), flush=True)
        for mode in ("retained_tree",):
            for tile in (128,):
                label = f"{prefix}/{mode}/n{tile}"
                try:
                    result = np.asarray(project(x, weight, tile_n=tile, mode=mode))
                    outputs[label] = result
                    row = {"label": label, **difference(expected, result)}
                except Exception as exc:
                    row = {"label": label, "error": str(exc)}
                rows.append(row)
                (args.output / "report.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    save_arrays(args.output / "outputs", outputs)


if __name__ == "__main__":
    main()
