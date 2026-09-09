"""Check real layer-0 mHC post against CPU arithmetic and both TPU backends."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from replay_deepseek_v4_8023 import difference
from sgl_jax.srt.kernels.mhc import mhc_post_fused, mhc_pre_fused
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    whole = load_arrays(args.fixture / "whole-first-token")
    chunk = load_arrays(args.fixture / "chunk-first-token")
    x, residual = whole["ffn.operator"], whole["attn.post"]
    np.testing.assert_array_equal(x, chunk["ffn.operator"])
    np.testing.assert_array_equal(residual, chunk["attn.post"])
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    fn, scale, base = (
        jnp.asarray(checkpoint.read_tensor("layers.0.hc_ffn_" + key))
        for key in ("fn", "scale", "base")
    )
    reports, gates = {}, {}
    for count in (128, 7936):
        pre, post, comb = mhc_pre_fused(
            jnp.asarray(np.repeat(residual, count, axis=0)),
            fn,
            scale,
            base,
            hc_mult=4,
            sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
            dot_precision=jax.lax.Precision.HIGHEST,
        )
        gates[count] = tuple(np.asarray(v[:1]) for v in (pre, post, comb))
    reports["pre_gates_whole_vs_chunk"] = {
        name: difference(a, b)
        for name, a, b in zip(("pre", "post", "comb"), gates[128], gates[7936], strict=True)
    }
    _, post, comb = gates[128]
    cpu64 = (
        post.astype(np.float64)[..., None] * x.astype(np.float64)[:, None]
        + np.einsum("nij,nid->njd", comb.astype(np.float64), residual.astype(np.float64))
    ).astype(jnp.bfloat16)
    reports["cpu64_vs_recorded_chunk"] = difference(chunk["ffn.post"], cpu64)
    reports["cpu64_vs_recorded_whole"] = difference(whole["ffn.post"], cpu64)
    for count, backend in ((128, "xla"), (128, "pallas"), (7936, "auto"), (7936, "xla")):
        output = mhc_post_fused(
            *(jnp.asarray(np.repeat(a, count, axis=0)) for a in (x, residual, post, comb)),
            backend=backend,
            precision=jax.lax.Precision.HIGHEST,
        )
        value = np.asarray(output[:1])
        label = f"{backend}-{count}"
        reports[label] = {
            "cpu64": difference(cpu64, value),
            "recorded_chunk": difference(chunk["ffn.post"], value),
            "recorded_whole": difference(whole["ffn.post"], value),
        }
        print(json.dumps({"backend": label, **reports[label]}), flush=True)
        save_arrays(args.output / label, {"output": value})
    # Model the FFN's FP32 sum -> BF16 result -> FP32 post multiplication.
    # An integrated compiler may elide this semantic rounding boundary even
    # though the materialized operator trace itself has the correct dtype.
    raw = jnp.asarray(np.repeat(x.astype(np.float32) * 1.001, 128, axis=0))
    res, p, c = (jnp.asarray(np.repeat(a, 128, axis=0)) for a in (residual, post, comb))
    rounded = np.asarray(raw.astype(jnp.bfloat16))
    want = (
        np.asarray(p, np.float64)[..., None] * rounded.astype(np.float64)[:, None]
        + np.einsum("nij,nid->njd", np.asarray(c, np.float64), np.asarray(res, np.float64))
    ).astype(jnp.bfloat16)
    for backend in ("xla", "pallas"):

        def integrated(raw, res, p, c, backend=backend):
            operator = raw.astype(jnp.bfloat16)
            return operator, mhc_post_fused(
                operator, res, p, c, backend=backend, precision=jax.lax.Precision.HIGHEST
            )

        operator, value = jax.jit(integrated)(raw, res, p, c)
        reports[f"integrated-{backend}"] = {
            "operator": difference(rounded, np.asarray(operator)),
            "post": difference(want, np.asarray(value)),
        }
        (args.output / f"integrated-{backend}.hlo.txt").write_text(
            jax.jit(integrated).lower(raw, res, p, c).compile().as_text()
        )
    save_arrays(
        args.output / "inputs",
        {
            "x": x,
            "residual": residual,
            "post": post,
            "comb": comb,
            "cpu64": cpu64,
            "recorded_chunk": chunk["ffn.post"],
            "recorded_whole": whole["ffn.post"],
        },
    )
    (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
