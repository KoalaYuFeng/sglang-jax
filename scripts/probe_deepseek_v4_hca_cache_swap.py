"""Causally isolate real HCA cache emission from attention consumption.

The original Pallas query must reproduce its captured uninstrumented output
before comparing a diagnostic replacement of compressed KV only.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4 import hca
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.test_deepseek_v4_paged import make_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.capture / "report.json").read_text())
    if source["source_fingerprint"] != framework_fingerprint():
        raise ValueError("cache-swap diagnosis requires a faithful current-source capture")
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    config = config_for_layer(checkpoint.config, args.layer, source["context"])
    stem = f"layer-{args.layer:02d}"
    reference = load_arrays(args.capture / f"{stem}-reference")
    trace = load_arrays(args.capture / f"{stem}-chunk{args.chunk}-trace")
    ref_cache = load_arrays(args.capture / f"{stem}-reference-cache")
    cache = load_arrays(args.capture / f"{stem}-chunk{args.chunk}-cache")
    different = np.argwhere(reference["attn.attention_value"] != trace["attn.attention_value"])
    if not len(different):
        raise ValueError("capture has no HCA attention difference")
    token = int(different[0, 0])
    np.testing.assert_array_equal(reference["attn.q"], trace["attn.q"])
    count = source["tokens"] // 128
    compressed_diff = np.argwhere(
        ref_cache["main.compressed"][:count] != cache["main.compressed"][1 : count + 1]
    )
    batch = make_batch(
        [token], [1], pages=[list(range(1, config.max_context // 128 + 1))], decode=True
    )
    meta = jax.tree.map(
        jnp.asarray, V4PagedBackend(max_context=config.max_context).get_forward_metadata(batch)
    )
    q = jnp.asarray(trace["attn.q"][token : token + 1])
    window = jnp.asarray(cache["window"])
    ids = jnp.asarray(trace["attn.indices"][token : token + 1, :128])
    positions = jnp.asarray([token], jnp.int32)
    sink = jnp.asarray(checkpoint.read_tensor(f"layers.{args.layer}.attn.attn_sink"))
    run = jax.jit(lambda c: hca.attend(q, window, c, ids, positions, sink, meta, config))
    original = run(jnp.asarray(cache["main.compressed"]))[0]
    fixed_cache = (
        jnp.asarray(cache["main.compressed"])
        .at[1 : count + 1]
        .set(ref_cache["main.compressed"][:count])
    )
    swapped = run(fixed_cache)[0]
    expected = reference["attn.attention_value"][token]
    report = {
        "source_fingerprint": framework_fingerprint(),
        "capture": str(args.capture),
        "layer": args.layer,
        "token": token,
        "attention_differences": len(different),
        "first_attention_differences": different[:16].tolist(),
        "compressed_differences": len(compressed_diff),
        "first_compressed_differences": compressed_diff[:16].tolist(),
        "query_faithful": compare_arrays(trace["attn.attention_value"][token], original),
        "original_vs_reference": compare_arrays(expected, original),
        "compressed_swap_vs_reference": compare_arrays(expected, swapped),
    }
    save_arrays(
        args.output / "query",
        {"q": q, "reference": expected, "original": original, "swapped": swapped},
    )
    if len(compressed_diff):
        group = int(compressed_diff[0, 0])
        save_arrays(
            args.output / "first-compressed",
            {
                "values": cache["main.snapshot_kv"][group + 1 : group + 2],
                "scores": cache["main.snapshot_score"][group + 1 : group + 2],
                "starts": np.asarray([group * 128], np.int32),
                "reference": ref_cache["main.compressed"][group : group + 1],
                "original": cache["main.compressed"][group + 1 : group + 2],
            },
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["query_faithful"]["bitwise_equal"]:
        raise AssertionError("isolated original HCA query does not reproduce its capture")


if __name__ == "__main__":
    main()
