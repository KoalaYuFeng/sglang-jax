"""Replay all HCA attention values in the immutable first-failure prefill.

Uses actual original Pallas attention, identical captured Q/KV, and retained
independent attention values. It does not load all 43 layers or change goldens.
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
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.test_deepseek_v4_paged import make_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=132)
    parser.add_argument("--reemit", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    description = args.capture / "report.json"
    context = json.loads(description.read_text())["context"] if description.exists() else 384
    config = config_for_layer(checkpoint.config, args.layer, context)
    stem = f"layer-{args.layer:02d}"
    reference = load_arrays(args.capture / f"{stem}-reference")
    trace = load_arrays(args.capture / f"{stem}-chunk{args.chunk}-trace")
    cache = load_arrays(args.capture / f"{stem}-chunk{args.chunk}-cache")
    q = jnp.asarray(trace["attn.q"])
    tokens = q.shape[0]
    pages = None if context == 384 else [list(range(1, context // 128 + 1))]
    batch = make_batch([0], [tokens], pages=pages)
    meta = jax.tree.map(
        jnp.asarray, V4PagedBackend(max_context=context).get_forward_metadata(batch)
    )
    sink = jnp.asarray(checkpoint.read_tensor(f"layers.{args.layer}.attn.attn_sink"))
    run = jax.jit(lambda q, w, c, ids, p, s, m: hca.attend(q, w, c, ids, p, s, m, config))
    window = jnp.asarray(cache["window"])
    indices = jnp.asarray(trace["attn.indices"][:, :128])

    def attend(compressed):
        if tokens <= 384:
            return run(
                q, window, compressed, indices, jnp.arange(tokens, dtype=jnp.int32), sink, meta
            )
        # Match the actual packed-token launch bound; do not turn an 8K
        # diagnostic into an unrelated 8K-scalar-prefetch kernel requirement.
        chunks = []
        backend = V4PagedBackend(max_context=context)
        for begin in range(0, tokens, 128):
            end = min(begin + 128, tokens)
            part = backend.get_forward_metadata(make_batch([begin], [end - begin], pages=pages))
            chunks.append(
                run(
                    q[begin:end],
                    window,
                    compressed,
                    indices[begin:end],
                    jnp.arange(begin, end, dtype=jnp.int32),
                    sink,
                    part,
                )
            )
        return jnp.concatenate(chunks)

    original = attend(jnp.asarray(cache["main.compressed"]))
    extra = {}
    if args.reemit:
        fidelity = compare_arrays(trace["attn.attention_value"], original)
        if not fidelity["bitwise_equal"]:
            raise AssertionError("attention-only replay must reproduce the old captured values")
        count = tokens // 128
        page_ids = np.asarray(meta.page_table)[0, :count]
        norm = jnp.asarray(
            checkpoint.read_tensor(f"layers.{args.layer}.attn.compressor.norm.weight")
        )

        @jax.jit
        def emit(v, s, n, p):
            value = hca.emit(v, s, n, p, jnp.ones(p.shape, jnp.bool_), config)
            return jnp.concatenate(
                (activation_fp8_roundtrip(value[:, :-64], 64), value[:, -64:]), axis=1
            )

        records = emit(
            jnp.asarray(cache["main.snapshot_kv"][page_ids]),
            jnp.asarray(cache["main.snapshot_score"][page_ids]),
            norm,
            jnp.arange(count, dtype=jnp.int32) * 128,
        )
        ref_cache = load_arrays(args.capture / f"{stem}-reference-cache")
        extra = {
            "old_attention_faithful": fidelity,
            "recomputed_records": count,
            "records": compare_arrays(ref_cache["main.compressed"][:count], records),
        }
        compressed = jnp.asarray(cache["main.compressed"]).at[page_ids].set(records)
        actual = attend(compressed)
        save_arrays(
            args.output / "records",
            {"expected": ref_cache["main.compressed"][:count], "actual": records},
        )
    else:
        actual = original
    expected = reference["attn.attention_value"]
    differences = np.argwhere(np.asarray(actual) != expected)
    report = {
        "source_fingerprint": framework_fingerprint(),
        "capture": str(args.capture),
        "scope": "identical immutable layer inputs, not a newly replayed complete model",
        **extra,
        "metrics": compare_arrays(expected, actual),
        "different_elements": len(differences),
        "first_differences": differences[:32].tolist(),
        "old_different_elements": int(np.count_nonzero(expected != trace["attn.attention_value"])),
    }
    if tokens <= 384:
        save_arrays(args.output / "values", {"expected": expected, "actual": actual})
    elif len(differences):
        selected = np.unique(differences[:, 0])[:8]
        save_arrays(
            args.output / "first-different-values",
            {"positions": selected, "expected": expected[selected], "actual": actual[selected]},
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if len(differences) or (args.reemit and not extra["records"]["bitwise_equal"]):
        raise AssertionError("real first-failure prefill still has BF16 attention differences")


if __name__ == "__main__":
    main()
