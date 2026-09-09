"""Read actual original-HCA online-softmax state on a failing real query.

The diagnostic wraps the existing Pallas program and adds an output copy of
its FP32 scratch; it does not implement another candidate attention kernel.
Both diagnostic outputs must first reproduce the saved uninstrumented values.
"""

import argparse
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4 import hca
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer
from sgl_jax.srt.kernels.hca.attention import _streaming_attention
from sgl_jax.srt.kernels.low_bit.formats import round_bf16
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.test_deepseek_v4_paged import make_batch

ORIGINAL_ATTENTION = importlib.import_module("sgl_jax.srt.kernels.hca.attention")


def reference_stats(q, kv, indices, sink):
    count = (indices.shape[0] + 63) // 64 * 64
    indices = jnp.pad(indices, (0, count - indices.shape[0]), constant_values=-1)
    initial = (
        jnp.full((64,), -1e30, jnp.float32),
        jnp.zeros((64,), jnp.float32),
        jnp.zeros((64, 512), jnp.float32),
    )

    def block(i, state):
        maximum, denominator, numerator = state
        ids = jax.lax.dynamic_slice_in_dim(indices, i * 64, 64)
        keys = kv[jnp.maximum(ids, 0)]
        score = jnp.matmul(q, keys.T, preferred_element_type=jnp.float32) * 512**-0.5
        valid = ids[None] >= 0
        score = jnp.where(valid, score, -1e30)
        next_max = jnp.maximum(maximum, jnp.max(score, axis=-1))
        alpha = jnp.exp(maximum - next_max)
        probability = jnp.where(valid, jnp.exp(score - next_max[:, None]), 0.0)
        numerator = numerator * alpha[:, None] + jnp.matmul(
            round_bf16(probability), keys, preferred_element_type=jnp.float32
        )
        denominator = denominator * alpha + jnp.sum(probability, axis=-1)
        return next_max, denominator, numerator

    maximum, denominator, numerator = jax.lax.fori_loop(0, count // 64, block, initial)
    rescale = jnp.exp(maximum - jnp.maximum(maximum, sink))
    denom = denominator * rescale + jnp.exp(sink - jnp.maximum(maximum, sink))
    value = round_bf16(numerator * rescale[:, None] / denom[:, None])
    return value, jnp.stack(
        (
            jnp.broadcast_to(maximum[:, None], numerator.shape),
            jnp.broadcast_to(denominator[:, None], numerator.shape),
            numerator,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--verify-fix", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    config = config_for_layer(checkpoint.config, args.layer, 384)
    stem = f"layer-{args.layer:02d}"
    reference = load_arrays(args.capture / f"{stem}-reference")
    actual = load_arrays(args.capture / f"{stem}-chunk132-trace")
    cache = load_arrays(args.capture / f"{stem}-chunk132-cache")
    differing = np.argwhere(reference["attn.attention_value"] != actual["attn.attention_value"])
    token = int(differing[0, 0])
    print(
        json.dumps(
            {"different_elements": len(differing), "first_indices": differing[:16].tolist()}
        ),
        flush=True,
    )
    batch = make_batch([token], [1], decode=True)
    meta = jax.tree.map(jnp.asarray, V4PagedBackend(max_context=384).get_forward_metadata(batch))
    q = jnp.asarray(actual["attn.q"][token : token + 1])
    window, compressed = jnp.asarray(cache["window"]), jnp.asarray(cache["main.compressed"])
    ids = jnp.asarray(actual["attn.indices"][token : token + 1])
    positions = jnp.asarray([token], jnp.int32)
    sink = jnp.asarray(checkpoint.read_tensor(f"layers.{args.layer}.attn.attn_sink"))
    ref_value, ref_stats = jax.jit(reference_stats)(
        q[0], jnp.concatenate((window, compressed)), ids[0], sink
    )
    uninstrumented = jax.jit(
        lambda q, w, c, ids, p, s, m: hca.attend(q, w, c, ids, p, s, m, config)
    )(q, window, compressed, ids[:, :128], positions, sink, meta)
    diagnostics = {}
    original_call = pl.pallas_call

    def instrument(kernel, *, grid_spec, out_shape, **kwargs):
        tokens, heads, dim = out_shape.shape
        num_inputs = grid_spec.num_scalar_prefetch + len(grid_spec.in_specs)
        new_grid = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=grid_spec.num_scalar_prefetch,
            grid=grid_spec.grid,
            in_specs=grid_spec.in_specs,
            out_specs=(
                grid_spec.out_specs,
                pl.BlockSpec((1, 6, heads, dim), lambda t, *_: (t, 0, 0, 0)),
            ),
            scratch_shapes=grid_spec.scratch_shapes,
        )

        def wrapped(*refs):
            stats_ref = refs[num_inputs + 1]
            stats_ref[...] = jnp.zeros(stats_ref.shape, jnp.float32)
            original_sum = jnp.sum
            original_v4_sum = ORIGINAL_ATTENTION._v4_probability_sum_64
            captured = []

            def capture_probability(a):
                stats_ref[0, 3 + len(captured)] = jnp.pad(a, ((0, 0), (0, dim - 64)))
                captured.append(1)

            def capture_sum(a, axis=None, **options):
                if a.shape == (heads, 64) and axis == 1:
                    capture_probability(a)
                return original_sum(a, axis=axis, **options)

            def capture_v4_sum(a):
                capture_probability(a)
                return original_v4_sum(a)

            with (
                patch.object(jnp, "sum", capture_sum),
                patch.object(ORIGINAL_ATTENTION, "_v4_probability_sum_64", capture_v4_sum),
            ):
                kernel(*refs[: num_inputs + 1], *refs[num_inputs + 2 :])
            maximum, denominator, acc = (ref[...] for ref in refs[-3:])
            stats_ref[0, 0] = jnp.broadcast_to(maximum[:, :1], acc.shape)
            stats_ref[0, 1] = jnp.broadcast_to(denominator[:, :1], acc.shape)
            stats_ref[0, 2] = acc

        call = original_call(
            wrapped,
            grid_spec=new_grid,
            out_shape=(out_shape, jax.ShapeDtypeStruct((tokens, 6, heads, dim), jnp.float32)),
            **kwargs,
        )

        def apply(*args):
            value, stats = call(*args)
            diagnostics["stats"] = stats
            return value

        return apply

    @jax.jit
    def run(q, w, c, ids, p, s, m):
        with (
            patch.object(pl, "pallas_call", instrument),
            patch.object(hca, "_streaming_attention", _streaming_attention.__wrapped__),
        ):
            value = hca.attend(q, w, c, ids, p, s, m, config)
        return value, diagnostics["stats"]

    value, stats = run(q, window, compressed, ids[:, :128], positions, sink, meta)

    @jax.jit
    def probabilities(q, kv, ids, maximum):
        score = (
            jnp.matmul(q, kv[jnp.maximum(ids, 0)].T, preferred_element_type=jnp.float32) * 512**-0.5
        )
        prob = jnp.where(ids[None] >= 0, jnp.exp(score - maximum[:, None]), 0.0)
        return prob, jnp.sum(prob, axis=-1)

    ref_probability, ref_sum = probabilities(
        q[0], jnp.concatenate((window, compressed)), ids[0, :64], ref_stats[0, :, 0]
    )
    p = np.asarray(stats[0, 3, :, :64])
    adjacent, halves = p.copy(), p.copy()
    while adjacent.shape[-1] > 1:
        pairs = adjacent.reshape(64, -1, 2)
        adjacent = pairs[..., 0] + pairs[..., 1]
        width = halves.shape[-1] // 2
        halves = halves[:, :width] + halves[:, width:]
    report = {
        "source_fingerprint": framework_fingerprint(),
        "token": token,
        "different_indices": differing.tolist(),
        "reference_faithful": compare_arrays(reference["attn.attention_value"][token], ref_value),
        "pallas_faithful": compare_arrays(uninstrumented[0], value[0]),
        "historical_pallas_comparison": compare_arrays(
            actual["attn.attention_value"][token], value[0]
        ),
        "fixed_value": compare_arrays(ref_value, value[0]),
        "first_block_probability": compare_arrays(ref_probability, p),
        "first_block_ref_sum": compare_arrays(ref_stats[1, :, 0], ref_sum),
        "numpy_adjacent_sum": compare_arrays(ref_stats[1, :, 0], adjacent[:, 0]),
        "numpy_halves_sum": compare_arrays(ref_stats[1, :, 0], halves[:, 0]),
        "stages": {
            key: compare_arrays(ref_stats[i], stats[0, i])
            for i, key in enumerate(("maximum", "denominator", "numerator"))
        },
    }
    save_arrays(
        args.output / "states",
        {
            "reference": ref_stats,
            "pallas": stats[0],
            "sink": sink,
            "reference_value": ref_value,
            "pallas_value": value[0],
            "reference_probability": ref_probability,
        },
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if (
        not report["reference_faithful"]["bitwise_equal"]
        or not report["pallas_faithful"]["bitwise_equal"]
    ):
        raise AssertionError(
            "instrumented query does not faithfully reproduce the captured outputs"
        )
    if args.verify_fix and (
        not report["fixed_value"]["bitwise_equal"]
        or any(not stage["bitwise_equal"] for stage in report["stages"].values())
    ):
        raise AssertionError("original HCA scratch/output does not match the retained query")


if __name__ == "__main__":
    main()
