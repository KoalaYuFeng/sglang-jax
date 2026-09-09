"""Inspect the actual original HCA emitter on a causally isolated real group."""

import argparse
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4 import hca
from sgl_jax.srt.kernels.deepseek_v4.numerics import (
    _fixed_tree_mean_last,
    config_for_layer,
    rope_angles,
)
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip, round_bf16
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint

COMP = importlib.import_module("sgl_jax.srt.kernels.hca.compressor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--verify-fix", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    config = config_for_layer(checkpoint.config, args.layer, 8192)
    fixture = load_arrays(args.fixture / "first-compressed")
    v, s, p = (jnp.asarray(fixture[key]) for key in ("values", "scores", "starts"))
    norm = jnp.asarray(checkpoint.read_tensor(f"layers.{args.layer}.attn.compressor.norm.weight"))
    quantize = jax.jit(
        lambda x: jnp.concatenate((activation_fp8_roundtrip(x[:, :-64], 64), x[:, -64:]), axis=1)
    )

    @jax.jit
    def reference(v, s, n, p):
        probability = jax.nn.softmax(s, axis=1)
        pool_raw = jnp.sum(v * probability, axis=1)
        pooled = round_bf16(pool_raw).astype(jnp.float32)
        inv = jax.lax.rsqrt(_fixed_tree_mean_last(pooled * pooled)[:, None] + config.eps)
        norm_raw = pooled * inv * n.astype(jnp.float32)
        normalized = round_bf16(norm_raw).astype(jnp.float32)
        phase = rope_angles(p, config)
        cosine, sine = jnp.cos(phase), jnp.sin(phase)
        pairs = normalized[:, -64:].reshape(-1, 32, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        rotated = jnp.stack((a * cosine - b * sine, a * sine + b * cosine), axis=-1).reshape(-1, 64)
        raw = jnp.concatenate((normalized[:, :-64], rotated), axis=1)
        tables = jnp.pad(jnp.concatenate((cosine, sine), axis=1), ((0, 0), (0, 448)))
        maximum = jnp.max(s, axis=1, keepdims=True)
        exponent = jnp.exp(s - maximum)
        denominator = jnp.sum(exponent, axis=1)
        stages = jnp.stack(
            (pool_raw, pooled, norm_raw, normalized, raw, tables, denominator, maximum[:, 0]),
            axis=1,
        )
        return round_bf16(raw), stages, probability, exponent

    expected, ref_stages, ref_probability, ref_exponent = reference(v, s, norm, p)
    uninstrumented = jax.jit(
        lambda v, s, n, p: hca.emit(v, s, n, p, jnp.ones(p.shape, jnp.bool_), config)
    )(v, s, norm, p)
    original_call = pl.pallas_call
    captured = {}

    def instrument(kernel, *, grid_spec, out_shape, **kwargs):
        entries, heads, lanes = out_shape.shape
        tile = grid_spec.out_specs.block_shape[0]
        num_inputs = len(grid_spec.in_specs)
        new_grid = pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid_spec.grid,
            in_specs=grid_spec.in_specs,
            out_specs=(
                grid_spec.out_specs,
                pl.BlockSpec((tile, 8, 512), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 128, 512), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 128, 512), lambda i: (i, 0, 0)),
            ),
        )

        def wrapped(*refs):
            stats, probs, exps = refs[num_inputs + 1 : num_inputs + 4]
            stats[...] = jnp.zeros(stats.shape, jnp.float32)
            old_round, old_pool = (
                COMP.round_bf16,
                COMP._pool_normalize_rotate,
            )
            rounds = []

            def capture_round(x):
                y = old_round(x)
                index = 2 * len(rounds)
                stats[:, index] = x.reshape(tile, 512)
                stats[:, index + 1] = y.astype(jnp.float32).reshape(tile, 512)
                rounds.append(1)
                return y

            def capture_softmax(x, *args, **options):
                if args or options != {"axis": 1}:
                    raise ValueError("only the original HCA axis-1 softmax is instrumented")
                maximum = jnp.max(x, axis=1, keepdims=True)
                exponent = jnp.exp(x - maximum)
                denominator = jnp.sum(exponent, axis=1, keepdims=True)
                y = exponent / denominator
                stats[:, 6] = denominator.reshape(tile, 512)
                stats[:, 7] = maximum.reshape(tile, 512)
                exps[...] = exponent.reshape(tile, 128, 512)
                probs[...] = y.reshape(tile, 128, 512)
                return y

            def capture_pool(*args, **options):
                y = old_pool(*args, **options)
                stats[:, 4] = y.reshape(tile, 512)
                return y

            with (
                patch.object(COMP, "round_bf16", capture_round),
                patch.object(COMP, "_pool_normalize_rotate", capture_pool),
                patch.object(jax.nn, "softmax", capture_softmax),
            ):
                kernel(*refs[: num_inputs + 1])
            stats[:, 5] = jnp.pad(refs[3][...].reshape(tile, 128), ((0, 0), (0, 384)))

        call = original_call(
            wrapped,
            grid_spec=new_grid,
            out_shape=(
                out_shape,
                jax.ShapeDtypeStruct((entries, 8, 512), jnp.float32),
                jax.ShapeDtypeStruct((entries, 128, 512), jnp.float32),
                jax.ShapeDtypeStruct((entries, 128, 512), jnp.float32),
            ),
            **kwargs,
        )

        def apply(*args):
            result, stages, probability, exponent = call(*args)
            captured.update(stages=stages, probability=probability, exponent=exponent)
            return result

        return apply

    @jax.jit
    def run(v, s, n, p):
        with (
            patch.object(pl, "pallas_call", instrument),
            patch.object(
                hca, "_hca_emit_selected_pallas", COMP._hca_emit_selected_pallas.__wrapped__
            ),
        ):
            result = hca.emit(v, s, n, p, jnp.ones(p.shape, jnp.bool_), config)
        return result, captured["stages"], captured["probability"], captured["exponent"]

    actual, stages, probability, exponent = run(v, s, norm, p)
    names = (
        "pooled_fp32",
        "pooled_bf16",
        "normalized_fp32",
        "normalized_bf16",
        "rotated_fp32",
        "cos_sin",
        "denominator",
        "maximum",
    )
    report = {
        "source_fingerprint": framework_fingerprint(),
        "fixture": str(args.fixture),
        "reference_faithful": compare_arrays(fixture["reference"], quantize(expected)),
        "original_faithful": compare_arrays(fixture["original"], quantize(uninstrumented)),
        "repaired_vs_reference": compare_arrays(fixture["reference"], quantize(uninstrumented)),
        "instrumented_faithful": compare_arrays(uninstrumented, actual),
        "probability": compare_arrays(ref_probability, probability[: len(v)]),
        "exponent": compare_arrays(ref_exponent, exponent[: len(v)]),
        "stages": {
            name: compare_arrays(ref_stages[:, i], stages[: len(v), i])
            for i, name in enumerate(names)
        },
    }
    if args.previous:
        previous = load_arrays(args.previous / "stages")
        report["previous_faithful"] = {
            "reference_stages": compare_arrays(previous["reference"], ref_stages[:, :6]),
            "pallas_stages": compare_arrays(previous["pallas"], stages[:, :6]),
            "reference_probability": compare_arrays(
                previous["reference_probability"], ref_probability
            ),
            "pallas_probability": compare_arrays(previous["pallas_probability"], probability),
        }
    save_arrays(
        args.output / "stages",
        {
            "reference": ref_stages,
            "pallas": stages,
            "reference_probability": ref_probability,
            "pallas_probability": probability,
            "reference_exponent": ref_exponent,
            "pallas_exponent": exponent,
            "values": v,
            "scores": s,
            "norm": norm,
            "starts": p,
        },
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not all(
        report[key]["bitwise_equal"]
        for key in (
            "reference_faithful",
            "repaired_vs_reference" if args.verify_fix else "original_faithful",
            "instrumented_faithful",
        )
    ):
        raise AssertionError("emitter diagnostic does not reproduce the immutable real outputs")
    if args.previous and any(
        not m["bitwise_equal"]
        for name, m in report["previous_faithful"].items()
        if not args.verify_fix or name.startswith("reference")
    ):
        raise AssertionError(
            "extra softmax instrumentation changes previous original intermediates"
        )


if __name__ == "__main__":
    main()
