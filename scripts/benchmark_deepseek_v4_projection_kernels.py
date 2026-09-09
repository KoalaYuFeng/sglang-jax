"""Real FP8 checkpoint A/B gates and warmed, alternating kernel timings.

Four chips execute the same local TP4 shard dimensions. No collectives,
scheduler, expert routing, or model-level throughput are measured here.
Random BF16 inputs are reproducible, not captured model hidden states.
"""

import argparse
import functools
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from run_deepseek_v4_framework import framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4 import normalization, numerics, projections
from sgl_jax.srt.kernels.deepseek_v4.fp8 import fp8_linear
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4, 128])
    parser.add_argument("--tiles", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=["linear", "norm", "wo_a", "merged"],
        default=["linear", "norm", "wo_a", "merged"],
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(m < 1 for m in args.tokens):
        raise ValueError("tokens and repeats must be positive")
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("requires the idle four-chip TPU host")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    config = numerics.config_for_layer(checkpoint.config, 2, 8192)
    groups, rank = config.groups // 4, config.o_rank
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    replicated = NamedSharding(mesh, P())
    rng = np.random.default_rng(8023)
    report = {
        "complete": False,
        "framework_source_fingerprint": framework_fingerprint(),
        "checkpoint": str(args.checkpoint),
        "jax": jax.__version__,
        "devices": [d.device_kind for d in jax.devices()],
        "scope": "four-chip replicated local TP4 shard; random BF16 activations; no model/scheduler/collectives",
        "timing": "alternating warmed host-dispatch-to-block_until_ready, compile and correctness checks excluded",
        "cases": [],
    }

    def save(event):
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    def array(value):
        return jax.device_put(value, replicated)

    def activation(shape):
        return array(rng.normal(size=shape).astype(ml_dtypes.bfloat16))

    def compare(label, baseline, candidate, values):
        record = {"label": label, "shapes": [list(v.shape) for v in values]}
        compiled = []
        for name, fn in (("baseline", baseline), ("candidate", candidate)):
            mapped = jax.jit(
                jax.shard_map(
                    fn,
                    mesh=mesh,
                    in_specs=(P(),) * len(values),
                    out_specs=P(),
                    check_vma=False,
                )
            )
            start = time.perf_counter()
            executable = mapped.lower(*values).compile()
            record[name + "_compile_s"] = time.perf_counter() - start
            stats = executable.memory_analysis()
            record[name + "_memory"] = {
                key: getattr(stats, key)
                for key in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                )
            }
            (args.output / (label + "." + name + ".hlo.txt")).write_text(
                executable.as_text()
            )
            compiled.append(executable)
        expected, actual = [np.asarray(executable(*values)) for executable in compiled]
        record["bitwise_equal"] = bool(
            np.array_equal(expected.view(np.uint16), actual.view(np.uint16))
        )
        record["finite"] = bool(
            np.isfinite(expected.astype(np.float32)).all()
            and np.isfinite(actual.astype(np.float32)).all()
        )
        record["mismatches"] = int(np.count_nonzero(expected != actual))
        report["cases"].append(record)
        save({"event": "correctness", **record})
        if not record["finite"] or not record["bitwise_equal"]:
            raise AssertionError("exact checkpoint arithmetic gate failed: " + label)
        samples = [[], []]
        for executable in compiled:
            for _ in range(3):
                jax.block_until_ready(executable(*values))
        for repeat in range(args.repeats):
            for i in (0, 1) if repeat % 2 == 0 else (1, 0):
                start = time.perf_counter()
                jax.block_until_ready(compiled[i](*values))
                samples[i].append((time.perf_counter() - start) * 1000)
        record.update(
            baseline_ms=float(np.median(samples[0])),
            candidate_ms=float(np.median(samples[1])),
            samples_ms=samples,
        )
        record["speedup"] = record["baseline_ms"] / record["candidate_ms"]
        save(
            {
                "event": "timing",
                "label": label,
                **{k: record[k] for k in ("baseline_ms", "candidate_ms", "speedup")},
            }
        )

    try:
        with jax.set_mesh(mesh):
            if "linear" in args.kinds:
                for prefix in (
                    "attn.wq_a",
                    "attn.wkv",
                    "attn.wq_b",
                    "attn.indexer.wq_b",
                    "attn.wo_b",
                    "ffn.shared_experts.w1",
                    "ffn.shared_experts.w2",
                ):
                    loaded = checkpoint.load_linear(
                        "layers.2." + prefix,
                        shard_count=4 if prefix == "attn.wq_b" else 1,
                    )
                    w, s = array(loaded.data), array(loaded.scales)
                    for m in args.tokens:
                        x = activation((m, loaded.logical_shape[1]))
                        for tile in args.tiles:
                            compare(
                                f"linear_{prefix}_m{m}_tm{tile}",
                                functools.partial(
                                    low_bit_matmul,
                                    weight_format="fp8",
                                    quantize_activation=True,
                                ),
                                functools.partial(fp8_linear, block_m=tile),
                                (x, w, s),
                            )
            for m in args.tokens:
                positions = array(np.arange(m, dtype=np.int32) + 8023)
                if "norm" in args.kinds:
                    x = activation((m, config.hidden))
                    weight = array(checkpoint.read_tensor("layers.2.attn_norm.weight"))
                    compare(
                        f"rms_norm_m{m}",
                        lambda x, w: numerics.rms_norm(x, w, config.eps),
                        lambda x, w: normalization.rms_norm(x, w, config.eps),
                        (x, weight),
                    )

                    def baseline_q(q, p):
                        from sgl_jax.srt.kernels.low_bit.formats import round_bf16

                        square = round_bf16(q.astype(jnp.float32) ** 2)
                        variance = round_bf16(
                            numerics._fixed_tree_mean_last(square.astype(jnp.float32))[
                                ..., None
                            ]
                        )
                        variance = round_bf16(variance.astype(jnp.float32) + config.eps)
                        inverse = round_bf16(
                            jax.lax.rsqrt(variance.astype(jnp.float32))
                        )
                        return numerics.rope(
                            round_bf16(
                                q.astype(jnp.float32) * inverse.astype(jnp.float32)
                            ),
                            p,
                            config,
                        )

                    def fused_q(q, p):
                        phase = numerics.rope_angles(p, config)
                        return normalization.qnorm_rope(
                            q, jnp.cos(phase), jnp.sin(phase), config.eps
                        )

                    compare(
                        f"qnorm_rope_m{m}",
                        baseline_q,
                        fused_q,
                        (
                            activation((m, config.heads // 4, config.head_dim)),
                            positions,
                        ),
                    )
                if "wo_a" in args.kinds:
                    loaded = checkpoint.load_linear("layers.2.attn.wo_a", shard_count=4)
                    x, w, s = (
                        activation((m, config.heads // 4, config.head_dim)),
                        array(loaded.data),
                        array(loaded.scales),
                    )

                    def baseline_wo(x, w, s, p):
                        x = numerics.rope(x, p, config, inverse=True).reshape(
                            x.shape[0], groups, -1
                        )
                        return jnp.concatenate(
                            [
                                low_bit_matmul(
                                    x[:, g],
                                    w[g * rank : (g + 1) * rank],
                                    s[g * rank // 128 : (g + 1) * rank // 128],
                                    weight_format="fp8",
                                )
                                for g in range(groups)
                            ],
                            axis=-1,
                        )

                    for tile in args.tiles:

                        def fused_wo(x, w, s, p, *, tile=tile):
                            phase = numerics.rope_angles(p, config)
                            return projections.inverse_rope_fp8_wo_a(
                                x,
                                w,
                                s,
                                jnp.cos(phase),
                                jnp.sin(phase),
                                groups=groups,
                                block_m=tile,
                            )

                        compare(
                            f"wo_a_m{m}_tm{tile}",
                            baseline_wo,
                            fused_wo,
                            (x, w, s, positions),
                        )
                if "merged" in args.kinds:
                    for target, sources in projections.MERGED_PROJECTIONS.items():
                        parts = [
                            checkpoint.load_linear("layers.2." + source)
                            for source in sources
                        ]
                        split = parts[0].logical_shape[0]
                        w, s = (
                            array(np.concatenate([part.data for part in parts])),
                            array(np.concatenate([part.scales for part in parts])),
                        )
                        x = activation((m, parts[0].logical_shape[1]))

                        def baseline_merged(x, w, s, *, split=split):
                            return jnp.concatenate(
                                (
                                    low_bit_matmul(
                                        x,
                                        w[:split],
                                        s[: split // 128],
                                        weight_format="fp8",
                                        quantize_activation=True,
                                    ),
                                    low_bit_matmul(
                                        x,
                                        w[split:],
                                        s[split // 128 :],
                                        weight_format="fp8",
                                        quantize_activation=True,
                                    ),
                                ),
                                axis=-1,
                            )

                        for tile in args.tiles:
                            compare(
                                f"merged_{target}_m{m}_tm{tile}",
                                baseline_merged,
                                functools.partial(fp8_linear, block_m=tile),
                                (x, w, s),
                            )
        report["complete"] = True
        save({"event": "complete", "cases": len(report["cases"])})
    except BaseException:
        report["error"] = traceback.format_exc()
        save({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
