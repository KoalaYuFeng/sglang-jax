"""Measured real-weight online vs preconverted matmul baseline, not optimization.

Only one selected matrix is preconverted at a time. Replicated mode matches the
local matrix dimensions in the EP reference runner; it excludes expert dispatch
and collectives. The same-tile BF16 baseline controls the Pallas tiling, while
the direct JAX baseline may use a different schedule. Report compiler temporary
HBM and explicitly labelled analytical traffic, not hardware-profiler traffic.
TPU host must be otherwise idle.
"""

import argparse
import functools
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.low_bit.formats import (
    activation_fp8_roundtrip,
    dequantize_fp4,
    dequantize_fp8,
)
from sgl_jax.srt.kernels.low_bit.matmul import low_bit_matmul
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def measured(compiled, args, repeats):
    jax.block_until_ready(compiled(*args))
    milliseconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = compiled(*args)
        jax.block_until_ready(result)
        milliseconds.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": float(np.median(milliseconds)),
        "p90_ms": float(np.percentile(milliseconds, 90)),
        "min_ms": min(milliseconds),
        "samples": milliseconds,
    }


def memory(compiled):
    stats = compiled.memory_analysis()
    return {
        key: getattr(stats, key)
        for key in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
            "alias_size_in_bytes",
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--layout", choices=("replicated", "output_sharded"), default="replicated")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("requires an otherwise-idle four-chip TPU host")
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    divisor = 4 if args.layout == "output_sharded" else 1
    weight_spec = P("tensor", None) if divisor == 4 else P()
    output_spec = P(None, "tensor") if divisor == 4 else P()
    ws = NamedSharding(mesh, weight_spec)
    report = {
        "checkpoint": str(args.checkpoint),
        "complete": False,
        "matrices": [],
        "layout": args.layout,
        "measurement_scope": "warmed host-dispatch-to-block_until_ready on four chips",
        "excluded_costs": "expert dispatch, EP collectives, attention, complete-model scheduling",
        "baseline_scope": "same Pallas tile plus independent direct JAX matmul; both retain A8 QAT",
        "traffic_scope": "analytical Pallas input loads; not measured memory traffic",
    }

    def save(event):
        print(json.dumps(event), flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for prefix in (
            "layers.2.ffn.experts.0.w1",
            "layers.2.ffn.experts.0.w2",
            "layers.2.attn.wq_a",
            "layers.2.attn.wo_b",
        ):
            loaded = checkpoint.load_linear(prefix)
            raw, scales = jax.device_put(loaded.data, ws), jax.device_put(loaded.scales, ws)
            dequant_fn = dequantize_fp4 if loaded.weight_format == "fp4" else dequantize_fp8
            dq = jax.jit(
                jax.shard_map(
                    dequant_fn,
                    mesh=mesh,
                    in_specs=(weight_spec, weight_spec),
                    out_specs=weight_spec,
                    check_vma=False,
                )
            )
            dq_compiled = dq.lower(raw, scales).compile()
            expanded = dq_compiled(raw, scales).block_until_ready()
            dq_time = measured(dq_compiled, (raw, scales), args.repeats)
            n, k = loaded.logical_shape
            for tokens in (1, 128):
                x = (
                    np.random.default_rng(821 + tokens)
                    .normal(size=(tokens, k))
                    .astype(ml_dtypes.bfloat16)
                )
                x = jax.device_put(x, NamedSharding(mesh, P()))
                online_fn = functools.partial(
                    low_bit_matmul, weight_format=loaded.weight_format, quantize_activation=True
                )
                online = jax.jit(
                    jax.shard_map(
                        online_fn,
                        mesh=mesh,
                        in_specs=(P(), weight_spec, weight_spec),
                        out_specs=output_spec,
                        check_vma=False,
                    )
                )

                def direct(x, weight):
                    return jnp.matmul(
                        activation_fp8_roundtrip(x), weight.T, preferred_element_type=jnp.float32
                    ).astype(jnp.bfloat16)

                baseline = jax.jit(
                    jax.shard_map(
                        direct,
                        mesh=mesh,
                        in_specs=(P(), weight_spec),
                        out_specs=output_spec,
                        check_vma=False,
                    )
                )

                def same_tile(x, weight):
                    return low_bit_matmul(
                        x, weight, None, weight_format="bf16", quantize_activation=True
                    )

                tiled = jax.jit(
                    jax.shard_map(
                        same_tile,
                        mesh=mesh,
                        in_specs=(P(), weight_spec),
                        out_specs=output_spec,
                        check_vma=False,
                    )
                )
                online_compiled = online.lower(x, raw, scales).compile()
                baseline_compiled = baseline.lower(x, expanded).compile()
                tiled_compiled = tiled.lower(x, expanded).compile()
                actual = np.asarray(online_compiled(x, raw, scales), np.float32)
                expected = np.asarray(baseline_compiled(x, expanded), np.float32)
                tiled_result = np.asarray(tiled_compiled(x, expanded), np.float32)
                nrmse = float(
                    np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
                )
                tiled_nrmse = float(
                    np.linalg.norm(actual - tiled_result) / max(np.linalg.norm(tiled_result), 1e-12)
                )
                if (
                    not all(np.all(np.isfinite(v)) for v in (actual, expected, tiled_result))
                    or max(nrmse, tiled_nrmse) > 0.005
                ):
                    raise AssertionError(
                        f"online/preconverted correctness failed: {prefix}, {tokens}, "
                        f"{nrmse}, {tiled_nrmse}"
                    )
                local_n = n // divisor
                bm, bn = 8, 128
                m_tiles, n_tiles = math.ceil(tokens / bm), math.ceil(local_n / bn)
                scale_load = (
                    loaded.scales.nbytes
                    // divisor
                    * (1 if loaded.weight_format == "fp4" else n_tiles)
                )
                row = {
                    "prefix": prefix,
                    "format": loaded.weight_format,
                    "shape": [n, k],
                    "local_weight_shape": [local_n, k],
                    "tokens": tokens,
                    "nrmse_online_vs_preconverted": nrmse,
                    "nrmse_online_vs_same_tile_preconverted": tiled_nrmse,
                    "checkpoint_unique_bytes": loaded.nbytes,
                    "checkpoint_physical_bytes_all_devices": loaded.nbytes * 4 // divisor,
                    "bf16_unique_bytes": n * k * 2,
                    "bf16_physical_bytes_all_devices": n * k * 2 * 4 // divisor,
                    "online": measured(online_compiled, (x, raw, scales), args.repeats),
                    "preconverted_same_tile": measured(tiled_compiled, (x, expanded), args.repeats),
                    "preconverted_direct": measured(baseline_compiled, (x, expanded), args.repeats),
                    "standalone_dequant": dq_time,
                    "online_compiler_hbm_per_device": memory(online_compiled),
                    "direct_compiler_hbm_per_device": memory(baseline_compiled),
                    "same_tile_compiler_hbm_per_device": memory(tiled_compiled),
                    "dequant_compiler_hbm_per_device": memory(dq_compiled),
                    "estimated_online_weight_and_scale_load_bytes_per_device": m_tiles
                    * (loaded.data.nbytes // divisor + scale_load),
                    "estimated_online_activation_load_bytes_per_device": m_tiles
                    * n_tiles
                    * bm
                    * k
                    * 2,
                    "logical_bf16_weight_tile_bytes_in_vmem": bn * k * 2,
                    "dequantized_weight_values_per_device": m_tiles * local_n * k,
                }
                report["matrices"].append(row)
                save(row)
            del raw, scales, expanded, loaded
        report["complete"] = True
    finally:
        save({"event": "benchmark_finished", "complete": report["complete"]})


if __name__ == "__main__":
    main()
