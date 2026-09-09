"""Controlled FP4 MoE diagnostics; no production backend or default is changed.

Use one real layer, captured routes/activations and all 256 official experts on
EP4. Compare the unchanged production call with a named diagnostic copy, then
move A8 roundtrip outside GMM and/or predecode this ONE layer's weights to BF16.
All comparisons must be bitwise. Counterfactual deltas are NOT additive stage
times: hoisting changes scheduling and predecoding increases weight DMA bytes.
Loading, predecoding, compilation, oracle and profiler overhead are excluded
from alternating warm timings. M4 contains four captured tokens, not a serving
batch; M128 is routed MoE only, not end-to-end prefill.
"""

import argparse
import hashlib
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4.moe_gmm import gmm_fp4_experts, pack_routes
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip, dequantize_fp4
from sgl_jax.srt.kernels.low_bit.gmm import CheckpointFP4Rhs
from sgl_jax.srt.model_executor.deepseek_v4_reference import source_fingerprint
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer
from validate_deepseek_v4_moe_gmm import captured_array, numpy_one_token


@dataclass(frozen=True)
class PredecodedRhs(CheckpointFP4Rhs):
    """Diagnostic BF16[N,K] storage, same driver/tile/dot/rounding as FP4."""

    name = "diagnostic_predecoded_bf16"

    def logical_shape(self, rhs):
        experts, n, k = rhs.shape
        return experts, k, n

    def validate(self, *, lhs, rhs, group_sizes, rhs_scale, rhs_bias):
        if lhs.dtype != jnp.bfloat16 or rhs.dtype != jnp.bfloat16:
            raise ValueError("diagnostic expects BF16 inputs and weights")
        if rhs.shape[2] != lhs.shape[1] or rhs_scale.shape != (rhs.shape[0], 1, 1):
            raise ValueError("invalid predecoded shape/dummy scale")
        if rhs_bias is not None or group_sizes.dtype != jnp.int32:
            raise ValueError("invalid grouped arguments")

    def block_specs(self, *, tk, tn, indices):
        def weight_index(*args):
            expert, k_tile, n_tile = indices(*args)
            return expert, n_tile, k_tile

        def scale_index(*args):
            expert, _, _ = indices(*args)
            return expert, 0, 0

        return pl.BlockSpec((None, tn, tk), weight_index), pl.BlockSpec(
            (None, 1, 1), scale_index
        )

    def dot(self, lhs, rhs, scales):
        del scales
        if self.quantize_activation:
            lhs = activation_fp8_roundtrip(lhs)
        return jax.lax.dot_general(
            lhs, rhs, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )


def diagnostic_moe(*values, predecoded=False, hoisted=False, limit=10.0, adapter=None):
    """Copy only the thin V4 glue; grouping and GMM are the production helpers."""
    x, w1, w3, w2, s1, s3, s2, ids, weights = values
    with jax.named_scope("DIAG_ROUTE_PACK_AND_GATHER"):
        permutation, sizes, sorted_ids, mixing = pack_routes(ids, weights, 256)
        rows = permutation // ids.shape[1]
        valid = (sorted_ids < 256) & (rows < x.shape[0])
        inputs = jnp.where(valid[:, None], x[jnp.minimum(rows, x.shape[0] - 1)], 0)
        first = jax.lax.axis_index("tensor") * w1.shape[0]
    if hoisted:
        with jax.named_scope("DIAG_GATE_UP_ACTIVATION_QAT_ONCE"):
            inputs = activation_fp8_roundtrip(inputs)
    if adapter is None:
        adapter = (PredecodedRhs if predecoded else CheckpointFP4Rhs)(
            quantize_activation=not hoisted
        )
    elif predecoded or hoisted:
        raise ValueError(
            "explicit candidate adapter cannot be combined with counterfactual flags"
        )
    tile_m, tile_n = getattr(adapter, "tile_m", 8), getattr(adapter, "tile_n", 128)
    padding = (-inputs.shape[0]) % tile_m
    if padding:
        inputs = jnp.pad(inputs, ((0, padding), (0, 0)))
        mixing = jnp.pad(mixing, (0, padding))
        sizes = sizes.at[-1].add(padding)

    def project(value, weight, scale, label):
        with jax.named_scope(label):
            return gmm(
                value,
                weight,
                sizes,
                preferred_element_type=jnp.bfloat16,
                rhs_scale=scale,
                tiling=(tile_m, value.shape[1], tile_n),
                group_offset=first,
                rhs_adapter=adapter,
            )

    gate = jnp.minimum(
        project(inputs, w1, s1, "DIAG_W1_GATE").astype(jnp.float32), limit
    )
    up = jnp.clip(
        project(inputs, w3, s3, "DIAG_W3_UP").astype(jnp.float32), -limit, limit
    )
    with jax.named_scope("DIAG_SWIGLU_AND_ROUTE_SCALE"):
        hidden = (mixing[:, None] * jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    if hoisted:
        with jax.named_scope("DIAG_DOWN_ACTIVATION_QAT_ONCE"):
            hidden = activation_fp8_roundtrip(hidden)
    projected = project(hidden, w2, s2, "DIAG_W2_DOWN").astype(jnp.float32)
    with jax.named_scope("DIAG_UNPERMUTE_AND_LOCAL_SUM"):
        inverse = (
            jnp.zeros_like(permutation)
            .at[permutation]
            .set(jnp.arange(permutation.size))
        )
        unsorted = projected[inverse][: ids.size].reshape(*ids.shape, x.shape[1])
        ordered = jnp.take_along_axis(
            unsorted, jnp.argsort(ids, axis=-1, stable=True)[..., None], axis=1
        )
        local = jnp.zeros(x.shape, jnp.float32)
        for choice in range(ids.shape[1]):
            local = local + ordered[:, choice]
    with jax.named_scope("DIAG_EP_ALL_REDUCE_INCLUDING_WAIT"):
        return jax.lax.psum(local, "tensor")


def route_geometry(ids, mixing):
    """Logical tile work from shared GMM group boundaries, NOT HW counters."""
    sizes = np.zeros(257, np.int64)
    for row_ids, row_mix in zip(ids, mixing, strict=True):
        for expert in set(row_ids.tolist()):
            if 0 <= expert < 256 and np.sum(row_mix[row_ids == expert]) != 0:
                sizes[expert] += 1
    sizes[-1] = (ids.size + 7) // 8 * 8 - int(sizes.sum())
    ends = np.cumsum(sizes)
    starts = ends - sizes
    tiles = np.where(sizes > 0, (ends + 7) // 8 - starts // 8, 0)
    chips = []
    for chip in range(4):
        local = slice(chip * 64, (chip + 1) * 64)
        tile_count, routes = int(tiles[local].sum()), int(sizes[local].sum())
        chips.append(
            {
                "chip": chip,
                "active_experts": int(np.count_nonzero(sizes[local])),
                "active_routes": routes,
                "m8_group_tiles_per_projection": tile_count,
                "useful_m_rows_fraction": routes / (8 * tile_count)
                if tile_count
                else None,
                "logical_fp4_and_scale_weight_tile_bytes": tile_count
                * 3
                * 2048
                * 4096
                * 17
                // 32,
                "logical_bf16_weight_tile_bytes": tile_count * 3 * 2048 * 4096 * 2,
                "logical_m8_bf16_dot_flops": tile_count * 3 * 2 * 8 * 2048 * 4096,
                "in_tile_qat_bf16_values": tile_count * 8 * (2 * 16 * 4096 + 32 * 2048),
            }
        )
    return {
        "scope": "static logical work, excludes DMA reuse, MXU internal padding and HW overlap",
        "group_sizes": sizes.tolist(),
        "chips": chips,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-prefix", type=Path, required=True)
    parser.add_argument("--correctness-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    fixture = json.loads(args.correctness_report.read_text())
    if (
        not fixture.get("complete")
        or fixture["framework_source_fingerprint"] != framework_fingerprint()
        or Path(fixture["checkpoint"]).resolve() != args.checkpoint.resolve()
        or fixture["capture"] != str(args.capture_prefix)
        or fixture["layer"] != args.layer
        or args.repeats < 10
    ):
        raise ValueError(
            "requires matching accepted real-MoE evidence and >=10 repeats"
        )
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("requires exclusive use of the four physical TPU chips")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "scope": __doc__,
        "source_fingerprint": source_fingerprint(),
        "framework_source_fingerprint": framework_fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "capture": str(args.capture_prefix),
        "correctness_report_sha256": hashlib.sha256(
            args.correctness_report.read_bytes()
        ).hexdigest(),
        "jax_version": jax.__version__,
        "devices": [str(d) + ": " + d.device_kind for d in jax.devices()],
        "checks": [],
        "runs": {},
        "captures": [],
    }

    def save(event):
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    def check(label, actual, expected):
        metrics = compare_arrays(expected, actual)
        metrics["all_finite"] = bool(
            np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
        )
        report["checks"].append({"label": label, **metrics})
        if not metrics["bitwise_equal"] or not metrics["all_finite"]:
            raise AssertionError(
                f"diagnostic changed numerical results: {label}: {metrics}"
            )

    try:
        selected = np.asarray(fixture["capture_rows"], np.int32)
        if len(selected) != 128:
            raise ValueError("expected accepted 128-row capture selection")
        x, ids, mixing = (
            captured_array(args.capture_prefix, key)[selected]
            for key in ("ffn.norm", "expert_ids", "routing_weights")
        )
        report["capture_rows"] = selected.tolist()
        checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
        limit = checkpoint.config["swiglu_limit"]
        mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
        spec = P("tensor", None, None)
        specs = (P(), *(spec for _ in range(6)), P(), P())
        with jax.set_mesh(mesh):
            loaded = load_layer(checkpoint, args.layer, mesh)
            raw = tuple(
                loaded["experts." + key] for key in ("w1", "w3", "w2", "s1", "s3", "s2")
            )
            dequant = jax.jit(
                jax.shard_map(
                    dequantize_fp4,
                    mesh=mesh,
                    in_specs=(spec, spec),
                    out_specs=spec,
                    check_vma=False,
                )
            )
            start = time.perf_counter()
            expanded = tuple(
                jax.block_until_ready(dequant(w, s))
                for w, s in zip(raw[:3], raw[3:], strict=True)
            )
            report["excluded_one_layer_predecode_compile_and_execute_seconds"] = (
                time.perf_counter() - start
            )
            dummy = jax.device_put(
                np.zeros((256, 1, 1), np.uint8), NamedSharding(mesh, spec)
            )
            bf16 = (*expanded, dummy, dummy, dummy)
            report["storage_per_chip_bytes"] = {
                "raw_weights_and_scales": sum(
                    w.addressable_shards[0].data.nbytes for w in raw
                ),
                "diagnostic_bf16_weights": sum(
                    w.addressable_shards[0].data.nbytes for w in expanded
                ),
            }
            save({"event": "one_layer_loaded_and_predecoded"})
            for tokens in (1, 4, 128):
                functions = {
                    "production": lambda *v: gmm_fp4_experts(*v, swiglu_limit=limit),
                    "fp4_named_control": lambda *v: diagnostic_moe(*v, limit=limit),
                    "fp4_qat_once": lambda *v: diagnostic_moe(
                        *v, hoisted=True, limit=limit
                    ),
                    "bf16_in_tile_qat": lambda *v: diagnostic_moe(
                        *v, predecoded=True, limit=limit
                    ),
                    "bf16_qat_once": lambda *v: diagnostic_moe(
                        *v, predecoded=True, hoisted=True, limit=limit
                    ),
                }
                row = {
                    "geometry": route_geometry(ids[:tokens], mixing[:tokens]),
                    "variants": {},
                }
                report["runs"][str(tokens)] = row
                calls, inputs, expected = {}, {}, None
                for name, fn in functions.items():
                    weights = bf16 if name.startswith("bf16") else raw
                    values = (x[:tokens], *weights, ids[:tokens], mixing[:tokens])
                    inputs[name] = tuple(
                        jax.device_put(v, NamedSharding(mesh, s))
                        for v, s in zip(values, specs, strict=True)
                    )
                    run = jax.jit(
                        jax.shard_map(
                            fn,
                            mesh=mesh,
                            in_specs=specs,
                            out_specs=P(),
                            check_vma=False,
                        )
                    )
                    start = time.perf_counter()
                    calls[name] = run.lower(*inputs[name]).compile()
                    actual = np.asarray(calls[name](*inputs[name]))
                    if expected is None:
                        expected = actual
                        if tokens == 1:
                            oracle = numpy_one_token(
                                checkpoint,
                                args.layer,
                                x[:1].astype(np.float32),
                                ids[:1],
                                mixing[:1],
                                limit,
                            )
                            check("production_M1_independent_numpy", actual, oracle)
                    check(f"M{tokens}/{name}", actual, expected)
                    hlo = calls[name].as_text()
                    (args.output / f"M{tokens}-{name}.hlo.txt").write_text(hlo)
                    stats = calls[name].memory_analysis()
                    row["variants"][name] = {
                        "compile_and_first_check_seconds": time.perf_counter() - start,
                        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
                        "compiler_temp_hbm_bytes_per_chip": stats.temp_size_in_bytes,
                        "samples_ms": [],
                    }
                    save(
                        {
                            "event": "compiled_and_bitwise_checked",
                            "tokens": tokens,
                            "variant": name,
                        }
                    )
                names = list(calls)
                for index in range(args.repeats):
                    order = names[index % len(names) :] + names[: index % len(names)]
                    for name in order:
                        start = time.perf_counter()
                        jax.block_until_ready(calls[name](*inputs[name]))
                        row["variants"][name]["samples_ms"].append(
                            (time.perf_counter() - start) * 1000
                        )
                for name, values in row["variants"].items():
                    values.update(
                        p50_ms=float(np.median(values["samples_ms"])),
                        p95_ms=float(np.percentile(values["samples_ms"], 95)),
                    )
                save(
                    {
                        "event": "timings_complete",
                        "tokens": tokens,
                        "p50_ms": {n: v["p50_ms"] for n, v in row["variants"].items()},
                    }
                )
                if args.profile:
                    for name in (
                        "production",
                        "fp4_named_control",
                        "fp4_qat_once",
                        "bf16_qat_once",
                    ):
                        label = f"M{tokens}_{name}"
                        options = jax.profiler.ProfileOptions()
                        options.host_tracer_level = 2
                        options.python_tracer_level = 0
                        options.enable_hlo_proto = True
                        options.raise_error_on_start_failure = True
                        options.advanced_configuration = {
                            "tpu_trace_mode": "TRACE_ONLY_XLA",
                            "tpu_num_chips_to_profile_per_task": 4,
                            "tpu_num_sparse_cores_to_trace": 0,
                            "tpu_perf_counters": True,
                        }
                        seconds = []
                        jax.profiler.start_trace(
                            str(args.output / "traces" / label),
                            profiler_options=options,
                        )
                        try:
                            for step in range(8):
                                start = time.perf_counter()
                                # Reuse the exporter's recognized timing markers;
                                # report scope/labels identify this isolated MoE.
                                with jax.profiler.StepTraceAnnotation(
                                    "V4_PREFILL" if tokens == 128 else "V4_DECODE",
                                    step_num=step,
                                ):
                                    result = jax.block_until_ready(
                                        calls[name](*inputs[name])
                                    )
                                seconds.append(time.perf_counter() - start)
                        finally:
                            jax.profiler.stop_trace()
                        check(label + "/profile_replay", np.asarray(result), expected)
                        report["captures"].append(
                            {"label": label, "seconds": seconds, "chips": 4}
                        )
                        save({"event": "capture_complete", "label": label})
            report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        save({"event": "finished", "complete": report["complete"]})


if __name__ == "__main__":
    main()
