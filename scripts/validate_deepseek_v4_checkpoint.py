"""Real-checkpoint numerical gates before full-model inference.

Run on the four-chip TPU host, after the synthetic operator test gate passes.
The independent oracle executes the pinned official PyTorch Attention class
with CPU implementations of its GPU-only kernels; no production JAX numerical
helpers are reused by the oracle.
"""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import numpy as np
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    attention,
    compiled_head,
    compiled_layer,
    config_for_layer,
    empty_cache,
    load_layer,
    source_fingerprint,
)
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.deepseek_v4_cpu_oracle import (
    load_module_weights,
    official_module,
    tensor_from_checkpoint,
)


def error_metrics(actual, expected):
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    if (
        actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected))
    ):
        raise AssertionError(f"shape/nonfinite mismatch: {actual.shape}, {expected.shape}")
    difference = actual - expected
    return {
        "nrmse": float(np.linalg.norm(difference) / max(float(np.linalg.norm(expected)), 1e-12)),
        "max_abs": float(np.max(np.abs(difference))),
        "shape": list(actual.shape),
    }


def attention_gate(checkpoint, mesh, report, emit):
    module, args = official_module(checkpoint)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(16)
    rng = np.random.default_rng(20260906)
    import ml_dtypes

    x = rng.normal(size=(132, args.dim)).astype(ml_dtypes.bfloat16)
    xd = jax.device_put(x, NamedSharding(mesh, P()))
    positions = jax.device_put(np.arange(len(x), dtype=np.int32), NamedSharding(mesh, P()))
    for layer_id in (0, 2, 3):
        config = config_for_layer(checkpoint.config, layer_id, 256)
        weights = load_layer(checkpoint, layer_id, mesh, include_experts=False)
        cache = jax.device_put(empty_cache(config), NamedSharding(mesh, P()))
        compute = jax.jit(
            jax.shard_map(
                functools.partial(attention, config=config),
                mesh=mesh,
                in_specs=(P(), P(), P(), P()),
                out_specs=(P(), P(), P()),
                check_vma=False,
            )
        )
        emit({"event": "real_attention_compile", "layer": layer_id, "ratio": config.ratio})
        start = time.perf_counter()
        actual, cache, trace = compute(xd, positions, weights, cache)
        jax.block_until_ready(actual)
        compile_run = time.perf_counter() - start
        official = load_module_weights(
            module.Attention(layer_id, args), checkpoint, f"layers.{layer_id}.attn."
        )
        captured = {}
        original_sparse = module.sparse_attn

        def capture_sparse(q, kv, sink, indices, scale):
            captured["q"] = q[0].float().clone().numpy()
            out = original_sparse(q, kv, sink, indices, scale)
            captured["attention_value"] = out[0].float().clone().numpy()
            return out

        module.sparse_attn = capture_sparse
        official.q_norm.register_forward_hook(
            lambda layer, inputs, output: captured.update(qr=output[0].float().clone().numpy())
        )
        original_fp4 = module.fp4_act_quant

        def capture_fp4(value, *args):
            label = "index_q_before_quant" if value.ndim == 4 else "index_compressed_before_quant"
            captured[label] = value[0].float().clone().numpy()
            result = original_fp4(value, *args)
            if value.ndim == 4:
                captured["index_q"] = value[0].float().clone().numpy()
            return result

        module.fp4_act_quant = capture_fp4
        emit({"event": "official_cpu_attention", "layer": layer_id})
        start = time.perf_counter()
        with torch.inference_mode():
            expected = (
                official(torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16)[None], 0)[0]
                .float()
                .numpy()
            )
        cpu_seconds = time.perf_counter() - start
        checks = {"output": error_metrics(actual, expected)}
        for name in captured:
            checks[name] = error_metrics(trace[name], captured[name])
        ring_ids = np.arange(len(x) - args.window_size, len(x)) % args.window_size
        checks["window_cache"] = error_metrics(
            cache["window"][len(x) - args.window_size : len(x)],
            official.kv_cache[0, ring_ids].float().numpy(),
        )
        if config.ratio:
            count = len(x) // config.ratio
            state_rows = config.ratio if config.ratio == 4 else len(x) % config.ratio
            checks["compressor_state_kv"] = error_metrics(
                cache["main.kv"][:state_rows],
                official.compressor.kv_state[0, :state_rows].float().numpy(),
            )
            checks["compressor_state_score"] = error_metrics(
                cache["main.score"][:state_rows],
                official.compressor.score_state[0, :state_rows].float().numpy(),
            )
            checks["compressed_cache"] = error_metrics(
                cache["main.compressed"][:count],
                official.kv_cache[0, args.window_size : args.window_size + count].float().numpy(),
            )
            if config.ratio == 4:
                checks["index_cache"] = error_metrics(
                    cache["index.compressed"][:count],
                    official.indexer.kv_cache[0, :count].float().numpy(),
                )
                for before, after in (
                    ("index_q_before_quant", trace["index_q"]),
                    ("index_compressed_before_quant", cache["index.compressed"][:count]),
                ):
                    identical_input = torch.from_numpy(
                        np.asarray(trace[before], np.float32).copy()
                    ).to(torch.bfloat16)
                    with torch.inference_mode():
                        original_fp4(identical_input, 32, True)
                    np.testing.assert_array_equal(
                        np.asarray(after, np.float32),
                        identical_input.float().numpy(),
                        err_msg=f"FP4 QAT must be bit-exact for identical input: {before}",
                    )
        row = {
            "layer": layer_id,
            "ratio": config.ratio,
            "prefill_tokens": len(x),
            "compile_and_run_seconds": compile_run,
            "cpu_seconds": cpu_seconds,
            "checks": checks,
        }
        report["attention"].append(row)
        emit(row)
        # BF16/QAT perturbations can amplify around quantizer boundaries. These
        # real-layer limits are explicit and separate from the unchanged 0.5%
        # synthetic GEMM and original four-template NumPy oracle thresholds.
        for name, metrics in checks.items():
            limit = 0.035 if name in ("index_q", "index_cache") else 0.025
            if metrics["nrmse"] > limit:
                raise AssertionError(f"layer {layer_id} {name}: {metrics['nrmse']:.6f} > {limit}")
        decode_checks = []
        for position in range(132, 136):
            token_x = rng.normal(size=(1, args.dim)).astype(ml_dtypes.bfloat16)
            actual, cache, _ = compute(
                jax.device_put(token_x, NamedSharding(mesh, P())),
                jax.device_put(np.array([position], np.int32), NamedSharding(mesh, P())),
                weights,
                cache,
            )
            with torch.inference_mode():
                expected = (
                    official(
                        torch.from_numpy(token_x.astype(np.float32)).to(torch.bfloat16)[None],
                        position,
                    )[0]
                    .float()
                    .numpy()
                )
            metrics = error_metrics(actual, expected)
            decode_checks.append({"position": position, **metrics})
            emit({"layer": layer_id, "decode": decode_checks[-1]})
            if metrics["nrmse"] > 0.025:
                raise AssertionError(f"layer {layer_id} decode {position}: {metrics}")
        row["decode"] = decode_checks
        if config.ratio == 128:
            # Cross an HCA emission boundary in decode, not only in prefill.
            boundary = load_module_weights(
                module.Attention(layer_id, args), checkpoint, f"layers.{layer_id}.attn."
            )
            boundary_cache = jax.device_put(empty_cache(config), NamedSharding(mesh, P()))
            _, boundary_cache, _ = compute(xd[:127], positions[:127], weights, boundary_cache)
            with torch.inference_mode():
                boundary(torch.from_numpy(x[:127].astype(np.float32)).to(torch.bfloat16)[None], 0)
            boundary_checks = []
            for position in range(127, 132):
                actual_boundary, boundary_cache, _ = compute(
                    xd[position : position + 1],
                    positions[position : position + 1],
                    weights,
                    boundary_cache,
                )
                with torch.inference_mode():
                    expected_boundary = (
                        boundary(
                            torch.from_numpy(x[position : position + 1].astype(np.float32)).to(
                                torch.bfloat16
                            )[None],
                            position,
                        )[0]
                        .float()
                        .numpy()
                    )
                metric = error_metrics(actual_boundary, expected_boundary)
                boundary_checks.append({"position": position, **metric})
                if metric["nrmse"] > 0.025:
                    raise AssertionError(f"HCA boundary decode {position}: {metric}")
            row["hca_boundary_decode"] = boundary_checks
            emit({"layer": layer_id, "hca_boundary_decode": boundary_checks})
            del boundary, boundary_cache
        module.sparse_attn, module.fp4_act_quant = original_sparse, original_fp4
        del actual, cache, weights, official, trace


def layer_gate(checkpoint, mesh, report, emit):
    module, args = official_module(checkpoint)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(16)
    ids = np.array([0, 1000, 2012, 42, 19, 501, 986, 777], np.int32)
    import ml_dtypes

    embedded = checkpoint.read_tensor("embed.weight", ids)
    streams = np.repeat(embedded[:, None, :], args.hc_mult, axis=1).astype(ml_dtypes.bfloat16)
    positions = jax.device_put(np.arange(len(ids), dtype=np.int32), NamedSharding(mesh, P()))
    device_ids = jax.device_put(ids, NamedSharding(mesh, P()))
    for layer_id in (0, 2, 3):
        config = config_for_layer(checkpoint.config, layer_id, 256)
        emit({"event": "real_layer_loading", "layer": layer_id})
        weights = load_layer(checkpoint, layer_id, mesh)
        cache = jax.device_put(empty_cache(config), NamedSharding(mesh, P()))
        start = time.perf_counter()
        emit({"event": "real_layer_compile", "layer": layer_id})
        actual, cache, trace = compiled_layer(config, mesh)(
            jax.device_put(streams, NamedSharding(mesh, P())), positions, device_ids, weights, cache
        )
        jax.block_until_ready(actual)
        compile_run = time.perf_counter() - start
        emit(
            {
                "event": "official_cpu_layer",
                "layer": layer_id,
                "compile_and_run_seconds": compile_run,
            }
        )
        official = load_module_weights(
            module.Block(layer_id, args), checkpoint, f"layers.{layer_id}."
        )
        captured = {}
        for label, submodule in (
            ("attn.norm", official.attn_norm),
            ("attn.operator", official.attn),
            ("ffn.norm", official.ffn_norm),
            ("ffn.operator", official.ffn),
        ):
            submodule.register_forward_hook(
                lambda layer, inputs, output, key=label: captured.update(
                    {key: output[0].float().clone().numpy()}
                )
            )
        official.ffn.gate.register_forward_hook(
            lambda layer, inputs, output: captured.update(
                routing_weights=output[0].float().numpy(), expert_ids=output[1].int().numpy()
            )
        )
        start = time.perf_counter()
        with torch.inference_mode():
            expected = (
                official(
                    torch.from_numpy(streams.astype(np.float32)).to(torch.bfloat16)[None],
                    0,
                    torch.from_numpy(ids)[None],
                )[0]
                .float()
                .numpy()
            )
        checks = {"output": error_metrics(actual, expected)}
        routes_match = True
        for key, value in captured.items():
            if key == "expert_ids":
                routes_match = bool(
                    np.array_equal(
                        np.sort(np.asarray(trace[key]), axis=-1), np.sort(value, axis=-1)
                    )
                )
            else:
                checks[key] = error_metrics(trace[key], value)
        row = {
            "layer": layer_id,
            "checks": checks,
            "cpu_seconds": time.perf_counter() - start,
            "compile_and_run_seconds": compile_run,
            "end_to_end_expert_routes_match": routes_match,
        }
        # Separate accumulated upstream differences from the MoE operator's
        # own correctness at discrete activation-quantization boundaries.
        with torch.inference_mode():
            local_input = torch.from_numpy(np.asarray(trace["ffn.norm"], np.float32).copy()).to(
                torch.bfloat16
            )
            local_routing, local_ids = official.ffn.gate(local_input, torch.from_numpy(ids))
            np.testing.assert_array_equal(
                np.sort(np.asarray(trace["expert_ids"]), axis=-1),
                np.sort(local_ids.int().numpy(), axis=-1),
                err_msg="identical-input expert routing must agree",
            )
            local_ffn = (
                official.ffn(
                    torch.from_numpy(np.asarray(trace["ffn.norm"], np.float32).copy()).to(
                        torch.bfloat16
                    )[None],
                    torch.from_numpy(ids)[None],
                )[0]
                .float()
                .numpy()
            )
        row["ffn_identical_input"] = error_metrics(trace["ffn.operator"], local_ffn)
        row["identical_input_routes_match"] = True
        report["layers"].append(row)
        emit(row)
        for key, metrics in checks.items():
            if key == "ffn.operator":
                continue  # Report propagated error; gate the local operator below.
            if metrics["nrmse"] > 0.025:
                raise AssertionError(f"real layer {layer_id} {key}: {metrics}")
        if row["ffn_identical_input"]["nrmse"] > 0.005:
            raise AssertionError(
                f"real layer {layer_id} identical-input MoE: {row['ffn_identical_input']}"
            )
        del weights, cache, trace, actual, official


def head_gate(checkpoint, mesh, report, emit):
    module, args = official_module(checkpoint)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(16)
    from types import SimpleNamespace

    import ml_dtypes

    value = (
        np.random.default_rng(883)
        .normal(size=(1, args.hc_mult, args.dim))
        .astype(ml_dtypes.bfloat16)
    )
    shared = {
        key: jax.device_put(checkpoint.read_tensor(key), NamedSharding(mesh, P()))
        for key in ("hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight", "head.weight")
    }
    actual = compiled_head(config_for_layer(checkpoint.config, 0, 256), mesh)(
        jax.device_put(value, NamedSharding(mesh, P())), shared
    )
    with torch.inference_mode():
        stream = torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)[None]
        collapsed = module.ParallelHead.hc_head(
            SimpleNamespace(norm_eps=args.norm_eps, hc_eps=args.hc_eps),
            stream,
            tensor_from_checkpoint(checkpoint, "hc_head_fn"),
            tensor_from_checkpoint(checkpoint, "hc_head_scale"),
            tensor_from_checkpoint(checkpoint, "hc_head_base"),
        )
        norm = load_module_weights(module.RMSNorm(args.dim, args.norm_eps), checkpoint, "norm.")
        normalized = norm(collapsed)[:, -1].float()
        expected = (
            normalized @ tensor_from_checkpoint(checkpoint, "head.weight").float().T
        ).numpy()
    metrics = error_metrics(actual, expected)
    report["head"] = metrics
    emit({"event": "official_head_check", **metrics})
    if metrics["nrmse"] > 0.005:
        raise AssertionError(f"official head mismatch: {metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--phase", choices=("attention", "layer", "all"), default="attention")
    options = parser.parse_args()
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("real checkpoint gates require four TPU devices")
    report = {
        "checkpoint": str(options.checkpoint),
        "devices": [str(d) for d in jax.devices()],
        "attention": [],
        "layers": [],
        "source_fingerprint": source_fingerprint(),
        "complete": False,
    }

    def emit(event):
        print(json.dumps(event), flush=True)
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report, indent=2) + "\n")

    try:
        mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
        checkpoint = DeepSeekV4Checkpoint(options.checkpoint)
        if options.phase in ("attention", "all"):
            attention_gate(checkpoint, mesh, report, emit)
        if options.phase in ("layer", "all"):
            layer_gate(checkpoint, mesh, report, emit)
            head_gate(checkpoint, mesh, report, emit)
        report["complete"] = True
    finally:
        emit({"event": "gate_finished", "complete": report["complete"]})


if __name__ == "__main__":
    main()
