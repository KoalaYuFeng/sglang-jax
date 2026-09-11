"""Read-only instrumentation of a production layer; independent CPU stage checks.

Tracing must reproduce the previously saved uninstrumented output bitwise.
No runtime sources, oracle arithmetic, or historical thresholds are changed.
"""

import argparse
import importlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_numerical_acceptance import tensor_metrics
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.kernels.deepseek_v4.dense import DenseKernels
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer, weight_specs
from sgl_jax.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, attention_uses_tp
from sgl_jax.test.deepseek_v4_cpu_oracle import load_module_weights, official_module
from sgl_jax.test.test_deepseek_v4_paged import make_batch, make_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    options = parser.parse_args()
    prior = json.loads((options.trajectory / "report.json").read_text())
    if fingerprint() != prior["source_fingerprint"]:
        raise ValueError("runtime fingerprint changed")
    options.output.mkdir(exist_ok=False)
    saved = np.load(options.trajectory / f"layer-{options.layer:02d}.npz")
    ids = np.load(options.trajectory / "input_ids.npy")
    chunk, count, context = prior["native_chunk"], prior["prefill"], prior["context"]
    cp = DeepSeekV4Checkpoint(prior["checkpoint"])
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    module, args = official_module(cp, context)
    cfg = config_for_layer(cp.config, options.layer, context)
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    if len(jax.devices()) != 4 or jax.default_backend() != "tpu":
        raise RuntimeError("requires four TPU chips")
    put = lambda x: jax.device_put(x, NamedSharding(mesh, P()))
    dense = DenseKernels(
        fp8_backend="gmm", fused_norm=True, merged_projections=True, fused_wo_a=True
    )
    ap = attention_uses_tp(cfg, True)
    weights = load_layer(
        cp,
        options.layer,
        mesh,
        attention_tp=ap,
        merged_projections=True,
        fp4_scale_transposed=True,
    )
    model_module = importlib.import_module("sgl_jax.srt.models.deepseek_v4")
    moe_module = importlib.import_module("sgl_jax.srt.kernels.deepseek_v4.moe")
    original_attn, original_moe = model_module.attention, model_module.moe
    original_pre, original_post = model_module.mhc_pre_fused, model_module.mhc_post
    original_route = moe_module.route

    def run(x, positions, token_ids, w, cache, metadata, locations):
        trace = {}
        phase = ["attn"]

        def pre(*a, **kw):
            result = original_pre(*a, **kw)
            for name, value in zip(
                ("pre", "post_weights", "comb"), result, strict=True
            ):
                trace[phase[0] + "." + name] = value
            return result

        def post(*a, **kw):
            value = original_post(*a, **kw)
            trace[phase[0] + ".post"] = value
            phase[0] = "ffn"
            return value

        def attn(x, *a, **kw):
            trace["attn.norm"] = x
            value, cache = original_attn(x, *a, **kw)
            trace["attn.operator"] = value
            return value, cache

        def moe(x, *a, **kw):
            trace["ffn.norm"] = x
            value = original_moe(x, *a, **kw)
            trace["ffn.operator"] = value
            return value

        def route(*a, **kw):
            indices, routing = original_route(*a, **kw)
            trace.update(expert_ids=indices, routing_weights=routing)
            return indices, routing

        with ExitStack() as stack:
            for target, key, fn in (
                (model_module, "attention", attn),
                (model_module, "moe", moe),
                (model_module, "mhc_pre_fused", pre),
                (model_module, "mhc_post", post),
                (moe_module, "route", route),
            ):
                stack.enter_context(patch.object(target, key, fn))
            out, cache = DeepseekV4DecoderLayer(
                cfg, attention_tp=True, moe_backend="gmm_tuned", dense_kernels=dense
            )(x, positions, token_ids, w, cache, metadata, locations)
        return out, cache, trace

    compute = jax.jit(
        jax.shard_map(
            run,
            mesh=mesh,
            in_specs=(
                P(),
                P(),
                P(),
                weight_specs(weights, attention_tp=ap),
                P(),
                P(),
                P(),
            ),
            out_specs=(P(), P(), P()),
            check_vma=False,
        )
    )
    backend = V4PagedBackend(max_context=context, mesh=mesh)
    print("native trace", flush=True)
    cache = put(make_cache(cfg, capacity=context + 128, requests=1))
    trace_rows = []
    for begin in range(0, count, chunk):
        end = min(begin + chunk, count)
        padding = chunk - (end - begin)
        batch = make_batch(
            [begin],
            [end - begin],
            pages=[list(range(1, 1 + context // 128))],
            padding=padding,
        )
        value = jnp.pad(
            jnp.asarray(saved["native_input"][begin:end], jnp.bfloat16),
            ((0, padding), (0, 0), (0, 0)),
        )
        out, cache, device_trace = compute(
            put(value),
            put(batch.positions),
            put(np.pad(ids[begin:end], (0, padding))),
            weights,
            cache,
            backend.get_forward_metadata(batch),
            put(batch.out_cache_loc),
        )
        actual = np.asarray(out, np.float32)[: end - begin]
        np.testing.assert_array_equal(
            actual,
            saved["native_output"][begin:end],
            err_msg="instrumentation changed output; do not interpret stages",
        )
        trace_rows.append(
            {
                k: np.asarray(v, np.float32)[: end - begin]
                for k, v in device_trace.items()
            }
        )
    trace = {k: np.concatenate([r[k] for r in trace_rows]) for k in trace_rows[0]}
    np.savez_compressed(options.output / "native.npz", **trace)
    print("official CPU stages", flush=True)
    official = load_module_weights(
        module.Block(options.layer, args), cp, f"layers.{options.layer}."
    )
    captured = {}
    for label, submodule in (
        ("attn.norm", official.attn_norm),
        ("attn.operator", official.attn),
        ("ffn.norm", official.ffn_norm),
        ("ffn.operator", official.ffn),
    ):
        submodule.register_forward_hook(
            lambda m, i, o, key=label: captured.update(
                {key: o[0].float().clone().numpy()}
            )
        )
    official.ffn.gate.register_forward_hook(
        lambda m, i, o: captured.update(
            routing_weights=o[0].float().clone().numpy(),
            expert_ids=o[1].int().clone().numpy(),
        )
    )
    with torch.inference_mode():
        cpu_out = (
            official(
                torch.from_numpy(saved["cpu_input"][: prior["prefill"]]).to(
                    torch.bfloat16
                )[None],
                0,
                torch.from_numpy(ids[: prior["prefill"]])[None],
            )[0]
            .float()
            .numpy()
        )
    np.testing.assert_array_equal(
        cpu_out,
        saved["cpu_output"][: prior["prefill"]],
        err_msg="CPU baseline not reproduced",
    )
    cpu_trace = {k: v[:count].copy() for k, v in captured.items()}
    report = {
        "source_fingerprint": fingerprint(),
        "layer": options.layer,
        "tokens": count,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "traced_output_bitwise": True,
        "cpu_baseline_bitwise": True,
        "stages": {
            k: tensor_metrics(trace[k], v)
            for k, v in cpu_trace.items()
            if k != "expert_ids"
        },
        "end_to_end_routes_equal": bool(
            np.array_equal(
                np.sort(trace["expert_ids"]), np.sort(cpu_trace["expert_ids"])
            )
        ),
    }
    # A separate local operator control: feed precisely the native MoE input to CPU.
    print("CPU same-input MoE", flush=True)
    with torch.inference_mode():
        local = (
            official.ffn(
                torch.from_numpy(trace["ffn.norm"].copy()).to(torch.bfloat16)[None],
                torch.from_numpy(ids[:count])[None],
            )[0]
            .float()
            .numpy()
        )
    report["same_input_moe"] = tensor_metrics(trace["ffn.operator"], local)
    report["same_input_routes_equal"] = bool(
        np.array_equal(np.sort(trace["expert_ids"]), np.sort(captured["expert_ids"]))
    )
    report["same_input_routing_weights"] = tensor_metrics(
        trace["routing_weights"], captured["routing_weights"]
    )
    np.savez_compressed(
        options.output / "official.npz", **cpu_trace, same_input_moe=local
    )
    assert fingerprint() == prior["source_fingerprint"]
    (options.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
