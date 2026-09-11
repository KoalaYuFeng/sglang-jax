"""Attribute frozen SWA attention residuals without modifying serving code.

Uses real production attention, an unchanged independent official CPU oracle,
and same-input local controls. Instrumented output must match the saved actual
DecoderLayer output bitwise. Measurements do not replace historical gates.
"""

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_attention_diagnostics import align_masked_indices
from deepseek_v4_numerical_acceptance import tensor_metrics
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.kernels.deepseek_v4 import projections
from sgl_jax.srt.kernels.deepseek_v4.attention import attention
from sgl_jax.srt.kernels.deepseek_v4.dense import DenseKernels
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer, weight_specs
from sgl_jax.test.deepseek_v4_cpu_oracle import load_module_weights, official_module
from sgl_jax.test.test_deepseek_v4_paged import make_batch, make_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    prior = json.loads((options.trajectory / "report.json").read_text())
    stages = json.loads((options.stages / "report.json").read_text())
    if (
        fingerprint() != prior["source_fingerprint"]
        or fingerprint() != stages["source_fingerprint"]
    ):
        raise ValueError("runtime changed since the frozen captures")
    options.output.mkdir(exist_ok=False)
    layer = stages["layer"]
    cp = DeepSeekV4Checkpoint(prior["checkpoint"])
    count, chunk, context = prior["prefill"], prior["native_chunk"], prior["context"]
    cfg = config_for_layer(cp.config, layer, context)
    if cfg.ratio != 0:
        raise ValueError(
            "this bounded diagnostic supports the replicated SWA layers only"
        )
    frozen = np.load(options.stages / "native.npz")
    x = frozen["attn.norm"]
    if x.shape[0] != count:
        raise ValueError("prefill capture length mismatch")
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("requires four TPU chips")
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    put = lambda a: jax.device_put(a, NamedSharding(mesh, P()))
    dense = DenseKernels(
        fp8_backend="gmm", fused_norm=True, merged_projections=True, fused_wo_a=True
    )
    weights = load_layer(
        cp, layer, mesh, include_experts=False, merged_projections=True
    )
    original_linear, original_merged, original_wo_a = (
        DenseKernels.linear,
        projections.merged_linear,
        projections.inverse_rope_fp8_wo_a,
    )

    def run(value, positions, w, cache, metadata, locations):
        trace = {}

        def linear(self, value, w, prefix, **kw):
            out = original_linear(self, value, w, prefix, **kw)
            trace[prefix + ".input"], trace[prefix + ".output"] = value, out
            return out

        def merged(*a, **kw):
            qa, kv = original_merged(*a, **kw)
            trace.update(qa=qa, raw_kv=kv)
            return qa, kv

        def wo_a(*a, **kw):
            projected = original_wo_a(*a, **kw)
            trace["projected"] = projected
            return projected

        with ExitStack() as stack:
            for target, key, fn in (
                (DenseKernels, "linear", linear),
                (projections, "merged_linear", merged),
                (projections, "inverse_rope_fp8_wo_a", wo_a),
            ):
                stack.enter_context(patch.object(target, key, fn))
            out, cache = attention(
                value,
                positions,
                w,
                cache,
                cfg,
                metadata,
                locations,
                trace=trace,
                hca_backend="pallas",
                csa_backend="pallas",
                dense_kernels=dense,
            )
        trace["output"] = out
        return out, cache, trace

    compute = jax.jit(
        jax.shard_map(
            run,
            mesh=mesh,
            in_specs=(P(), P(), weight_specs(weights), P(), P(), P()),
            out_specs=(P(), P(), P()),
            check_vma=False,
        )
    )
    backend = V4PagedBackend(max_context=context, mesh=mesh)
    cache = put(make_cache(cfg, capacity=context + 128, requests=1))
    rows = []
    print("native attention stages", flush=True)
    for begin in range(0, count, chunk):
        end = min(begin + chunk, count)
        padding = chunk - end + begin
        batch = make_batch(
            [begin],
            [end - begin],
            pages=[list(range(1, 1 + context // 128))],
            padding=padding,
        )
        value = jnp.pad(jnp.asarray(x[begin:end], jnp.bfloat16), ((0, padding), (0, 0)))
        out, cache, trace = compute(
            put(value),
            put(batch.positions),
            weights,
            cache,
            backend.get_forward_metadata(batch),
            put(batch.out_cache_loc),
        )
        np.testing.assert_array_equal(
            np.asarray(out, np.float32)[: end - begin],
            frozen["attn.operator"][begin:end],
            err_msg="instrumentation changed production output",
        )
        rows.append(
            {k: np.asarray(v, np.float32)[: end - begin] for k, v in trace.items()}
        )
    native = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(options.output / "native.npz", **native)
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    module, args = official_module(cp, context)
    official = load_module_weights(
        module.Attention(layer, args), cp, f"layers.{layer}.attn."
    )
    captured = {}

    def array(t):
        return t.detach().float().clone().numpy()

    def tx(a):
        return torch.from_numpy(np.array(a, dtype=np.float32)).to(torch.bfloat16)

    handles = []
    for label, obj in (
        ("qa", official.wq_a),
        ("raw_kv", official.wkv),
        ("qr", official.q_norm),
        ("attn.wq_b.output", official.wq_b),
    ):
        handles.append(
            obj.register_forward_hook(
                lambda m, i, o, key=label: captured.update({key: array(o[0])})
            )
        )
    handles.append(
        official.wo_b.register_forward_pre_hook(
            lambda m, i: captured.update(projected=array(i[0][0]))
        )
    )
    sparse = module.sparse_attn

    def capture_sparse(q, kv, sink, indices, scale):
        captured.update(q=array(q[0]), kv=array(kv[0]), indices=array(indices[0]))
        result = sparse(q, kv, sink, indices, scale)
        captured["attention_value"] = array(result[0])
        return result

    print("official CPU attention stages", flush=True)
    with torch.inference_mode(), patch.object(module, "sparse_attn", capture_sparse):
        captured["output"] = array(official(tx(x)[None], 0)[0])
    for handle in handles:
        handle.remove()
    # Physical page 1 starts at 128. This fixture never wraps its SWA window.
    logical_indices = np.where(
        native["indices"] >= 0, native["indices"] - 128, -1
    ).astype(np.int32)
    canonical_native, canonical_cpu = align_masked_indices(
        logical_indices, captured["indices"]
    )
    np.testing.assert_array_equal(
        canonical_native, canonical_cpu, err_msg="attention key order/mask differs"
    )
    checks = {
        k: tensor_metrics(native[k], v) for k, v in captured.items() if k != "indices"
    }
    local, same_input = {}, {}

    def record(key, actual_key, value):
        local[key] = array(value)
        same_input[key] = tensor_metrics(native[actual_key], local[key])

    print("same-input CPU suboperators", flush=True)
    with torch.inference_mode():
        record("q_norm", "qr", official.q_norm(tx(native["qa"])))
        record("wq_b", "attn.wq_b.output", official.wq_b(tx(native["qr"])))
        q = tx(native["attn.wq_b.output"]).reshape(
            1, count, official.n_local_heads, official.head_dim
        )
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + official.eps)
        module.apply_rotary_emb(
            q[..., -official.rope_head_dim :], official.freqs_cis[:count]
        )
        record("qnorm_rope", "q", q[0])
        kv = official.kv_norm(tx(native["raw_kv"]))[None]
        module.apply_rotary_emb(
            kv[..., -official.rope_head_dim :], official.freqs_cis[:count]
        )
        module.act_quant(
            kv[..., : -official.rope_head_dim],
            64,
            module.scale_fmt,
            module.scale_dtype,
            True,
        )
        record("kv_norm_rope_qat", "kv", kv[0])
        value = sparse(
            tx(native["q"])[None],
            tx(native["kv"])[None],
            official.attn_sink,
            torch.from_numpy(logical_indices)[None],
            official.softmax_scale,
        )
        record("sparse_attention", "attention_value", value[0])
        value = tx(native["attention_value"])[None]
        module.apply_rotary_emb(
            value[..., -official.rope_head_dim :], official.freqs_cis[:count], True
        )
        value = value.reshape(1, count, official.n_local_groups, -1)
        wo_a = official.wo_a.weight.reshape(
            official.n_local_groups, official.o_lora_rank, -1
        )
        value = torch.einsum("bsgd,grd->bsgr", value, wo_a).flatten(2)[0]
        record("inverse_rope_wo_a", "projected", value)
        record("wo_b", "output", official.wo_b(tx(native["projected"])))
    report = {
        "source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "layer": layer,
        "tokens": count,
        "chunk": chunk,
        "traced_output_bitwise": True,
        "indices_equal": True,
        "raw_indices_shapes": [
            list(logical_indices.shape),
            list(captured["indices"].shape),
        ],
        "indices_comparison": "strict equality after appending only trailing -1 masks",
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "propagated": checks,
        "same_input": same_input,
    }
    np.savez_compressed(options.output / "official.npz", **captured)
    np.savez_compressed(options.output / "same-input.npz", **local)
    assert fingerprint() == prior["source_fingerprint"]
    (options.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
