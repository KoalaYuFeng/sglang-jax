"""Isolated Q-norm-only A/B, with frozen independent CPU trajectories.

Replays production DecoderLayers bitwise before interpreting instrumentation.
No serving source, exp implementation, CPU oracle or acceptance gate changes.
The truncated head is diagnostic and is not deployed-model accuracy.
"""

import argparse
import gc
import hashlib
import importlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import torch
from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_attention_candidates import make_compensated_qnorm
from deepseek_v4_numerical_acceptance import tensor_metrics
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.kernels.deepseek_v4 import normalization
from sgl_jax.srt.kernels.deepseek_v4.dense import DenseKernels
from sgl_jax.srt.kernels.deepseek_v4.mhc import head_collapse
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
    options = parser.parse_args()
    prior = json.loads((options.trajectory / "report.json").read_text())
    assert (
        prior["complete"] and not prior["full_model"] and prior["layers_requested"] == 4
    )
    assert fingerprint() == prior["source_fingerprint"]
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("requires four TPU chips")
    options.output.mkdir(parents=True, exist_ok=False)
    cp = DeepSeekV4Checkpoint(prior["checkpoint"])
    assert (
        hashlib.sha256(
            (Path(prior["checkpoint"]) / "inference/model.py").read_bytes()
        ).hexdigest()
        == prior["official_model_sha256"]
    )
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    count, chunk, context = prior["prefill"], prior["native_chunk"], prior["context"]
    module, args = official_module(cp, context)
    ids = np.load(options.trajectory / "input_ids.npy")
    assert hashlib.sha256(ids.tobytes()).hexdigest() == prior["input_ids_sha256"]
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    put = lambda value: jax.device_put(value, NamedSharding(mesh, P()))
    dense = DenseKernels(
        fp8_backend="gmm", fused_norm=True, merged_projections=True, fused_wo_a=True
    )
    baseline_qnorm = normalization.qnorm_rope
    candidate_qnorm = make_compensated_qnorm()
    model = importlib.import_module("sgl_jax.srt.models.deepseek_v4")
    original_attention = model.attention
    backend = V4PagedBackend(max_context=context, mesh=mesh)
    pages = [list(range(1, 1 + context // 128))]
    spans = [(i, min(i + chunk, count), False) for i in range(0, count, chunk)]
    spans += [(i, i + 1, True) for i in range(count, len(ids))]
    report = {
        "complete": False,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "source_fingerprint": fingerprint(),
        "source_trajectory": str(options.trajectory),
        "checkpoint": prior["checkpoint"],
        "prefill": count,
        "decode": prior["decode"],
        "chunk": chunk,
        "layers_requested": prior["layers_requested"],
        "full_model": False,
        "cpu_reference": "frozen independently propagated official CPU trajectory; fresh same-input attention",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            Path(__file__).with_name("deepseek_v4_attention_candidates.py").read_bytes()
        ).hexdigest(),
        "execution": prior["execution"],
        "layers": [],
    }
    report["input_manifest"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in options.trajectory.iterdir()
        if p.suffix in (".json", ".npy", ".npz")
    }

    def emit(event, **values):
        print(json.dumps(dict(event=event, time=time.time(), **values)), flush=True)
        (options.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def compare(a, b):
        return {
            "all": tensor_metrics(a, b),
            "prefill": tensor_metrics(a[:count], b[:count]),
            "decode": tensor_metrics(a[count:], b[count:]),
        }

    try:
        propagated = None
        for layer in range(prior["layers_requested"]):
            emit("loading", layer=layer)
            saved = np.load(options.trajectory / f"layer-{layer:02d}.npz")
            native_input = saved["native_input"]
            if propagated is None:
                propagated = native_input
            cfg = config_for_layer(cp.config, layer, context)
            ap = attention_uses_tp(cfg, True)
            weights = load_layer(
                cp,
                layer,
                mesh,
                attention_tp=ap,
                merged_projections=True,
                fp4_scale_transposed=True,
            )
            specs = weight_specs(weights, attention_tp=ap)

            def build(decode, qnorm, cfg=cfg, specs=specs):
                def run(x, positions, token_ids, w, cache, metadata, locations):
                    trace = {}

                    def attention(value, *a, **kw):
                        trace["attention_input"] = value
                        result, cache = original_attention(value, *a, **kw)
                        trace["attention_output"] = result
                        return result, cache

                    with (
                        patch.object(normalization, "qnorm_rope", qnorm),
                        patch.object(model, "attention", attention),
                    ):
                        out, cache = DeepseekV4DecoderLayer(
                            cfg,
                            attention_tp=True,
                            csa_decode_batch=decode,
                            moe_backend="gmm_tuned",
                            dense_kernels=dense,
                        )(x, positions, token_ids, w, cache, metadata, locations)
                    return out, cache, trace

                return jax.jit(
                    jax.shard_map(
                        run,
                        mesh=mesh,
                        in_specs=(P(), P(), P(), specs, P(), P(), P()),
                        out_specs=(P(), P(), P()),
                        check_vma=False,
                    )
                )

            runs = {
                name: (build(False, qnorm), build(True, qnorm))
                for name, qnorm in (
                    ("baseline", baseline_qnorm),
                    ("candidate", candidate_qnorm),
                )
            }

            def execute(
                name, values, label, cfg=cfg, layer=layer, runs=runs, weights=weights
            ):
                cache = put(make_cache(cfg, capacity=context + 128, requests=1))
                outputs, traces = [], []
                for begin, end, decode in spans:
                    padding = 0 if decode else chunk - (end - begin)
                    batch = make_batch(
                        [begin],
                        [end - begin],
                        pages=pages,
                        padding=padding,
                        decode=decode,
                    )
                    x = jnp.pad(
                        jnp.asarray(values[begin:end], jnp.bfloat16),
                        ((0, padding), (0, 0), (0, 0)),
                    )
                    emit("native", layer=layer, variant=label, begin=begin, end=end)
                    out, cache, trace = runs[name][int(decode)](
                        put(x),
                        put(batch.positions),
                        put(np.pad(ids[begin:end], (0, padding))),
                        weights,
                        cache,
                        backend.get_forward_metadata(batch),
                        put(batch.out_cache_loc),
                    )
                    outputs.append(np.asarray(out, np.float32)[: end - begin])
                    traces.append(
                        {
                            key: np.asarray(value, np.float32)[: end - begin]
                            for key, value in trace.items()
                        }
                    )
                return (
                    np.concatenate(outputs),
                    {
                        key: np.concatenate([row[key] for row in traces])
                        for key in traces[0]
                    },
                    {key: np.asarray(value) for key, value in cache.items()},
                )

            baseline, bt, bc = execute("baseline", native_input, "baseline")
            np.testing.assert_array_equal(
                baseline,
                saved["native_output"],
                err_msg="baseline instrumentation changed frozen output",
            )
            local, lt, lc = execute("candidate", native_input, "candidate_same_input")
            np.testing.assert_array_equal(bt["attention_input"], lt["attention_input"])
            for key in bc:
                np.testing.assert_array_equal(
                    bc[key],
                    lc[key],
                    err_msg="Q-only change altered same-input cache: " + key,
                )
            if layer == 0:
                candidate, ct, cc = local, lt, lc
            else:
                candidate, ct, cc = execute(
                    "candidate", propagated, "candidate_propagated"
                )
            assert (
                normalization.qnorm_rope is baseline_qnorm
                and model.attention is original_attention
            )

            emit("cpu_same_input_attention", layer=layer)
            official = load_module_weights(
                module.Attention(layer, args), cp, f"layers.{layer}.attn."
            )
            tx = torch.from_numpy(bt["attention_input"].copy()).to(torch.bfloat16)[None]
            with torch.inference_mode():
                pieces = [official(tx[:, :count], 0)[0].float().numpy()]
                for position in range(count, len(ids)):
                    pieces.append(
                        official(tx[:, position : position + 1], position)[0]
                        .float()
                        .numpy()
                    )
            cpu_attn = np.concatenate(pieces)
            row = {
                "layer": layer,
                "ratio": cfg.ratio,
                "baseline_replay_bitwise": True,
                "same_input_attention_input_bitwise": True,
                "same_input_final_cache_bitwise": True,
                "baseline_vs_cpu": compare(baseline, saved["cpu_output"]),
                "candidate_same_input_vs_cpu_trajectory": compare(
                    local, saved["cpu_output"]
                ),
                "candidate_propagated_vs_cpu": compare(candidate, saved["cpu_output"]),
                "candidate_propagated_vs_baseline": compare(candidate, baseline),
                "baseline_attention_vs_same_input_cpu": compare(
                    bt["attention_output"], cpu_attn
                ),
                "candidate_attention_vs_same_input_cpu": compare(
                    lt["attention_output"], cpu_attn
                ),
            }
            np.savez_compressed(
                options.output / f"layer-{layer:02d}.npz",
                baseline_input=native_input,
                candidate_input=propagated,
                baseline=baseline,
                candidate_same_input=local,
                candidate_propagated=candidate,
                cpu=saved["cpu_output"],
                cpu_attention_same_input=cpu_attn,
                **{"baseline_" + k: v for k, v in bt.items()},
                **{"candidate_same_input_" + k: v for k, v in lt.items()},
                **{"candidate_propagated_" + k: v for k, v in ct.items()},
            )
            np.savez_compressed(
                options.output / f"cache-{layer:02d}.npz",
                **{"baseline_" + k: v for k, v in bc.items()},
                **{"candidate_same_input_" + k: v for k, v in lc.items()},
                **{"candidate_propagated_" + k: v for k, v in cc.items()},
            )
            report["layers"].append(row)
            emit("layer_complete", **row)
            propagated = candidate
            del weights, runs, execute, official, tx, pieces, bc, lc, cc
            gc.collect()

        emit("diagnostic_head")
        shared = {
            key: put(cp.read_tensor(key))
            for key in (
                "hc_head_fn",
                "hc_head_scale",
                "hc_head_base",
                "norm.weight",
                "head.weight",
            )
        }

        def head(x, w):
            collapsed = head_collapse(
                x,
                w["hc_head_fn"],
                w["hc_head_scale"],
                w["hc_head_base"],
                eps=args.norm_eps,
                hc_eps=args.hc_eps,
                backend="pallas",
            )
            hidden = dense.norm(collapsed, w["norm.weight"], args.norm_eps)
            return jnp.matmul(
                hidden, w["head.weight"].T, preferred_element_type=jnp.float32
            )

        compiled = jax.jit(
            jax.shard_map(
                head,
                mesh=mesh,
                in_specs=(P(), {k: P() for k in shared}),
                out_specs=P(),
                check_vma=False,
            )
        )
        old_logits = np.load(options.trajectory / "logits.npz")
        baseline_logits = np.asarray(
            compiled(put(jnp.asarray(baseline[count - 1 :], jnp.bfloat16)), shared),
            np.float32,
        )
        np.testing.assert_array_equal(baseline_logits, old_logits["native"])
        candidate_logits = np.asarray(
            compiled(put(jnp.asarray(propagated[count - 1 :], jnp.bfloat16)), shared),
            np.float32,
        )
        report["logits"] = {
            "scope": "four-layer diagnostic head, not full-model logits",
            "baseline_replay_bitwise": True,
        }
        for name, values in (
            ("baseline", baseline_logits),
            ("candidate", candidate_logits),
        ):
            report["logits"][name] = dict(
                **tensor_metrics(values, old_logits["cpu"]),
                top1_agreement=float(
                    np.mean(values.argmax(-1) == old_logits["cpu"].argmax(-1))
                ),
                top1=values.argmax(-1).tolist(),
            )
        report["logits"]["cpu_top1"] = old_logits["cpu"].argmax(-1).tolist()
        np.savez_compressed(
            options.output / "logits.npz",
            baseline=baseline_logits,
            candidate=candidate_logits,
            cpu=old_logits["cpu"],
        )
        assert fingerprint() == prior["source_fingerprint"]
        report["complete"] = True
        emit("complete", logits=report["logits"])
    except BaseException:
        import traceback

        report["error"] = traceback.format_exc()
        emit("failed", error=report["error"])
        raise


if __name__ == "__main__":
    main()
