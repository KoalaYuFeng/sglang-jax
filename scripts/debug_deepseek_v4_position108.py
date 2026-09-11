"""Same-input and route-intervention diagnosis of the frozen position 108.

CPU-only experiment using official MoE/mHC control flow and original weights.
Interventions are diagnostic counterfactuals, never acceptance references.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_numerical_acceptance import tensor_metrics
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.deepseek_v4_cpu_oracle import (
    load_module_weights,
    official_module,
    tensor_from_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    base, out = options.profiles, options.output
    trajectory = base / "v4-cpu-trajectory-20260911-02"
    prior = json.loads((trajectory / "report.json").read_text())
    assert fingerprint() == prior["source_fingerprint"]
    out.mkdir(exist_ok=False)
    cp = DeepSeekV4Checkpoint(prior["checkpoint"])
    assert (
        hashlib.sha256((cp.directory / "inference/model.py").read_bytes()).hexdigest()
        == prior["official_model_sha256"]
    )
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    module, args = official_module(cp, prior["context"])
    stage = base / "v4-cpu-stages-20260911-01"
    native = np.load(stage / "native.npz")
    cpu = np.load(stage / "official.npz")
    local = np.load(base / "v4-cpu-local-stages-20260911-01/same-input.npz")
    layer = np.load(trajectory / "layer-03.npz")
    ids = np.load(trajectory / "input_ids.npy")[:127]
    position = 108
    report = {
        "complete": False,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "position": position,
        "layer": 3,
        "source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stages": {},
        "gate": {},
        "interventions": [],
    }
    captures = {}

    def emit(event, **values):
        print(json.dumps(dict(event=event, time=time.time(), **values)), flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def tx(value):
        return torch.from_numpy(np.array(value, np.float32)).to(torch.bfloat16)

    def fp(value):
        return torch.from_numpy(np.array(value, np.float32))

    def array(value):
        return value.detach().float().numpy()

    def metric(a, b):
        return tensor_metrics(a[position : position + 1], b[position : position + 1])

    try:
        for i in range(4):
            saved = np.load(trajectory / f"layer-{i:02d}.npz")
            report["stages"][f"layer_{i}_input"] = metric(
                saved["native_input"], saved["cpu_input"]
            )
            report["stages"][f"layer_{i}_output"] = metric(
                saved["native_output"], saved["cpu_output"]
            )
        for key in ("attn.norm", "attn.operator", "ffn.norm", "ffn.operator"):
            report["stages"][key] = metric(native[key], cpu[key])
        report["stages"]["same_input_attention"] = metric(
            native["attn.operator"], local["attention"]
        )
        report["stages"]["same_input_moe"] = metric(
            native["ffn.operator"], cpu["same_input_moe"]
        )
        emit("frozen_stages", stages=report["stages"])
        emit("loading_moe")
        moe = load_module_weights(module.MoE(3, args), cp, "layers.3.ffn.")
        inputs = {"native": tx(native["ffn.norm"]), "cpu": tx(cpu["ffn.norm"])}
        token_ids = torch.from_numpy(ids)
        gate_weight = moe.gate.weight.detach().float()
        bias = moe.gate.bias.detach().float()
        routing = {}
        with torch.inference_mode():
            for name, value in inputs.items():
                weights, selected = moe.gate(value, token_ids)
                expected_ids = (native if name == "native" else cpu)[
                    "expert_ids"
                ].astype(np.int64)
                np.testing.assert_array_equal(selected.numpy(), expected_ids)
                raw = value.float() @ gate_weight.T
                score = torch.nn.functional.softplus(raw).sqrt()
                adjusted = score + bias
                score64 = (
                    np.asarray(value[position].float(), np.float64)
                    @ np.asarray(gate_weight, np.float64).T
                )
                adjusted64 = np.sqrt(np.logaddexp(0, score64)) + np.asarray(
                    bias, np.float64
                )
                order64 = np.argsort(-adjusted64)[: args.n_activated_experts]
                np.testing.assert_array_equal(order64, selected[position].numpy())
                routing[name] = (weights.clone(), selected.clone(), score.clone())
                order = torch.argsort(adjusted[position], descending=True)[:10]
                report["gate"][name] = {
                    "selected": selected[position].tolist(),
                    "fp64_selected": order64.tolist(),
                    "top10": order.tolist(),
                    "adjusted_top10": array(adjusted[position, order]).tolist(),
                    "margin_214_minus_199": float(
                        adjusted[position, 214] - adjusted[position, 199]
                    ),
                    "fp64_margin_214_minus_199": float(
                        adjusted64[214] - adjusted64[199]
                    ),
                    "same_input_route_ids_reproduced": True,
                    "routing_vs_captured": metric(
                        array(weights),
                        (native if name == "native" else cpu)["routing_weights"],
                    ),
                }
                captures[name + "_gate_scores"] = array(adjusted)
                captures[name + "_gate_fp64_position108"] = adjusted64
            emit("gate", gate=report["gate"])

            ctx = SimpleNamespace(
                norm_eps=args.norm_eps,
                hc_eps=args.hc_eps,
                hc_mult=args.hc_mult,
                hc_sinkhorn_iters=args.hc_sinkhorn_iters,
            )
            attn_weights = [
                tensor_from_checkpoint(cp, "layers.3.hc_attn_" + suffix)
                for suffix in ("fn", "scale", "base")
            ]
            _, post, comb = module.Block.hc_pre(
                ctx, tx(layer["cpu_input"][:127])[None], *attn_weights
            )
            cpu_residual = module.Block.hc_post(
                ctx,
                tx(cpu["attn.operator"])[None],
                tx(layer["cpu_input"][:127])[None],
                post,
                comb,
            )
            ffn_weights = [
                tensor_from_checkpoint(cp, "layers.3.hc_ffn_" + suffix)
                for suffix in ("fn", "scale", "base")
            ]
            pre, cpu_post, cpu_comb = module.Block.hc_pre(
                ctx, cpu_residual, *ffn_weights
            )
            norm = load_module_weights(
                module.RMSNorm(args.dim, args.norm_eps), cp, "layers.3.ffn_norm."
            )
            np.testing.assert_array_equal(array(norm(pre)[0]), cpu["ffn.norm"])
            cpu_layer = module.Block.hc_post(
                ctx, tx(cpu["ffn.operator"])[None], cpu_residual, cpu_post, cpu_comb
            )
            np.testing.assert_array_equal(
                array(cpu_layer[0]), layer["cpu_output"][:127]
            )
            report["cpu_residual_reconstruction_bitwise"] = True

            def native_post(value):
                return array(
                    module.Block.hc_post(
                        ctx,
                        tx(value)[None],
                        tx(native["attn.post"])[None],
                        fp(native["ffn.post_weights"])[None],
                        fp(native["ffn.comb"])[None],
                    )[0]
                )

            np.testing.assert_array_equal(
                native_post(native["ffn.operator"]), local["ffn.post"]
            )
            report["native_same_input_post_replay_bitwise"] = True
            original_gate = moe.gate.forward
            experiments = (
                ("cpu", "cpu", "own"),
                ("native", "native", "own"),
                ("native", "cpu", "recomputed"),
                ("native", "cpu", "donor"),
                ("cpu", "native", "recomputed"),
            )
            for input_name, route_name, weight_mode in experiments:
                label = f"{input_name}-input_{route_name}-route_{weight_mode}"
                emit("intervention", variant=label)

                def gate(
                    value,
                    token_ids,
                    input_name=input_name,
                    route_name=route_name,
                    weight_mode=weight_mode,
                ):
                    weights, selected = original_gate(value, token_ids)
                    if route_name != input_name:
                        weights, selected = weights.clone(), selected.clone()
                        selected[position] = routing[route_name][1][position]
                        if weight_mode == "donor":
                            weights[position] = routing[route_name][0][position]
                        else:
                            s = routing[input_name][2][position, selected[position]]
                            weights[position] = s / s.sum() * args.route_scale
                    return weights, selected

                with patch.object(moe.gate, "forward", gate):
                    result = array(moe(inputs[input_name][None], token_ids[None])[0])
                if input_name == route_name:
                    expected = (
                        cpu["ffn.operator"]
                        if input_name == "cpu"
                        else cpu["same_input_moe"]
                    )
                    np.testing.assert_array_equal(result, expected)
                if input_name == "native":
                    post_result = native_post(result)
                else:
                    post_result = array(
                        module.Block.hc_post(
                            ctx, tx(result)[None], cpu_residual, cpu_post, cpu_comb
                        )[0]
                    )
                captures[label + "_moe"] = result
                captures[label + "_layer"] = post_result
                report["interventions"].append(
                    {
                        "variant": label,
                        "baseline_replay_bitwise": True
                        if input_name == route_name
                        else None,
                        "moe_vs_cpu": metric(result, cpu["ffn.operator"]),
                        "moe_vs_native": metric(result, native["ffn.operator"]),
                        "layer_vs_cpu": metric(post_result, layer["cpu_output"][:127]),
                        "layer_vs_native": metric(
                            post_result, layer["native_output"][:127]
                        ),
                    }
                )
                emit("intervention_complete", **report["interventions"][-1])
        np.savez_compressed(out / "captures.npz", **captures)
        assert fingerprint() == prior["source_fingerprint"]
        report["complete"] = True
        emit("complete")
    except BaseException:
        import traceback

        report["error"] = traceback.format_exc()
        emit("failed", error=report["error"])
        raise


if __name__ == "__main__":
    main()
