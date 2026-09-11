"""Frozen layer-0 position-108 origin and FP8-activation/FP4-weight MoE audit.

Runs official CPU MoE at original full-prefill shapes, preserving frozen
references. No runtime changes and no replacement accuracy thresholds.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_attention_diagnostics import fp8_roundtrip_cpu
from deepseek_v4_numerical_acceptance import tensor_metrics
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.deepseek_v4_cpu_oracle import (
    _act_quant,
    _scales,
    load_module_weights,
    official_module,
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
    stage = base / "v4-cpu-stages-20260911-02"
    native, cpu = (np.load(stage / (name + ".npz")) for name in ("native", "official"))
    local = np.load(base / "v4-cpu-local-stages-20260911-02/same-input.npz")
    attn = base / "v4-cpu-attention-stages-20260911-02"
    an, ac, al = (
        np.load(attn / (name + ".npz")) for name in ("native", "official", "same-input")
    )
    cp = DeepSeekV4Checkpoint(prior["checkpoint"])
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    module, args = official_module(cp, 256)
    position = 108
    report = {
        "complete": False,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "position": position,
        "layer": 0,
        "source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stages": {},
        "attention": {},
        "attention_same_input": {},
        "experts": [],
    }

    def metric(a, b):
        return tensor_metrics(a[position : position + 1], b[position : position + 1])

    def emit(event, **values):
        print(json.dumps(dict(event=event, time=time.time(), **values)), flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def array(value):
        return value.detach().float().numpy()

    try:
        for key in ("attn.norm", "attn.operator", "ffn.norm", "ffn.operator"):
            report["stages"][key] = metric(native[key], cpu[key])
        report["stages"]["same_input_moe"] = metric(
            native["ffn.operator"], cpu["same_input_moe"]
        )
        for key in (
            "attn.pre",
            "attn.norm",
            "attn.post",
            "ffn.pre",
            "ffn.norm",
            "ffn.post",
        ):
            if key in local.files:
                report["stages"]["same_input_" + key] = metric(native[key], local[key])
        for key in (
            "qa",
            "raw_kv",
            "qr",
            "attn.wq_b.output",
            "q",
            "kv",
            "attention_value",
            "projected",
            "output",
        ):
            report["attention"][key] = metric(an[key], ac[key])
        for actual, control in (
            ("qr", "q_norm"),
            ("attn.wq_b.output", "wq_b"),
            ("q", "qnorm_rope"),
            ("kv", "kv_norm_rope_qat"),
            ("attention_value", "sparse_attention"),
            ("projected", "inverse_rope_wo_a"),
            ("output", "wo_b"),
        ):
            report["attention_same_input"][control] = metric(an[actual], al[control])
        emit(
            "frozen_stages",
            stages=report["stages"],
            attention=report["attention"],
            attention_same_input=report["attention_same_input"],
        )
        moe = load_module_weights(module.MoE(0, args), cp, "layers.0.ffn.")
        ids = torch.from_numpy(np.load(trajectory / "input_ids.npy")[:127])
        expected_ids = native["expert_ids"].astype(np.int64)
        np.testing.assert_array_equal(expected_ids, cpu["expert_ids"].astype(np.int64))
        selected = expected_ids[position]
        report["selected_experts"] = selected.tolist()
        captures = {}
        with torch.inference_mode():
            for name, saved in (("native", native), ("cpu", cpu)):
                x = torch.from_numpy(saved["ffn.norm"].copy()).to(torch.bfloat16)
                mixing, routes = moe.gate(x, ids)
                np.testing.assert_array_equal(routes.numpy(), expected_ids)
                captures[name + "_input"] = array(x)
                captures[name + "_routing"] = array(mixing)
                handles = []
                for expert in selected:
                    rows, _ = torch.where(routes == int(expert))
                    index = int(torch.where(rows == position)[0].item())
                    for projection in ("w1", "w3", "w2"):
                        linear = getattr(moe.experts[int(expert)], projection)
                        prefix = f"{name}_e{expert}_{projection}"

                        def hook(m, inputs, output, index=index, prefix=prefix):
                            value = inputs[0][index : index + 1]
                            q, scales = _act_quant(value, 128)
                            dequant = q.float() * _scales(scales).repeat_interleave(
                                128, dim=-1
                            )
                            independent, _ = fp8_roundtrip_cpu(array(value))
                            np.testing.assert_array_equal(array(dequant), independent)
                            captures[prefix + "_input"] = array(value)
                            captures[prefix + "_qbytes"] = (
                                q.view(torch.uint8).numpy().copy()
                            )
                            captures[prefix + "_sbytes"] = (
                                scales.view(torch.uint8).numpy().copy()
                            )
                            captures[prefix + "_dequant"] = array(dequant)
                            captures[prefix + "_output"] = array(
                                output[index : index + 1]
                            )

                        handles.append(linear.register_forward_hook(hook))
                emit("cpu_moe", input=name)
                result = array(moe(x[None], ids[None])[0])
                for handle in handles:
                    handle.remove()
                np.testing.assert_array_equal(
                    result,
                    cpu["same_input_moe"] if name == "native" else cpu["ffn.operator"],
                )
                captures[name + "_moe_output"] = result
                emit("baseline_reproduced", input=name)
        for expert in selected:
            for projection in ("w1", "w3", "w2"):
                a, b = (f"{name}_e{expert}_{projection}" for name in ("native", "cpu"))
                row = {
                    "expert": int(expert),
                    "projection": projection,
                    "input": tensor_metrics(
                        captures[a + "_input"], captures[b + "_input"]
                    ),
                    "dequant": tensor_metrics(
                        captures[a + "_dequant"], captures[b + "_dequant"]
                    ),
                    "output": tensor_metrics(
                        captures[a + "_output"], captures[b + "_output"]
                    ),
                    "changed_fp8_codes": int(
                        np.count_nonzero(
                            captures[a + "_qbytes"] != captures[b + "_qbytes"]
                        )
                    ),
                    "changed_scales": int(
                        np.count_nonzero(
                            captures[a + "_sbytes"] != captures[b + "_sbytes"]
                        )
                    ),
                    "both_quantizers_match_independent": True,
                }
                report["experts"].append(row)
        np.savez_compressed(out / "captures.npz", **captures)
        assert fingerprint() == prior["source_fingerprint"]
        report["complete"] = True
        emit("complete", experts=report["experts"])
    except BaseException:
        import traceback

        report["error"] = traceback.format_exc()
        emit("failed", error=report["error"])
        raise


if __name__ == "__main__":
    main()
