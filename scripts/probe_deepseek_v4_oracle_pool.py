"""Independent CPU adjudication of a faithfully captured decode compressor.

Run with JAX_PLATFORMS=cpu. Only post-projection inputs are injected: official
Compressor pooling, normalization, RoPE and low-bit QAT control flow is intact.
"""

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import torch

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from replay_deepseek_v4_8023 import difference, logical_cache
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedMetadata
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.deepseek_v4_cpu_oracle import load_module_weights, official_module


def numpy_tensor(tensor):
    value = tensor.detach().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().view(ml_dtypes.bfloat16).copy()
    return value.numpy().copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = json.loads((args.replay / "report.json").read_text())
    if not replay["finished"] or not replay["faithful"]:
        raise ValueError("first require faithful full layer replay")
    layer, position = replay["first_hidden_difference"], replay["position"]
    checkpoint = DeepSeekV4Checkpoint(replay["checkpoint"])
    config = config_for_layer(checkpoint.config, layer, 8192)
    if config.ratio != 4:
        raise ValueError("this probe requires a CSA compression boundary")
    metadata = V4PagedMetadata(**load_arrays(args.replay / "metadata"))
    native_trace = load_arrays(args.replay / "first-native-trace")
    reference_trace = load_arrays(args.replay / "first-reference-trace")
    np.testing.assert_array_equal(native_trace["attn.norm"], reference_trace["attn.norm"])
    states = {}
    for side in ("native", "reference"):
        states[side] = {}
        for when in ("before", "after"):
            values = load_arrays(args.replay / f"first-{side}-{when}")
            states[side][when] = (
                logical_cache(values, metadata, 0, config, after=when == "after")
                if side == "native"
                else values
            )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    official, cfg = official_module(checkpoint, max_context=8192)
    report = {"layer": layer, "position": position, "prefixes": {}}
    group = np.flatnonzero(
        (np.asarray(metadata.group4_starts) >= 0) & (np.asarray(metadata.group4_requests) == 0)
    )[0]
    for prefix, dim, rotate in (
        ("main", config.head_dim, False),
        ("index", config.index_dim, True),
    ):
        weight_prefix = "attn.indexer.compressor." if rotate else "attn.compressor."
        entries, payloads = {}, {}
        for side in ("native", "reference"):
            before, after = states[side]["before"], states[side]["after"]
            values, scores = (
                np.array(before[prefix + "." + suffix], copy=True) for suffix in ("kv", "score")
            )
            slot = 4 + position % 4
            current_kv, current_score = (
                np.array(after[prefix + "." + suffix][slot], copy=True)
                for suffix in ("kv", "score")
            )
            values[slot], scores[slot] = current_kv, current_score
            pv = np.concatenate((values[:4, :dim], values[4:, dim:]))
            ps = np.concatenate((scores[:4, :dim], scores[4:, dim:]))
            if side == "native":
                np.testing.assert_array_equal(
                    pv, native_trace[f"compressor.{prefix}.group_values"][group]
                )
                np.testing.assert_array_equal(
                    ps, native_trace[f"compressor.{prefix}.group_scores"][group]
                )
            probability = np.exp(ps.astype(np.float64) - ps.max(axis=0))
            probability /= probability.sum(axis=0)
            pooled64 = (pv.astype(np.float64) * probability).sum(axis=0).astype(ml_dtypes.bfloat16)
            payloads[side] = {"group_values": pv, "group_scores": ps}

            comp = load_module_weights(
                official.Compressor(cfg, 4, dim, rotate),
                checkpoint,
                f"layers.{layer}.{weight_prefix}",
            )
            comp.kv_cache = torch.zeros((1, 8192 // 4, dim), dtype=torch.bfloat16)
            comp.freqs_cis = official.precompute_freqs_cis(
                cfg.rope_head_dim,
                8192,
                cfg.original_seq_len,
                cfg.compress_rope_theta,
                cfg.rope_factor,
                cfg.beta_fast,
                cfg.beta_slow,
            )
            comp.kv_state.copy_(torch.from_numpy(before[prefix + ".kv"])[None])
            comp.score_state.copy_(torch.from_numpy(before[prefix + ".score"])[None])
            # Inject already-biased captured FP32 projections, so no CPU/TPU
            # GEMV reduction difference can disguise a pooling/state error.
            with torch.no_grad():
                comp.ape.zero_()
            comp.wkv.forward = lambda _, value=current_kv: torch.from_numpy(value)[
                None, None
            ].clone()
            comp.wgate.forward = lambda _, value=current_score: torch.from_numpy(value)[
                None, None
            ].clone()
            captured = {}
            comp.norm.register_forward_pre_hook(
                lambda _, inputs, target=captured: target.update(
                    pooled=numpy_tensor(inputs[0])[0, 0]
                )
            )
            comp.norm.register_forward_hook(
                lambda _, inputs, out, target=captured: target.update(
                    normalized=numpy_tensor(out)[0, 0]
                )
            )
            with torch.inference_mode():
                result = comp(
                    torch.from_numpy(native_trace["attn.norm"].astype(np.float32)).to(
                        torch.bfloat16
                    )[None],
                    position,
                )
            result = numpy_tensor(result)[0, 0]
            recorded_native = states["native"]["after"][prefix + ".compressed"][-1]
            recorded_reference = states["reference"]["after"][prefix + ".compressed"][position // 4]
            native_pooled = native_trace[f"compressor.{prefix}.pooled"][group]
            entries[side] = {
                "cpu_pool_vs_numpy64": difference(pooled64, captured["pooled"]),
                "native_pool_vs_numpy64": difference(pooled64, native_pooled),
                "cpu_pool_vs_native": difference(native_pooled, captured["pooled"]),
                "cpu_final_vs_native": difference(recorded_native, result),
                "cpu_final_vs_reference": difference(recorded_reference, result),
            }
            save_arrays(
                args.output / f"{prefix}-{side}",
                {**payloads[side], **captured, "pooled64": pooled64, "final": result},
            )
        entries["input_differences"] = {
            key: difference(payloads["native"][key], payloads["reference"][key])
            for key in payloads["native"]
        }
        report["prefixes"][prefix] = entries
        print(json.dumps({"prefix": prefix, **entries}), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
