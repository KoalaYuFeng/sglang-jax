"""Identical real HCA inputs: original Pallas vs the retained V4 attention path.

Historical captures are immutable operator fixtures, NOT a same-source model
replay. Full 43-layer new-source numerical acceptance is a separate gate.
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4 import hca
from sgl_jax.srt.kernels.deepseek_v4.attention import attention
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer, rms_norm, rope
from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip, round_bf16
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedMetadata
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.capture / "report.json").read_text())
    replay = json.loads((args.replay / "report.json").read_text())
    if (
        not replay["faithful"]
        or replay["framework_source_fingerprint"] != source["framework_source_fingerprint"]
    ):
        raise ValueError("requires a faithful replay of the matching historical capture")
    checkpoint = DeepSeekV4Checkpoint(source["checkpoint"])
    layers = [i for i, ratio in enumerate(checkpoint.config["compress_ratios"]) if ratio == 128]
    selected = [layers[0], layers[len(layers) // 2], layers[-1]]
    report = {
        "complete": False,
        "scope": __doc__,
        "checkpoint": source["checkpoint"],
        "source_fingerprint": framework_fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capture": str(args.capture),
        "fixture_source_fingerprint": source["framework_source_fingerprint"],
        "layers": selected,
        "checks": [],
    }

    def check(label, expected, actual, tolerance, *, state=False):
        a, b = np.asarray(expected), np.asarray(actual)
        metrics = compare_arrays(a, b)
        # State scores may contain identical -inf sentinels; no NaN or differing
        # infinity is accepted. Ordinary activations must be entirely finite.
        finite = np.isfinite(a) & np.isfinite(b)
        valid = np.all(a[~finite] == b[~finite]) if state else np.all(finite)
        row = {
            "label": label,
            "nonfinite_contract_passed": bool(valid),
            "tolerance": tolerance,
            **metrics,
        }
        report["checks"].append(row)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if not valid or not metrics["finite_mask_equal"] or metrics["nrmse"] > tolerance:
            save_arrays(args.output / "failure", {"expected": a, "actual": b})
            raise AssertionError(label)

    try:
        mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",))
        with jax.set_mesh(mesh):
            for layer in selected:
                config = config_for_layer(checkpoint.config, layer, 8192)
                weights = load_layer(checkpoint, layer, mesh, include_experts=False)

                def build(backend, config=config):
                    @jax.jit
                    def run(hidden, p, w, c, m, loc):
                        trace = {}
                        output, cache = attention(
                            hidden, p, w, c, config, m, loc, hca_backend=backend, trace=trace
                        )
                        return output, cache, trace

                    return run

                reference, candidate = build("reference"), build("pallas")
                for frame in ("B1-case1", "B2"):
                    folder = args.capture / frame
                    inputs = load_arrays(folder / "inputs")
                    meta = V4PagedMetadata(**load_arrays(folder / "metadata"))
                    before = load_arrays(folder / "before" / f"layer-{layer:02d}")
                    x = load_arrays(args.replay / f"{frame}-layer-{layer:02d}-trace")["attn.norm"]
                    call = jax.tree.map(
                        jnp.asarray,
                        (x, inputs["positions"], weights, before, meta, inputs["locations"]),
                    )
                    expected, ref_cache, ref_trace = reference(*call)
                    actual, got_cache, got_trace = candidate(*call)
                    label = f"layer{layer}/{frame}"
                    check(
                        label + "/attention_value",
                        ref_trace["attention_value"],
                        got_trace["attention_value"],
                        2e-4,
                    )
                    check(label + "/attention_output", expected, actual, 0.005)
                    for suffix in ("kv", "score"):
                        check(
                            label + "/main." + suffix,
                            ref_cache["main." + suffix],
                            got_cache["main." + suffix],
                            5e-7,
                            state=True,
                        )
                        slots = meta.req_slots[meta.token_requests]
                        offsets = inputs["positions"] % 128
                        check(
                            label + "/new_projection." + suffix,
                            ref_cache["main." + suffix][slots, offsets],
                            got_cache["main." + suffix][slots, offsets],
                            5e-7,
                            state=True,
                        )
                    if frame == "B1-case1":
                        # Re-emit all complete real groups (0..7935), not just
                        # this non-boundary decode token, with checkpoint YaRN.
                        count = int(meta.prefix_lens[0]) // 128
                        pages = np.asarray(meta.page_table[0, :count])
                        values = jnp.asarray(before["main.snapshot_kv"][pages])
                        scores = jnp.asarray(before["main.snapshot_score"][pages])
                        norm = weights["attn.compressor.norm.weight"]
                        starts = jnp.arange(count, dtype=jnp.int32) * 128
                        expected_emit = jax.jit(
                            lambda v, s, n, p, cfg=config: rope(
                                rms_norm(
                                    round_bf16(jnp.sum(v * jax.nn.softmax(s, axis=1), axis=1)),
                                    n,
                                    cfg.eps,
                                ),
                                p,
                                cfg,
                            )
                        )(values, scores, norm, starts)
                        got_emit = jax.jit(
                            lambda v, s, n, p, cfg=config: hca.emit(
                                v,
                                s,
                                n,
                                p,
                                jnp.ones((p.shape[0],), jnp.bool_),
                                cfg,
                            )
                        )(values, scores, norm, starts)
                        check(label + "/real_boundary_emit", expected_emit, got_emit, 2e-4)
                        quantize = jax.jit(
                            lambda value: jnp.concatenate(
                                (activation_fp8_roundtrip(value[:, :-64], 64), value[:, -64:]),
                                axis=1,
                            )
                        )
                        check(
                            label + "/real_boundary_fp8_qat",
                            quantize(expected_emit),
                            quantize(got_emit),
                            0.005,
                        )
                    del call, expected, actual, ref_cache, got_cache, ref_trace, got_trace
                del weights
        report["complete"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
