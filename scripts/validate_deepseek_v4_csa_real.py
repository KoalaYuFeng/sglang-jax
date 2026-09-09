"""Identical historical real CSA inputs, not a same-source whole-model replay.

Compare the original Pallas integration with the retained V4 path at position
8023 on early/middle/last CSA layers, including index ranking and FP32 state.
Full 43-layer and cold 8K acceptance remain separate gates.
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
from sgl_jax.srt.kernels.deepseek_v4 import csa
from sgl_jax.srt.kernels.deepseek_v4.attention import attention
from sgl_jax.srt.kernels.deepseek_v4.compressor import compress
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer, rope
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend, V4PagedMetadata
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer
from sgl_jax.test.test_deepseek_v4_paged import make_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-fixture", type=Path)
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
    layers = [i for i, ratio in enumerate(checkpoint.config["compress_ratios"]) if ratio == 4]
    selected = [layers[0], layers[len(layers) // 2], layers[-1]]
    report = {
        "complete": False,
        "scope": __doc__,
        "checkpoint": source["checkpoint"],
        "source_fingerprint": framework_fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "capture": str(args.capture),
        "replay": str(args.replay),
        "fixture_source_fingerprint": source["framework_source_fingerprint"],
        "layers": selected,
        "projection_fixture": str(args.projection_fixture) if args.projection_fixture else None,
        "checks": [],
    }

    def check(label, expected, actual, tolerance, *, state=False, exact=False):
        a, b = np.asarray(expected), np.asarray(actual)
        metrics = compare_arrays(a, b)
        finite = np.isfinite(a) & np.isfinite(b)
        valid = np.all(a[~finite] == b[~finite]) if state else np.all(finite)
        row = {
            "label": label,
            "nonfinite_contract_passed": bool(valid),
            "tolerance": tolerance,
            "exact_required": exact,
            **metrics,
        }
        report["checks"].append(row)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if (
            not valid
            or not metrics["finite_mask_equal"]
            or metrics["nrmse"] > tolerance
            or (exact and not metrics["bitwise_equal"])
        ):
            save_arrays(args.output / "failure", {"expected": a, "actual": b})
            raise AssertionError(label)

    try:
        mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",))
        with jax.set_mesh(mesh):
            if args.projection_fixture:
                fixture = load_arrays(args.projection_fixture)
                valid_starts = np.asarray(fixture["starts"])[np.asarray(fixture["valid"])]
                if valid_starts.size != 1 or fixture["activation"].shape != (1, 4096):
                    raise ValueError("projection regression fixture must contain one real token")
                position = int(valid_starts[0]) + 3
                metadata = V4PagedBackend(max_context=8192).get_forward_metadata(
                    make_batch([position], [1], pages=[list(range(1, 65))], decode=True)
                )
                config = config_for_layer(checkpoint.config, selected[0], 8192)
                for prefix in ("main", "index"):
                    actual = jax.jit(lambda x, a, b, m: csa.project(x, a, b, m, config))(
                        jnp.asarray(fixture["activation"]),
                        jnp.asarray(fixture[f"{prefix}.wkv.weight"]),
                        jnp.asarray(fixture[f"{prefix}.wgate.weight"]),
                        metadata,
                    )
                    for suffix, value in zip(("kv", "scores"), actual, strict=True):
                        check(
                            f"regression/position{position}/{prefix}/{suffix}",
                            fixture[f"{prefix}.reference.{suffix}"],
                            value,
                            0,
                            exact=True,
                        )
            for layer in selected:
                config = config_for_layer(checkpoint.config, layer, 8192)
                weights = load_layer(checkpoint, layer, mesh, include_experts=False)

                def build(backend, config=config):
                    @jax.jit
                    def run(activation, p, w, c, m, loc):
                        trace = {}
                        output, cache = attention(
                            activation, p, w, c, config, m, loc, csa_backend=backend, trace=trace
                        )
                        return output, cache, trace

                    return run

                reference, candidate = build("reference"), build("pallas")
                for frame in ("B1-case1", "B2"):
                    folder = args.capture / frame
                    inputs = load_arrays(folder / "inputs")
                    meta = V4PagedMetadata(**load_arrays(folder / "metadata"))
                    before = load_arrays(folder / "before" / f"layer-{layer:02d}")
                    hidden = load_arrays(args.replay / f"{frame}-layer-{layer:02d}-trace")[
                        "attn.norm"
                    ]
                    call = jax.tree.map(
                        jnp.asarray,
                        (hidden, inputs["positions"], weights, before, meta, inputs["locations"]),
                    )
                    expected, ref_cache, ref_trace = reference(*call)
                    actual, got_cache, got_trace = candidate(*call)
                    label = f"layer{layer}/{frame}"
                    for key in ("index_q", "index_score", "index_selected"):
                        check(
                            label + "/" + key,
                            ref_trace[key],
                            got_trace[key],
                            0,
                            state=key == "index_score",
                            exact=True,
                        )
                    check(
                        label + "/attention_value",
                        ref_trace["attention_value"],
                        got_trace["attention_value"],
                        2e-4,
                    )
                    check(label + "/attention_output", expected, actual, 0.005)
                    for prefix in ("main", "index"):
                        for suffix in ("kv", "score"):
                            key = f"{prefix}.{suffix}"
                            check(
                                label + "/" + key, ref_cache[key], got_cache[key], 5e-7, state=True
                            )
                            # Do not dilute a fresh projection error among
                            # untouched private state or padded request slots.
                            slots = meta.req_slots[meta.token_requests]
                            offsets = 4 + inputs["positions"] % 4
                            live = np.asarray(meta.token_valid)
                            check(
                                label + "/new_projection." + key,
                                np.asarray(ref_cache[key][slots, offsets])[live],
                                np.asarray(got_cache[key][slots, offsets])[live],
                                5e-7,
                                state=True,
                            )
                        key = f"{prefix}.compressed"
                        check(label + "/" + key, ref_cache[key], got_cache[key], 2e-4)

                    # Isolate the actual emitter from projection differences:
                    # both operators receive identical real overlap windows.
                    for index in (False, True):

                        def emit_pair(x, w, c, m, config=config, index=index):
                            trace = {}
                            compress(x, w, dict(c), config, m, index=index, trace=trace)
                            prefix = "attn.indexer.compressor" if index else "attn.compressor"
                            expected = rope(
                                trace["normalized"], jnp.maximum(m.group4_starts, 0), config
                            )
                            actual = csa.emit(
                                trace["group_values"],
                                trace["group_scores"],
                                w[prefix + ".norm.weight"],
                                m.group4_starts,
                                m.group4_starts >= 0,
                                config,
                            )
                            fused = csa.emit(
                                trace["raw_group_values"],
                                trace["raw_group_scores"],
                                w[prefix + ".norm.weight"],
                                m.group4_starts,
                                m.group4_starts >= 0,
                                config,
                                raw_overlap=True,
                            )
                            return expected, actual, fused

                        expected, actual, fused = jax.jit(emit_pair)(
                            call[0], call[2], call[3], call[4]
                        )
                        live = np.asarray(meta.group4_starts) >= 0
                        check(
                            label + f"/identical_emitter_{index}",
                            np.asarray(expected)[live],
                            np.asarray(actual)[live],
                            0,
                            exact=True,
                        )
                        check(
                            label + f"/identical_raw_emitter_{index}",
                            np.asarray(actual),
                            np.asarray(fused),
                            0,
                            exact=True,
                        )
                del weights, reference, candidate
                jax.clear_caches()
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
