"""Faithful multi-token, layer-major replay of the CSA whole-model A/B capture.

Intermediate differences are diagnostic only until all captured final logits
reproduce the uninstrumented ModelRunner. No acceptance tolerance is changed.
"""

import argparse
import functools
import hashlib
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from replay_deepseek_v4_8023 import difference, logical_cache, request_trace, traced_step
from run_deepseek_v4_framework import framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4.mhc import head_collapse
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer, rms_norm
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedMetadata
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer, weight_specs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def cache_difference(left, right):
    result = {}
    for key in left:
        comparison = difference(left[key], right[key])
        if comparison["different_elements"]:
            result[key] = comparison
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--resume-head", type=Path)
    args = parser.parse_args()
    capture = json.loads((args.capture / "report.json").read_text())
    fingerprint = framework_fingerprint()
    if not capture["complete"] or capture["framework_source_fingerprint"] != fingerprint:
        raise ValueError("requires a complete same-source whole-model A/B capture")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "faithful": False,
        "cache_only": args.cache_only,
        "capture": str(args.capture),
        "framework_source_fingerprint": fingerprint,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": capture["checkpoint"],
        "resume_head": str(args.resume_head) if args.resume_head else None,
        "events": [],
        "layers": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    try:
        checkpoint = DeepSeekV4Checkpoint(capture["checkpoint"])
        configs = [config_for_layer(checkpoint.config, layer, 8192) for layer in range(43)]
        positions = tuple(range(capture["capture_begin"], capture["position"] + 1))
        frames = {}
        for mode in ("reference", "pallas"):
            for position in positions:
                path = args.capture / mode / f"step-{position}"
                frames[mode, position] = {
                    "path": path,
                    "inputs": load_arrays(path / "inputs"),
                    "metadata": V4PagedMetadata(**load_arrays(path / "metadata")),
                }
        for position in positions:
            left, right = (frames[mode, position] for mode in ("reference", "pallas"))
            for key in left["inputs"]:
                if not np.array_equal(left["inputs"][key], right["inputs"][key]):
                    raise ValueError(f"nonidentical A/B inputs: {position}/{key}")

        if not args.cache_only:
            mesh = create_device_mesh([1, 4], [1, 1])
            put = lambda value: jax.device_put(value, NamedSharding(mesh, P()))
            if args.resume_head:
                prior = json.loads((args.resume_head / "report.json").read_text())
                if (
                    prior["framework_source_fingerprint"] != fingerprint
                    or prior["capture"] != str(args.capture)
                    or len(prior["layers"]) != 43
                ):
                    raise ValueError("head resume requires all 43 matching-source layer traces")
                report["layers"] = prior["layers"]
                report["first_hidden_difference"] = prior["first_hidden_difference"]
                for (mode, position), frame in frames.items():
                    frame["streams"] = put(
                        load_arrays(args.resume_head / f"{mode}-{position}-layer-42-trace")[
                            "ffn.post"
                        ]
                    )
            else:
                embedding = checkpoint.read_tensor("embed.weight")
                for frame in frames.values():
                    frame["streams"] = put(
                        np.repeat(embedding[frame["inputs"]["ids"]][:, None], configs[0].hc, axis=1)
                    )
                del embedding
            for frame in frames.values():
                frame["device_metadata"] = jax.tree.map(put, frame["metadata"])

            @functools.lru_cache(maxsize=12)
            def compiled(config, specs, mode):
                return jax.jit(
                    jax.shard_map(
                        functools.partial(
                            traced_step,
                            config=config,
                            mhc_backend="pallas",
                            hca_backend="pallas",
                            csa_backend=mode,
                        ),
                        mesh=mesh,
                        in_specs=(P(), P(), P(), dict(specs), P(), P(), P()),
                        out_specs=P(),
                        check_vma=False,
                    )
                )

        first_hidden = {}
        for layer, config in enumerate(() if args.resume_head else configs):
            layer_report = {"layer": layer, "ratio": config.ratio}
            before, logical_before = {}, {}
            for mode in ("reference", "pallas"):
                frame = frames[mode, positions[0]]
                before[mode] = load_arrays(frame["path"] / "before" / f"layer-{layer:02d}")
                logical_before[mode] = logical_cache(
                    before[mode], frame["metadata"], 0, config, after=False
                )
            layer_report["before"] = cache_difference(
                logical_before["reference"], logical_before["pallas"]
            )
            if args.cache_only:
                report["layers"].append(layer_report)
                emit({"event": "cache_compared", **layer_report})
                continue

            weights = load_layer(checkpoint, layer, mesh)
            data = {}
            for mode in ("reference", "pallas"):
                cache = jax.tree.map(put, before[mode])
                run = compiled(config, tuple(weight_specs(weights).items()), mode)
                for position in positions:
                    frame = frames[mode, position]
                    stem = args.output / f"{mode}-{position}-layer-{layer:02d}"
                    save_arrays(stem.with_name(stem.name + "-input"), {"streams": frame["streams"]})
                    streams, cache, trace = run(
                        frame["streams"],
                        put(frame["inputs"]["positions"]),
                        put(frame["inputs"]["ids"]),
                        weights,
                        cache,
                        frame["device_metadata"],
                        put(frame["inputs"]["locations"]),
                    )
                    jax.block_until_ready((streams, cache, trace))
                    frame["streams"] = streams
                    trace = jax.device_get(trace)
                    save_arrays(stem.with_name(stem.name + "-trace"), trace)
                    data[mode, position] = {
                        "streams": np.asarray(streams),
                        "trace": request_trace(
                            trace, frame["metadata"], 0, config, before[mode]["window"].shape[0]
                        ),
                        "cache": logical_cache(
                            jax.device_get(cache), frame["metadata"], 0, config, after=True
                        ),
                    }
            layer_report["steps"] = {}
            for position in positions:
                left, right = (data[mode, position] for mode in ("reference", "pallas"))
                hidden = difference(left["streams"], right["streams"])
                if hidden["different_elements"] and position not in first_hidden:
                    first_hidden[position] = {"layer": layer, **hidden}
                layer_report["steps"][str(position)] = {
                    "hidden": hidden,
                    "after": cache_difference(left["cache"], right["cache"]),
                    "stages": {
                        key: difference(left["trace"][key], right["trace"][key])
                        for key in left["trace"].keys() & right["trace"].keys()
                    },
                }
            report["layers"].append(layer_report)
            report["first_hidden_difference"] = first_hidden
            emit(
                {
                    "event": "layer_compared",
                    "layer": layer,
                    "before_different_fields": list(layer_report["before"]),
                    "hidden": {k: v["hidden"] for k, v in layer_report["steps"].items()},
                }
            )
            del weights, cache, data, before, logical_before

        if not args.cache_only:
            shared = {
                name: put(checkpoint.read_tensor(name))
                for name in ("hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight")
            }
            head_weight = jax.device_put(
                checkpoint.read_tensor("head.weight"), NamedSharding(mesh, P("tensor", None))
            )

            @functools.partial(
                jax.shard_map, mesh=mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False
            )
            def collapse(streams, shared):
                config = configs[0]
                collapsed = head_collapse(
                    streams,
                    shared["hc_head_fn"],
                    shared["hc_head_scale"],
                    shared["hc_head_base"],
                    eps=config.eps,
                    hc_eps=config.hc_eps,
                    backend="pallas",
                )
                return rms_norm(collapsed, shared["norm.weight"], config.eps)

            @jax.jit
            def head(streams, shared, weight):
                hidden = collapse(streams, shared)
                return jnp.matmul(
                    hidden,
                    weight.T,
                    preferred_element_type=jnp.float32,
                    out_sharding=NamedSharding(mesh, P("data", "tensor")),
                )

            checks = {}
            for (mode, position), frame in frames.items():
                logits = np.asarray(head(frame["streams"], shared, head_weight), np.float32)
                expected = load_arrays(frame["path"] / "production")["logits"]
                check = difference(expected, logits)
                check["top1_equal"] = bool(
                    np.array_equal(np.argmax(expected, -1), np.argmax(logits, -1))
                )
                check["all_finite"] = bool(np.all(np.isfinite(logits)))
                checks[f"{mode}/{position}"] = check
                save_arrays(args.output / f"{mode}-{position}-logits", {"logits": logits})
            report["production_logit_checks"] = checks
            report["faithful"] = all(
                c["finite_mask_equal"] and c["all_finite"] and c["nrmse"] < 1e-5 and c["top1_equal"]
                for c in checks.values()
            )
            if not report["faithful"]:
                raise AssertionError("instrumented replay does not reproduce production logits")
        report["complete"] = True
        emit({"event": "replay_complete", "faithful": report["faithful"]})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
