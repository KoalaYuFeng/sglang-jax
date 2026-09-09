"""Layer/state replay of a frozen production failure, not a performance path.

Replay must match the captured production logits before its intermediate
comparisons can be trusted. Instrumented results alone do not prove a fix.
"""

import argparse
import functools
import importlib
import json
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from debug_deepseek_v4_8023 import load_arrays, save_arrays
from debug_deepseek_v4_native_layers import native_step
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4.mhc import head_collapse
from sgl_jax.srt.kernels.deepseek_v4.numerics import (
    config_for_layer,
    rms_norm,
)
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedMetadata
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer, weight_specs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

ATTENTION = importlib.import_module("sgl_jax.srt.kernels.deepseek_v4.attention")
COMPRESS = ATTENTION.compress


def difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    result = compare_arrays(a, b)
    different = a != b
    result["different_elements"] = int(np.count_nonzero(different))
    result["elements"] = a.size
    result["first_different"] = np.argwhere(different)[:4].tolist()
    return result


def logical_cache(cache, metadata, row, config, *, after):
    """Compare request-owned content, not unequal physical addresses/free pages."""
    length = int(metadata.seq_lens[row] if after else metadata.prefix_lens[row])
    positions = np.arange(length)
    pages = np.asarray(metadata.page_table[row])
    physical = pages[positions // 128] * 128 + positions % 128
    slot = int(metadata.req_slots[row])
    result = {"window": cache["window"][physical]}
    for key, value in cache.items():
        if key == "window":
            continue
        if key.endswith(".compressed"):
            terminal = physical[config.ratio - 1 :: config.ratio]
            result[key] = value[terminal // config.ratio]
        elif ".snapshot_" in key:
            result[key] = value[pages[: length // 128]]
        else:
            result[key] = value[slot]
    return result


def logical_attention_indices(indices, metadata, row, config, window_rows):
    reverse = {int(page): logical for logical, page in enumerate(metadata.page_table[row]) if page}
    result = []
    for index in np.asarray(indices).tolist():
        if index < 0:
            result.append(-1)
            continue
        compressed = index >= window_rows
        physical = (index - window_rows) * config.ratio if compressed else index
        position = reverse[physical // 128] * 128 + physical % 128
        result.append(config.max_context + position // config.ratio if compressed else position)
    return np.asarray(result, np.int32)


def request_trace(trace, metadata, row, config, window_rows):
    result = {}
    groups = metadata.group4_requests if config.ratio == 4 else metadata.group128_requests
    starts = metadata.group4_starts if config.ratio == 4 else metadata.group128_starts
    for name, value in trace.items():
        if name == "attn.indices":
            result[name] = logical_attention_indices(value[row], metadata, row, config, window_rows)
        elif name.startswith("compressor."):
            suffix = name.rsplit(".", 1)[1]
            if suffix == "destination":
                continue  # physical target is checked through logical cache contents
            result[name] = (
                value[row] if suffix in ("kv", "scores") else value[(groups == row) & (starts >= 0)]
            )
        else:
            result[name] = value[row]
    return result


def traced_step(
    *args, config, mhc_backend="reference", hca_backend="reference", csa_backend="reference"
):
    traces = {}

    def compress(*args, index=False, hca_backend="reference", csa_backend="reference"):
        details = {}
        cache = COMPRESS(
            *args, index=index, trace=details, hca_backend=hca_backend, csa_backend=csa_backend
        )
        prefix = "index" if index else "main"
        traces.update({f"compressor.{prefix}.{name}": value for name, value in details.items()})
        return cache

    with patch.object(ATTENTION, "compress", compress):
        streams, cache, stages = native_step(
            *args,
            config=config,
            mhc_backend=mhc_backend,
            hca_backend=hca_backend,
            csa_backend=csa_backend,
        )
    return streams, cache, {**stages, **traces}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads((args.capture / "report.json").read_text())
    fingerprint = framework_fingerprint()
    if not capture["finished"] or capture["framework_source_fingerprint"] != fingerprint:
        raise ValueError("requires a complete same-source production reproduction")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "finished": False,
        "faithful": False,
        "framework_source_fingerprint": fingerprint,
        "capture": str(args.capture),
        "mhc_backend": capture.get("mhc_backend", "reference"),
        "hca_backend": capture.get("hca_backend", "reference"),
        "csa_backend": capture.get("csa_backend", "reference"),
        "events": [],
        "layers": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    try:
        checkpoint = DeepSeekV4Checkpoint(capture["checkpoint"])
        mesh = create_device_mesh([1, 4], [1, 1])
        put = lambda a: jax.device_put(a, NamedSharding(mesh, P()))
        configs = [config_for_layer(checkpoint.config, layer, 8192) for layer in range(43)]
        embedding = checkpoint.read_tensor("embed.weight")
        frames = {}
        for name in capture["frames"]:
            path = args.capture / name
            inputs = load_arrays(path / "inputs")
            metadata = V4PagedMetadata(**load_arrays(path / "metadata"))
            frames[name] = {
                "path": path,
                "inputs": inputs,
                "metadata": metadata,
                "device_metadata": jax.tree.map(put, metadata),
                "streams": put(np.repeat(embedding[inputs["ids"]][:, None], configs[0].hc, axis=1)),
            }
        del embedding

        @functools.lru_cache(maxsize=8)
        def compiled(config, specs):
            return jax.jit(
                jax.shard_map(
                    functools.partial(
                        traced_step,
                        config=config,
                        mhc_backend=report["mhc_backend"],
                        hca_backend=report["hca_backend"],
                        csa_backend=report["csa_backend"],
                    ),
                    mesh=mesh,
                    in_specs=(P(), P(), P(), dict(specs), P(), P(), P()),
                    out_specs=P(),
                    check_vma=False,
                )
            )

        first_hidden = None
        for layer_id, config in enumerate(configs):
            weights = load_layer(checkpoint, layer_id, mesh)
            run = compiled(config, tuple(weight_specs(weights).items()))
            data, layer_report = (
                {},
                {"layer": layer_id, "ratio": config.ratio, "production_cache": {}},
            )
            for name, frame in frames.items():
                before = load_arrays(frame["path"] / "before" / f"layer-{layer_id:02d}")
                save_arrays(
                    args.output / f"{name}-layer-{layer_id:02d}-input",
                    {"streams": frame["streams"]},
                )
                streams, updated, trace = run(
                    frame["streams"],
                    put(frame["inputs"]["positions"]),
                    put(frame["inputs"]["ids"]),
                    weights,
                    jax.tree.map(put, before),
                    frame["device_metadata"],
                    put(frame["inputs"]["locations"]),
                )
                jax.block_until_ready((streams, updated, trace))
                frame["streams"] = streams
                trace = jax.device_get(trace)
                save_arrays(args.output / f"{name}-layer-{layer_id:02d}-trace", trace)
                after = jax.device_get(updated)
                production = load_arrays(frame["path"] / "after" / f"layer-{layer_id:02d}")
                layer_report["production_cache"][name] = {
                    key: difference(production[key], value)
                    for key, value in after.items()
                    if not np.array_equal(production[key], value)
                }
                data[name] = {
                    "before": before,
                    "after": after,
                    "trace": trace,
                    "streams": np.asarray(streams),
                }
            comparisons = {}
            order = frames["B2"]["inputs"]["request_order"].tolist()
            for case in (0, 1):
                name, row = f"B1-case{case}", order.index(case)
                left, right = data[name], data["B2"]
                comparison = {"hidden": difference(left["streams"][0], right["streams"][row])}
                if comparison["hidden"]["different_elements"] and first_hidden is None:
                    first_hidden = {"layer": layer_id, "case": case, **comparison["hidden"]}
                for when in ("before", "after"):
                    a = logical_cache(
                        left[when], frames[name]["metadata"], 0, config, after=when == "after"
                    )
                    b = logical_cache(
                        right[when], frames["B2"]["metadata"], row, config, after=when == "after"
                    )
                    comparison[when] = {key: difference(a[key], b[key]) for key in a}
                a = request_trace(
                    left["trace"],
                    frames[name]["metadata"],
                    0,
                    config,
                    left["before"]["window"].shape[0],
                )
                b = request_trace(
                    right["trace"],
                    frames["B2"]["metadata"],
                    row,
                    config,
                    right["before"]["window"].shape[0],
                )
                comparison["stages"] = {key: difference(a[key], b[key]) for key in a}
                comparisons[str(case)] = comparison
            layer_report["requests"] = comparisons
            report["layers"].append(layer_report)
            report["first_hidden_difference"] = first_hidden
            emit(
                {
                    "event": "layer_compared",
                    "layer": layer_id,
                    "hidden": {k: v["hidden"] for k, v in comparisons.items()},
                    "production_cache_different_fields": {
                        k: len(v) for k, v in layer_report["production_cache"].items()
                    },
                }
            )
            del weights, updated, data

        shared = {
            name: put(checkpoint.read_tensor(name))
            for name in ("hc_head_fn", "hc_head_scale", "hc_head_base", "norm.weight")
        }
        head_weight = jax.device_put(
            checkpoint.read_tensor("head.weight"), NamedSharding(mesh, P("tensor", None))
        )

        @jax.jit
        def head(streams, shared, weight):
            config = configs[0]
            collapsed = head_collapse(
                streams,
                shared["hc_head_fn"],
                shared["hc_head_scale"],
                shared["hc_head_base"],
                eps=config.eps,
                hc_eps=config.hc_eps,
                backend=report["mhc_backend"],
            )
            hidden = rms_norm(collapsed, shared["norm.weight"], config.eps)
            return jnp.matmul(
                hidden,
                weight.T,
                preferred_element_type=jnp.float32,
                out_sharding=NamedSharding(mesh, P("data", "tensor")),
            )

        checks = {}
        for name, frame in frames.items():
            logits = np.asarray(head(frame["streams"], shared, head_weight), np.float32)
            expected = load_arrays(frame["path"] / "production")["logits"]
            checks[name] = difference(expected, logits)
            checks[name]["top1_equal"] = bool(
                np.array_equal(np.argmax(expected, -1), np.argmax(logits, -1))
            )
            save_arrays(args.output / f"{name}-logits", {"logits": logits})
        report["production_logit_checks"] = checks
        report["faithful"] = all(
            c["finite_mask_equal"] and c["nrmse"] < 1e-5 and c["top1_equal"]
            for c in checks.values()
        )
        report["finished"] = True
        emit(
            {
                "event": "replay_finished",
                "faithful": report["faithful"],
                "checks": checks,
                "first_hidden_difference": first_hidden,
            }
        )
        if not report["faithful"]:
            raise AssertionError("instrumented replay does not reproduce production logits")
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
