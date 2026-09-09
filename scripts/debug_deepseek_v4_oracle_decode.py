"""Faithful native/reference replay of the first from-empty-cache decode failure."""

import argparse
import dataclasses
import functools
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from debug_deepseek_v4_8023 import save_arrays
from replay_deepseek_v4_8023 import difference, logical_cache, traced_step
from run_deepseek_v4_8k_native import CAPACITY, CONTEXT, PROMPT
from run_deepseek_v4_framework import framework_fingerprint
from run_deepseek_v4_paged import ModelWorker, PagedWorkerSession, server_args
from sgl_jax.srt.kernels.deepseek_v4.mhc import head_collapse
from sgl_jax.srt.kernels.deepseek_v4.numerics import (
    config_for_layer,
    rms_norm,
)
from sgl_jax.srt.model_executor.deepseek_v4_reference import DeepSeekV4Reference
from sgl_jax.srt.model_loader.deepseek_v4_native import weight_specs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native_report.read_text())
    oracle = json.loads(args.oracle_report.read_text())
    fingerprint = framework_fingerprint()
    if any(d["framework_source_fingerprint"] != fingerprint for d in (native, oracle)):
        raise ValueError("requires the unchanged failing sources")
    failure = oracle["checks"][-1]
    case, position = failure["case"], failure["position"]
    index = position - PROMPT
    with np.load(args.native_report.parent / "golden_logits.npz") as data:
        golden = data[f"case{case}"]
    with np.load(args.oracle_report.parent / "failure.npz") as data:
        frozen_expected = data["expected"]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "finished": False,
        "faithful": False,
        "framework_source_fingerprint": fingerprint,
        "checkpoint": native["checkpoint"],
        "mhc_backend": native.get("mhc_backend", "reference"),
        "hca_backend": native.get("hca_backend", "reference"),
        "position": position,
        "case": case,
        "layers": [],
        "events": [],
        "first_hidden_difference": None,
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    try:
        sa = server_args(native["checkpoint"], CONTEXT)
        sa.json_model_override_args = json.dumps(
            {"v4_mhc_backend": report["mhc_backend"], "v4_hca_backend": report["hca_backend"]}
        )
        sa.max_total_tokens = CAPACITY
        mesh = create_device_mesh([1, 4], [1, 1])
        put = lambda a: jax.device_put(a, NamedSharding(mesh, P()))
        emit({"event": "loading_native"})
        worker = ModelWorker(sa, mesh)
        runner = worker.model_runner
        session = PagedWorkerSession(worker)
        session.new(0)
        prompt = native["prompts"][case]
        for begin in range(0, PROMPT, 128):
            session.step([(0, prompt[begin : begin + 128])], bucket=1)
            if begin % 2048 == 0:
                emit({"event": "native_prefill", "position": begin})
        for step in range(index):
            session.step([(0, [int(np.argmax(golden[step]))])], decode=True, bucket=1)

        saved = {}
        original = worker.forward_batch_generation

        def capture(batch):
            saved["metadata"] = runner.attn_backend.get_forward_metadata(batch)
            saved["inputs"] = jax.device_get(
                {
                    "ids": batch.input_ids,
                    "positions": batch.positions,
                    "locations": batch.out_cache_loc,
                }
            )
            saved["before"] = jax.device_get(runner.token_to_kv_pool.layers)
            return original(batch)

        worker.forward_batch_generation = capture
        token = int(np.argmax(golden[index]))
        actual, _ = session.step([(0, [token])], decode=True, bucket=1)
        worker.forward_batch_generation = original
        report["native_fidelity"] = difference(golden[index + 1], actual[0])
        if not report["native_fidelity"]["bitwise_equal"]:
            raise AssertionError("native call no longer reproduces its frozen logits")
        save_arrays(args.output / "inputs", saved["inputs"])
        save_arrays(args.output / "metadata", dataclasses.asdict(saved["metadata"]))
        emit({"event": "native_captured", **report["native_fidelity"]})

        # Share only read-only checkpoint arrays, not any activation or cache.
        ref = DeepSeekV4Reference(
            native["checkpoint"], max_context=CONTEXT, progress=lambda _: None
        )
        ref.mesh = mesh
        ref.layers = list(runner.model.layers.get_value())
        ref.shared = {
            **runner.model.shared.get_value(),
            "embed.weight": runner.model.embed_tokens.embedding.get_value(),
            "head.weight": jax.reshard(
                runner.model.lm_head.embedding.get_value(), NamedSharding(mesh, P())
            ),
        }
        ref.reset()
        ref.progress = lambda event: (
            print(json.dumps(event), flush=True) if event.get("tokens", 0) > 1 else None
        )
        emit({"event": "independent_whole_prefill"})
        ref.step(prompt)
        for step in range(index):
            ref.step([int(np.argmax(golden[step]))])
        before_ref = jax.device_get(ref.cache)
        expected, reference_traces = ref.step([token], trace_layers=range(43))
        report["reference_fidelity"] = difference(frozen_expected, np.asarray(expected)[0])
        if not report["reference_fidelity"]["bitwise_equal"]:
            raise AssertionError("shared-weight reference no longer reproduces frozen oracle")
        emit({"event": "independent_failure_reproduced", **report["reference_fidelity"]})

        inputs, metadata = saved["inputs"], saved["metadata"]
        streams = jnp.repeat(ref.shared["embed.weight"][put(inputs["ids"])][:, None], 4, axis=1)

        @functools.lru_cache(maxsize=16)
        def compiled(config, specs):
            return jax.jit(
                jax.shard_map(
                    functools.partial(
                        traced_step,
                        config=config,
                        mhc_backend=report["mhc_backend"],
                        hca_backend=report["hca_backend"],
                    ),
                    mesh=mesh,
                    in_specs=(P(), P(), P(), dict(specs), P(), P(), P()),
                    out_specs=P(),
                    check_vma=False,
                )
            )

        for layer_id, weights in enumerate(ref.layers):
            config = config_for_layer(ref.checkpoint.config, layer_id, CONTEXT)
            initial_streams = streams
            streams, cache, trace = compiled(config, tuple(weight_specs(weights).items()))(
                streams,
                put(inputs["positions"]),
                put(inputs["ids"]),
                weights,
                jax.tree.map(put, saved["before"][layer_id]),
                jax.tree.map(put, metadata),
                put(inputs["locations"]),
            )
            jax.block_until_ready((streams, cache, trace))
            cache, trace = jax.device_get((cache, trace))
            production = jax.device_get(runner.token_to_kv_pool.layers[layer_id])
            row = {
                "layer": layer_id,
                "ratio": config.ratio,
                "hidden": difference(reference_traces[layer_id]["ffn.post"], np.asarray(streams)),
                "production_cache_differences": {
                    k: difference(production[k], v)
                    for k, v in cache.items()
                    if not np.array_equal(production[k], v)
                },
                "stages": {},
                "cache": {},
            }
            for name in trace.keys() & reference_traces[layer_id].keys():
                a, b = reference_traces[layer_id][name], trace[name]
                if a.shape == b.shape:
                    row["stages"][name] = difference(a, b)
            for when, a, b in (
                ("before", before_ref[layer_id], saved["before"][layer_id]),
                ("after", jax.device_get(ref.cache[layer_id]), cache),
            ):
                logical = logical_cache(b, metadata, 0, config, after=when == "after")
                row["cache"][when] = {}
                for name, values in a.items():
                    value = logical[name]
                    expected_value = (
                        values[: len(value)]
                        if name == "window" or name.endswith(".compressed")
                        else values
                    )
                    row["cache"][when][name] = difference(expected_value, value)
            if not row["hidden"]["bitwise_equal"] and report["first_hidden_difference"] is None:
                report["first_hidden_difference"] = layer_id
                save_arrays(args.output / "first-native-input", {"streams": initial_streams})
                save_arrays(args.output / "first-native-before", saved["before"][layer_id])
                save_arrays(args.output / "first-native-after", cache)
                save_arrays(args.output / "first-reference-before", before_ref[layer_id])
                save_arrays(args.output / "first-reference-after", ref.cache[layer_id])
                save_arrays(args.output / "first-native-trace", trace)
                save_arrays(args.output / "first-reference-trace", reference_traces[layer_id])
            report["layers"].append(row)
            emit(
                {
                    "event": "layer_compared",
                    "layer": layer_id,
                    "hidden": row["hidden"],
                    "production_cache_different_fields": len(row["production_cache_differences"]),
                }
            )

        @jax.jit
        def head(streams, shared, weight):
            cfg = config_for_layer(ref.checkpoint.config, 0, CONTEXT)
            s = shared
            x = head_collapse(
                streams,
                s["hc_head_fn"],
                s["hc_head_scale"],
                s["hc_head_base"],
                eps=cfg.eps,
                hc_eps=cfg.hc_eps,
                backend=report["mhc_backend"],
            )
            x = rms_norm(x, s["norm.weight"], cfg.eps)
            return jnp.matmul(
                x,
                weight.T,
                preferred_element_type=jnp.float32,
                out_sharding=NamedSharding(mesh, P("data", "tensor")),
            )

        report["replay_fidelity"] = difference(
            actual,
            np.asarray(
                head(
                    streams,
                    runner.model.shared.get_value(),
                    runner.model.lm_head.embedding.get_value(),
                )
            ),
        )
        report["faithful"] = report["replay_fidelity"]["bitwise_equal"] and all(
            not r["production_cache_differences"] for r in report["layers"]
        )
        report["finished"] = True
        emit(
            {
                "event": "replay_finished",
                "faithful": report["faithful"],
                "first_hidden_difference": report["first_hidden_difference"],
                **report["replay_fidelity"],
            }
        )
        if not report["faithful"]:
            raise AssertionError("native layer replay differs from production")
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
