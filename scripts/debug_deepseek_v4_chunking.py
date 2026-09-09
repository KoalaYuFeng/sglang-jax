"""Locate the first real-weight layer where whole/chunked prefill diverges."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import AutoTokenizer

from debug_deepseek_v4_8023 import save_arrays
from run_deepseek_v4_8k_worker import CHUNK, CONTEXT, make_tokens
from run_deepseek_v4_framework import compare_arrays

from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    compiled_layer,
    config_for_layer,
    empty_cache,
    load_layer,
)
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint


def first_different_token(a, b):
    a, b = np.asarray(a), np.asarray(b)
    unequal = a != b
    if np.issubdtype(a.dtype, np.floating):
        unequal |= np.isnan(a) != np.isnan(b)
    if not np.any(unequal):
        return None
    per_token = unequal.reshape(a.shape[0], -1).any(axis=1)
    return int(np.flatnonzero(per_token)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=271)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--save-dir", type=Path)
    options = parser.parse_args()
    checkpoint = DeepSeekV4Checkpoint(options.checkpoint)
    if options.fixture:
        prompt = json.loads(options.fixture.read_text())["prompts"][options.case]
        ids = np.asarray(prompt[: options.tokens], np.int32)
        if len(ids) != options.tokens:
            raise ValueError("fixture is shorter than the requested token count")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            options.checkpoint, local_files_only=True, trust_remote_code=False
        )
        ids = np.asarray(make_tokens(tokenizer, options.tokens), np.int32)
    if options.save_dir:
        options.save_dir.mkdir(parents=True, exist_ok=False)
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    sharding = NamedSharding(mesh, P())
    ids_device = jax.device_put(ids, sharding)
    positions = jax.device_put(np.arange(len(ids), dtype=np.int32), sharding)
    embedding = jax.device_put(checkpoint.read_tensor("embed.weight"), sharding)
    whole = jnp.repeat(embedding[ids_device, None, :], 4, axis=1)
    chunked = whole
    report = {"tokens": len(ids), "layers": [], "first_divergent_layer": None}

    for layer_id in range(options.layers):
        config = config_for_layer(checkpoint.config, layer_id, CONTEXT)
        start = time.perf_counter()
        weights = load_layer(checkpoint, layer_id, mesh)
        whole_output, whole_cache, whole_trace = compiled_layer(config, mesh)(
            whole, positions, ids_device, weights, jax.device_put(empty_cache(config), sharding)
        )
        jax.block_until_ready(whole_output)
        print(json.dumps({"event": "whole_layer_ready", "layer": layer_id}), flush=True)
        chunk_cache = jax.device_put(empty_cache(config), sharding)
        chunk_outputs = []
        chunk_traces = []
        for begin in range(0, len(ids), CHUNK):
            end = min(begin + CHUNK, len(ids))
            output, chunk_cache, trace = compiled_layer(config, mesh)(
                chunked[begin:end],
                positions[begin:end],
                ids_device[begin:end],
                weights,
                chunk_cache,
            )
            chunk_outputs.append(output)
            chunk_traces.append(trace)
            if end % 2048 == 0:
                print(
                    json.dumps({"event": "chunk_progress", "layer": layer_id, "end": end}),
                    flush=True,
                )
        chunk_output = jnp.concatenate(chunk_outputs)
        jax.block_until_ready((whole_output, chunk_output, whole_cache, chunk_cache))
        output_check = compare_arrays(whole_output, chunk_output)
        row = {
            "layer": layer_id,
            "ratio": config.ratio,
            "seconds": time.perf_counter() - start,
            "output": output_check,
        }
        if not output_check["bitwise_equal"]:
            report["first_divergent_layer"] = layer_id
            row["first_different_output_token"] = first_different_token(
                jax.device_get(whole_output), jax.device_get(chunk_output)
            )
            if options.save_dir:
                token = row["first_different_output_token"] or 0
                save_arrays(options.save_dir / "input", {"streams": whole, "ids": ids})
                save_arrays(options.save_dir / "whole-cache", whole_cache)
                save_arrays(options.save_dir / "chunk-cache", chunk_cache)
                save_arrays(
                    options.save_dir / "whole-attention-input", {"x": whole_trace["attn.norm"]}
                )
                save_arrays(
                    options.save_dir / "chunk-attention-input",
                    {"x": jnp.concatenate([t["attn.norm"] for t in chunk_traces])},
                )
                save_arrays(
                    options.save_dir / "whole-first-token",
                    {
                        k: v[token : token + 1]
                        for k, v in whole_trace.items()
                        if v.ndim and v.shape[0] == len(ids)
                    },
                )
                save_arrays(
                    options.save_dir / "chunk-first-token",
                    {
                        k: jnp.concatenate([t[k] for t in chunk_traces])[token : token + 1]
                        for k, v in whole_trace.items()
                        if v.ndim and v.shape[0] == len(ids)
                    },
                )
            row["trace"] = {}
            for name, expected in whole_trace.items():
                if expected.ndim == 0 or expected.shape[0] != len(ids):
                    continue
                actual = jnp.concatenate([trace[name] for trace in chunk_traces])
                expected_host, actual_host = jax.device_get((expected, actual))
                check = compare_arrays(expected_host, actual_host)
                check["first_different_token"] = first_different_token(expected_host, actual_host)
                if name == "index_selected" and not check["bitwise_equal"]:
                    token = check["first_different_token"]
                    check["different_entries"] = int(np.count_nonzero(expected_host != actual_host))
                    check["whole_first_20"] = expected_host[token, :20].tolist()
                    check["chunked_first_20"] = actual_host[token, :20].tolist()
                row["trace"][name] = check
            report["layers"].append(row)
            print(json.dumps(row), flush=True)
            break
        report["layers"].append(row)
        whole, chunked = whole_output, chunk_output
        del weights, whole_cache, chunk_cache, whole_trace, chunk_traces
        gc.collect()
        print(json.dumps(row), flush=True)

    options.report.parent.mkdir(parents=True, exist_ok=True)
    options.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(options.report), **report}), flush=True)


if __name__ == "__main__":
    main()
