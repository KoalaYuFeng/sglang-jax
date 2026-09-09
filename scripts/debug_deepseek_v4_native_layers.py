"""Isolate real-weight layer arithmetic from paged/chunked serving adaptation."""

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from sgl_jax.srt.kernels.deepseek_v4.attention import attention
from sgl_jax.srt.kernels.deepseek_v4.mhc import post as mhc_post
from sgl_jax.srt.kernels.deepseek_v4.moe import moe, route
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer, rms_norm
from sgl_jax.srt.kernels.mhc import mhc_pre_fused
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.model_executor.deepseek_v4_reference import layer_step, empty_cache
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer, weight_specs
from sgl_jax.test.test_deepseek_v4_paged import make_batch, make_cache


def native_step(
    streams,
    positions,
    ids,
    weights,
    cache,
    meta,
    locations,
    *,
    config,
    mhc_backend="reference",
    hca_backend="reference",
    csa_backend="reference",
):
    trace = {}
    for sublayer in ("attn", "ffn"):
        residual = streams
        x, post, comb = mhc_pre_fused(
            streams,
            weights["hc_" + sublayer + "_fn"],
            weights["hc_" + sublayer + "_scale"],
            weights["hc_" + sublayer + "_base"],
            hc_mult=config.hc,
            sinkhorn_iters=config.sinkhorn_iters,
            norm_eps=config.eps,
            hc_eps=config.hc_eps,
            dot_precision=jax.lax.Precision.HIGHEST,
        )
        trace[sublayer + ".pre"] = x
        x = rms_norm(x, weights[sublayer + "_norm.weight"], config.eps)
        trace[sublayer + ".norm"] = x
        if sublayer == "attn":
            attn_trace = {}
            x, cache = attention(
                x,
                positions,
                weights,
                cache,
                config,
                meta,
                locations,
                trace=attn_trace,
                hca_backend=hca_backend,
                csa_backend=csa_backend,
            )
            trace.update({"attn." + k: v for k, v in attn_trace.items()})
        else:
            indices, routing = route(x, ids, weights, config, meta)
            trace.update(expert_ids=indices, routing_weights=routing)
            x = moe(x, ids, weights, config, meta)
        trace[sublayer + ".operator"] = x
        streams = mhc_post(x, residual, post, comb, backend=mhc_backend)
        trace[sublayer + ".post"] = streams
    return streams, cache, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--stop-on-difference", action="store_true")
    parser.add_argument("--mhc-backend", choices=("pallas", "reference"), default="pallas")
    parser.add_argument("--hca-backend", choices=("pallas", "reference"), default="pallas")
    parser.add_argument("--csa-backend", choices=("pallas", "reference"), default="pallas")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context", type=int, default=384)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--chunks", type=int, nargs="+", default=[132, 31])
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--resume-capture", type=Path)
    parser.add_argument("--stop-on-hca-attention-difference", action="store_true")
    parser.add_argument("--uncompressed-capture", action="store_true")
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    ids = np.asarray(json.loads(args.fixture.read_text())["prompts"][0], np.int32)
    if args.prompt_tokens is not None:
        ids = ids[: args.prompt_tokens]
    if len(ids) > args.context or any(chunk <= 0 for chunk in args.chunks):
        raise ValueError("prompt must fit context and chunks must be positive")
    pages = None if args.context == 384 else [list(range(1, (args.context + 127) // 128 + 1))]
    report = {
        "source_fingerprint": framework_fingerprint(),
        "fixture": str(args.fixture),
        "context": args.context,
        "tokens": len(ids),
        "chunks": args.chunks,
        "checkpoint": args.checkpoint,
        "start_layer": args.start_layer,
        "requested_layers": args.layers,
        "resume_capture": str(args.resume_capture) if args.resume_capture else None,
        "compressed_capture": not args.uncompressed_capture,
        "mhc_backend": args.mhc_backend,
        "hca_backend": args.hca_backend,
        "csa_backend": args.csa_backend,
        "events": [],
    }

    def emit(row):
        report["events"].append(row)
        if args.output:
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    put = lambda value: jax.device_put(value, NamedSharding(mesh, P()))
    if args.start_layer:
        from debug_deepseek_v4_8023 import load_arrays

        if args.resume_capture is None:
            raise ValueError("a resumed layer requires its immutable reference capture")
        prior = json.loads((args.resume_capture / "report.json").read_text())
        if any(
            prior[key] != report[key]
            for key in ("source_fingerprint", "context", "tokens", "fixture")
        ):
            raise ValueError("resume must use matching source, fixture and context")
        streams = put(
            load_arrays(args.resume_capture / f"layer-{args.start_layer - 1:02d}-reference")[
                "ffn.post"
            ]
        )
    else:
        embed = put(checkpoint.read_tensor("embed.weight"))
        streams = jnp.repeat(embed[put(ids)][:, None], 4, axis=1)
    backend = V4PagedBackend(max_context=args.context, mesh=mesh)
    for layer in range(args.start_layer, args.layers):
        config = config_for_layer(checkpoint.config, layer, args.context)
        weights = load_layer(checkpoint, layer, mesh)
        emit({"layer": layer, "event": "loaded"})
        reference = jax.jit(
            jax.shard_map(
                partial(layer_step, config=config),
                mesh=mesh,
                in_specs=(P(), P(), P(), weight_specs(weights), P()),
                out_specs=P(),
                check_vma=False,
            )
        )
        expected, ref_cache, ref_trace = reference(
            streams,
            put(np.arange(len(ids), dtype=np.int32)),
            put(ids),
            weights,
            put(empty_cache(config)),
        )
        if args.output and config.ratio == 128:
            from debug_deepseek_v4_8023 import save_arrays as write_arrays

            save_arrays = partial(write_arrays, compressed=not args.uncompressed_capture)

            save_arrays(args.output / f"layer-{layer:02d}-inputs", {"streams": streams, "ids": ids})
            save_arrays(args.output / f"layer-{layer:02d}-reference-cache", ref_cache)
            save_arrays(args.output / f"layer-{layer:02d}-reference", ref_trace)
        native = jax.jit(
            jax.shard_map(
                partial(
                    native_step,
                    config=config,
                    mhc_backend=args.mhc_backend,
                    hca_backend=args.hca_backend,
                    csa_backend=args.csa_backend,
                ),
                mesh=mesh,
                in_specs=(P(), P(), P(), weight_specs(weights), P(), P(), P()),
                out_specs=P(),
                check_vma=False,
            )
        )
        layer_differs = False
        hca_attention_differs = False
        for chunk_size in args.chunks:
            cache = put(make_cache(config, capacity=max(1536, args.context)))
            pieces, traces = [], []
            for begin in range(0, len(ids), chunk_size):
                end = min(begin + chunk_size, len(ids))
                padding = 128 - (end - begin) if chunk_size < 128 else 0
                batch = make_batch([begin], [end - begin], padding=padding, pages=pages)
                x = jnp.pad(streams[begin:end], ((0, padding), (0, 0), (0, 0)))
                output, cache, trace = native(
                    x,
                    put(batch.positions),
                    put(np.pad(ids[begin:end], (0, padding))),
                    weights,
                    cache,
                    backend.get_forward_metadata(batch),
                    put(batch.out_cache_loc),
                )
                pieces.append(output[: end - begin])
                traces.append({k: v[: end - begin] for k, v in trace.items()})
            actual = jnp.concatenate(pieces)
            comparison = compare_arrays(expected, actual)
            layer_differs |= not comparison["values_equal"]
            emit(
                {
                    "layer": layer,
                    "chunk": chunk_size,
                    "stage": "output",
                    **comparison,
                }
            )
            for key, ref_value in ref_trace.items():
                if key not in traces[0]:
                    continue
                got = jnp.concatenate([trace[key] for trace in traces])
                difference = compare_arrays(ref_value, got)
                difference["first_different"] = np.argwhere(
                    np.asarray(ref_value) != np.asarray(got)
                )[:8].tolist()
                if difference["values_equal"]:
                    continue
                if config.ratio == 128 and key == "attn.attention_value":
                    hca_attention_differs = True
                emit(
                    {
                        "layer": layer,
                        "chunk": chunk_size,
                        "stage": key,
                        **difference,
                    }
                )
            if args.output and config.ratio == 128:
                save_arrays(args.output / f"layer-{layer:02d}-chunk{chunk_size}-cache", cache)
                save_arrays(
                    args.output / f"layer-{layer:02d}-chunk{chunk_size}-trace",
                    {k: jnp.concatenate([trace[k] for trace in traces]) for k in traces[0]},
                )
        streams = expected
        if layer_differs and args.stop_on_difference:
            raise AssertionError(f"first layer-level numerical divergence: {layer}")
        if hca_attention_differs and args.stop_on_hca_attention_difference:
            raise AssertionError(f"first HCA attention numerical divergence: {layer}")


if __name__ == "__main__":
    main()
