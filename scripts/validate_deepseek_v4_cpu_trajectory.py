"""Layer-major CPU/TPU teacher-forced trajectory; never a performance benchmark.

Uses the production DecoderLayer, independently propagated CPU Block outputs,
and bounded one-layer weight staging. A truncated run's head is diagnostic,
not the deployed model's logits. Historical compressor gates are not replaced.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch
from analyze_deepseek_v4_native_profile import fingerprint
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.kernels.deepseek_v4.dense import DenseKernels
from sgl_jax.srt.kernels.deepseek_v4.mhc import head_collapse
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer, weight_specs
from sgl_jax.srt.models.deepseek_v4 import DeepseekV4DecoderLayer, attention_uses_tp
from sgl_jax.test.deepseek_v4_cpu_oracle import (
    load_module_weights,
    official_module,
    tensor_from_checkpoint,
)
from sgl_jax.test.test_deepseek_v4_paged import make_batch, make_cache


def metrics(actual, expected):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError(f"invalid comparison: {a.shape}, {b.shape}")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        raise ValueError("nonfinite trajectory")
    diff = a - b
    return {
        "shape": list(a.shape),
        "finite": finite,
        "equal": bool(np.array_equal(a, b)),
        "nrmse": float(np.linalg.norm(diff) / max(np.linalg.norm(b), 1e-12)),
        "max_vector_nrmse": float(
            np.max(
                np.linalg.norm(diff, axis=-1)
                / np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
            )
        ),
        "max_abs": float(np.max(np.abs(diff))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--prefill", type=int, default=127)
    parser.add_argument("--decode", type=int, default=5)
    parser.add_argument("--chunk", type=int, default=64)
    options = parser.parse_args()
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("requires four TPU chips")
    if not (0 < options.prefill and 0 < options.decode and 0 < options.chunk <= 128):
        raise ValueError("invalid sequence settings")
    cp = DeepSeekV4Checkpoint(options.checkpoint)
    if not 0 < options.layers <= cp.config["num_hidden_layers"]:
        raise ValueError("invalid layer count")
    total = options.prefill + options.decode
    context = ((total + 127) // 128) * 128
    options.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    module, args = official_module(cp, context)
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    put = lambda x: jax.device_put(x, NamedSharding(mesh, P()))
    dense = DenseKernels(
        fp8_backend="gmm", fused_norm=True, merged_projections=True, fused_wo_a=True
    )
    # Fixed teacher forcing; no selection from either candidate's logits.
    ids = np.random.default_rng(20260912).integers(
        0, cp.config["vocab_size"], size=total, dtype=np.int32
    )
    embedded = cp.read_tensor("embed.weight", ids)
    native = np.repeat(embedded[:, None, :], args.hc_mult, axis=1)
    expected = native.astype(np.float32)
    report = {
        "complete": False,
        "diagnostic_only": True,
        "historical_gate_replaced": False,
        "source_fingerprint": fingerprint(),
        "checkpoint": str(options.checkpoint),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_model_sha256": hashlib.sha256(
            (options.checkpoint / "inference/model.py").read_bytes()
        ).hexdigest(),
        "torch_version": torch.__version__,
        "layers_requested": options.layers,
        "full_model": options.layers == cp.config["num_hidden_layers"],
        "prefill": options.prefill,
        "decode": options.decode,
        "native_chunk": options.chunk,
        "context": context,
        "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "embedding_sha256": hashlib.sha256(embedded.tobytes()).hexdigest(),
        "execution": {
            "attention_tp": True,
            "ep": 4,
            "dp": 1,
            "moe_backend": "gmm_tuned",
            "fp4_scale_transposed": True,
            "mhc": "pallas",
            "csa": "pallas",
            "hca": "pallas",
            "dense": dense.__dict__,
        },
        "layers": [],
    }
    np.save(options.output / "input_ids.npy", ids)

    def emit(event, **data):
        print(json.dumps(dict(event=event, time=time.time(), **data)), flush=True)
        temporary = options.output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(options.output / "report.json")

    pages = [list(range(1, 1 + context // 128))]
    backend = V4PagedBackend(max_context=context, mesh=mesh)
    try:
        for layer in range(options.layers):
            cfg = config_for_layer(cp.config, layer, context)
            ap = attention_uses_tp(cfg, True)
            emit("loading", layer=layer)
            weights = load_layer(
                cp,
                layer,
                mesh,
                attention_tp=ap,
                merged_projections=True,
                fp4_scale_transposed=True,
            )
            specs = weight_specs(weights, attention_tp=ap)
            cache = put(make_cache(cfg, capacity=context + 128, requests=1))

            def build(decode, cfg=cfg, specs=specs):
                def run(x, positions, token_ids, w, state, metadata, locations):
                    return DeepseekV4DecoderLayer(
                        cfg,
                        attention_tp=True,
                        csa_decode_batch=decode,
                        moe_backend="gmm_tuned",
                        dense_kernels=dense,
                    )(x, positions, token_ids, w, state, metadata, locations)

                return jax.jit(
                    jax.shard_map(
                        run,
                        mesh=mesh,
                        in_specs=(P(), P(), P(), specs, P(), P(), P()),
                        out_specs=(P(), P()),
                        check_vma=False,
                    )
                )

            pref, dec = build(False), build(True)
            pieces = []
            spans = [
                (i, min(i + options.chunk, options.prefill), False)
                for i in range(0, options.prefill, options.chunk)
            ]
            spans += [(i, i + 1, True) for i in range(options.prefill, total)]
            for begin, end, decode in spans:
                padding = 0 if decode else options.chunk - (end - begin)
                batch = make_batch(
                    [begin], [end - begin], pages=pages, padding=padding, decode=decode
                )
                value = jnp.asarray(native[begin:end], jnp.bfloat16)
                value = jnp.pad(value, ((0, padding), (0, 0), (0, 0)))
                emit("native", layer=layer, begin=begin, end=end)
                out, cache = (dec if decode else pref)(
                    put(value),
                    put(batch.positions),
                    put(np.pad(ids[begin:end], (0, padding))),
                    weights,
                    cache,
                    backend.get_forward_metadata(batch),
                    put(batch.out_cache_loc),
                )
                pieces.append(np.asarray(out, np.float32)[: end - begin])
            actual = np.concatenate(pieces)
            # CPU follows its own preceding layer's output, never candidate hidden state.
            emit("cpu_loading", layer=layer)
            official = load_module_weights(
                module.Block(layer, args), cp, f"layers.{layer}."
            )
            tx = torch.from_numpy(expected).to(torch.bfloat16)[None]
            ti = torch.from_numpy(ids)[None]
            emit("cpu", layer=layer)
            with torch.inference_mode():
                rows = [
                    official(tx[:, : options.prefill], 0, ti[:, : options.prefill])[0]
                    .float()
                    .numpy()
                ]
                for position in range(options.prefill, total):
                    rows.append(
                        official(
                            tx[:, position : position + 1],
                            position,
                            ti[:, position : position + 1],
                        )[0]
                        .float()
                        .numpy()
                    )
            cpu = np.concatenate(rows)
            row = {
                "layer": layer,
                "ratio": cfg.ratio,
                "incoming": metrics(native, expected),
                "output": metrics(actual, cpu),
                "prefill": metrics(actual[: options.prefill], cpu[: options.prefill]),
                "decode": metrics(actual[options.prefill :], cpu[options.prefill :]),
            }
            np.savez_compressed(
                options.output / f"layer-{layer:02d}.npz",
                native_input=native.astype(np.float32),
                cpu_input=expected,
                native_output=actual,
                cpu_output=cpu,
            )
            report["layers"].append(row)
            emit("layer_complete", **row)
            native, expected = actual, cpu
            del official, weights, cache, tx, rows, pref, dec, pieces, out
            gc.collect()

        emit("head_loading", full_model=report["full_model"])
        shared = {
            key: put(cp.read_tensor(key))
            for key in (
                "hc_head_fn",
                "hc_head_scale",
                "hc_head_base",
                "norm.weight",
                "head.weight",
            )
        }

        def head(x, w):
            collapsed = head_collapse(
                x,
                w["hc_head_fn"],
                w["hc_head_scale"],
                w["hc_head_base"],
                eps=args.norm_eps,
                hc_eps=args.hc_eps,
                backend="pallas",
            )
            hidden = dense.norm(collapsed, w["norm.weight"], args.norm_eps)
            return jnp.matmul(
                hidden, w["head.weight"].T, preferred_element_type=jnp.float32
            )

        compiled = jax.jit(
            jax.shard_map(
                head,
                mesh=mesh,
                in_specs=(P(), {key: P() for key in shared}),
                out_specs=P(),
                check_vma=False,
            )
        )
        actual_logits = np.asarray(
            compiled(
                put(jnp.asarray(native[options.prefill - 1 :], jnp.bfloat16)), shared
            ),
            np.float32,
        )
        with torch.inference_mode():
            collapsed = module.ParallelHead.hc_head(
                SimpleNamespace(norm_eps=args.norm_eps, hc_eps=args.hc_eps),
                torch.from_numpy(expected[options.prefill - 1 :]).to(torch.bfloat16)[
                    None
                ],
                *(
                    tensor_from_checkpoint(cp, key)
                    for key in ("hc_head_fn", "hc_head_scale", "hc_head_base")
                ),
            )
            norm = load_module_weights(
                module.RMSNorm(args.dim, args.norm_eps), cp, "norm."
            )
            cpu_logits = (
                norm(collapsed)[0].float()
                @ tensor_from_checkpoint(cp, "head.weight").float().T
            ).numpy()
        report["logits"] = dict(
            **metrics(actual_logits, cpu_logits),
            top1_agreement=float(
                np.mean(actual_logits.argmax(-1) == cpu_logits.argmax(-1))
            ),
            candidate_top1=actual_logits.argmax(-1).tolist(),
            cpu_top1=cpu_logits.argmax(-1).tolist(),
            scope="full-model teacher-forced logits"
            if report["full_model"]
            else "truncated-model diagnostic head; not deployed-model logits",
        )
        np.savez_compressed(
            options.output / "logits.npz", native=actual_logits, cpu=cpu_logits
        )
        assert fingerprint() == report["source_fingerprint"]
        report["complete"] = True
        emit("complete", logits=report["logits"])
    except BaseException:
        import traceback

        report["error"] = traceback.format_exc()
        emit("failed", error=report["error"])
        raise


if __name__ == "__main__":
    main()
