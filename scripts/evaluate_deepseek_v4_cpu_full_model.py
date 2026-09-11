"""Full 43-layer official-Python/CPU-shim logits on a frozen natural prompt.

Not official GPU execution. CPU follows its own states; no FP64 replacement,
no TPU hidden-state injection, no modified acceptance thresholds.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from analyze_deepseek_v4_native_profile import fingerprint
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.deepseek_v4_cpu_oracle import (
    load_module_weights,
    official_module,
    tensor_from_checkpoint,
)


def prepare(base, out):
    from transformers import AutoTokenizer

    old = base / "v4-mmlu-accuracy-20260911-01"
    protocol = json.loads((old / "protocol.json").read_text())
    model = Path(protocol["checkpoint"])
    tokenizer_dir = base / "v4-standard-bench-20260910-01/client-tokenizer"
    for name, expected in protocol["tokenizer_sha256"].items():
        assert (
            hashlib.sha256((tokenizer_dir / name).read_bytes()).hexdigest() == expected
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    encoder_path = model / "encoding/encoding_dsv4.py"
    assert (
        hashlib.sha256(encoder_path.read_bytes()).hexdigest()
        == protocol["encoder_sha256"]
    )
    spec = importlib.util.spec_from_file_location("pinned_v4_encoder", encoder_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    content = "Read these background notes, then answer the final question.\n"
    content += "\n".join(
        f"Note {i:02d}: The meeting room has blue chairs." for i in range(12)
    )
    content += (
        "\nFinal question: Compute 17 + 25. State the calculation in one sentence."
    )
    prompt = encoder.encode_messages(
        [{"role": "user", "content": content}], thinking_mode="chat"
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    continuation = "17 + 25 = 42."
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    assert len(prompt_ids) > 128 and len(prompt_ids) + len(continuation_ids) < 512
    fixture = {
        "checkpoint": str(model),
        "prompt": prompt,
        "content": content,
        "prompt_ids": prompt_ids,
        "continuation": continuation,
        "continuation_ids": continuation_ids,
        "thinking_mode": "chat",
        "vocab_size": DeepSeekV4Checkpoint(model).config["vocab_size"],
        "selection": "Fixed natural-language arithmetic prompt with 12 background notes; fixed continuation before either runtime executes",
        "source_fingerprint": fingerprint(),
        "tokenizer_sha256": protocol["tokenizer_sha256"],
        "encoder_sha256": protocol["encoder_sha256"],
    }
    (out / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "fixture",
                "prefill": len(prompt_ids),
                "teacher_tokens": len(continuation_ids),
            }
        ),
        flush=True,
    )


def run(out):
    fixture = json.loads((out / "fixture.json").read_text())
    assert fingerprint() == fixture["source_fingerprint"]
    target = out / "cpu-reference"
    target.mkdir(exist_ok=False)
    cp = DeepSeekV4Checkpoint(Path(fixture["checkpoint"]))
    ids = np.asarray(fixture["prompt_ids"] + fixture["continuation_ids"], np.int64)
    prefill = len(fixture["prompt_ids"])
    context = (len(ids) + 127) // 128 * 128
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.bfloat16)
    module, args = official_module(cp, context)
    expected = np.repeat(
        cp.read_tensor("embed.weight", ids)[:, None, :], args.hc_mult, axis=1
    ).astype(np.float32)
    report = {
        "complete": False,
        "full_model": True,
        "layers_requested": 43,
        "layers": [],
        "reference": "Official Python model with existing independent CPU GPU-kernel shims; not official GPU runtime",
        "source_fingerprint": fingerprint(),
        "prefill": prefill,
        "teacher_tokens": len(ids) - prefill,
        "fixture_sha256": hashlib.sha256(
            (out / "fixture.json").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    assert args.n_layers == 43

    def emit(event, **values):
        (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": event, "time": time.time(), **values}), flush=True)

    try:
        with torch.inference_mode():
            for layer in range(43):
                emit("loading", layer=layer)
                block = load_module_weights(
                    module.Block(layer, args), cp, f"layers.{layer}."
                )
                tx = torch.from_numpy(expected).to(torch.bfloat16)[None]
                ti = torch.from_numpy(ids)[None]
                emit("forward", layer=layer)
                pieces = [block(tx[:, :prefill], 0, ti[:, :prefill])[0].float().numpy()]
                for position in range(prefill, len(ids)):
                    pieces.append(
                        block(
                            tx[:, position : position + 1],
                            position,
                            ti[:, position : position + 1],
                        )[0]
                        .float()
                        .numpy()
                    )
                expected = np.concatenate(pieces)
                assert np.isfinite(expected).all()
                np.savez_compressed(target / f"layer-{layer:02d}.npz", hidden=expected)
                report["layers"].append(
                    {
                        "layer": layer,
                        "finite": True,
                        "max_abs": float(np.abs(expected).max()),
                    }
                )
                emit("layer_complete", layer=layer)
                del block, tx, ti, pieces
                gc.collect()
            collapsed = module.ParallelHead.hc_head(
                SimpleNamespace(norm_eps=args.norm_eps, hc_eps=args.hc_eps),
                torch.from_numpy(expected[prefill - 1 :]).to(torch.bfloat16)[None],
                *(
                    tensor_from_checkpoint(cp, key)
                    for key in ("hc_head_fn", "hc_head_scale", "hc_head_base")
                ),
            )
            norm = load_module_weights(
                module.RMSNorm(args.dim, args.norm_eps), cp, "norm."
            )
            logits = (
                norm(collapsed)[0].float()
                @ tensor_from_checkpoint(cp, "head.weight").float().T
            ).numpy()
            assert logits.shape == (len(ids) - prefill + 1, cp.config["vocab_size"])
            assert np.isfinite(logits).all()
            np.savez_compressed(target / "logits.npz", logits=logits)
            report.update(
                complete=True,
                logits_shape=list(logits.shape),
                top1=logits.argmax(-1).tolist(),
            )
            assert fingerprint() == fixture["source_fingerprint"]
            emit("complete", top1=report["top1"])
    except BaseException:
        import traceback

        report["error"] = traceback.format_exc()
        emit("failed", error=report["error"])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    opts = parser.parse_args()
    prepare(opts.profiles, opts.output) if opts.mode == "prepare" else run(opts.output)
