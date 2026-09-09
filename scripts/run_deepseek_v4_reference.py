"""Gated full-checkpoint forward/decode on 4xv5p, with cached-state replay.

Requires a successful real-layer report from the same numerical source tree.
This is a bounded-context, batch-one correctness milestone, not a serving or
optimized-throughput claim. All 43 main layers and all 256 experts/layer are
loaded; no CPU offload or synthetic/repeated layer weights are substituted.
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import jax
import numpy as np
from transformers import AutoTokenizer

import sgl_jax.srt.model_executor.deepseek_v4_reference as reference_module
from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    DeepSeekV4Reference,
    source_fingerprint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer-gate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prefill-tokens", type=int, default=132)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--diagnostic-trace", action="store_true")
    parser.add_argument("--warm-repeats", type=int, default=0)
    parser.add_argument("--chat-prompt")
    parser.add_argument("--chat-max-new-tokens", type=int, default=16)
    parser.add_argument("--expect-chat", help="optional exact stripped content assertion")
    args = parser.parse_args()
    gate = json.loads(args.layer_gate.read_text())
    if (
        not gate.get("complete")
        or not gate.get("head")
        or {r["layer"] for r in gate.get("attention", [])} != {0, 2, 3}
        or {r["layer"] for r in gate.get("layers", [])} != {0, 2, 3}
        or gate.get("source_fingerprint") != source_fingerprint()
        or Path(gate["checkpoint"]).resolve() != args.checkpoint.resolve()
    ):
        raise RuntimeError(
            "full inference is locked until same-source real attention/block gates pass"
        )
    if (
        args.prefill_tokens < 128
        or args.decode_steps < 4
        or args.prefill_tokens + args.decode_steps > args.max_context
        or args.warm_repeats < 0
        or args.chat_max_new_tokens < 1
    ):
        raise ValueError(
            "milestone requires >=128 prefill tokens, >=4 decode steps, within cache capacity"
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "source_fingerprint": source_fingerprint(),
        "complete": False,
        "events": [],
        "prefill_tokens": args.prefill_tokens,
        "decode_steps": args.decode_steps,
        "max_context": args.max_context,
        "scope": "all main layers, batch one, expert parallel, replicated attention, no MTP",
    }
    trace_phase = "prefill"
    traces = {}
    phase_counts = {}
    original_compiled_layer = reference_module.compiled_layer
    if args.diagnostic_trace:

        def instrument(config, mesh):
            compiled = original_compiled_layer(config, mesh)

            def run(*inputs):
                output, cache, trace = compiled(*inputs)
                layer = phase_counts.get(trace_phase, 0)
                phase_counts[trace_phase] = layer + 1
                for key, value in trace.items():
                    if value.ndim and value.shape[0] == output.shape[0]:
                        traces[f"{trace_phase}/{layer}/{key}"] = np.asarray(value[-1], np.float32)
                traces[f"{trace_phase}/{layer}/streams"] = np.asarray(output[-1], np.float32)
                if trace_phase in ("prefill", "replay"):
                    traces[f"{trace_phase}/{layer}/prefix_window"] = np.asarray(
                        cache["window"][: args.prefill_tokens], np.float32
                    )
                return output, cache, trace

            return run

        reference_module.compiled_layer = instrument

    def emit(event):
        event["time"] = time.time()
        report["events"].append(event)
        print(json.dumps(event), flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, local_files_only=True, trust_remote_code=False
        )
        seed = tokenizer.encode(
            "This is a correctness test of numerical computation on TPU. Please continue the sequence one two three four. ",
            add_special_tokens=False,
        )
        tokens = [int(tokenizer.bos_token_id or 0)] + (
            seed * (args.prefill_tokens // len(seed) + 1)
        )
        tokens = tokens[: args.prefill_tokens]
        report["input_ids"] = list(tokens)
        model = DeepSeekV4Reference(args.checkpoint, max_context=args.max_context, progress=emit)
        start = time.perf_counter()
        model.load()
        report["load_seconds"] = time.perf_counter() - start
        report["loaded_main_layers"] = len(model.layers)
        report["hbm_after_load"] = [d.memory_stats() for d in jax.devices()]
        emit(
            {
                "event": "all_weights_loaded",
                "layers": len(model.layers),
                "seconds": report["load_seconds"],
            }
        )
        start = time.perf_counter()
        logits, _ = model.step(tokens)
        report["first_prefill_seconds_including_compile"] = time.perf_counter() - start
        generated = []
        decode_seconds = []
        for step in range(args.decode_steps):
            trace_phase = f"decode{step}"
            host_logits = np.asarray(logits, np.float32)
            if host_logits.shape != (1, model.checkpoint.config["vocab_size"]) or not np.all(
                np.isfinite(host_logits)
            ):
                raise AssertionError("invalid full-model logits")
            token = int(np.argmax(host_logits[0]))
            tokens.append(token)
            generated.append(token)
            start = time.perf_counter()
            logits, _ = model.step([token])
            decode_seconds.append(time.perf_counter() - start)
            emit(
                {
                    "event": "decode_completed",
                    "step": step,
                    "token": token,
                    "text": tokenizer.decode([token]),
                    "seconds": decode_seconds[-1],
                }
            )
        cached_logits = np.asarray(logits, np.float32)
        if not np.all(np.isfinite(cached_logits)):
            raise AssertionError("nonfinite final cached decode logits")
        cached_kv = jax.device_get(model.cache)
        report.update(
            generated_ids=generated,
            generated_text=tokenizer.decode(generated),
            decode_seconds=decode_seconds,
            hbm_after_decode=[d.memory_stats() for d in jax.devices()],
        )
        model.reset()
        trace_phase = "replay"
        emit({"event": "independent_full_prefill_replay", "tokens": len(tokens)})
        replay_logits, _ = model.step(tokens)
        replay = np.asarray(replay_logits, np.float32)
        nrmse = float(np.linalg.norm(replay - cached_logits) / max(np.linalg.norm(replay), 1e-12))
        report["cached_vs_full_prefill"] = {
            "nrmse": nrmse,
            "max_abs": float(np.max(np.abs(replay - cached_logits))),
            "argmax_equal": bool(np.argmax(replay) == np.argmax(cached_logits)),
        }
        cache_checks = []
        for layer, (old, new) in enumerate(zip(cached_kv, jax.device_get(model.cache))):
            checks = {}
            for key in old:
                # Compare visible KV tensors; partial compressor scratch may
                # intentionally differ between vectorized prefill and decode.
                if key != "window" and not key.endswith(".compressed"):
                    continue
                a, b = old[key].astype(np.float32), new[key].astype(np.float32)
                checks[key] = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
                if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                    raise AssertionError(f"nonfinite KV in layer {layer}: {key}")
            cache_checks.append({"layer": layer, "nrmse": checks})
        report["cache_checks"] = cache_checks
        if args.diagnostic_trace:
            trace_checks = []
            for layer in range(len(model.layers)):
                checks = {}
                for key in (
                    "attn.pre",
                    "attn.norm",
                    "attn.qr",
                    "attn.q",
                    "attn.kv",
                    "attn.attention_value",
                    "attn.operator",
                    "attn.post",
                    "ffn.pre",
                    "ffn.norm",
                    "expert_ids",
                    "routing_weights",
                    "ffn.operator",
                    "ffn.post",
                    "streams",
                ):
                    a = traces[f"decode{args.decode_steps - 1}/{layer}/{key}"]
                    b = traces[f"replay/{layer}/{key}"]
                    checks[key] = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
                a, b = (
                    traces[f"prefill/{layer}/prefix_window"],
                    traces[f"replay/{layer}/prefix_window"],
                )
                checks["existing_prefix_window"] = float(
                    np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
                )
                trace_checks.append({"layer": layer, "nrmse": checks})
                emit({"event": "trace_comparison", **trace_checks[-1]})
            report["trace_checks"] = trace_checks
        if not np.all(np.isfinite(replay)) or nrmse > 0.025:
            raise AssertionError(
                f"cached decode vs independent full prefill failed: {report['cached_vs_full_prefill']}"
            )
        reference_module.compiled_layer = original_compiled_layer
        # Exclude diagnostic device_get and per-layer JSON/file writes from
        # optional timings. The reference still synchronizes between layers;
        # these are batch-one reference measurements, not serving throughput.
        model.progress = lambda event: None
        if args.warm_repeats:
            prefill_times, decode_times = [], []
            for repeat in range(args.warm_repeats):
                model.reset()
                start = time.perf_counter()
                warm_logits, _ = model.step(report["input_ids"])
                prefill_times.append(time.perf_counter() - start)
                warm_ids = []
                for _ in range(args.decode_steps):
                    start = time.perf_counter()
                    token = int(np.argmax(np.asarray(warm_logits, np.float32)[0]))
                    warm_ids.append(token)
                    warm_logits, _ = model.step([token])
                    decode_times.append(time.perf_counter() - start)
                if warm_ids != generated or not np.array_equal(
                    np.asarray(warm_logits, np.float32), cached_logits
                ):
                    raise AssertionError("warmed execution changed the correctness fixture")
                emit({"event": "warm_repeat_completed", "repeat": repeat})
            report["warm_reference_timing"] = {
                "scope": "batch one, all 43 layers, per-layer synchronization; no trace/JSON I/O",
                "excludes": "checkpoint load, compilation, reset and KV-cache allocation",
                "decode_includes": "host greedy token selection and full cached forward",
                "prefill_seconds": prefill_times,
                "decode_seconds": decode_times,
                "prefill_tokens_per_second": args.prefill_tokens / float(np.median(prefill_times)),
                "decode_tokens_per_second": 1.0 / float(np.median(decode_times)),
                "hbm": [d.memory_stats() for d in jax.devices()],
            }
            emit({"event": "warm_reference_timing", **report["warm_reference_timing"]})
        if args.chat_prompt:
            encoding_path = args.checkpoint / "encoding" / "encoding_dsv4.py"
            spec = importlib.util.spec_from_file_location("pinned_deepseek_encoding", encoding_path)
            encoding = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(encoding)
            prompt_ids = tokenizer.encode(
                encoding.encode_messages(
                    [{"role": "user", "content": args.chat_prompt}], thinking_mode="chat"
                )
            )
            if len(prompt_ids) + args.chat_max_new_tokens > args.max_context:
                raise ValueError("chat prompt and output budget exceed reference cache capacity")
            model.reset()
            start = time.perf_counter()
            chat_logits, _ = model.step(prompt_ids)
            completion = []
            for step in range(args.chat_max_new_tokens):
                host_logits = np.asarray(chat_logits, np.float32)
                if not np.all(np.isfinite(host_logits)):
                    raise AssertionError("nonfinite chat logits")
                token = int(np.argmax(host_logits[0]))
                completion.append(token)
                if token == tokenizer.eos_token_id:
                    break
                if step + 1 < args.chat_max_new_tokens:
                    chat_logits, _ = model.step([token])
            raw_text = tokenizer.decode(completion)
            ended_with_eos = completion[-1] == tokenizer.eos_token_id
            parsed = (
                encoding.parse_message_from_completion_text(raw_text, thinking_mode="chat")
                if ended_with_eos
                else {"content": tokenizer.decode(completion, skip_special_tokens=True)}
            )
            report["chat_smoke"] = {
                "prompt": args.chat_prompt,
                "prompt_ids": prompt_ids,
                "completion_ids": completion,
                "raw_text": raw_text,
                "parsed": parsed,
                "ended_with_eos": ended_with_eos,
                "seconds_including_new_shape_compile": time.perf_counter() - start,
                "scope": "official chat encoding, greedy, one request; not a quality benchmark",
            }
            emit({"event": "chat_smoke", **report["chat_smoke"]})
            if (
                args.expect_chat is not None
                and parsed.get("content", "").strip() != args.expect_chat
            ):
                raise AssertionError("chat smoke did not match the requested expected content")
        report["complete"] = True
    finally:
        if args.diagnostic_trace and traces:
            np.savez_compressed(args.report.with_suffix(".trace.npz"), **traces)
        reference_module.compiled_layer = original_compiled_layer
        emit({"event": "full_inference_finished", "complete": report["complete"]})


if __name__ == "__main__":
    main()
