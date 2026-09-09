"""Gate the native V4 ModelWorker path against the unchanged reference.

One checkpoint allocation is shared read-only between paths; caches are
independent. All numerical work runs sequentially on the same four TPU chips.
Output directories must be new, so a failed experiment cannot erase a report.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import psutil

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerSamplingInfo
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    DeepSeekV4Reference,
    source_fingerprint,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def framework_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(source_fingerprint().encode())
    paths = [
        "configs/deepseek_v4.py",
        "configs/model_config.py",
        "hf_transformers_utils.py",
        "models/deepseek_v4.py",
        "model_loader/deepseek_v4_native.py",
        "layers/attention/deepseek_v4_backend.py",
        "layers/attention/deepseek_v4_paged_backend.py",
        "mem_cache/deepseek_v4_pool.py",
        "mem_cache/deepseek_v4_paged_pool.py",
        "model_executor/model_runner.py",
        "model_executor/model_runner_kv_cache_mixin.py",
        "model_executor/compilation_manager.py",
        "kernels/gmm/routing.py",
        "kernels/gmm/megablox_gmm_kernel/gmm.py",
        "kernels/gmm/megablox_gmm_kernel/common.py",
        "kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py",
    ]
    paths.extend(
        str(p.relative_to(root / "python/sgl_jax/srt"))
        for p in sorted((root / "python/sgl_jax/srt/kernels/deepseek_v4").glob("*.py"))
    )
    paths.extend(
        str(p.relative_to(root / "python/sgl_jax/srt"))
        for p in sorted((root / "python/sgl_jax/srt/kernels/hca").glob("*.py"))
    )
    paths.extend(
        str(p.relative_to(root / "python/sgl_jax/srt"))
        for p in sorted((root / "python/sgl_jax/srt/kernels/csa").glob("*.py"))
    )
    paths.append("kernels/dsa/streamindex_topk.py")
    for name in paths:
        digest.update(
            name.encode() + b"\0" + (root / "python/sgl_jax/srt" / name).read_bytes()
        )
    return digest.hexdigest()


def server_args(checkpoint, *, inspect_only=False):
    return ServerArgs(
        model_path=str(checkpoint),
        context_length=256,
        tp_size=4,
        ep_size=4,
        dp_size=1,
        device="cpu" if inspect_only else "tpu",
        dtype="bfloat16",
        moe_backend="epmoe",
        max_running_requests=1,
        max_total_tokens=256,
        max_prefill_tokens=136,
        page_size=1,
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        disable_hybrid_swa_memory=True,
        disable_overlap_schedule=True,
        attention_backend="deepseek_v4",
        random_seed=42,
        mem_fraction_static=0.85,
        disable_precompile=True,
        skip_tokenizer_init=True,
        precompile_token_paddings=[18, 132, 136],
        precompile_bs_paddings=[1],
    )


def memory_snapshot():
    return {
        "devices": [{"id": d.id, "stats": d.memory_stats()} for d in jax.devices()],
        "host_rss_bytes": psutil.Process().memory_info().rss,
    }


def compare_arrays(expected, actual):
    raw_a, raw_b = np.asarray(expected), np.asarray(actual)
    a, b = raw_a.astype(np.float32), raw_b.astype(np.float32)
    if a.shape != b.shape:
        raise AssertionError(f"different shapes: {a.shape} vs {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    diff = a[finite] - b[finite]
    return {
        "bitwise_equal": raw_a.dtype == raw_b.dtype
        and raw_a.tobytes() == raw_b.tobytes(),
        "values_equal": bool(np.array_equal(a, b)),
        "finite_mask_equal": bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
        "nrmse": float(np.linalg.norm(diff) / max(np.linalg.norm(a[finite]), 1e-12)),
        "max_abs": float(np.max(np.abs(diff), initial=0)),
    }


def compare_caches(expected, actual, *, visible_only=False):
    checks = []
    for layer, (a, b) in enumerate(zip(expected, actual, strict=True)):
        for key in a:
            if visible_only and key != "window" and not key.endswith(".compressed"):
                continue
            checks.append(
                {"layer": layer, "field": key, **compare_arrays(a[key], b[key])}
            )
    return {"bitwise_equal": all(c["bitwise_equal"] for c in checks), "checks": checks}


class WorkerSession:
    """Exercise actual request/token allocators and ModelWorker generation."""

    def __init__(self, worker):
        self.worker = worker
        self.runner = worker.model_runner
        self.request = None
        self.position = 0
        self.locations = []
        self.last_batch = None

    def reset(self):
        if self.request is not None:
            self.runner.token_to_kv_pool_allocator.free(
                np.asarray(self.locations, np.int32)
            )
            self.runner.req_to_token_pool.free(self.request)
        self.request = SimpleNamespace(
            req_pool_idx=None, is_chunked=0, kv_committed_len=0
        )
        if self.runner.req_to_token_pool.alloc([self.request]) != [0]:
            raise AssertionError("failed to allocate the single request slot")
        self.position, self.locations = 0, []
        # Cache reset is part of prefill's compiled program, not a host clear.

    def batch(self, ids):
        ids = np.asarray(ids, np.int32)
        count = ids.size
        if self.request is None or (self.position and count != 1):
            raise ValueError(
                "reset before whole prefill; subsequent calls must decode one token"
            )
        allocated = self.runner.token_to_kv_pool_allocator.alloc(count)
        if allocated is None:
            raise RuntimeError("token allocator exhausted")
        self.locations.extend(int(i) for i in allocated)
        self.runner.req_to_token_pool.write(
            (0, slice(self.position, self.position + count)), allocated
        )
        mode = ForwardMode.EXTEND if self.position == 0 else ForwardMode.DECODE
        batch = self.worker.compilation_manager._make_dummy_batch(
            1,
            count,
            mode,
            self.worker.compilation_manager.cache_loc_buckets[0],
            dp_size=1,
            per_dp_bs_size=1,
        )
        batch.bid = self.position + 1
        batch.input_ids = ids
        batch.real_input_ids_len = count
        batch.positions = np.arange(
            self.position, self.position + count, dtype=np.int32
        )
        batch.seq_lens = np.asarray([self.position + count], np.int32)
        batch.out_cache_loc = allocated
        batch.cache_loc[: len(self.locations)] = self.locations
        batch.return_output_logprob_only = False
        batch.sampling_info = (
            ModelWorkerSamplingInfo.generate_for_precompile_all_greedy(
                1, self.worker.model_config.vocab_size
            )
        )
        batch.sampling_info.vocab_mask = None
        if mode == ForwardMode.EXTEND:
            batch.extend_seq_lens = np.asarray([count], np.int32)
            batch.logits_indices = np.asarray([count - 1], np.int32)
        self.position += count
        return batch

    def step(self, ids, *, device_sample=True):
        batch = self.batch(ids)
        start = time.perf_counter()
        output, sampled, misses = self.worker.forward_batch_generation(
            batch, skip_sample=not device_sample
        )
        jax.block_until_ready((output, sampled))
        elapsed = time.perf_counter() - start
        logits = np.asarray(output.next_token_logits, np.float32)
        if not np.all(np.isfinite(logits)):
            raise AssertionError("nonfinite native full-model logits")
        token = (
            int(np.asarray(sampled).reshape(-1)[0])
            if device_sample
            else int(np.argmax(logits[0]))
        )
        if token != int(np.argmax(logits[0])):
            raise AssertionError(
                "framework greedy sampler disagrees with logits argmax"
            )
        self.last_batch = batch
        return logits, token, elapsed, misses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.warm_repeats < 1:
        raise ValueError("warm-repeats must be positive")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    fixture = json.loads(args.reference_report.read_text())
    if (
        not fixture.get("complete")
        or fixture["source_fingerprint"] != source_fingerprint()
        or Path(fixture["checkpoint"]).resolve() != args.checkpoint.resolve()
        or fixture["loaded_main_layers"] != 43
    ):
        raise ValueError(
            "requires the same-source complete official-checkpoint reference gate"
        )
    sa = server_args(args.checkpoint, inspect_only=args.inspect_only)
    if args.inspect_only:
        from sgl_jax.srt.configs.deepseek_v4 import validate_v4_server_args
        from sgl_jax.srt.model_loader.arch import get_model_architecture

        config = ModelConfig.from_server_args(sa)
        validate_v4_server_args(sa, config)
        print(
            get_model_architecture(config)[1],
            config.quantization_config,
            source_fingerprint(),
        )
        return
    if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
        raise RuntimeError("requires exactly four TPU devices")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "source_fingerprint": source_fingerprint(),
        "reference_source_fingerprint": source_fingerprint(),
        "framework_source_fingerprint": framework_fingerprint(),
        "validation_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "checkpoint": str(args.checkpoint),
        "jax": jax.__version__,
        "server_args": dataclasses.asdict(sa),
        "events": [],
        "captures": [],
        "scope": "native ModelWorker, B1 EP4 replicated attention, cache=256, original low-bit weights",
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(json.dumps(event, default=str), flush=True)

    try:
        mesh = create_device_mesh([1, 4], [1, 1])
        emit({"event": "loading_native_model"})
        start = time.perf_counter()
        worker = ModelWorker(sa, mesh)
        runner = worker.model_runner
        report["load_seconds"] = time.perf_counter() - start
        report["hbm_after_native_load"] = memory_snapshot()
        emit({"event": "native_model_loaded", "seconds": report["load_seconds"]})

        reference = DeepSeekV4Reference(args.checkpoint, progress=lambda event: None)
        reference.mesh = mesh
        reference.shared = runner.model.shared.get_value()
        reference.layers = list(runner.model.layers.get_value())
        reference.reset()

        def reference_step(ids):
            # Match the production Explicit mesh, including the eager embed
            # and preparation outside the reference's per-layer JITs.
            with jax.set_mesh(mesh):
                return reference.step(ids)

        from profile_deepseek_v4_reference import array_inventory

        report["native_array_inventory"] = array_inventory(
            SimpleNamespace(
                layers=reference.layers,
                shared=reference.shared,
                cache=runner.token_to_kv_pool.layers,
            )
        )
        session = WorkerSession(worker)
        inputs = fixture["input_ids"]
        generated = fixture["generated_ids"]
        report["input_ids"], report["teacher_forced_ids"] = inputs, generated
        report["correctness"] = []
        gold_logits = []
        session.reset()
        for step, ids in enumerate([inputs, *([token] for token in generated)]):
            ref_logits, _ = reference_step(ids)
            gold_logits.append(np.asarray(ref_logits, np.float32))
            emit({"event": "native_forward_start", "step": step, "tokens": len(ids)})
            logits, sampled, elapsed, misses = session.step(ids)
            check = {
                "step": step,
                "logits": compare_arrays(ref_logits, logits),
                "cache": compare_caches(
                    jax.device_get(reference.cache),
                    jax.device_get(runner.token_to_kv_pool.layers),
                ),
                "seconds_including_compile": elapsed,
                "sampled_id": sampled,
                "pjit_cache_misses": misses,
            }
            report["correctness"].append(check)
            emit(
                {
                    "event": "native_forward_checked",
                    "step": step,
                    "seconds": elapsed,
                    "logits": check["logits"],
                    "cache_equal": check["cache"]["bitwise_equal"],
                }
            )
            if (
                not check["logits"]["bitwise_equal"]
                or not check["cache"]["bitwise_equal"]
            ):
                raise AssertionError(
                    "whole-model compilation changed numerical output/state"
                )

        cached_logits = logits
        cached_state = jax.device_get(runner.token_to_kv_pool.layers)
        session.reset()
        emit({"event": "native_full_prefill_replay_start"})
        replay_logits, _, _, _ = session.step(inputs + generated)
        report["cached_vs_replay"] = {
            "logits": compare_arrays(cached_logits, replay_logits),
            "cache": compare_caches(
                cached_state,
                jax.device_get(runner.token_to_kv_pool.layers),
                visible_only=True,
            ),
        }
        if not all(
            report["cached_vs_replay"][key]["bitwise_equal"]
            for key in ("logits", "cache")
        ):
            raise AssertionError("native cache/replay mismatch")
        emit({"event": "native_correctness_passed"})

        chat = fixture["chat_smoke"]
        session.reset()
        emit({"event": "official_chat_fixture_start"})
        _, token, _, _ = session.step(chat["prompt_ids"])
        completion = [token]
        for _ in range(len(chat["completion_ids"]) - 1):
            _, token, _, _ = session.step([token])
            completion.append(token)
        report["chat_smoke"] = {
            "prompt": chat["prompt"],
            "prompt_ids": chat["prompt_ids"],
            "completion_ids": completion,
            "matches_reference": completion == chat["completion_ids"],
            "scope": "official encoded fixture through ModelWorker and its greedy sampler",
        }
        if not report["chat_smoke"]["matches_reference"]:
            raise AssertionError("framework greedy chat fixture changed")
        emit({"event": "official_chat_fixture_passed", "completion_ids": completion})

        timings = {}
        for path in ("reference", "framework"):
            prefill_times, decode_times, misses_seen = [], [], []
            for repeat in range(args.warm_repeats):
                reference.reset() if path == "reference" else session.reset()
                for step, ids in enumerate([inputs, *([token] for token in generated)]):
                    start = time.perf_counter()
                    if path == "reference":
                        out, _ = reference_step(ids)
                        token = int(np.argmax(np.asarray(out, np.float32)[0]))
                    else:
                        out, token, _, misses = session.step(ids, device_sample=False)
                        misses_seen.append(misses)
                    elapsed = time.perf_counter() - start
                    if not np.array_equal(
                        np.asarray(out, np.float32), gold_logits[step]
                    ):
                        raise AssertionError(
                            f"{path} warmed numerical result changed at step {step}"
                        )
                    (prefill_times if step == 0 else decode_times).append(
                        elapsed * 1000
                    )
            timings[path] = {
                "prefill_ms": prefill_times,
                "decode_ms": decode_times,
                "decode_p50_ms": float(np.median(decode_times)),
                "decode_p95_ms": float(np.percentile(decode_times, 95)),
                "pjit_cache_misses": misses_seen,
                "includes": "host batch preparation/dispatch, ready wait and host argmax in BOTH paths",
                "sampling": "framework device sampler verified separately, excluded from this A/B",
            }
            emit({"event": "warm_timing", "path": path, **timings[path]})
        report["warm_timing"] = timings
        report["hbm_warm_shared_weights_two_small_caches"] = memory_snapshot()

        # Inspect the exact same JIT entry used by ModelRunner._forward.
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

        batch = session.last_batch
        forward_batch = ForwardBatch.init_new(batch, runner)
        runner.attn_backend.forward_metadata = runner.attn_backend.get_forward_metadata(
            batch
        )
        logits_metadata = LogitsMetadata.from_model_worker_batch(batch, mesh)
        with jax.set_mesh(mesh):
            lowered = runner._jitted_run_model.lower(
                runner._model_def,
                runner._model_state_def,
                runner.model_state_leaves,
                forward_batch,
                runner.memory_pools,
                logits_metadata,
            )
            compiled = lowered.compile()
        report["decode_compiler_memory"] = str(compiled.memory_analysis())
        (args.output / "decode_optimized_hlo.txt").write_text(compiled.as_text())
        emit({"event": "compiler_exported", "memory": report["decode_compiler_memory"]})

        if args.profile:
            for path in ("reference", "framework"):
                reference.reset() if path == "reference" else session.reset()
                if path == "reference":
                    reference_step(inputs)
                else:
                    session.step(inputs, device_sample=False)
                label = path + "_decode_xla"
                emit({"event": "decode_trace_start", "path": path})
                jax.profiler.start_trace(str(args.output / "traces" / label))
                seconds = []
                try:
                    for step, token in enumerate(generated):
                        marker = (
                            "V4_DECODE"
                            if path == "reference"
                            else "V4_FRAMEWORK_DECODE"
                        )
                        with jax.profiler.TraceAnnotation(marker, step_num=step):
                            start = time.perf_counter()
                            if path == "reference":
                                out, _ = reference_step([token])
                                int(np.argmax(np.asarray(out, np.float32)[0]))
                            else:
                                session.step([token], device_sample=False)
                            seconds.append(time.perf_counter() - start)
                finally:
                    jax.profiler.stop_trace()
                report["captures"].append({"label": label, "seconds": seconds})
                emit({"event": "decode_trace_exported", "path": path})
        report["complete"] = True
        emit({"event": "framework_milestone_complete"})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
