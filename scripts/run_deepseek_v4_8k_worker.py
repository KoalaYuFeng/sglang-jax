"""Real-checkpoint 8K chunked-prefill gate through the native ModelWorker.

This process owns all four TPU chips. Run the separate Engine gate only after
this process exits, so two complete checkpoints never compete for HBM.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from transformers import AutoTokenizer

from run_deepseek_v4_framework import compare_arrays, framework_fingerprint

from sgl_jax.srt.managers.schedule_batch import ModelWorkerSamplingInfo
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    DeepSeekV4Reference,
    source_fingerprint,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

CONTEXT = 8192
CHUNK = 128


def server_args(checkpoint: Path) -> ServerArgs:
    return ServerArgs(
        model_path=str(checkpoint),
        context_length=CONTEXT,
        tp_size=4,
        ep_size=4,
        dp_size=1,
        device="tpu",
        dtype="bfloat16",
        moe_backend="epmoe",
        max_running_requests=1,
        max_total_tokens=CONTEXT,
        max_prefill_tokens=CHUNK,
        page_size=1,
        chunked_prefill_size=CHUNK,
        disable_radix_cache=True,
        disable_hybrid_swa_memory=True,
        disable_overlap_schedule=True,
        attention_backend="deepseek_v4",
        random_seed=42,
        mem_fraction_static=0.85,
        disable_precompile=True,
        skip_tokenizer_init=True,
        precompile_token_paddings=[CHUNK],
        precompile_bs_paddings=[1],
        watchdog_timeout=900,
    )


def memory_snapshot():
    return {
        "devices": [{"id": d.id, "stats": d.memory_stats()} for d in jax.devices()],
        "host_rss_bytes": psutil.Process().memory_info().rss,
    }


def make_tokens(tokenizer, count: int) -> list[int]:
    seed = tokenizer.encode(
        "DeepSeek V4 Flash long context correctness test on TPU. "
        "Preserve compressed attention state across every scheduler chunk. ",
        add_special_tokens=False,
    )
    if not seed:
        raise AssertionError("tokenizer produced an empty deterministic fixture")
    bos = tokenizer.bos_token_id
    values = ([] if bos is None else [int(bos)]) + seed * (count // len(seed) + 2)
    return [int(x) for x in values[:count]]


def _relevant_cache(cache, configs, tokens):
    relevant = []
    for layer, config in zip(cache, configs, strict=True):
        fields = {"window": layer["window"][:tokens]}
        if config.ratio:
            limit = min(tokens // config.ratio + 1, layer["main.compressed"].shape[0])
            for prefix in ("main", "index"):
                if prefix + ".compressed" not in layer:
                    continue
                fields[prefix + ".kv"] = layer[prefix + ".kv"]
                fields[prefix + ".score"] = layer[prefix + ".score"]
                fields[prefix + ".compressed"] = layer[prefix + ".compressed"][:limit]
        relevant.append(fields)
    return tuple(relevant)


def compare_caches(expected, actual):
    checks = []
    for layer, (a, b) in enumerate(zip(expected, actual, strict=True)):
        if a.keys() != b.keys():
            raise AssertionError(f"cache field mismatch at layer {layer}")
        for field in a:
            checks.append({"layer": layer, "field": field, **compare_arrays(a[field], b[field])})
    return {"bitwise_equal": all(x["bitwise_equal"] for x in checks), "checks": checks}


class ChunkedWorkerSession:
    """Direct ModelWorker harness with the same metadata emitted by ScheduleBatch."""

    def __init__(self, worker):
        self.worker = worker
        self.runner = worker.model_runner
        self.request = None
        self.position = 0
        self.locations: list[int] = []

    def reset(self):
        if self.request is not None:
            self.runner.token_to_kv_pool_allocator.free(np.asarray(self.locations, np.int32))
            self.runner.req_to_token_pool.free(self.request)
        self.request = SimpleNamespace(req_pool_idx=None, is_chunked=0, kv_committed_len=0)
        if self.runner.req_to_token_pool.alloc([self.request]) != [0]:
            raise AssertionError("failed to allocate V4 request slot zero")
        self.position = 0
        self.locations = []

    def _batch(self, ids, mode):
        ids = np.asarray(ids, np.int32)
        if ids.ndim != 1 or not ids.size:
            raise ValueError("a nonempty one-dimensional token chunk is required")
        if mode == ForwardMode.EXTEND and self.position and self.position % CHUNK:
            raise ValueError("continuation prefill must begin on a 128-token boundary")
        if mode == ForwardMode.DECODE and ids.size != 1:
            raise ValueError("decode consumes one token")
        allocated = self.runner.token_to_kv_pool_allocator.alloc(ids.size)
        if allocated is None:
            raise RuntimeError("V4 token allocator exhausted")
        begin = self.position
        self.locations.extend(int(i) for i in allocated)
        self.runner.req_to_token_pool.write(
            (0, slice(begin, begin + ids.size)), allocated
        )
        batch = self.worker.compilation_manager._make_dummy_batch(
            1,
            ids.size,
            mode,
            self.worker.compilation_manager.cache_loc_buckets[0],
            dp_size=1,
            per_dp_bs_size=1,
        )
        batch.bid = begin + 1
        batch.input_ids = ids
        batch.real_input_ids_len = ids.size
        batch.positions = np.arange(begin, begin + ids.size, dtype=np.int32)
        batch.seq_lens = np.asarray([begin + ids.size], np.int32)
        batch.out_cache_loc = allocated
        batch.cache_loc[: len(self.locations)] = self.locations
        batch.return_output_logprob_only = False
        batch.sampling_info = ModelWorkerSamplingInfo.generate_for_precompile_all_greedy(
            1, self.worker.model_config.vocab_size
        )
        batch.sampling_info.vocab_mask = None
        if mode == ForwardMode.EXTEND:
            batch.extend_prefix_lens = np.asarray([begin], np.int32)
            batch.extend_seq_lens = np.asarray([ids.size], np.int32)
            batch.logits_indices = np.asarray([ids.size - 1], np.int32)
        self.position += ids.size
        return batch

    def step(self, ids, mode):
        batch = self._batch(ids, mode)
        start = time.perf_counter()
        output, _, misses = self.worker.forward_batch_generation(batch, skip_sample=True)
        jax.block_until_ready(output)
        seconds = time.perf_counter() - start
        logits = np.asarray(output.next_token_logits, np.float32)
        if logits.shape != (1, self.worker.model_config.vocab_size) or not np.all(
            np.isfinite(logits)
        ):
            raise AssertionError("invalid native V4 logits")
        return logits, seconds, int(misses)

    def prefill(self, ids, *, profile_dir: Path | None = None):
        records = []
        logits = None
        for begin in range(0, len(ids), CHUNK):
            chunk = ids[begin : begin + CHUNK]
            trace_this = profile_dir is not None and (
                begin == CHUNK or begin == ((len(ids) - 1) // CHUNK - 1) * CHUNK
            )
            if trace_this:
                label = "early" if begin == CHUNK else "late"
                target = profile_dir / f"{label}_prefill"
                jax.profiler.start_trace(str(target))
            try:
                with jax.profiler.TraceAnnotation("V4_8K_PREFILL_CHUNK", start=begin):
                    logits, seconds, misses = self.step(chunk, ForwardMode.EXTEND)
            finally:
                if trace_this:
                    jax.profiler.stop_trace()
            records.append(
                {
                    "start": begin,
                    "tokens": len(chunk),
                    "seconds": seconds,
                    "tokens_per_second": len(chunk) / seconds,
                    "pjit_cache_misses": misses,
                    "profiled": trace_this,
                }
            )
        return logits, records

    def decode(self, token, *, profile_dir: Path | None = None):
        if profile_dir is not None:
            jax.profiler.start_trace(str(profile_dir / "decode"))
        try:
            with jax.profiler.TraceAnnotation("V4_8K_DECODE", position=self.position):
                return self.step([token], ForwardMode.DECODE)
        finally:
            if profile_dir is not None:
                jax.profiler.stop_trace()


def validate_long_cache(cache, configs, tokens):
    host = jax.device_get(cache)
    checks = []
    for layer_id, (layer, config) in enumerate(zip(host, configs, strict=True)):
        row = {
            "layer": layer_id,
            "ratio": config.ratio,
            "window_prefix_finite": bool(np.all(np.isfinite(layer["window"][:tokens]))),
            "window_suffix_zero": bool(np.all(layer["window"][tokens:] == 0)),
        }
        if config.ratio:
            available = tokens // config.ratio
            row.update(
                available_compressed_rows=available,
                compressed_prefix_finite=bool(
                    np.all(np.isfinite(layer["main.compressed"][:available]))
                ),
                compressed_suffix_zero=bool(
                    np.all(layer["main.compressed"][available:] == 0)
                ),
            )
            if config.ratio == 4:
                row.update(
                    index_candidates_exceed_topk=available > config.index_topk,
                    index_prefix_finite=bool(
                        np.all(np.isfinite(layer["index.compressed"][:available]))
                    ),
                    index_suffix_zero=bool(
                        np.all(layer["index.compressed"][available:] == 0)
                    ),
                )
        checks.append(row)
    required = (
        all(x["window_prefix_finite"] and x["window_suffix_zero"] for x in checks)
        and all(
            x.get("compressed_prefix_finite", True) and x.get("compressed_suffix_zero", True)
            for x in checks
        )
        and all(
            x.get("index_prefix_finite", True) and x.get("index_suffix_zero", True)
            for x in checks
        )
        and any(x.get("index_candidates_exceed_topk", False) for x in checks)
    )
    return {"passed": required, "checks": checks}


def timing_summary(records):
    steady = [
        row
        for row in records
        if row["tokens"] == CHUNK and row["start"] >= 2 * CHUNK and not row["profiled"]
    ]
    ms = [row["seconds"] * 1000 for row in steady]
    total_tokens = sum(row["tokens"] for row in records)
    total_seconds = sum(row["seconds"] for row in records)
    return {
        "chunks": len(records),
        "total_tokens": total_tokens,
        "model_forward_seconds": total_seconds,
        "aggregate_tokens_per_second": total_tokens / total_seconds,
        "steady_full_chunks": len(steady),
        "steady_chunk_p50_ms": statistics.median(ms),
        "steady_chunk_p95_ms": float(np.percentile(ms, 95)),
        "steady_tokens_per_second": CHUNK / (statistics.median(ms) / 1000),
        "full_chunk_position_recompile_free": all(
            row["pjit_cache_misses"] == 0 for row in records if row["start"] >= CHUNK and row["tokens"] == CHUNK
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-tokens", type=int, default=271)
    # TpModelWorker reserves one token plus five safety slots, so the normal
    # Engine's public input limit is context_length - 6.
    parser.add_argument("--long-tokens", type=int, default=CONTEXT - 6)
    parser.add_argument("--profile", action="store_true")
    options = parser.parse_args()
    if not CHUNK * 2 < options.oracle_tokens < CONTEXT:
        raise ValueError("oracle-tokens must cross two chunk boundaries")
    if not 4096 < options.long_tokens <= CONTEXT - 6:
        raise ValueError("long-tokens must be in (4096, context_length - 6]")
    if options.output.exists():
        raise FileExistsError(f"refusing to overwrite {options.output}")
    options.output.mkdir(parents=True)
    report = {
        "complete": False,
        "checkpoint": str(options.checkpoint.resolve()),
        "source_fingerprint": source_fingerprint(),
        "framework_source_fingerprint": framework_fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "context_length": CONTEXT,
        "chunked_prefill_size": CHUNK,
        "oracle_tokens": options.oracle_tokens,
        "long_tokens": options.long_tokens,
        "events": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (options.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(json.dumps(event, default=str), flush=True)

    try:
        if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
            raise RuntimeError("requires exactly four TPU devices")
        tokenizer = AutoTokenizer.from_pretrained(
            options.checkpoint, local_files_only=True, trust_remote_code=False
        )
        oracle_ids = make_tokens(tokenizer, options.oracle_tokens)
        long_ids = make_tokens(tokenizer, options.long_tokens)
        report["oracle_input_sha256"] = hashlib.sha256(
            np.asarray(oracle_ids, np.int32).tobytes()
        ).hexdigest()
        report["long_input_ids"] = long_ids

        mesh = create_device_mesh([1, 4], [1, 1])
        emit({"event": "native_model_load_start"})
        start = time.perf_counter()
        worker = ModelWorker(server_args(options.checkpoint), mesh)
        report["model_load_seconds"] = time.perf_counter() - start
        report["hbm_after_load"] = memory_snapshot()
        emit({"event": "native_model_loaded", "seconds": report["model_load_seconds"]})

        runner = worker.model_runner
        reference = DeepSeekV4Reference(
            options.checkpoint, max_context=CONTEXT, progress=lambda _: None
        )
        reference.mesh = mesh
        reference.shared = runner.model.shared.get_value()
        reference.layers = list(runner.model.layers.get_value())
        reference.reset()
        emit({"event": "bounded_whole_oracle_start", "tokens": len(oracle_ids)})
        start = time.perf_counter()
        with jax.set_mesh(mesh):
            expected_logits, _ = reference.step(oracle_ids)
        oracle_seconds = time.perf_counter() - start

        session = ChunkedWorkerSession(worker)
        session.reset()
        emit({"event": "bounded_chunked_worker_start", "tokens": len(oracle_ids)})
        actual_logits, oracle_chunks = session.prefill(oracle_ids)
        expected_cache, actual_cache = jax.device_get(
            (
                _relevant_cache(reference.cache, reference.configs, len(oracle_ids)),
                _relevant_cache(runner.token_to_kv_pool.layers, reference.configs, len(oracle_ids)),
            )
        )
        bounded = {
            "whole_reference_seconds": oracle_seconds,
            "chunk_records": oracle_chunks,
            "logits": compare_arrays(expected_logits, actual_logits),
            "cache": compare_caches(expected_cache, actual_cache),
        }
        report["bounded_whole_vs_chunked"] = bounded
        if not bounded["logits"]["bitwise_equal"] or not bounded["cache"]["bitwise_equal"]:
            raise AssertionError("bounded whole/chunked real-checkpoint equivalence failed")
        emit({"event": "bounded_whole_vs_chunked_passed"})

        del reference, expected_cache, actual_cache
        gc.collect()
        session.reset()
        profile_dir = options.output / "traces" if options.profile else None
        emit({"event": "long_chunked_prefill_start", "tokens": len(long_ids)})
        long_logits, long_records = session.prefill(long_ids, profile_dir=profile_dir)
        prefill_token = int(np.argmax(long_logits[0]))
        cache_gate = validate_long_cache(
            runner.token_to_kv_pool.layers, runner.model.configs, len(long_ids)
        )
        report["long_cache_gate"] = cache_gate
        report["long_prefill_records"] = long_records
        report["long_prefill_timing"] = timing_summary(long_records)
        report["hbm_after_long_prefill"] = memory_snapshot()
        if not cache_gate["passed"]:
            raise AssertionError("8K cache/index boundary validation failed")
        if not report["long_prefill_timing"]["full_chunk_position_recompile_free"]:
            raise AssertionError("full 128-token chunks recompiled as position changed")
        emit({"event": "long_chunked_prefill_passed", **report["long_prefill_timing"]})

        # Compile/warm decode without profiling.  Capturing the first decode
        # folds compilation into the trace and leaves no device dispatch in
        # the measured window on a cold cache.
        second_logits, decode_seconds, decode_misses = session.decode(prefill_token)
        second_token = int(np.argmax(second_logits[0]))
        report["long_expected_output_ids"] = [prefill_token, second_token]
        report["long_decode"] = {
            "position": len(long_ids),
            "seconds": decode_seconds,
            "pjit_cache_misses": decode_misses,
        }
        if profile_dir is not None:
            third_logits, hot_decode_seconds, hot_decode_misses = session.decode(
                second_token, profile_dir=profile_dir
            )
            report["profiled_hot_decode"] = {
                "position": len(long_ids) + 1,
                "input_token": second_token,
                "output_token": int(np.argmax(third_logits[0])),
                "seconds": hot_decode_seconds,
                "pjit_cache_misses": hot_decode_misses,
            }
            if hot_decode_misses:
                raise AssertionError("hot decode recompiled when only the position changed")
        report["hbm_after_decode"] = memory_snapshot()

        short_ids = oracle_ids[:CHUNK]
        session.reset()
        short_logits, short_records = session.prefill(short_ids)
        short_first = int(np.argmax(short_logits[0]))
        short_second_logits, short_decode_seconds, _ = session.decode(short_first)
        short_second = int(np.argmax(short_second_logits[0]))
        report["short_reset_case"] = {
            "input_ids": short_ids,
            "expected_output_ids": [short_first, short_second],
            "prefill_records": short_records,
            "decode_seconds": short_decode_seconds,
        }
        report["complete"] = True
        emit({"event": "v4_8k_modelworker_gate_passed"})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
