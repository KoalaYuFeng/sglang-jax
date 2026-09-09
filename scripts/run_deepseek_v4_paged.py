"""Real-checkpoint native paging gate; one process owns the four TPU devices.

The independent reference shares only read-only checkpoint arrays. Results
include source fingerprints; old B1 reports are not acceptance for this gate.
"""

import argparse
import dataclasses
import json
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
from deepseek_v4_execution_options import (
    HELPER_SHA256,
    add_execution_arguments,
    assert_runner_options,
    model_overrides,
    options_from_namespace,
)
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from run_deepseek_v4_framework import (
    compare_arrays,
    framework_fingerprint,
    memory_snapshot,
)
from sgl_jax.srt.managers.schedule_batch import ModelWorkerSamplingInfo
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.deepseek_v4_reference import DeepSeekV4Reference
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from transformers import AutoTokenizer


def server_args(checkpoint, context=384):
    return ServerArgs(
        model_path=str(checkpoint),
        context_length=context,
        tp_size=4,
        ep_size=4,
        dp_size=1,
        device="tpu",
        dtype="bfloat16",
        moe_backend="epmoe",
        attention_backend="deepseek_v4",
        page_size=128,
        max_running_requests=4,
        max_total_tokens=context * 4,
        max_prefill_tokens=128,
        chunked_prefill_size=128,
        disable_radix_cache=False,
        disable_hybrid_swa_memory=True,
        disable_overlap_schedule=True,
        random_seed=42,
        mem_fraction_static=0.85,
        disable_precompile=True,
        skip_tokenizer_init=True,
        precompile_token_paddings=[128],
        precompile_bs_paddings=[1, 2, 4],
        watchdog_timeout=1800,
    )


def reference_layers(runner):
    """Restore only attention sharding for the unchanged independent oracle.

    Keep raw FP4/FP8 bytes/dtypes and EP expert ownership. This may temporarily
    replicate FP8 head weights, but never expands expert weights to BF16.
    """
    with jax.set_mesh(runner.mesh):
        from sgl_jax.srt.kernels.deepseek_v4.projections import unpack_merged_weights
        from sgl_jax.srt.model_loader.deepseek_v4_native import original_fp4_scale_view

        return [
            {
                key: value
                if key.startswith("experts.")
                else jax.reshard(value, NamedSharding(runner.mesh, P()))
                for key, value in unpack_merged_weights(
                    original_fp4_scale_view(
                        layer, transposed=runner.model.moe_backend == "gmm_tuned"
                    )
                ).items()
            }
            for layer in runner.model.layers.get_value()
        ]


class PagedWorkerSession:
    """Real request/page allocators with the scheduler's packed batch contract."""

    def __init__(self, worker):
        self.worker, self.runner = worker, worker.model_runner
        self.requests = {}
        self.bid = 0

    def new(self, key):
        request = SimpleNamespace(
            req_pool_idx=None, is_chunked=0, kv_committed_len=0, locations=[]
        )
        if self.runner.req_to_token_pool.alloc([request]) is None:
            raise RuntimeError("request pool exhausted")
        self.requests[key] = request

    def free(self, key):
        request = self.requests.pop(key)
        self.runner.token_to_kv_pool_allocator.free(
            np.asarray(request.locations, np.int32)
        )
        self.runner.req_to_token_pool.free(request)

    def step(self, items, *, decode=False, bucket=4, token_bucket=None):
        bs = len(items)
        counts = np.asarray([len(ids) for _, ids in items], np.int32)
        tokens = paged_test_token_bucket(
            counts, decode=decode, bucket=bucket, token_bucket=token_bucket
        )
        prefixes = np.asarray(
            [len(self.requests[key].locations) for key, _ in items], np.int32
        )
        lengths = prefixes + counts
        last = [
            self.requests[key].locations[-1] if self.requests[key].locations else -1
            for key, _ in items
        ]
        allocator = self.runner.token_to_kv_pool_allocator
        locations = (
            allocator.alloc_decode(lengths, last)
            if decode
            else allocator.alloc_extend(prefixes, lengths, last, int(counts.sum()))
        )
        if locations is None:
            raise RuntimeError("paged KV allocator exhausted")
        mode = ForwardMode.DECODE if decode else ForwardMode.EXTEND
        batch = self.worker.compilation_manager._make_dummy_batch(
            bucket,
            tokens,
            mode,
            self.worker.compilation_manager.cache_loc_buckets[-1],
            dp_size=1,
            per_dp_bs_size=bucket,
        )
        self.bid += 1
        batch.bid, batch.real_bs, batch.real_bs_per_dp = self.bid, bs, [bs]
        batch.real_input_ids_len = int(counts.sum())
        batch.input_ids = np.zeros(tokens, np.int32)
        batch.positions = np.zeros(tokens, np.int32)
        batch.out_cache_loc = np.full(tokens, -1, np.int32)
        batch.seq_lens = np.pad(lengths, (0, bucket - bs))
        batch.req_pool_indices = np.zeros(bucket, np.int32)
        batch.cache_loc[:] = 0
        row, cache_offset = 0, 0
        for i, ((key, ids), prefix, count) in enumerate(
            zip(items, prefixes, counts, strict=True)
        ):
            request = self.requests[key]
            request.locations.extend(int(v) for v in locations[row : row + count])
            self.runner.req_to_token_pool.write(
                (request.req_pool_idx, slice(prefix, prefix + count)),
                locations[row : row + count],
            )
            batch.input_ids[row : row + count] = ids
            batch.positions[row : row + count] = np.arange(prefix, prefix + count)
            batch.out_cache_loc[row : row + count] = locations[row : row + count]
            batch.req_pool_indices[i] = request.req_pool_idx
            pages = np.asarray(request.locations[::128], np.int32)
            all_locations = (pages[:, None] + np.arange(128)).ravel()
            batch.cache_loc[cache_offset : cache_offset + len(all_locations)] = (
                all_locations
            )
            cache_offset += len(all_locations)
            row += int(count)
        if not decode:
            batch.extend_prefix_lens = np.pad(prefixes, (0, bucket - bs))
            batch.extend_seq_lens = np.pad(counts, (0, bucket - bs))
            batch.logits_indices = np.pad(
                np.cumsum(counts) - 1, (0, bucket - bs)
            ).astype(np.int32)
        batch.logits_indices_selector = np.arange(bs, dtype=np.int32)
        batch.return_output_logprob_only = False
        batch.sampling_info = (
            ModelWorkerSamplingInfo.generate_for_precompile_all_greedy(
                bucket, self.worker.model_config.vocab_size
            )
        )
        batch.sampling_info.vocab_mask = None
        start = time.perf_counter()
        output, sampled, misses = self.worker.forward_batch_generation(batch)
        jax.block_until_ready((output, sampled))
        logits = np.asarray(output.next_token_logits, np.float32)[:bs]
        if not np.all(np.isfinite(logits)):
            raise AssertionError("nonfinite model logits")
        np.testing.assert_array_equal(
            np.asarray(sampled)[:bs], np.argmax(logits, axis=-1)
        )
        self.last_batch = batch
        return logits, {
            "seconds": time.perf_counter() - start,
            "cache_misses": int(misses),
            "requests": bs,
            "tokens": int(counts.sum()),
            "decode": decode,
        }


def paged_test_token_bucket(counts, *, decode, bucket, token_bucket=None):
    """Diagnostic packed capacity; default 128 and serving algorithms are unchanged.

    Validate before allocating pages so an invalid experiment cannot leak them.
    """
    counts = np.asarray(counts)
    tokens = (bucket if decode else 128) if token_bucket is None else token_bucket
    if (
        counts.ndim != 1
        or not np.issubdtype(counts.dtype, np.integer)
        or type(tokens) is not int
        or tokens <= 0
        or type(bucket) is not int
        or bucket <= 0
        or len(counts) == 0
        or len(counts) > bucket
        or np.any(counts <= 0)
        or sum(counts) > tokens
        or (decode and (tokens != bucket or np.any(counts != 1)))
    ):
        raise ValueError("invalid test bucket")
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=384)
    add_execution_arguments(parser)
    parser.add_argument("--check-kernel-dispatch", action="store_true")
    args = parser.parse_args()
    selected = options_from_namespace(args)
    args.output.mkdir(parents=True, exist_ok=False)
    sa = server_args(args.checkpoint, args.context)
    sa.json_model_override_args = json.dumps(model_overrides(selected))
    report = {
        "complete": False,
        "source_fingerprint": framework_fingerprint(),
        "server_args": dataclasses.asdict(sa),
        "checkpoint": str(args.checkpoint),
        **selected,
        "execution_helper_sha256": HELPER_SHA256,
        "scope": "real 43-layer ModelWorker; B1/B2/B4 packed prefill, decode, reorder, slot reuse",
        "events": [],
        "checks": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(json.dumps(event, default=str), flush=True)

    def check(label, expected, actual):
        comparison = {
            "label": label,
            **compare_arrays(expected, actual),
            "all_finite": bool(
                np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))
            ),
            "top1_equal": bool(
                np.array_equal(np.argmax(expected, axis=-1), np.argmax(actual, axis=-1))
            ),
        }
        report["checks"].append(comparison)
        emit({"event": "comparison", **comparison})
        if (
            not comparison["all_finite"]
            or not comparison["top1_equal"]
            or comparison["nrmse"] > 0.005
        ):
            raise AssertionError(f"full-model numerical gate failed: {label}")

    try:
        mesh = create_device_mesh([1, 4], [1, 1])
        emit({"event": "model_loading"})
        worker = ModelWorker(sa, mesh)
        runner = worker.model_runner
        report["actual_execution"] = assert_runner_options(runner, selected)
        emit(
            {
                "event": "model_loaded",
                "memory": memory_snapshot(),
                "cache_bytes_per_device": runner.token_to_kv_pool.get_kv_size_bytes(),
            }
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.checkpoint), local_files_only=True
        )
        seeds = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in (
                "The capital of France is Paris. Here is a short explanation of geography. ",
                "One plus one equals two. Here is a simple arithmetic calculation. ",
            )
        ]
        prompts = [
            (seed * (132 // len(seed) + 1))[: 132 - i] for i, seed in enumerate(seeds)
        ]
        report["prompts"] = prompts
        references, expected_prefill, expected_decode, teacher = [], [], [], []
        for i, prompt in enumerate(prompts):
            ref = DeepSeekV4Reference(
                args.checkpoint, max_context=args.context, progress=lambda _: None
            )
            ref.mesh, ref.layers = mesh, reference_layers(runner)
            ref.shared = {
                **runner.model.shared.get_value(),
                "embed.weight": runner.model.embed_tokens.embedding.get_value(),
                "head.weight": jax.reshard(
                    runner.model.lm_head.embedding.get_value(), NamedSharding(mesh, P())
                ),
            }
            ref.reset()
            with jax.set_mesh(mesh):
                logits, _ = ref.step(prompt)
                expected_prefill.append(np.asarray(logits, np.float32))
                token = int(np.argmax(logits[0]))
                teacher.append(token)
                logits, _ = ref.step([token])
                expected_decode.append(np.asarray(logits, np.float32))
            references.append(ref)
            emit({"event": "reference_ready", "request": i})
        report["expected_output_ids"] = [
            [teacher[i], int(np.argmax(expected_decode[i][0]))]
            for i in range(len(prompts))
        ]
        session = PagedWorkerSession(worker)
        for batch_size in (1, 2, 4):
            for key in range(batch_size):
                session.new(key)
            # Unequal first chunks plus reordered continuation exercise request
            # identity independently from row order and physical page order.
            positions = [0] * batch_size
            final_logits = {}
            while any(positions[i] < len(prompts[i % 2]) for i in range(batch_size)):
                items = []
                for key in reversed(range(batch_size)):
                    stop = min(positions[key] + 31 + key % 2, len(prompts[key % 2]))
                    if stop > positions[key]:
                        items.append((key, prompts[key % 2][positions[key] : stop]))
                        positions[key] = stop
                logits, timing = session.step(items, bucket=batch_size)
                for row, (key, _) in enumerate(items):
                    final_logits[key] = logits[row : row + 1]
                emit({"event": "native_prefill", "batch": batch_size, **timing})
            for key in range(batch_size):
                check(
                    f"B{batch_size}/request{key}/prefill",
                    expected_prefill[key % 2],
                    final_logits[key],
                )
            order = list(reversed(range(batch_size)))
            logits, timing = session.step(
                [(key, [teacher[key % 2]]) for key in order],
                decode=True,
                bucket=batch_size,
            )
            emit({"event": "native_decode", "batch": batch_size, **timing})
            for row, key in enumerate(order):
                check(
                    f"B{batch_size}/request{key}/decode",
                    expected_decode[key % 2],
                    logits[row : row + 1],
                )
                session.free(key)
        if args.check_kernel_dispatch:
            from deepseek_v4_kernel_evidence import export_mhc_evidence

            report["kernel_evidence"] = export_mhc_evidence(session, args.output)
            emit({"event": "kernel_dispatch_verified", **report["kernel_evidence"]})
        report["complete"] = True
        emit({"event": "paged_modelworker_passed", "memory": memory_snapshot()})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
