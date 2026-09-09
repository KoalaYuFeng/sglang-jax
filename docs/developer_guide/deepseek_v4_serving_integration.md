# DeepSeek V4 native serving integration

The [existing-kernel integration](deepseek_v4_original_kernel_integration.md)
now uses original Pallas mHC pre/Sinkhorn/post/head and all 20 HCA layers'
projection/compressor/attention. The repaired HCA source passes 213 TPU tests,
2424 native 43-layer 8K full-logit checks and 514 independent reference rows.
Maximum independent logit NRMSE is 0.010527%, with finite values and matching
top-1 tokens. The same-source short/ragged worker rerun also passes all 14
full-logit checks. Original CSA integration remains pending. The results below
are the preserved pre-migration baseline, not new timing/pressure measurements.

Preserved baseline: **fixed 43-layer 8K native numerical fixtures passed** on four TPU
v5p devices: 1910 B1/B2/B4 cold-cache full-logit comparisons, plus 514
independently constructed whole-prefill/decode rows bitwise equal. See the
[position 8023 correctness report](deepseek_v4_8023_correctness_report.md).
This does **not** yet revalidate Engine pressure/performance, HTTP behavior or
official model benchmark accuracy. Earlier failed measurements remain failed.

Historical pre-repair 8K run (2026-09-07): the full 43-layer B1/B2/B4
concurrency, queue, genuine
KV-exhaustion/recovery and eviction measurements are now finished. Greedy
tokens match and request lifecycle checks complete, but B2/B4 full-logit NRMSE
reaches 47.19% against the 0.5% gate. **Full 8K numerical acceptance failed.**
The lifecycle observations and timings are diagnostics of a failing model,
not accepted inference, pressure correctness or valid model benchmarks. See
the [native 8K regression report](deepseek_v4_8k_native_regression.md) for the
failure, HBM, steady-state timings and cold-prefill latency limitations.

The follow-up [position 8023 correctness report](deepseek_v4_8023_correctness_report.md)
records the exact reproduction, layer/state diagnosis, minimal compressor fix
and fresh full-model numerical validation status.

The current implementation adds V4-only kernels, a paged backend/pool, and a
native model wrapper with standard Embed/ParallelLMHead/LogitsProcessor. Only
the existing V4 dispatch hooks in ModelRunner and its pool mixin change; the
scheduler, allocator and RadixCache are unmodified.

## Contract and change budget

Keep the original FP4/FP8/E8M0 checkpoint and online dequantization. Keep the
existing Engine, scheduler, sampler and whole-model ModelRunner JIT. Changes to
shared infrastructure must be small dispatch hooks; model-specific semantics
belong to V4 modules and kernels. The initial hardware gate uses the existing
four-device v5p slice, with expert parallelism and replicated shared attention.

## Work and acceptance gates

1. Add a request-aware V4 backend with dynamic ragged metadata, padding masks,
   per-request state, physical token locations and standard 128-token pages.
   Token length and batch composition must not be static metadata.
2. Add V4 KV buffers and compressor snapshots owned by physical pages. Standard
   RadixCache only shares complete pages. A new request restores its compressor
   state from the last shared page; active partial pages remain request-owned.
   All writes must honor valid-token masks. Validate allocation, noncontiguous
   pages, prefix forks, slot reuse, cancellation and retraction.
3. Implement packed-batch attention/compression using the original V4 numerical
   recipe. CSA indexer uses Hadamard + FP4 QAT, not the older template's FP8
   index-cache format. Compare SWA/CSA/HCA against independent requests and the
   retained reference, including arbitrary chunk boundaries and 8K top-k.
4. Use standard model/layer/logits interfaces, preserving the output head's
   FP32 accumulation. Produce one sampled row per request, support logprobs and
   hidden-state capture, and keep all layers/cache updates/head in one JIT.
5. Adapt expert execution to token grouping without changing routing, SwiGLU
   clipping or route scaling before the down projection's activation QAT.
6. Run CPU contracts, real TPU kernels, complete real-checkpoint ModelWorker and
   Engine/HTTP gates. Compare B=1/2/4, reordered and interleaved requests, reused
   prefixes, padding buckets, 8K and streaming. Report HBM and warm timings.

## Cache design

Window KV uses ordinary token locations. Compressed KV uses the physical
location of each compression group's terminal token divided by its ratio;
128-token page alignment preserves group boundaries for both ratios 4 and 128.
Page-end compressor snapshots are immutable while a prefix page is shared.
Current partial compressor state is indexed by request slot. Prefix restoration
and per-request resets execute inside the donated model call, not on the host.

Snapshots introduce measured HBM overhead; capacity must account for them rather
than pretending they are free. The snapshot layout avoids a new scheduler or
radix implementation. Larger topology support, MTP, LoRA and PD are separate
extensions and must not be implied by the four-device serving gate.

## Numerical regression discovered during integration

The historical reference's CSA index-score path used ordinary BF16-to-FP32
cast pairs. Under fused JIT these could retain excess precision: the CPU repro
returned `-0.26549375` at a required BF16 boundary instead of `-0.265625`.
This changed tied top-k order. Both the reference and native path now make the
official rounding boundaries explicit, with an independent NumPy representability
test. Historical report fingerprints are therefore intentionally invalidated;
new full-model reports must be regenerated. The 0.5% logit NRMSE gate is not relaxed.

The v5p lowering of concatenated, doubly-sliced rank-3 compressor tensors also
selected an incorrect channel half. The V4 adapter uses explicit 2-D channel
selection; ragged compressor tests cover the affected path on actual TPU hardware.

The 8K synthetic test now requires bitwise-equal visible attention outputs
**and FP32 compressor scratch**. The earlier four-float32-epsilon scratch
allowance was insufficient: persistent ulps can cross a BF16 pooling boundary
and amplify in later layers. The position-8023 repair adds strict batch/order
invariance regressions and preserves the unchanged full-model logit gate.

## Verified real-checkpoint result (2026-09-07)

The native ModelWorker report `v4-paged-worker-20260907-02/report.json` contains
14 passing full-logit comparisons across B1/B2/B4 prefill and decode. B1 is
bitwise equal to the corrected independent reference. B2/B4 have identical
top-1 tokens, maximum NRMSE 1.499e-7 and maximum absolute error 1.526e-5.
Fixtures use two distinct 131/132-token prompts, interleaved 31/32-token chunks,
reordered requests, padding and request-slot reuse. This is an integration
correctness gate, not an accuracy benchmark against an official hosted API.

The source fingerprint is
`c7ec7e1357f41b4710f5505fa76ed95fc458693998df6ebab4ec5b0fc39ce983`.
At context capacity 384 per request and four request slots, loaded weights plus
native cache occupy about 42.05 GiB per device; the cache itself is 290.3 MiB.
This is not an 8K memory measurement. The worker's later peak includes extra
independent-reference buffers and must not be labeled production resident HBM.
The buffer-layout calculation for four full 8K requests is 4.54 GiB of cache
per device, including snapshots and private scratch; compiler temporaries and
other runtime allocations are additional and still require an 8K measurement.

Whole-model compilation takes roughly 133–154 seconds per new tested bucket.
Warm 128-padded-token prefill calls take about 0.70–0.88 seconds in this small
fixture, depending on valid token count and batch size. The one-token decode
comparisons were cold calls; they are not a hot decode throughput benchmark.

The standard Engine report `v4-paged-engine-20260907-01/report.json` has nine
passing output comparisons: cold B2, prefix-shared B4, streaming logprobs,
post-cancellation slot reuse and post-flush recomputation. All four prefix
requests record a real 128-token RadixCache hit. Overlap was disabled in this
first report; the strengthened gate below additionally verifies the abort
acknowledgement and a full 32-token retract/resume comparison.

`v4-paged-engine-overlap-20260907-01/report.json` passes the same cases with
standard overlap scheduling enabled, plus retract/resume (10 output comparisons
total). The abort case records a real `finish_reason.type=abort`. At retraction,
the scheduler reports zero running requests, one queued request and all four
request slots free. After resumption, all 32 generated tokens equal the
uninterrupted baseline. This is a short-context state-lifecycle regression,
not a concurrency or throughput stress test.

CPU regression receipts: 67 V4 framework/paging tests, 35 shared allocator/radix
tests, and 33 reference/low-bit/platform tests passed. The latter CPU run skips
18 TPU-only tests. The final combined real-TPU run passes all 118 V4
framework/paging/reference/low-bit/platform tests, with no skips or failures
(`v4-native-all-tpu-20260907-01.xml`, 89.54 seconds). This includes the 18 tests
skipped on CPU and the full 8K synthetic index/top-k boundary gate. The earlier
57-pass/1-fail TPU report is diagnostic history, superseded by this final run.

Ruff, Python syntax compilation and `git diff --check` also pass. Source and
reports are synchronized to the local workspace and existing Spot TPU; this
work does not create a Git commit or push to GitHub.

## Running the gates

`scripts/run_deepseek_v4_paged.py` exercises real checkpoint loading and ordinary
ModelWorker/request/page allocator calls at B=1/2/4. It writes a fresh report
directory, source fingerprint, HBM statistics, timings and numerical checks.
Use the separate Engine gate only after the ModelWorker process exits; do not
allocate two complete checkpoints on the same four-device slice.

The old `run_deepseek_v4_framework.py` / `run_deepseek_v4_8k_worker.py` B1
experiment harnesses used a page-size-one allocator and are historical scripts,
not the new serving launch path. Current gates:

```bash
python scripts/run_deepseek_v4_paged.py --checkpoint /path/to/original-checkpoint --output /new/worker-report
python scripts/run_deepseek_v4_paged_engine.py --worker-report /new/worker-report/report.json --output /new/engine-report
```

The Engine gate checks cold B2, radix-shared B4, streaming/output logprobs,
acknowledged cancellation and slot reuse, retract/resume, and cache flush.
Its optional `--overlap` uses the standard overlap scheduler; a result without
that flag does not establish overlap correctness. Reports are saved locally
under the ignored `GCP_login/results` directory, not committed with checkpoints.

The equivalent standard server configuration (overlap and radix enabled) is:

```bash
python -m sgl_jax.launch_server \
  --model-path /path/to/original-checkpoint --device tpu --dtype bfloat16 \
  --tp-size 4 --ep-size 4 --dp-size 1 --moe-backend epmoe \
  --attention-backend deepseek_v4 --page-size 128 \
  --context-length 384 --max-running-requests 4 --max-total-tokens 1536 \
  --chunked-prefill-size 128 --max-prefill-tokens 128 \
  --disable-hybrid-swa-memory --mem-fraction-static 0.85 \
  --precompile-token-paddings 128 --precompile-bs-paddings 1 2 4 \
  --disable-precompile --skip-server-warmup --watchdog-timeout 1800
```

`--disable-precompile` defers compilation to first use; it does not disable
whole-model JIT. Add `--disable-overlap-schedule` to reproduce the first Engine
gate. HTTP protocol behavior itself has not been separately tested here.
The special AOT dispatcher stays
disabled for V4 until its keys include static mode/logits/capture metadata.
The ordinary JIT cache already handles those distinctions.

Do not use these short-context reports to claim complete 8K serving acceptance.
The 8K unit gate checks physical pages, 2048 candidates/top-512, batched terminal
decode and compressor boundaries; a full 43-layer 8K Engine regression is separate.

The full-model 8K token/lifecycle, actual pressure/retraction/eviction and
steady-state performance/profile experiments above were measured on the
pre-repair source in the separate native 8K report. The B2/B4 numerical repair
and independent complete-context gates now pass; Engine lifecycle/pressure
must still be rerun on this fixed source, alongside HTTP protocol smoke and
mixed-prefill/decode validation. Explicit pause/retract alone does not
substitute for the real
memory-pressure gate recorded in that report. Larger or sustained soak
workloads remain untested.
