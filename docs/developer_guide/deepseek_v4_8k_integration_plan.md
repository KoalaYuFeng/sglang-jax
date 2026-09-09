# DeepSeek-V4-Flash 8K chunked-prefill integration plan

Date: 2026-09-07. Branch: `integration/deepseek-v4`.

Status: the bounded B1/TP4/EP4 8K milestone is complete. The ordinary Engine
scheduler, cross-chunk compressor state, 8K cache/index/top-k gate, performance
profile, official checkpoint oracle, and 141-test TPU regression all pass. See
[the completed integration report](deepseek_v4_8k_integration.md) for exact
measurements and remaining limitations. The sections below retain the original
work plan and acceptance criteria.

## Goal

Run a complete DeepSeek-V4-Flash request with up to 8192 total tokens on one
4-chip TPU v5p slice through the normal SGLang-JAX scheduler and ModelWorker.
Checkpoint FP4/FP8/block scales remain packed; dequantization stays online.
This milestone remains batch one, TP=4/EP=4, DP=1, radix cache disabled, and
overlap scheduling disabled. It does not claim concurrent serving or prefix
sharing.

## Work plan and gates

### 1. Scheduler and model state contract

- Use the scheduler's standard 128-token chunked-prefill path.
- Treat `extend_prefix_lens[0]` as the already committed absolute prefix and
  `extend_seq_lens[0]` as the current chunk length.
- Reset every layer's request-owned cache only when the first absolute position
  is zero. Every continuation chunk inherits window KV, compressed KV, main
  compressor scratch, and ratio-4 indexer scratch.
- Keep absolute token positions dynamic so all full 128-token continuation
  chunks reuse one executable.

Gate: host metadata tests cover first, middle, final, and near-8K chunks;
misaligned or out-of-capacity chunks fail before model execution.

### 2. Compressor, cache, index, and top-k correctness

- Make ratio-4 overlap pooling consume the previous chunk's completed group.
- Write compressed rows at the absolute `position // ratio`, not at row zero.
- Preserve ratio-128 partial groups for subsequent decode.
- Always expose the full 128-token sliding window to a short final chunk.
- Validate available compressed rows and top-k masking at 4/128 compression
  boundaries and when the ratio-4 candidate set exceeds `index_topk=512`.

Gates:

1. Whole prefill and `[128, 128, tail]` prefill produce bitwise-equal
   compressor scratch and compressed caches for ratio 4, index ratio 4, and
   ratio 128.
2. With real checkpoint weights, whole and chunked execution produce the same
   final logits and all 43 layer cache fields on a bounded prompt.
3. Near the inclusive 8K public prompt limit (`context_length - 6`, because
   ModelWorker reserves one token plus five safety slots), all reads/writes stay in range;
   rows beyond the available compressed prefix remain untouched and output
   logits are finite.

### 3. Normal Engine path

- Launch with `context_length=8192`, `max_total_tokens=8192`,
  `chunked_prefill_size=128`, and `max_prefill_tokens>=128`.
- Submit one long request through `Engine.generate`; do not call the reference
  runner or manually split the Engine input.
- Compare the Engine's greedy token with the direct ModelWorker gate for the
  identical 8K input, proving scheduler chunk ownership and final sampling.

Gate: prompt accounting, finish reason, output token, and cache lifecycle all
match; a second short request proves reset/reuse after the 8K request.

### 4. Performance and regression

- Record cold compile, first chunk, steady middle chunks, final tail, and first
  decode separately. Never mix model loading or compilation into hot latency.
- Track compilation-cache misses by chunk shape. The 128-token steady path must
  not recompile as absolute position changes.
- Capture per-chip HBM after loading, after the bounded oracle, and near 8K.
- Profile an early and a late 128-token chunk plus decode; attribute time to
  low-bit GEMMs/dequant, compressor/index projection, top-k/sparse attention,
  MoE, mHC, collectives, and host gaps.
- Run the existing low-bit, mHC/HCA/CSA, vertical-slice, framework, and scheduler
  regressions after the new gates.

Performance acceptance is evidence-based rather than a fixed speed promise:
no position-dependent recompilation, no unbounded intermediate allocation, and
a report of steady prefill tokens/s, decode latency, HBM, and the largest traced
bottleneck. Any top-k tiling/fusion change must retain the correctness gates.

## Delivery artifacts

- Source and unit/integration tests in this branch.
- A reproducible 8K ModelWorker gate and a separate Engine gate.
- JSON timing/memory/correctness reports plus optional XPlane traces under the
  ignored `GCP_login/results` directory.
- Updated integration documentation stating exactly what is supported and what
  remains out of scope.
