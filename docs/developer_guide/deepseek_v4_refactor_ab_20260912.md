# V4 source-refactor TPU A/B — 2026-09-12

Within the four tested cases, the refactor preserves captured logits exactly,
greedy generations and logical KV/compressor state hashes. Warm TTFT changes
are below 0.1% and TPOT changes below 0.6%; measured allocator allocation
snapshots match on every chip. This is a bounded source-regression check, not
a new public-task accuracy score or HTTP/production-stress certification.

## Setting and source identity

- One newly restored `v5p-8` Spot in `us-east5-a`: four physical v5p chips,
  TP4 / EP4 / DP1. Original independent 500 GiB data disk reused without
  formatting; original checkpoint and compilation cache retained.
- Complete 43-layer official Instruct checkpoint
  `deepseek-ai/DeepSeek-V4-Flash`, revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`.
- Original packed FP4/FP8/block-scale storage and online dequantization;
  BF16 activations, no whole-model BF16 weight expansion.
- Python 3.12.14, JAX/jaxlib 0.11.1, libtpu 0.0.46.1, NumPy 2.2.6,
  Flax 0.12.9; original inference package freeze restored and checked.
- Baseline: commit `e8e1eb8896f49c138cffb377f848f15be8dff5c1` in an independent
  source directory. Candidate: uncommitted `integration/deepseek-v4` refactor,
  frozen as a baseline Git bundle, tracked binary patch, new-file archive and
  SHA-256 manifest. Candidate verification covered 1,271 source files and
  absence of 16 removed paths. No commit or push was made for this test.
- Baseline runtime fingerprint:
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.
- Candidate runtime fingerprint:
  `5d9585a3ec6e7823e17827e5d6881b4f11be9e89d6c46e85eaf272014a369f64`.
- Identical driver SHA-256:
  `3fd998036a7bed616ee068437422b73d5fe96423404ef46331e55b5f9f52e1a6`.

Both arms ran serially on the same node and environment, through the native
SGLang-JAX ModelWorker with real request/page allocators. Execution was checked
against explicit choices: mHC/HCA/CSA `pallas`, MoE `gmm_tuned`, attention TP
and CSA decode batching enabled, FP8 `gmm`, fused normalization, merged
projections and fused `wo_a` enabled. These remain explicit options, not newly
enabled defaults.

Context capacity is 8192, page size 128, KV capacity 33,280 tokens, maximum
four requests, static memory fraction 0.85 and seed 42. The harness uses
`disable_precompile=True`, `disable_overlap_schedule=True` and deterministic
synthetic token fixtures. It is not an HTTP scheduler workload or an official
chat-template/public-benchmark evaluation.

## Protocol and correctness

Cases are 128/B1, 1024/B1, 1024/B4 and 8064/B1, followed by exactly 32 greedy
output tokens (EOS ignored). Each arm uses two warmups followed by three
measured runs per case. All 24 measured runs across both arms have zero
reported compilation misses. No measured wave was discarded.

The global prefill bucket contains 128 tokens: B1 advances 128 tokens per
call, while the lockstep B4 harness advances 32 tokens per request. This is
not the older benchmark's 128 tokens **per request**. TTFT is prefill wall time
to the first generated token; TPOT is wall time for the subsequent 31 decode
steps divided by 31. Timings include harness metadata handling and host logit
copies. State capture and disk writes occur outside timing. Loading and cold
compilation are excluded; startup/cache effects are not an A/B conclusion.

The first warmup captures every prefill-boundary and decode-step logit array,
all 32 generated IDs per request, and logical state after generation. Each
arm additionally requires identical generated IDs across its five repetitions.
The long-input case crosses position 8023 and ends below the 8192 total limit.

Independent local audit of downloaded artifacts confirms:

- **228 logit-array comparisons**, representing **417 rows / 53,909,760
  float32 values**, all finite and bitwise equal; maximum absolute difference 0.
- **224 generated token IDs** across seven requests, all identical.
- **2,471 logical KV/compressor state fields**, matching shape, dtype and
  byte-content hashes. Raw state tensors were not saved; these are comparisons
  of the captured logical-state hashes.
- Same input IDs, checkpoint path, driver bytes, server configuration and
  actual selected backends; both arm reports are complete.

Before the full-model A/B, a four-device sharded BF16 matmul passed, followed
by **73 passing TPU projection/option/module-boundary tests**. The earlier
two-value CPU QNorm tail-bit failure did not recur in this TPU test run.

## Warm performance and allocator snapshots

Values are medians of three measured runs. Positive percentage means slower.
HBM is the largest per-chip median `bytes_in_use` snapshot, not solely weight
storage, a directly measured traffic value or total physical capacity. All four
per-chip snapshots match exactly between arms, not just the rounded maximum.

| Input / batch | Baseline TTFT (s) | Refactor TTFT (s) | Change | Baseline TPOT (ms) | Refactor TPOT (ms) | Change | HBM both (GiB/chip, max) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 / B1 | 0.31165 | 0.31170 | +0.014% | 41.315 | 41.552 | +0.574% | 44.6162 |
| 1024 / B1 | 2.42157 | 2.41993 | -0.068% | 41.389 | 41.225 | -0.395% | 44.6178 |
| 1024 / B4 | 10.18784 | 10.18313 | -0.046% | 61.505 | 61.365 | -0.227% | 44.8017 |
| 8064 / B1 | 19.20965 | 19.19510 | -0.076% | 41.654 | 41.447 | -0.496% | 44.8031 |

| Input / batch | Prefill tokens/s before → after | Decode tokens/s before → after |
| --- | ---: | ---: |
| 128 / B1 | 410.71 → 410.66 | 24.20 → 24.07 |
| 1024 / B1 | 422.87 → 423.15 | 24.16 → 24.26 |
| 1024 / B4 | 402.05 → 402.23 | 65.04 → 65.18 |
| 8064 / B1 | 419.79 → 420.11 | 24.01 → 24.13 |

Throughput is aggregate across the batch. Prefill rate is input tokens / TTFT;
decode rate is batch size × 31 / decode wall time. These are not the published
HTTP benchmark's whole-request-duration throughput definitions. Three samples
are not a statistical equivalence certificate or evidence of a speedup.

## Evidence and remaining scope

Private local evidence is under
`GCP_login/v4-refactor-recovery-20260912-01/`; the original disk retains
`profiles/v4-refactor-ab-20260912-01/`. Both source arms, all recorded logits,
input/generated arrays, state-hash records and incremental reports were
retained. The local audit recomputes tensor equality, timing medians and
throughput, and checks protocol identity and compilation-miss exclusion.

- Baseline report SHA-256:
  `82536da5725e912b79f82076820904ddba77ba22b015f7b01c4d10a38fd69924`.
- Candidate report SHA-256:
  `802e7f93cd6bc6de8a50465992f116b8c3915ee106a9060841c937b852548fbb`.

The first local audit was started before the copy completed and stopped on a
missing NPY file; after transfer completion the independent audit passed.
No tensor file or numerical tolerance was changed to obtain that result.

This closes the bounded refactor A/B, not the broader 8K concurrent pressure,
Engine/HTTP lifecycle/stress or full public-accuracy rerun. Historical GPQA,
GSM8K and HumanEval scores remain measurements of their original snapshots.
The test processes exited normally; the restored Spot node remains online.
