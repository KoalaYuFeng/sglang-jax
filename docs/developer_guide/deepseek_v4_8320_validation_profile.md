# V4 8192-input validation and post-EP-fix profiling

Status: complete. All ten numerical/performance cases and final audit passed.
Evidence archived on the persistent disk and verified/extracted locally.
No production context-limit change has been accepted.

## Scope and retained configuration

Experiment `v4-8320-profile-20260910-02` runs on four physical TPU v5p chips
(TP4/EP4, DP1), the full 43-layer official checkpoint, original packed
FP4/FP8 weights and the explicitly enabled Pallas attention/compressors,
GMM-tuned MoE, FP8 GMM and fused dense helpers. Runtime framework fingerprint:
`2543e8b0e98fc56658912cd2d14f05e4d4ae4e00cc961f07bc150744070e0c0f`.

The total-context guard remains 8192 in production. Diagnostic processes
override only that guard to 8320, the page-aligned capacity needed for 8192
input tokens, 32 generated tokens and the benchmark's two reserved slots.
No scheduler algorithms, kernels, checkpoint format or numerical tolerances
are changed by this experiment.

## Acceptance and timing protocol

- Boundary operator tests cover noncontiguous pages, the 65th HCA record,
  CSA top-k over 2080 candidates, BF16 arithmetic and compressor state.
- New 8192-input B1 reference uses the independent ring EP implementation,
  with four fixed prompts. It shares the unchanged rest of the model and is
  not an independent official GPU accuracy benchmark.
- An extra B1 trajectory generates 129 outputs, executing through position
  8319 and the completion of the 65th HCA compressed record.
- Candidate B1/4/8/16/32 runs use exact 8192- and 4096-token inputs and 32
  greedy outputs. Every checked output compares the full vocabulary with
  the independent EP reference: finite, same top-1, NRMSE <= 0.005.
  B1 checks four prompts sequentially; larger batches cycle these four
  frozen prompts across actual concurrent slots. These controlled fixtures
  are not a representative traffic/expert-routing distribution.
- Each case owns an exact `batch * context` page pool. Requests are packed
  concurrently, not queued as B1; there is no batch shrink, input truncation
  or offload. OOM is distinguished from numerical and other failures.
- Prefill advances 128 tokens **per request** per call (`batch * 128` packed
  tokens). This is ModelWorker fixed-batch timing, not the HTTP scheduler's
  global chunk-128 configuration.
- TTFT includes page/batch preparation through the first output argmax;
  TPOT averages the remaining 31 output intervals. Both include worker,
  sampling, synchronization and full-vocabulary host copy/checks, but exclude
  checkpoint loading, warmup and reference comparison. Three whole waves
  must have zero compilation misses before their timings can be reported.
- B1/B16 capture early prefill, late prefill and decode in a separate warmed
  wave. Instrumented timings are not mixed with the three benchmark rounds.

## Analysis safeguards

`scripts/analyze_deepseek_v4_ep_profile.py` reuses the existing XProf exporter
and HLO exclusive-self-time accounting. It separates ordered MoE all-gather
(including wait), local FP32 addition/layout, and attention TP communication.
Coverage requires 8 TensorCore planes, 43 EP gathers and 129 routed FP4 GEMM
calls per full-model call per TensorCore. Empty SparseCore planes are excluded.
Raw traces and grouping witnesses are retained. HLO FLOP/byte estimates do
not establish actual HBM bandwidth; fused GEMM time does not isolate dequant.

CPU analysis accounting tests: 16 passed. Boundary gates: CPU 2 passed and
TPU 15 passed (including 8192 and 8320 cases).

The 8192-input candidate B1 matches all four independent-reference prompts
exactly (128 full-vocabulary output vectors). Its additional 129-output
trajectory through position 8319 is also bitwise equal, NRMSE zero.
Three compile-free B1 timing rounds give median TTFT 20.9194 s and TPOT
41.2743 ms (about 24.23 output tokens/s); the corresponding prefill rate is
391.60 input tokens/s. These are the fixed prompt-0 waves, not an average
across all four correctness prompts. All five 8K batches passed, including
an independent recomputation from saved arrays and an audit of every formal
wave's check receipts. All five 4K batches also passed. No case reported OOM.

| Input tokens | Actual batch | Median TTFT, s | Median TPOT, ms | Aggregate prefill tokens/s | Aggregate decode tokens/s |
| --- | --- | --- | --- | --- | --- |
| 8192 | 1 | 20.9194 | 41.2743 | 391.60 | 24.23 |
| 8192 | 4 | 54.4761 | 73.0088 | 601.51 | 54.79 |
| 8192 | 8 | 91.3877 | 85.0899 | 717.12 | 94.02 |
| 8192 | 16 | 168.9587 | 119.5262 | 775.76 | 133.86 |
| 8192 | 32 | 326.0461 | 177.0044 | 804.01 | 180.79 |
| 4096 | 1 | 10.3114 | 40.5541 | 397.23 | 24.66 |
| 4096 | 4 | 25.9632 | 72.7261 | 631.05 | 55.00 |
| 4096 | 8 | 45.4480 | 84.0715 | 721.00 | 95.16 |
| 4096 | 16 | 83.8812 | 114.5887 | 781.30 | 139.63 |
| 4096 | 32 | 161.8229 | 166.3749 | 809.97 | 192.34 |

The 8K audit verifies unchanged historical reference and source hashes.
B4/8/16/32 maximum NRMSE is 1.54459e-7. Final HBM allocator snapshots are
41.18, 44.64, 49.22, 58.38 and 76.72 GiB per physical chip respectively;
these are allocation snapshots, not a time-resolved physical-memory trace.
The corresponding 4K snapshots are 40.62, 42.40, 44.75, 49.46 and 58.87 GiB
per chip. Maximum 4K NRMSE is 1.54642e-7. The final audit recomputes saved
warmup/long-boundary arrays and validates all formal-wave check receipts,
zero compilation misses, three-round medians and unchanged source/reference
hashes. Not every formal wave's full logits are saved; those waves retain
their in-process numerical comparison receipts.

The B1 capture passes all full-model dispatch, GMM and EP coverage checks:

| B1 phase | Profiled host call, ms | Mean device-module union, ms | Routed gate/up + down GMM, ms | EP gather including wait, ms | Compressor/state, ms | FP8 projection, ms |
| --- | --- | --- | --- | --- | --- | --- |
| Early prefill, position 128, 128 tokens | 316.71 | 302.94 | 151.74 | 24.32 | 23.54 | 29.71 |
| Late prefill, position 8064, 128 tokens | 326.34 | 314.44 | 163.81 | 20.89 | 23.54 | 29.71 |
| Decode, position 8208 | 43.77 | 33.34 | 5.93 | 5.49 | 4.90 | 4.75 |

Component values are exclusive HLO self-time per TensorCore per call,
averaged across eight TensorCores, not a sum of eight chips' times. They
exclude other components and are not an exhaustive additive wall-time table.
The local ordered EP sum/layout takes 0.37 ms in prefill and 0.118 ms in
decode. EP gather includes waiting, so these measurements do not separate
wire transfer from expert-load imbalance. The host/device difference includes
the worker, sampling, synchronization and diagnostic full-logit host copy;
there is no HTTP scheduler in this fixed-batch measurement.
The decode trace is a single non-compression-boundary sample, not a
boundary-frequency-weighted average of all 31 timed decode calls.

B1 CPU trace analysis finished before the B4 first wave completed, during
loading/compilation, and not during the formal B4 timing rounds. The analysis
start/observed-completion receipts and benchmark event timestamps are retained.

B16 analysis also passes all coverage checks. The first CPU export reached
its 180-second analysis budget after finishing early prefill; its partial
results/error receipt are retained. A bounded resume exported late prefill
and decode successfully, finishing while B32's first warmup was still pending.
This analysis timeout is not a model numerical/performance failure.

| B16 phase (2048 packed prefill tokens) | Profiled host call, ms | Device-module union, ms | MoE gate/up + down GMM, ms | FP8 projection, ms | Compressor/state, ms | EP gather including wait, ms |
| --- | --- | --- | --- | --- | --- | --- |
| Early prefill | 2617.81 | 2595.51 | 601.99 | 465.17 | 323.86 | 105.99 |
| Late prefill | 2649.92 | 2629.68 | 595.97 | 465.17 | 323.87 | 119.49 |
| Decode, position 8208 | 114.97 | 99.80 | 21.34 | 8.71 | 32.91 | 9.87 |

At late prefill B16, sparse attention is another 326.29 ms and attention
projection/helpers 320.77 ms. MoE GMM is about 23% of exclusive HLO time,
not the roughly 52% observed in B1 late prefill. Decode's largest single
category is now compressor/state handling, about 33% of HLO self-time.
The ordered local EP addition/layout is only 0.136 ms in that decode sample.

An additional HLO accounting pass (`compressor-breakdown.json`) subdivides
the 32.9052 ms decode compressor category without double-counting. CSA main
projection is 16.4141 ms and CSA index projection 2.1516 ms; CSA pooling/
emission is 1.6874 ms. HCA projection-related instructions are 0.9829 ms
and HCA pooling/emission 0.4330 ms. Other compressor custom fusions and
loop fusions account for 6.7635 and 3.0881 ms respectively; their raw source
witnesses remain available, without calling them all pure memory traffic.

The active `csa_project_decode_pallas` uses a `(tokens, width / 128)` grid
of the retained single-row `_csa_v4_gemv_kernel`. It batches dispatch but
still performs independent per-request GEMVs with the fixed FP32 reduction
tree, rather than a shared-weight multi-row MXU dot. Inputs/weights consumed
by this body are BF16 and products/accumulators FP32; this main-projection
cost is not an FP4-unpacking measurement. This is an evidence-backed target
for a subsequent isolated kernel study, not a kernel change in this run.

The 4K B1 captures also pass all coverage checks. The same accounting gives:

| 4K B1 phase | Host call, ms | Device-module union, ms | Routed MoE GMM, ms | EP gather including wait, ms | Compressor/state, ms | FP8 projection, ms |
| --- | --- | --- | --- | --- | --- | --- |
| Early prefill, position 128 | 313.32 | 300.97 | 151.67 | 24.55 | 23.70 | 29.51 |
| Late prefill, position 3968 | 331.57 | 320.18 | 170.14 | 23.95 | 23.71 | 29.51 |
| Decode, position 4112 | 43.20 | 33.22 | 5.94 | 5.65 | 4.83 | 4.82 |

These decode samples show similar device time at 4K and 8K; they do not
establish identical cost at every compression boundary or for larger batches.
The 4K B1 CPU analysis finished in 46 seconds during B4's first warmup,
before formal timings, and passed full dispatch/GMM/EP coverage.

The 4K B16 captures give the same qualitative bottleneck ordering:

| 4K B16 phase (2048 packed prefill tokens) | Host call, ms | Device-module union, ms | Routed MoE GMM, ms | FP8 projection, ms | Compressor/state, ms | EP gather including wait, ms |
| --- | --- | --- | --- | --- | --- | --- |
| Early prefill, position 128 | 2616.10 | 2589.40 | 601.99 | 465.25 | 327.17 | 106.04 |
| Late prefill, position 3968 | 2627.83 | 2607.98 | 593.40 | 465.24 | 327.16 | 129.33 |
| Decode, position 4112 | 114.81 | 99.87 | 21.84 | 8.70 | 33.21 | 10.54 |

Within decode compressor time, CSA main/index projection takes 16.4149 /
2.1516 ms, closely matching the 8K capture. Local ordered EP sum/layout is
0.1352 ms. Late-prefill sparse attention and attention projection/helpers
take another 326.23 and 322.76 ms respectively.

The 4K B16 first CPU analysis reached its 180-second budget after early
prefill. Its attempted bounded resume was refused before launching analysis
because B32's first warmup had already finished. Remaining captures were
successfully exported in 39.6 seconds **after all formal matrix waves ended**.
All three captures pass dispatch/GMM/EP coverage; timeout/refusal receipts
are retained and no heavy CPU trace analysis overlaps formal timing.

Static pool specifications are also calculated independently of trace timing
(`memory-breakdown.json`), with totals checked against `pool_size_bytes`.
For B32, 8320-token capacity per request, the replicated pool is 36.6615 GiB
per physical chip. HCA page-end snapshots account for 20.3223 GiB, all raw
window buffers for 10.9232 GiB, CSA snapshots for 3.3341 GiB, compressed
KV/index buffers for 1.7068 GiB, and active request state for 0.3751 GiB.
This is **allocation size**, not measured HBM traffic, a bandwidth estimate,
or proof that the largest allocation dominates execution time. It excludes
weights and other runtime/executable allocations.

## Interpretation and bounded next experiments

- Small-batch prefill: routed FP4 MoE GMM remains the largest category.
  Its time includes fused conversion, so this trace alone cannot attribute
  the majority to FP4 unpacking or justify replacing the reused GMM design.
- Larger-batch decode: investigate CSA main/index projection in isolation.
  A shared-weight multi-row implementation must preserve the validated V4
  BF16 rounding and FP32 reduction contract; replacing it with an arbitrary
  matrix dot is not assumed numerically equivalent.
- Larger-batch prefill: FP8 projection, sparse attention, attention helpers
  and compressor handling are substantial alongside MoE. Optimizing only
  MoE does not address most exclusive device time at B16.
- HBM capacity: page-end state snapshots are a large allocation. Any change
  must retain concurrent requests, chunk continuation and page ownership;
  allocation accounting is not evidence of a corresponding bandwidth cost.

No such kernel, memory-layout or scheduler optimization is implemented by
this validation/profiling task. No fresh before/after A/B of the EP fix is
claimed; historical pre-fix numbers use a different reduction contract.

The initial `v4-8320-profile-20260910-01` attempt stopped before checkpoint
loading because its helper referenced a constant at the wrong module scope.
It automatically restored the original server. This is a retained harness
error, not an OOM or model numerical failure. Attempt `-02` corrects the
process-local override; the first attempt and historical results are intact.

## Final audit, service and evidence

The controller completed all 13 stages: two boundary test stages, independent
8K EP reference, and ten candidate cases. Final saved-array audit passed;
all twelve B1/B16 trace captures pass full-model dispatch/GMM/EP coverage.
The original HTTP service was restored as PID 259000, loopback port 30126,
with 8192 total context, four request slots, 33280 KV tokens, global prefill
chunk 128, overlap enabled and mixed prefill/decode disabled. Its owned PID/
command and source receipt match; ready/idle checks and full KV release pass.
This is a restoration check, not a new HTTP accuracy or pressure benchmark.

Full evidence archive on the persistent data disk:
`profiles/v4-8320-profile-20260910-evidence.tar.gz` (3,604,281,034 bytes).
SHA-256: `3abe3ac87810a85c2a254cec8423ed00e2a2ab053cef171d899e1c9dba54f556`.
It includes both harness attempts, raw XPlane and derived HLO tables,
reference/candidate arrays, check receipts, tests, source snapshots and
explicit reference dependencies, without checkpoint weights or credentials.
The restored live server log is a prefix snapshot at archival time.

The local copy is verified and extracted under ignored
`GCP_login/results/v4-8320-profile-20260910-02/`; archive file `evidence.tar.gz`
has the same byte count and SHA-256. `local-verification.json` confirms 1152
safe archive members, 763 source files matching the current local checkout,
six verified reference dependencies, and twelve raw-trace/report hash pairs.
This local check verifies provenance and the retained remote numerical audit;
it does not claim a second local TPU or NumPy numerical rerun.
No commit or push is performed by this validation task.
