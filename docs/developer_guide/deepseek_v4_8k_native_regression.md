# V4 native 8K concurrency / pressure / performance regression

Historical run status: **FAILED — B2/B4 full-model 8K numerical correctness.**
This source version is not accepted for 8K multi-request inference. Matching
greedy token IDs and
successful request completion/resource recovery are diagnostic observations,
not passing inference or pressure-correctness acceptance. All performance
numbers below describe the failing implementation, not a validated model
benchmark. No model, kernel, scheduler, cache or allocator production code is
changed by this gate.

The subsequent [position 8023 diagnosis and repair](deepseek_v4_8023_correctness_report.md)
tracks the compressor fix, fused mHC BF16-boundary repair, reference pooling
correction and fresh numerical gates. It does not turn
the failed measurements below into accepted performance results.
The repaired source now passes 1910 cold-cache native comparisons within the
unchanged tolerance, and 514 independent full-context comparisons bitwise;
see that report for exact scope.
Engine pressure and performance have not been rerun on the repaired source.

## Workload and evidence boundaries

- Original complete 43-layer FP4/FP8 checkpoint, online dequant, EP4/DP1.
- Native worker: 7,936 prompt tokens plus 256 generated tokens reaches 8,192 total tokens.
  The direct worker also consumes the last generated token to exercise cache
  position 8,191 / the 2,048th CSA candidate. The standard Engine reserves two
  positions (`max_req_len=context-1`, generation capped at `max_req_len-input-1`),
  so public requests generate 254 tokens and total 8,190. Its guard is unchanged.
- Native B1 runs create two full-logit goldens. B2/B4 decode restores those real
  prefill pages outside timing and compares every decode logit row, alternating
  request order. B4 has two copies of each of the two fixtures. This is not a
  four-distinct-input or concurrent-prefill benchmark; the Engine tests those.
- The independent 43-layer oracle at position 8,191 receives a logical view of
  the native prefix cache. It validates long-context decode computation, not a
  separately recomputed full 8K prefill or official hosted-API accuracy.
- The standard Engine separately tests cold B1/B2/B4 prefill + decode. Its four
  inputs have different first pages, so cold B4 cannot silently reuse a common
  prefix. The two additional inputs receive independent serial Engine baselines.
- Two cold B4 rounds, long-prefix B4 reuse, and eight queued requests exercise
  actual scheduling, request-slot reuse and queueing. Every completed case
  checks that all slots are free and free KV + radix-owned KV equals capacity.

## Genuine allocator pressure

The separate pressure process has a 16,256-token KV pool, enough for one full
8K request but 128 tokens short of two. Only that process uses the existing
`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=64` admission knob to under-reserve future
generation. This is not a recommended production setting.

Numerical correctness is a prerequisite for pressure acceptance; it failed
in this run, so the following observations cannot establish pressure
correctness. The diagnostic checks require a real `KV cache pool is full` event with at least one
retracted request, zero aborted requests, eventual completion and exact output
agreement with the wide-pool baseline. `SGLANG_TEST_RETRACT` and manual pause are
not used. Later distinct 8K requests must evict old prefixes, whose recomputation
is checked again. This stresses the configured KV allocator, not physical HBM
exhaustion or a destructive hardware OOM.

## Diagnostic timing and profiling — correctness failed

Cold/cache-miss calls and profiled calls are excluded from warm p50/p95. Direct
worker measurements synchronize device completion and include the ordinary
ModelWorker call and output transfer. Engine measurements include scheduling,
tokenizer/IPC and streaming; TTFT includes queueing and prefill. Coalesced stream
events are recorded and their normalized intervals are not labeled exact
per-token device latency. Raw XPlane profiles cover two warmed decode steps per
batch; HLO self-times are measured, while FLOP/byte values are compiler estimates.
Fused low-bit GEMM timings include online dequant and cannot isolate its cost.

The wide pool has 33,280 token slots (four 8K contexts plus four page-alignment /
admission headroom pages). Its extra capacity does not increase context length.

## Run, one TPU owner at a time

```bash
python scripts/run_deepseek_v4_8k_native.py --checkpoint /path/to/checkpoint --output /new/native --profile
python scripts/run_deepseek_v4_8k_serving.py --worker-report /new/native/report.json --output /new/serving
SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=64 python scripts/run_deepseek_v4_8k_serving.py \
  --worker-report /new/native/report.json --serving-report /new/serving/report.json \
  --pressure --output /new/pressure --run-log /new/pressure.log > /new/pressure.log 2>&1
```

After capture, use the separate profile-tools environment:

```bash
python scripts/analyze_deepseek_v4_native_profile.py --profile /new/native
```

The source fingerprint must match every report. Reports and traces are backed
up under the ignored local `GCP_login/results` directory, not committed to Git.

## Failure observed, not waived

`v4-native-8k-20260907-01/report.json` fails B2 request 1, decode index 87
(zero-based cache position 8,023): full-logit NRMSE 0.10763054 and maximum
absolute difference 3.1207333. Top-1 still matches at this step. The preceding
step has NRMSE 1.418e-7. Both independent B1 final-position checks are bitwise
equal. This is a batch/long-decode numerical inconsistency; the responsible
kernel was not yet isolated when this diagnostic report was recorded. It must
not be reported as passing just
because the sampled token matches.

Diagnostic continuation uses `--reuse-goldens <first-report>` and
`--continue-on-numerical-failure` in a fresh native report. It retains the
same 0.5% numerical gate and records every failure; `finished=true` only means
the experiment ran to completion, while `complete=false` preserves failed
acceptance. The Engine's explicit `--allow-failed-gates` follows the same rule
to collect remaining serving/pressure measurements without certifying the model.

## Native measurement results

`v4-native-8k-20260907-02/report.json` finished all 256 decode steps at B2/B4.
B2 fails 333 of 512 full-logit rows; B4 fails 666 of 1,024. Maximum NRMSE is
0.471915 in both batches. All top-1 IDs still match the teacher-forced B1
goldens. The two B4 copies of each fixture reproduce the B2 failure, not a
different failure caused only by four-way scheduling. Root cause was not yet
isolated in this historical run, and its 0.5% gate remains failed.

Inspecting the four saved failure NPZs confirms this is not merely a harmless
constant logit offset: subtracting the mean difference from the first failure
still leaves 10.35% relative RMS error (versus 10.76% raw). These repetitive
stress prompts have very confident next-token predictions: top-1 probability
is above 0.9999 in those four rows, and softmax total-variation differences are
only 9.0e-7 to 1.17e-5. That explains why greedy text can agree despite the
logit failure; it neither estimates benchmark accuracy loss nor validates
less-confident prompts or nonzero-temperature sampling.

| Native warm decode | p50 step ms | p95 step ms | Aggregate tokens/s |
| --- | ---: | ---: | ---: |
| B1, fixture 0 | 157.87 | 158.85 | 6.33 |
| B1, fixture 1 | 158.26 | 159.34 | 6.32 |
| B2 | 199.93 | 202.28 | 10.01 |
| B4 | 213.22 | 215.31 | 18.78 |

These are measurements of the currently numerically failing implementation,
not accepted throughput. Each row has 253 warm, unprofiled samples. B1 prefill
is about 115.5–115.8 tokens/s with 128-token chunks. No per-position cache miss
occurs after first-use compilation. B4's first call takes 141.85 seconds and
is excluded; that includes compilation and execution, not pure compile time.

Loaded HBM is 46.37 GiB per chip, after prefill approximately 46.46 GiB and
after B4 approximately 46.62 GiB. The native diagnostic process peaks at
51.11 GiB; prefix restoration and test/profile temporaries are included in
that peak. Reference-only temporary allocations from the earlier B1 oracle
run are not labeled production resident memory.

The historical single-request/page-size-one 8K experiment reported 64.81 ms
decode and 46.28 prefill tokens/s. The new native B1 decode is roughly 2.44x
that latency, while native prefill is faster. This is a regression warning,
not a controlled same-source/input A/B test: cache layout, capacity, numerical
fixes and fixtures differ. The old report cannot certify current correctness.

### Native B1 hardware attribution

Two warmed steps across all eight TensorCores contain all 688 expected layer
MoE all-reduces, plus 48 output/sampling collectives. HLO self-time per core
per step is 148.46 ms, of which routed experts account for 103.89 ms (70.0%).
Dynamic-slice-related fusions total 86.63 ms. The FP4 GEMM custom calls,
including their fused online dequant, total only 7.04 ms within that stage.
Weight selection/layout and movement deserve inspection; these data do not
prove a specific physical HBM-traffic volume or isolate dequant cost.

Other measured stages: attention projection/index 18.80 ms, compressor/paged
state 7.17 ms, MoE collectives including wait 6.38 ms, normalization 4.15 ms,
sparse attention 2.55 ms and mHC 1.73 ms. A per-core HLO sum is not a host
critical-path timeline, so its difference from worker latency is not directly
labeled scheduler idle time.

| Measured HLO self-time, ms/core/step | B1 | B2 | B4 |
| --- | ---: | ---: | ---: |
| Routed FP4 experts, total | 103.89 | 131.57 | 131.76 |
| FP4 GEMM including online dequant, subset above | 7.04 | 12.61 | 12.61 |
| Attention projection/index | 18.80 | 20.41 | 22.22 |
| Compressor and paged state | 7.17 | 11.48 | 15.00 |
| Sparse attention | 2.55 | 5.35 | 10.64 |
| MoE all-reduce including wait | 6.38 | 8.30 | 8.44 |
| All HLO self-time | 148.46 | 190.33 | 202.76 |

All three captures contain two model dispatches and two sampler dispatches on
each TensorCore. The separate raw timeline measures B1 at 162.23 ms host wall,
149.72 ms device-module union and 12.50 ms outside modules per step, averaged
over cores. B4 is 218.88 / 203.54 / 15.34 ms respectively. These are instrumented
timings, not the uninstrumented p50 values. Outside-module time includes worker
preparation and transfers; it is not all scheduler idle. The grouped expert
workload depends on active expert diversity, so duplicated B4 fixtures do not
establish worst-case four-request throughput.

Reporting/attribution CPU tests: 28 passed (`v4-8k-reporting-cpu-20260907-03.xml`).

## Standard Engine observations — not correctness acceptance

`v4-serving-8k-20260907-01/report.json` finished all eight cases: 25 requests,
6,350 output tokens. Two requests create extra serial baselines; the other
23 requests compare all 5,842 output tokens and match their individual
baselines. All finish normally, and every case returns all request slots with
exact free KV + radix-owned KV accounting. This does not override the failed native
full-logit gate: the Engine report correctly retains `complete=false`.
Its per-case `passed` field only summarizes greedy-ID/lifecycle checks. It
must not be interpreted as model correctness or used to promote this build.

| Case | Batch wall seconds | Simultaneous decode tokens/s | Greedy IDs match only |
| --- | ---: | ---: | --- |
| B1, first use including cold compilation | 397.98 | 6.68 | yes |
| Extra serial baseline 2, compiled | 107.28 | 6.69 | baseline creation |
| Extra serial baseline 3, compiled | 106.84 | 6.68 | baseline creation |
| Cold-cache B2, including first B2 compilation | 328.30 | 10.45 | yes |
| Cold-cache B4, first round / first B4 compilation | 472.20 | 18.28 | yes |
| Cold-cache B4, reordered second round / compiled | 331.65 | 18.28 | yes |
| Long-prefix B4 | 59.85 | 18.28 | yes |
| Eight queued requests, four slots | 119.70 | no common eight-way interval | yes |

Both cold B4 rounds observe four running requests and zero cached input
tokens. Prefix B4 records a 7,808-token hit on each request, with first-token
latency 1.12–4.44 seconds. The queued burst completes all 2,032 output tokens;
its 16.98 output tokens/s is end-to-end aggregate throughput, not eight-way
simultaneous decode. The same telemetry observation contains four running
and four waiting requests, so this is not inferred from separate maxima.
Lowest free KV is 512 tokens in B4 and queued cases.

Important latency limitation: `enable_mixed_chunk=false` in this tested
configuration. Long prefills are prioritized while earlier requests wait to
resume decode. In the compiled second B4 round, first-token latency ranges
from 69.26 to 276.24 seconds, and the earliest request has a 207.19-second
output gap even without cold compilation. Thus 18.28 decode tokens/s does not
describe cold-cache end-to-end serving: that round produces only 3.06 output
tokens/s including prefill. Mixed-prefill/decode support and its correctness
regression are separate follow-up work, not certified by this run.

## KV pressure observations — not correctness acceptance

The 16,256-token pressure process reaches zero available KV tokens and logs
one actual capacity-induced retraction with zero aborted requests. Both
requests subsequently finish, and all 508 output IDs match the normal-capacity
baselines. The first pressure case takes 633.88 seconds, including new
capacity-specific compilation and the first B2-to-B1 executable transition.
Its client decode window includes retraction and compilation; it is not a
steady-state throughput measurement.

Two distinct long requests then evict older prefixes, each producing 254
matching output tokens in 106.48 / 106.04 seconds. Resubmitting the original
prompt records zero cached tokens, recomputes the complete long prefill, and
produces the same 254 output IDs in 106.44 seconds. All slots and KV accounting
return correctly after every case. The pressure report is
`v4-pressure-8k-20260907-01/report.json`; all five requests / 1,270 output tokens
match their wide-pool baselines, but `complete=false` preserves the failed
upstream numerical gate.

## Acceptance and next work

This is a bounded 30-request long-context Engine regression (two serial
baseline creations plus 28 exact-output comparisons), not a long-running soak,
an official accuracy benchmark, an HTTP protocol gate or proof of correctness
for all prompts/sampling settings. The native worker separately tests exact
cache position 8,191; the standard Engine's existing safety allowance ends
these public requests at 8,190 total tokens.

The next milestone is correctness, not optimization. Reproduce the first
failure at decode index 87 / zero-based position 8,023, and compare B1 versus
B2 layer outputs and compressor/cache states to isolate the first divergence.
Fix that cause, retain the 0.5% full-logit gate and original packed weights,
and rerun the full long-context numerical gates before accepting any serving,
pressure or performance result. Only after that should the routed-expert
selection/layout and mixed-prefill/decode optimizations be considered.
No production code, tolerances or shared framework behavior were changed to
make these measurements pass.

The final combined real-TPU regression is **146 passed, zero failures/skips**
in 60.67 seconds (`v4-8k-final-tpu-20260907-01.xml`): the existing 118
framework/paging/reference/low-bit/platform tests plus 28 reporting/attribution
tests. Five non-failing warnings concern Flax variable-access deprecation and
JUnit property compatibility. This passing suite does not supersede the failed real
long-decode logit gate and shows a coverage gap that the new harness exposes.
Ruff, syntax compilation and `git diff --check` also pass.

All reports use framework fingerprint
`c7ec7e1357f41b4710f5505fa76ed95fc458693998df6ebab4ec5b0fc39ce983`.
Source, receipts, failure NPZs and raw/XProf traces are synchronized locally
under ignored `GCP_login/results`; no Git commit or GitHub push is made.
