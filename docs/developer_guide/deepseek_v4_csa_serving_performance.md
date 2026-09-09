# Original-CSA V4: Engine, HTTP and performance acceptance

Status: measured on the existing four-chip TPU v5p Spot slice, 2026-09-08.
The overlap Engine lifecycle gate passes all 10 output comparisons, including
acknowledged abort, real explicit retract/resume and post-eviction recompute.
The normal 8K Engine gate also passes all eight scenarios (25 requests / 6350
output tokens). Natural KV-pressure rerun `02` passes all four scenarios after
a harness-only observer correction; failed attempt `01` is preserved.
HTTP workload/soak cases pass, but full attempt `01` fails its status-observer
timeout during first logprobs compilation; a fresh separate lifecycle subset passes.
Native B1/B2/B4 performance and numerical checks pass; B4 interior/boundary
offline profiles are exported and checked. HTTP client and profile/report
contracts pass 49 CPU tests, including separate boundary/interior trace attribution,
whole-case compilation-log tracking and cooperative-observer teardown.
No new cloud resource or production model/scheduler change is part of this
test phase. Preserve the accepted production/profile fingerprint:
`cf77664b90f217c3bf1d1bdaa2cb112b59bc70d757a7223992d8920a6c86f7e0`.

The prerequisites are complete: 261 V4/template tests, 106 bitwise real-input
checks, 14 short 43-layer checks, 2424 native cold 8K B1/B2/B4 comparisons and
514 independently constructed full-logit reference rows. See the
[CSA integration record](deepseek_v4_csa_integration.md). New serving tests
must use these same-source receipts, not the failed historical 8K benchmark.

## Plan and acceptance boundaries

1. Engine lifecycle with overlap scheduling: cold B2, prefix B4, streaming
   logprobs, acknowledged cancellation, slot reuse, explicit retract/resume,
   eviction and recomputation, all with expected output IDs.
2. Complete 43-layer 8K Engine: four distinct prompts (two fixed numerical
   fixtures and two rotated variants with serial Engine baselines), cold
   B1/B2/B4, repeated B4, prefix reuse and B8 queueing on four active slots.
   Every request must finish with exact baseline IDs, and idle request/page
   accounting must return correctly.
3. Genuine allocator pressure: a separate 16256-token pool with admission
   estimate clip 64 must actually exhaust KV pages, naturally retract without
   aborting requests, resume correctly, evict older prefixes and recompute
   them. Manual pause or forced test-retract is not this acceptance criterion.
4. Owned HTTP server on `127.0.0.1` only: native streaming/nonstreaming,
   cold B4, warm prefix B4, B8 queueing for at least five minutes and three
   rounds, explicit cancellation, client disconnect, slot reuse, invalid
   over-context rejection, cache flush/recompute, and OpenAI-compatible
   completion streaming/nonstreaming. No public deployment or external
   load target. Shut down only the process group created by this gate.
5. Native performance/profile: warmed full-model decode B1/B2/B4 near 8K,
   p50/p95, aggregate tokens/s, allocator HBM snapshots/peaks, actual compiled
   original-kernel evidence and bounded XPlane captures. Keep full-logit
   checks enabled. Export traces with the separate CPU profile-tools runtime.

One TPU-owning process at a time. CPU-only parser/report tests and local test
development can run alongside it. Stop acceptance on numerical or lifecycle
failure; do not widen tolerance or silently switch to reference kernels.

## Measurement semantics

- Cold compilation, model loading and cache-hit prefill must not be mixed
  into warm kernel latency. Native `cache_misses` and profiled steps are
  excluded from warm p50/p95 and throughput.
- `--profile-boundary` separately captures position 8063 (CSA/HCA boundary),
  while the ordinary trace covers 8064/8065. Do not average the deliberately
  oversampled boundary trace into ordinary decode. The other 128-token
  boundary, 8191, remains unprofiled for the latency record.
- Native wall time includes submission, completion wait and output transfer;
  device timeline partitions are reported separately. Compilation counts and
  eight TensorCore timelines are not eight requests or eight physical chips.
- Engine/HTTP TTFT and end-to-end throughput include scheduling and transport.
  SSE can coalesce tokens; normalized stream-event intervals are not isolated
  per-token hardware latency. B8 is an eight-request queue with at most four
  active requests, not simultaneous B8 inference.
- HBM allocator snapshots/peaks are measured allocations. XProf estimated
  FLOPs/bytes are compiler estimates, not physical hardware traffic counters.
- This is a bounded synthetic regression/soak, not an official model-accuracy
  benchmark, unbounded reliability claim or production-capacity SLA.

## Receipts

Receipts under ignored local `GCP_login/results` (same names remotely):

- `v4-csa-engine-lifecycle-20260908-01/report.json`: complete, 10 comparisons.
- `v4-csa-serving-report-cpu-20260908-01.xml`: 44 passed.
- `v4-csa-serving-report-cpu-20260908-02.xml`: 45 passed, boundary sampling added.
- `v4-csa-engine-8k-20260908-01/report.json`: complete, 8 scenarios / 25 requests.
- `v4-csa-serving-report-cpu-20260908-03.xml`: 46 passed.
- `v4-csa-serving-report-cpu-20260908-04.xml`: 31 selected tests passed.
- `v4-csa-serving-report-cpu-20260908-05.xml`: all 48 reporting/client tests passed.
- `v4-csa-serving-report-cpu-20260908-06.xml`: all 49 tests passed, cold/idle timeout policy added.
- `v4-csa-pressure-8k-20260908-01/report.json`: incomplete/failed, preserved.
- `v4-csa-pressure-8k-20260908-02/report.json`: complete, four scenarios / five requests.
- `v4-csa-http-8k-20260908-01/report.json`: seven workload/abort cases and protocols pass;
  full receipt fails on the first-logprobs status timeout, preserved.
- `v4-csa-http-lifecycle-20260908-02/report.json`: complete **lifecycle subset**, six cases.
- `v4-csa-performance-8k-20260908-01/report.json`: complete, 2052 full-logit checks pass;
  eight boundary/interior trace captures saved.

The new HTTP driver is `scripts/run_deepseek_v4_http_stress.py`; it consumes
the complete normal Engine report and launches the ordinary
`sgl_jax.launch_server` with explicit Pallas mHC/HCA/CSA backends. It does not
instantiate a second in-process model in the client.

## Normal Engine result

All eight scenarios pass, including four distinct cold prompts in two request
orders, 7808-token prefix hits, and eight queued requests on four active slots.
Every checked completion matches its fixed baseline, with no request-slot or
KV-capacity leak. Two rotated prompt variants first establish serial Engine
baselines; they are not additional independently-oracled full-logit fixtures.
The standard public limit leaves two tokens of headroom: 7936 input tokens
plus 254 generated tokens per request.

| Engine scenario | Common active-decode aggregate tokens/s | End-to-end output tokens/s |
| --- | ---: | ---: |
| Cold B1, includes first compilation before decode | 6.696 | 0.737 |
| Serial variants, already compiled | 6.704 / 6.706 | 2.988 / 3.002 |
| Cold B2, includes first B2 compilation | 10.624 | 1.864 |
| Cold B4 first round, includes first B4 compilation | 18.795 | 2.725 |
| Cold B4 second order, already compiled | 18.793 | 4.204 |
| Long-prefix B4 | 18.792 | 17.838 |
| Queued B8, four active slots | No common eight-request window | 17.838 |

The two compiled serial variants have TTFT 47.125 / 46.730 seconds. Compiled
cold B4 TTFT is 47.155 / 94.262 / 141.081 / 187.795 seconds; prefill admission
is serial under this 128-token chunk configuration. Prefix-hit B4 TTFT is
0.781 / 1.537 / 2.297 / 3.058 seconds. B8 finishes all 2032 output tokens in
113.913 seconds; the second queued group sees first tokens after roughly
57.735–60.013 seconds. Do not present its aggregate throughput as B8 decoding.

Final response `cache_miss_count` is not a cumulative record of an entire
request: it can be zero after a cold compilation earlier in that request.
Cold/compiled distinctions above use the actual scheduler logs and phase
sequence. The new HTTP gate records per-case server-log byte ranges and all
logged compilation misses; its warm-soak minimum excludes any round with a
logged miss. Raw logs and per-request events remain preserved.

## Status-observer cancellation finding

Pressure attempt `01` genuinely exhausted the 16256-token KV pool, naturally
retracted one request without aborting, and recovered both exact 254-token
outputs. The first unique-prefix eviction request also passed. The next case
stopped progressing after generation logs ended; the scheduler's forward thread
was idle, not compiling or executing a TPU kernel. The case never completed its
final status/accounting check, so this is **not a passing pressure receipt**.

The harness canceled its monitor task when generation finished, potentially
interrupting an in-flight `Engine.async_get_server_info()` call. A deterministic
CPU reproduction on the unchanged production `_Communicator` confirms that
canceling a sent query leaves its result event uncleared; even after the reply,
the next query queues forever. Stack dumps plus that trigger identify a strong
explanation for the live hang, but the stacks alone do not expose asyncio task
locals. The reproduction and three process stacks are saved with failed attempt
`01`. Its owned Engine was interrupted gracefully after diagnostics.

Only test drivers are changed: finish/cancel owned generation tasks, let the
observer's current RPC drain and its loop exit naturally, then perform the final
idle check. Observer drain and final Engine status checks now have 60-second
bounds. Request timing stops before observer teardown. The HTTP harness uses
the same cooperative drain. CPU tests exercise both normal draining and a
timeout without canceling the live RPC, then verify a subsequent query works.

The shared production `_Communicator` itself remains unchanged. Cancellation
safety of that control-plane channel is a diagnosed framework limitation, not
a CSA numerical/kernel fix, and requires a separately scoped production change.

The fresh cooperative-observer rerun `02` passes all four pressure scenarios:
real exhaustion/recovery B2, two unique 8K prefix evictions, and recomputation
of the original fully evicted prefix (`cached_tokens=0`). All five requests /
1270 output tokens match their baselines. The pool reaches zero free tokens,
with one natural retraction and zero aborts. Idle request slots and KV capacity
account correctly after every case. Exhaustion/recovery takes 211.466 seconds;
the three eviction/recompute requests take 84.125 / 83.718 / 84.148 seconds.
These include prefill and recovery and are not isolated warm-decode latencies.

## HTTP workload result and cold-control-plane limitation

`v4-csa-http-8k-20260908-01/report.json` remains **incomplete/failed overall**.
Before its lifecycle-tail failure, these cases passed:

- Over-context input rejected with HTTP 400 and no lasting allocation.
- Native nonstream B1 and OpenAI-compatible streaming/nonstreaming completions
  agree on output text, finish reason and token accounting.
- Cold streaming B4: all 1016 output IDs match, four active requests observed,
  259.979 seconds including a compilation miss.
- Warm-prefix B4: 256 output IDs match in 16.422 seconds, no compilation miss.
- Three B8 queued rounds: all 24 requests / 6096 output IDs match; 342.191
  seconds total soak, with no compilation misses in any round. Round throughput
  is 17.8681 / 17.8680 / 17.8667 output tokens/s. Four active slots and a waiting
  queue are observed; no idle slot/KV accounting leak.
- Explicit HTTP abort is acknowledged; its emitted three-token prefix matches.

The following first 8K request with logprobs starts prefill at 07:02:31 UTC;
at 07:05:03 the scheduler logs prefill/decode compilation misses, then resumes
normal decoding. Its status observer had already exceeded a 60-second HTTP
read deadline during the roughly 152-second compilation window. The generation
coroutine returned before observer teardown raised that saved `ReadTimeout`;
its row was not persisted by the initial harness, so **do not count that request
as a stored passing receipt**. Client disconnect, subsequent reuse and final
flush/recompute were not reached. The test-owned server was shut down; shutdown
signal logs are not evidence of a spontaneous model crash.

This reveals cold-path control-plane unresponsiveness, not a 60-second service
SLA pass. The test observer now shares generation's bounded 1800-second cold
budget and records query latency; final idle checks retain their 60-second
budget. Validated generation rows are persisted before observer teardown so
future monitor failures cannot discard them. Production code is unchanged.
A fresh `--lifecycle-only` receipt primes its own B2 prefixes and exercises the
remaining cancellation/logprobs/disconnect/reuse/flush cases. Its scope is
explicitly a subset; it does not relabel the failed full run or rerun the soak.

That fresh subset, `v4-csa-http-lifecycle-20260908-02`, passes all six cases:
cold B2 priming, acknowledged abort, post-abort logprobs/slot reuse, client
disconnect, post-disconnect reuse, and cold flush/recompute. All five completed
64-token requests match (320 tokens), and the two intentionally canceled
prefixes match (two and one tokens). Logprobs are finite; idle accounting is
correct throughout and final flush releases all 33280 KV tokens. The longest
observed status query is 35.185 seconds during cached-artifact loading/compilation
of the logprobs path; its full case takes 45.131 seconds. Warm post-disconnect
reuse takes 10.181 seconds; cold flush/recompute takes 56.554 seconds.

Thus all planned workload and lifecycle scenarios have passing evidence across
the two receipts, but there is **no single all-green full HTTP run**. In
particular, no claim is made that the original 60-second cold status-query
deadline passes. Both owned HTTP servers are stopped after their runs.

## Native warm performance and measured HBM

The complete performance receipt has 2052 passing full-logit comparisons,
516 bitwise equal, maximum NRMSE `1.6165743943474808e-7`, all finite and top-1
equal. This includes compatibility against the immutable accepted 8K baseline,
two read-only reconstructed independent checks at position 8191, and B2/B4
comparisons on every decode step. Actual optimized HLO confirms all 21 original
CSA layers' projection, main/index emission, StreamIndex and joint attention,
alongside original mHC and all 20 HCA layers; no fallback or relaxed tolerance.

| Native full-logit path | Warm samples | p50 step ms | p95 step ms | Aggregate tokens/s |
| --- | ---: | ---: | ---: | ---: |
| B1 prompt 0 | 252 | 157.182 | 158.039 | 6.3614 |
| B1 prompt 1 | 253 | 157.558 | 158.631 | 6.3444 |
| B2 | 252 | 197.309 | 199.514 | 10.1413 |
| B4 | 252 | 206.835 | 208.957 | 19.3433 |

Each workload runs all 256 decode steps near 8K. First cache-miss calls and
profiled steps 127/128/129 are excluded from the warm table. Position 8191
remains an ordinary unprofiled sample. B2/B4 restore exact B1 prefixes outside
timing; this phase is not another cold-concurrent-prefill benchmark. The native
two-prompt fixture and full-logit device-to-host transfer differ from the
four-distinct-prompt Engine/HTTP generation workload, so their timings are not
interchangeable or a direct framework-overhead subtraction.

Per-chip allocator measurements (maximum among the four chips):

| Snapshot | Allocated GiB/chip |
| --- | ---: |
| Model + configured KV pool loaded | 46.3684 |
| First 8K prefill completed, before independent oracle | 46.4751 |
| B4 completed | 46.7570 |
| Whole validation-run allocation peak | 51.2401 |

Runtime allocator limit is 95.7275 GiB/chip. End-of-B4 allocated total is about
187.026 GiB across four chips. Peak is for the entire **validation** process,
including prefix snapshot restoration, reconstructed oracle and profiling;
it is not a separately isolated serving-decode peak. Live allocated bytes and
allocator-reserved/free bytes are separately preserved in raw snapshots.
Original FP4/FP8/E8M0 weights stay packed; KV is BF16/QAT, not packed KV.

## B4 device profile and the next optimization target

All eight raw captures are retained. Detailed XProf exports in this phase cover
`B4` (8064/8065, two steps) and `B4_boundary8063` (one step). Both export all six
available selected tools without errors: overview, HLO stats, op profile,
framework-op stats, roofline model and memory profile. B1/B2 captures remain
raw evidence; no per-stage B1/B2 claim is inferred from the B4 table.

All eight TensorCore timelines show exactly one `jit_jitted_run_model` plus
one sampler dispatch **per step**, not 43 separately dispatched layer programs.
All 43-layer MoE sums are present: 688 collective occurrences in the two-step
interior trace, and 344 in the boundary trace. Additional output/sampling
collectives are counted separately. Eight TensorCore lanes are four physical
TPU v5p chips, not eight chips.

The interior trace has 213.787 ms host annotation time/step, an average 197.564
ms module-union time/TensorCore/step, and 16.223 ms outside modules (7.59%).
That outside interval includes submission, host bookkeeping, synchronization,
full-logit transfer and profiler effects; it is **not all inter-layer scheduling
idle time**. Do not subtract these short profiled captures directly from the
206.835 ms unprofiled p50.

Full-XPlane HLO self-times, averaged per TensorCore per interior decode step:

| Attributed stage | ms | Share of HLO self-time |
| --- | ---: | ---: |
| Routed FP4 expert path, including weight selection and online dequant GEMMs | 131.734 | 66.9% |
| CSA/HCA compressor plus paged state operations | 20.739 | 10.5% |
| Attention projections | 19.547 | 9.9% |
| MoE all-reduce, including wait | 8.479 | 4.3% |
| Normalization | 4.564 | 2.3% |
| mHC path | 2.101 | 1.1% |
| Remaining head/residual/router/index/top-k/attention/shared/unattributed | 9.659 | 4.9% |
| Total HLO self-time | 196.823 | 100% |

These disjoint self-times are not summed nested kernel durations or utilization
counters. In particular, the routed path's 131.734 ms is **not dequant-only
cost**. Its named FP4 Pallas GEMM self-time is 12.513 ms, including fused online
dequant; no separate dequant-only experiment was performed. Attention/shared
FP8 GEMMs account for 7.917 / 1.647 ms within their respective parent stages.

The main actionable finding is expert-weight slicing. In `hlo_stats.json`, the
129 loop-fusion rows whose source is `kernels/deepseek_v4/moe.py:72` (43 layers
times three weight projections) total **73.8927 ms/TensorCore/step**. Every one
has 1024 occurrences: 64 local experts × eight TensorCore lanes × two steps.
The B4 routing tensor is `[4, 6]`, so at most 24 expert choices can be active;
the weight slices nevertheless execute for all 64 local expert iterations.

The optimized HLO confirms the reason, beyond a source-level guess. For
example, `%constant_dynamic-slice_fusion.650` selects a packed
`u8[1,2048,2048]` weight from `u8[64,2048,2048]` in
`%wide.region_596.614_spmd.sunk`, **before** `%while.3248`, the inner loop over
active token tiles. The dynamic inner bound skips GEMM work, but the compiler
has moved the invariant per-expert slices outside that loop. Thus inactive
experts still incur slicing/data-movement work. Raw HLO, source fingerprint,
XPlane counts and HLO self-times are all saved in the performance receipt.

The next performance experiment should gate/perform weight selection only for
active experts, potentially loading packed expert tiles directly inside the
low-bit kernel. Recheck the compiled HLO and numerical gates before claiming
any benefit; 73.893 ms is measured attribution, not a guaranteed speedup.
No such optimization is implemented in this testing phase.

The separately captured 8063 boundary has 209.681 ms host time, 195.542 ms
module union and 20.804 ms compressor/state self-time. These few samples show
no large boundary spike, but are insufficient to estimate a statistically
reliable boundary-only overhead. The unprofiled 8191 sample remains in the
warm latency population. Physical VMEM/HBM traffic and MXU utilization counters
were not available through these selected exports; roofline bytes/FLOPs and
scoped-VMEM sizes are compiler estimates, not measured bus traffic.

## Handoff

Production sources and the accepted fingerprint remain unchanged. Only test
drivers, observer/report helpers, CPU tests and documentation changed in this
phase. The two failed first attempts remain failed and auditable; the bounded
rerun/subset scopes are explicit above. Results and raw traces are mirrored to
ignored local `GCP_login/results`, with source snapshots under `GCP_login/snapshots`.
No commit, GitHub push, new cloud resource or production serving deployment is
part of this task. Test-owned inference processes are shut down; the Spot VM
is retained. Official GPU/API benchmark accuracy, a single all-green full HTTP
rerun, cancellation-safe shared status RPCs and cold-compilation responsiveness
remain separate follow-up work.

Final backup check: all six primary report JSONs and all eight raw XPlane files
have matching SHA-256 hashes locally and remotely. Test PIDs are gone and port
30124 is no longer listening. The VM is reachable. Its boot filesystem has
6.7 GiB free after retaining the traces; future large profiles should go on the
already-mounted data disk rather than consuming that remaining boot space.
No unrelated files or old failure receipts were removed.
