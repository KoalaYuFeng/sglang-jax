# V4 CSA compressor: independent correctness and fusion gate

## Scope and baseline

Work starts from framework fingerprint
`a8b53fea756d23a8e714b4c5d79814c6fef1b5e499fdb9d8f117f58ba199cb4c`.
The pre-change compressor SHA256 is
`f8a8eff726f9739f4a62e91cbc1f3f6ca92cabbcefee8f02af49d06567d0b438`.
The isolated pre-change projection/emitter audit is
`GCP_login/results/v4-csa-compressor-audit-20260908-01.xml`: 16 passed.

This change targets deterministic channel-half selection before CSA emission,
not indexer top-k gathering, projection arithmetic, weight storage, or cache
ownership. Keep request-private FP32 state, page snapshots, official BF16
rounding boundaries, and the accepted V4 reduction order. The fixed reduction
tree is the TPU reference contract, not a claim about a unique official GPU
instruction order. Preserve the legacy CSA interface and numerical defaults.

## Ordered acceptance gates

1. **Independent oracle and fixtures.** Test raw `[groups,8,2D]` inputs, with
   `D=128/512`, against NumPy channel selection and FP64 pooling/RMS/RoPE with
   explicit BF16 boundaries. Freeze the existing position-7979 real traces,
   including both historical projection variants; never regenerate expected
   outputs with the new kernel. Check selected values/scores, masked groups,
   preceding-window absence, lane/batch permutations and chunk-built windows.
2. **Standalone fused emitter.** Add a Pallas entry accepting raw overlap
   windows. Select halves in VMEM and immediately reuse the existing V4
   pool/norm/RoPE routine. Require bitwise equality with the accepted emitter
   on identical inputs and unchanged oracle tolerance. No Engine or model
   allocation is needed for this gate.
3. **Standalone A/B benchmark.** Compare the actual V4 gather-plus-emitter
   boundary with the fused raw-window entry on the same arrays, masks, shapes,
   and hardware. Exclude compilation, report host-ready and device timings
   separately, and distinguish measured memory from logical traffic estimates.
   Include decode-sized and prefill-sized groups. Do not substitute the old
   packed-cache CSA benchmark or promise that all historical 6.68 ms disappears.
4. **Thin integration only after gates 1–3 pass.** Change only the V4 compressor
   call boundary, keeping reference behavior and trace semantics explicit.
   Run state/batch/chunk/page regressions before full 43-layer/8K acceptance
   and an apples-to-apples model profile. Do not modify scheduler/allocator.

All four gates are complete for the tested Flash/EP4 configuration on four
physical v5p chips: independent kernel correctness, standalone A/B, thin
integration, full 43-layer/8K numerical acceptance and same-workload profile.
Immutable receipts and diagnostic failures are recorded below.

## Baseline oracle adjudication

`v4-csa-overlap-baseline-20260908-01.xml` is preserved as failed (12 passed,
1 failed). The new random D128/G9 fixture has FP64-output NRMSE
`0.0005310518458438276`, above the unchanged `2e-4` gate. The existing kernel
was not changed. Isolated receipt `v4-csa-oracle-diagnostic-20260908-01`
establishes that Pallas, independent JAX FP32, and the official typed primitive
sequence run with PyTorch CPU FP32 have **identical final output** on this
input. FP32 pooling rounds element `[3,63]` to `-1.0078125`; FP64 rounds to
`-1.0`, and normalization propagates that boundary crossing to five output
elements. Input SHA256:
`a3fdc3711a2f56a4ecfd56127501ddffe6cc7247a74da7268ff92e56a5d0fd5f`.

This specific immutable fixture is an explicit **CPU FP32 bitwise** regression,
not a passing FP64-tolerance case. Its failed FP64 comparison continues to be
reported. Other FP64 cases keep `2e-4`; no automatic alternate oracle is
selected for arbitrary future failures. Production arithmetic, the existing
reference, and model tolerances remain unchanged.

## Independent baseline frozen

`v4-csa-overlap-baseline-20260908-03.xml`: 13 passed. The preceding `...02.xml`
is retained as failed: the new CPU oracle had not masked signed zero in inert
outputs; fixing its explicit invalid-row contract did not change the kernel.
`GCP_login/results/v4-csa-overlap-baseline-20260908-03` contains 20 accepted
cases (16 synthetic, four historical raw windows), their input digests, HLO,
compiler memory estimates, host-ready timings, and frozen BF16 output bits.
The expected archive SHA256, verified after local download, is
`32790bd87fdb606facce8a6ae53d0ac7951dffc34c2e655cc4b8ba47b70ec87e`.

The first overlap candidate hit a Mosaic i1 shape-cast compile error, before
execution (`v4-csa-overlap-kernel-20260908-01.xml`: 13 baseline passes, 21
candidate compile failures). Constructing the row predicate directly as a
two-dimensional integer iota removes that unsupported boolean reshape.
`v4-csa-overlap-selection-20260908-02.xml`: both D128/D512 selection probes
pass, including poisoned unused halves and bitwise final emitter output.

## Standalone fusion acceptance

`v4-csa-overlap-kernel-20260908-02.xml`: **36 passed**, including tile sizes
1/2/4/8, independently checked channel selection with poisoned unused halves,
the actual 7979 raw windows, order/partition invariance, and replicated
execution on all four physical v5p chips. Optimized standalone HLO has the
new Pallas call and no outer gather. The old selected emitter and shared V4
pool/norm/RoPE arithmetic are unchanged.

Both immutable A/B receipts are complete and match **all 20 frozen cases
bitwise**, with the same expected-output archive SHA256 as the baseline:

- `GCP_login/results/v4-csa-overlap-candidate-20260908-01/`: tile 4.
- `GCP_login/results/v4-csa-overlap-candidate-t1-20260908-01/`: tile 1.

These are **single-chip emission-boundary** measurements, not four-chip
model throughput. Thirty alternating dispatch-to-ready samples, four input
buffers; separate ten-call XPlane captures have full two-TensorCore coverage.
The table uses tile-1's same-process paired baseline. Device times are the
per-call active-TensorCore module envelope medians, not host latency.

| Shape | Gather + selected device ms | Raw fused device ms | Host-ready ms, old → new | Compiler temporary bytes, old → new |
| --- | ---: | ---: | ---: | ---: |
| Index D128, G5 | 0.105416 | 0.034078 | 0.227061 → 0.151110 | 356128 → 206208 |
| Main D512, G5 | 0.281225 | 0.037577 | 0.412459 → 0.161946 | 618272 → 203648 |
| Index D128, G1985 | 69.052293 | 3.214015 | 69.117279 → 3.358893 | 25867968 → 10284032 |
| Main D512, G1985 | 444.465476 | 4.771785 | 444.658229 → 4.929543 | 97894144 → 12381120 |

The oversized prefill fixture is a boundary stress test, not the scheduler's
current 128-token chunk. It exposes particularly expensive standalone gather
lowering; do not extrapolate those speedups to the model. G5 is the B4 decode
metadata capacity (including an inert group).

Tile 4 is faster for large G (main G1985: 2.739 ms), but padding the raw
FP32 windows increases its temporary estimate to 139573152 bytes, above the
baseline. **Tile 1 is the conservative production default**: no group padding,
lower temporary allocation across these shapes, and small decode latency.
Larger tiles remain explicitly opt-in. Temporary bytes are compiler memory
analysis, not physical HBM traffic; raw tile sizes are input footprints, not
measured peak VMEM. No claim of globally optimal scheduling is made.

## Thin integration acceptance

Only the V4 emitter adapter and its compressor call boundary change. The
Pallas path passes raw `[G,8,2D]` windows directly, while reference/HCA behavior,
projection arithmetic, FP32 state, snapshots and FP4/FP8 QAT stay unchanged.
There is no scheduler, allocator or weight-loader change.

Traces preserve actual `raw_group_values`, `raw_group_scores`, and
`csa_emitted`. Reconstructed selected views have explicitly different names,
`selected_view_values`/`selected_view_scores`; they are not represented as
observed Pallas intermediates. With tracing disabled, no selection gather is
created in the V4 adapter. Existing selected-window diagnostic callers retain
their API through `csa.emit(..., raw_overlap=False)`.

The first integration-test receipt `v4-csa-overlap-integration-20260908-01.xml`
is preserved (56 passed, one failed). A new test had compared a JIT-compiled
raw adapter with an **eager** selected adapter, so their RoPE table generation
used different compilation boundaries. The isolated log
`GCP_login/logs/v4-csa-adapter-phase-diagnostic-20260908-01.log` checks D128/D512:
JIT raw = JIT selected bitwise; raw with the eager table = eager selected
bitwise; raw with the JIT table = JIT selected bitwise. The 12/13 differing
output elements are exclusively in the RoPE channels (maximum table
difference 0.01327088475227356). This is not a same-table emitter discrepancy.
The new test now uses the existing production JIT boundary on **both** sides;
no production RoPE arithmetic or tolerance was changed.

`v4-csa-overlap-integration-20260908-02.xml`: **92 passed, no skips/failures**.
This includes the 36 standalone cases (now default tile 1), 44 V4 integration
cases, two original CSA end-to-end/4-chip NumPy regressions and ten diagnostic
contract tests. It verifies that trace-enabled/disabled calls give identical
state and that the untraced production compressor no longer contains the
channel `take_along_axis` operations.

`GCP_login/results/v4-csa-overlap-real-20260908-01/report.json`: **118 checks,
all bitwise**, using the historical 8023 B1/B2 inputs on layers 2/22/42 and
the real 7979 projection fixture. This includes fresh FP32 projections, private
state, compressed KV, ranking/top-k, attention and both same-input emitters.

Production/profile source fingerprint:
`9b9ddeca670489a95cbd127f1aa9edead3aee3609cc580057b78a66a92853bd7`.
Comparison with the frozen source snapshot confirms exactly three changed
production files: `kernels/csa/compressor.py`,
`kernels/deepseek_v4/compressor.py`, and `kernels/deepseek_v4/csa.py`. Every
pre-existing CSA compressor function has an unchanged AST; the shared kernel
file only adds three functions. The scheduler/allocator/model/loader sources
are unchanged.
## Full-model numerical gate

`GCP_login/results/v4-csa-overlap-native-cold-20260908-01/report.json` is
complete: **2424 checks pass**, no numerical failures, maximum NRMSE
`1.618478790987865e-7` (unchanged full-logit limit `0.005`). Its 516 B1 checks
are bitwise; the other 1908 cold concurrent prefill/decode rows are not claimed
bitwise. B1/B2/B4 all start from empty caches, use 128 packed prefill tokens
per call (128/64/32 per request), alternate request order, and decode through
position 8191. Full checkpoint revision and the opt-in `gmm` MoE backend
match the preceding measured baseline.

The actual executed B4 HLO has **21 main and 21 index raw overlap emitters**,
all `-v4-overlap-t1`, with unique SSA ownership covering layers 2,4,...,42.
HLO SHA256:
`e56f60d3c007f879a1a9a9b1f237031899ff9939ad58898e52cd834eed0b8553`.
The raw receipt, HLO, full-vocabulary goldens and prefill checkpoints have
been copied locally. This run did not enable profiling.

`GCP_login/results/v4-csa-overlap-oracle-20260908-01/report.json` is complete:
**514 full-vocabulary rows pass**, all finite and top-1 equal; 435 are bitwise.
Maximum NRMSE is `0.00010527185077080503` (0.01053%), below the unchanged
0.5% gate. The reference recomputes both full prompts and every teacher-forced
decode step from empty cache, sharing no native hidden states, prefix KV or
compressor scratch. Its source fingerprint remains unchanged:
`9ef78243dcc50567444feb4bc4dae313000b15651401b46a7ebe6e89d96727d5`.
This is a numerical regression fixture, not an official GPU/API accuracy
benchmark. The reference full logits and receipt have been copied locally.

Immutable numerical receipt SHA256s:

- Cold native: `5e54b78b93a00b8713fdf47310a4da1820132646aaa870e21fc7aa9a35fe70a7`.
- Independent: `b687ac29bfdd9c42b39d054d617a6380aa4247debc3ff2d241863f228bfebf5b`.

## Same-workload model performance

`GCP_login/results/v4-csa-overlap-native-profile-20260908-01/` is complete.
It ran **after** both numerical gates passed, reusing the new same-source B1
goldens and matching `v4-moe-breakdown-native-20260908-01`'s GMM backend,
checkpoint, prompts, 7936-token prefixes and B2/B4 prefix-restoration workload.
Its additional **1538 numerical checks** pass (maximum NRMSE
`1.6165743943474808e-7`). Its executed HLO SHA256 is identical to the accepted
cold run. Receipt SHA256:
`93f7adcd5a31a72c6d003f2558626063a26889cb53b8a37c361943959d848864`.

Each batch has 252 unprofiled warm calls, excluding first compilation and
three profiled calls. Throughput is aggregate tokens divided by elapsed time,
not per-request generation rate. This is a native ModelRunner benchmark,
**not Engine/HTTP throughput**. The old and new full-model runs are historical
same-workload measurements, not interleaved A/B samples; the standalone
boundary experiment above is interleaved.

| Metric | B2 old → fused | B4 old → fused |
| --- | ---: | ---: |
| Warm decode p50, ms | 82.189 → 77.662 | 92.544 → 85.475 |
| Warm decode p95, ms | 83.925 → 79.479 | 94.363 → 87.196 |
| Aggregate throughput, tokens/s | 24.377 → 25.764 | 43.251 → 46.830 |
| p50 latency reduction | 5.51% | 7.64% |
| Throughput increase | 5.69% | 8.27% |

The following device numbers are HLO self-time in ms/call **averaged over
eight TensorCores**, using the two interior steps 8064/8065. They are not host
latency and are not summed across chips. The 8063 boundary capture remains
separate.

| Device work | B2 old → fused | B4 old → fused |
| --- | ---: | ---: |
| External channel-half gather | 4.185216 → 0 | 6.683690 → 0 |
| Selected emitter → raw fused emitter | 0.204401 → 0.273256 | 0.369602 → 0.448310 |
| Gather + emitter boundary | 4.389616 → 0.273256 | 7.053292 → 0.448310 |
| Total device HLO self-time | 72.354458 → 68.108934 | 82.715276 → 75.651082 |
| Mean module-union time | 73.235577 → 68.916528 | 83.532846 → 76.450887 |

In B4 the boundary saves **6.604983 ms**, not the entire old 6.683690 ms:
selection/masking now adds a small amount of work inside the emitter. The old
84 external gather instructions (1344 occurrences) are absent. All **42 raw
emitters have 672 occurrences** (`21 layers × main/index × 8 lanes × 2 calls`).
Do not attribute every other change in model latency solely to these kernel
instructions; scheduling/layout effects and run-to-run variation remain.

Loaded HBM is unchanged at **46.368 GiB/chip**. The profiled allocator peak is
51.176 GiB/chip (old profile: 51.196); this is not a claim of substantial
model-memory savings, and profiler peaks are not unprofiled inference peaks.
Raw FP4/FP8/E8M0 checkpoint storage and online weight dequantization are
unchanged.

### Exporter and raw-timeline cross-check

XProf emits a nonfatal `async-update` operand-arity compatibility warning
while reading HLO protos. The completed exports retain full model dispatch
coverage, all 2064 GMM occurrences and 688 MoE collectives for each interior
capture (half those counts at the boundary), excluding empty SparseCore
planes. No FLOP/byte estimate is treated as a hardware counter.

`GCP_login/logs/v4-csa-overlap-raw-timeline-20260908-03.log` independently reads
the original XPlane with JAX `ProfileData`, without the XProf HLO converter.
Each of eight lanes has exactly two model calls, two **separate sampler**
calls, and 84 emission events with 42 distinct raw-overlap names. Summing raw
`device_duration_ps` (including its time-scale multiplier) exactly reproduces
the exported 0.273256171875 ms B2 and 0.448309921875 ms B4 emission totals.

The earlier raw-reader logs `...01`/`...02` are retained as failed diagnostic
probes: raw operation names are full HLO instructions rather than bare names,
samplers must not be counted as model calls, and `duration_ns` truncates
sub-nanosecond timing (a 19 ns aggregate discrepancy in B2). The successful
cross-check uses the instruction LHS and original picosecond counters; no
captured data or production code was changed to resolve those reader issues.

All numerical/performance receipts, optimized HLO, full logits and traces are
stored locally under Git-ignored `GCP_login/`; the final profile copy has been
verified with checksum-based synchronization. No Git commit or GitHub push
was made. Engine/HTTP pressure was not rerun in this kernel-focused change,
and tile 1 is a measured conservative default, not a global optimality claim.
