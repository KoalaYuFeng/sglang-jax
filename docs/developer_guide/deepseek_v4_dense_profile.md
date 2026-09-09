# Opt-in V4 dense kernels: whole-model A/B profile

## Experiment contract

The control and candidate run sequentially on the same four physical v5p chips,
in separate ModelWorker processes. This is the real 43-layer SGLang-JAX model
forward and sampler, not a layer loop or isolated kernel benchmark. The core
scheduler and serving defaults are unchanged; this does not measure HTTP or
Engine end-to-end scheduling latency.

Both configurations retain raw checkpoint FP4/FP8/E8M0 storage, online dequant,
FP4 GMM expert parallelism across four chips, CSA/HCA head-TP4 and batched CSA
decode. Only these four candidate options change:

```text
--fp8-backend gmm --fused-norm --merged-projections --fused-wo-a
```

The actual loaded model options and the final executed B4 decode HLO must
confirm the selection. No silent fallback or default promotion is permitted.

## Workload and correctness

- Context capacity 8192; the original two 7936-token fixture prompts followed
  by 256 teacher-forced decode calls, through position 8191.
- Two rounds per configuration. Serial B1 prefill uses 128-token chunks;
  B1/B2/B4 decode restores the real computed prefix pages outside timing.
  Multi-request order alternates on each decode step.
- Every 128-token prefill boundary and every decode row is compared against
  the immutable earlier native full-vocabulary logits. Those fixtures have
  a complete independent 514-row oracle check and the unchanged reference
  fingerprint. Historical production fingerprints are retained explicitly:
  these are compatibility fixtures, not fabricated same-source receipts.
- The original finite/top-1/NRMSE <= 0.005 gate is unchanged. Actual output
  bytes are hashed for exact same-workload control/candidate comparisons.
  Any numerical failure aborts the process and preserves a failure artifact.
- The timing process does not instantiate the diagnostic reference model.
  This test does not replace cold concurrent prefill, full internal-state
  regression, natural memory pressure, Engine or HTTP acceptance.

## Timing and attribution

The timed interval includes worker generation, device completion, full-logit
transfer and the existing finite/argmax checks. Page/metadata preparation,
compatibility comparisons, prefix copy/restore and trace start/stop are outside
that interval. Compilation/cache-miss calls, the first three warmup calls of
each round and all profiled calls are excluded from warm statistics. All
remaining outliers are retained. Aggregate throughput uses total tokens divided
by total included time, not batch size divided by median latency.

The second round captures two late-prefill chunks beginning at 7168/7296,
decode positions 8064/8065 for each batch, and a separate B4 boundary at 8063.
Exclusive HLO self-time is averaged across active TensorCore timelines and
captured calls. Collective durations include waiting. Online dequantization
inside a Pallas GEMM is not separately measurable from these traces.

The attribution helper separates FP8 GMM calls and their metadata/layout
operations from routed FP4 MoE, including SSA-consumer-based input-copy
ownership. Shared constants are not charged entirely to MoE. HBM snapshots
are allocator measurements; process high-water includes prefix restoration
and compiled shapes, not only decode intermediates. Hardware HBM/VMEM traffic
and MXU utilization must not be inferred from these data.

## Reproduction

Use `scripts/profile_deepseek_v4_dense.py` with the same official checkpoint,
`--native-report` and matching `--oracle-report`, two rounds and:

```text
--moe-backend gmm --attention-tp --csa-decode-batch --rounds 2
```

First run the control into a new output directory. Then pass its completed
receipt as `--control-report` and enable the four candidate flags above. Each
process writes its own complete/incomplete receipt; a failed candidate never
changes the control's result. Analyze only after the timing processes exit,
using the separate CPU XProf environment and
`scripts/analyze_deepseek_v4_fp4_moe.py --profile <directory> --export`.

## Results

The 2026-09-09 experiment completed on `sglang-jax-v5p8-spot`, node ID
`3079367133054764732`, without replacement between the runs. Both receipts
pass all 3832 full-vocabulary checks. Every candidate output row is bitwise
identical to its same-workload control row, including all prefill boundaries,
multi-request order reversals and decode through position 8191. Maximum NRMSE
against the historical fixture is 1.6165744e-7 for both configurations; not
every control output is bitwise identical to that earlier fixture.

Warm worker timings, with the exclusions described above:

| Workload | Control median ms | Enabled median ms | Control tokens/s | Enabled tokens/s | Throughput gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 prefill, fixture 0, 128-token chunk | 487.62 | 397.49 | 262.46 | 322.07 | 22.7% |
| B1 prefill, fixture 1, 128-token chunk | 482.47 | 392.49 | 265.28 | 326.20 | 23.0% |
| B1 decode | 54.77 | 44.17 | 18.25 | 22.62 | 24.0% |
| B2 decode | 68.68 | 58.63 | 29.14 | 34.13 | 17.1% |
| B4 decode | 75.15 | 64.39 | 53.22 | 62.12 | 16.7% |

Decode throughput is aggregate across the batch, not per request. The two
prefill cases contribute 116/118 unprofiled warm calls; B1/B2/B4 contribute
504/504/503 calls each. Enabled decode p95 is 45.05/60.53/66.19 ms, compared
with 55.74/70.51/76.97 ms in the control. These are sequential same-machine
comparisons, not an interleaved A/B or an HTTP throughput measurement.

### Where the improvement comes from

The following are exclusive device HLO self-times, averaged over eight active
TensorCores and two calls. The prefill capture starts at positions 7168/7296;
the B4 decode capture uses positions 8064/8065.

| Device-time group | Prefill control → enabled, ms | B4 decode control → enabled, ms |
| --- | ---: | ---: |
| FP8 projection kernels, including grouped wo_a | 72.180 → 34.149 | 5.320 → 5.396 |
| Normalization/RoPE and projection/index surrounding operations | 65.760 → 14.553 | 13.856 → 3.283 |
| Routed FP4 expert GMM | 205.079 → 205.074 | 13.065 → 13.054 |
| MoE all-reduce, including wait | 51.643 → 51.616 | 8.094 → 8.145 |
| Compressor and paged state | 23.286 → 23.444 | 11.538 → 11.488 |
| All exclusive HLO self-time | 474.155 → 384.754 | 65.104 → 54.610 |

The surrounding-operations group combines the legacy `normalization` and
`attention_projection_and_index` buckets with the candidate's fused norm,
normalization helpers, FP8 metadata, padding, scale-layout copies and projection
glue. It is not a pure RMSNorm timer: fusion moves these arithmetic boundaries.
The FP8 kernel group includes `inverse_rope_fp8_wo_a` on the candidate and the
legacy FP8 wo_a calls on the control. The table does not sum isolated speedups.

**Decode's gain comes primarily from fusion and removal of surrounding
programs, not a faster FP8 GEMM alone.** FP8 kernel time including wo_a is
essentially unchanged at this tiny batch size. Prefill benefits from both the
FP8 projection implementation/tile choice and fusion; this four-switch
experiment does not isolate each individual switch's causal contribution.

Mean device module-union time falls from 65.889 to 55.307 ms for B4 and from
475.025 to 385.485 ms for prefill. Profiled B4 host calls average 80.147/72.868
ms; do not subtract device time from an unprofiled median and call the result
"scheduler idle." Profiling overhead, host output transfer/checks and the
timing scopes differ.

### Remaining bottlenecks

For enabled B4 decode, relative to 54.610 ms exclusive HLO self-time:

| Stage | ms | Share |
| --- | ---: | ---: |
| FP4 expert gate/up/down GMM | 13.054 | 23.9% |
| Compressor and paged-state operations | 11.488 | 21.0% |
| MoE all-reduce, including wait | 8.145 | 14.9% |

All routed-MoE-attributed work plus router totals 28.545 ms (52.3%), including
GMM metadata, packing/combine and scale-layout copies; this subtotal excludes
the FP8 shared-expert projection kernels. Sparse attention itself is only
0.729 ms (1.3%), while attention TP all-gather is 0.250 ms (0.46%). Compressor
and paged state is a broad combined bucket, not an isolated CSA core-kernel
measurement. MoE all-reduce includes imbalance/synchronization waiting, so it
does not establish an ICI bandwidth bottleneck.

Enabled prefill is more strongly MoE-bound: FP4 GMM takes 205.074 ms (53.3%)
and MoE all-reduce including wait 51.616 ms (13.4%). The next measured targets
are FP4 MoE for prefill, and routed MoE plus compressor/state for decode.
Mature kernel reuse alone does not remove these measured bottlenecks. No
scheduler algorithm or production kernel was changed during this experiment.

### HBM and dispatch evidence

Maximum allocator bytes in use across the four physical chips:

| Snapshot | Control GiB/chip | Enabled GiB/chip |
| --- | ---: | ---: |
| Loaded weights + 8K pool | 44.446 | 44.446 |
| After B4 completion | 44.864 | 44.775 |
| Process high-water, including prefix restoration | 49.430 | 49.342 |

This is approximately unchanged HBM usage, not a substantial weight-storage
saving. Raw checkpoint bytes and online dequant remain selected. Actual model
options confirm all four new switches. The executed enabled B4 HLO contains
236 FP8 GMM, 173 exact norm, 43 QNorm/RoPE and 43 inverse-RoPE/wo_a calls; these
new calls are absent in the control. Both retain the 43-layer original mHC,
21 CSA, 20 HCA and 129 routed FP4 GMM call structure with verified head/expert
ownership. Static instruction counts are not host dispatch counts.

All ten captures verify eight active TensorCore timelines, one model and one
sampler dispatch per call, and complete dynamic coverage of 129 FP4 GMM calls
and 43 MoE sums per call. Empty SparseCore planes are excluded. XProf still
logs its async-update operand-arity parsing warning; raw XPlanes, HLO tables,
module coverage and grouping witnesses are retained. No hardware performance
counter export is available, so there is no measured VMEM/HBM bandwidth or
MXU utilization claim.

### Artifacts and boundaries

The complete receipts are mirrored locally under `GCP_login/results/` and on
the persistent model disk under `profiles/`:

- `v4-dense-control-profile-20260909-01`
- `v4-dense-enabled-profile-20260909-01`

Both include optimized HLO, five raw trace captures, exported XProf tables,
per-core module timelines and source-attribution witnesses. Analysis results
bind the analyzer files and capture receipt by SHA256. A separate manifest
checks all artifacts and logs between local and remote copies.

Production fingerprint:
`ee32d17da94f49d7ae035d8df662b6588b32e006d109e63775c1d374b4fcb4fc`.
Unchanged independent reference:
`9ef78243dcc50567444feb4bc4dae313000b15651401b46a7ebe6e89d96727d5`.
Control HLO SHA256:
`18cdd92095ec6226787ca58ff9fd98d49c2d3781bd4ab00bbd60deb8183cc2dc`.
Enabled HLO SHA256:
`9e1589a8e20d62cedf42df90206b625d7af74bc24e3ad651937fbea9b35f52d9`.

The profiling/receipt contracts pass 39 CPU tests before execution; the final
attribution tests pass 13 cases, plus 24 existing profile-analysis cases.
All inference and analysis processes exit. This experiment explicitly enables
the new path, but does not change serving defaults or leave an HTTP server up.
Fresh cold concurrent-prefill/state, Engine and HTTP acceptance for these four
switches remain separate gates before default promotion.
