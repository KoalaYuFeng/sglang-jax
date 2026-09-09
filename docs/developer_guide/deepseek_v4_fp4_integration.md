# V4 tuned FP4 adapter: explicit integration and recovery retest

Current status (2026-09-09 UTC / 2026-09-10 Singapore): replacement-node
ModelWorker A/B is COMPLETE. Both fresh `-02` runs pass 3832 checks, and every
candidate output hash equals its same-node control bitwise. All ten profile
captures pass coverage and independent raw timing audits. Prefill improves
39.7–40.6%; B1/B2/B4 aggregate decode improves 10.9%/11.1%/9.6%.
The `gmm_tuned` option remains opt-in; tuned Engine/HTTP acceptance is still
pending. The previous interrupted `-01` candidate is preserved, not relabeled.

The replacement is one same-configuration `v5p-8` Spot (four physical chips),
node `8772628354803788712`, with the original independent disk and exact pinned
environments. It is READY, queue ACTIVE, through the after-candidate receipt.
All 94 TPU checks pass on it. The earlier deleted node's cause remains
unresolved; it is not automatically attributed to Spot preemption.

## Scope

This follows the isolated gate in `deepseek_v4_fp4_tuning.md`. The new V4-only
`gmm_tuned` option connects that adapter to the existing model and loader. It
does not modify core scheduling, routing, expert ownership, collectives, or
the `legacy`/`gmm` defaults. It is opt-in, not a deployment-default promotion.

Packed FP4 weights remain uint8 `[E,N,K/2]`. At loading, compact E8M0 scales
are permuted once from `[E,N,K/32]` to `[E,K/32,N]`, before device placement.
The loader does not retain a second on-device scale collection. Tile-local
dequantization preserves FP32 scaling and the original BF16 rounding boundary.
The diagnostic `original_fp4_scale_view` reverses only that byte permutation;
it is outside inference and never constructs BF16 expert collections.

The shape policy uses M8 below 128 input-token capacity, M32 at/above 128,
full K, and N256 (N128 for the smaller unit-test output width). Extra routed
rows belong to the existing inactive sentinel and are discarded on unpermutation.
This is a conservative V4 policy, not a general optimum for arbitrary models.

## Recovered environment

The isolated and cold numerical gates used a `v5p-8` Spot in `us-east5-a`, node
`6548387651919761446`, four physical v5p chips. The original 500 GB independent
disk is mounted without formatting. Python 3.12.14, JAX/jaxlib 0.11.1 and
libtpu 0.0.46.1 are restored from a byte-identical package freeze. Four-chip
BF16 smoke passes. Original checkpoint revision:
`60d8d70770c6776ff598c94bb586a859a38244f1`.

Fresh complete `-02` performance measurements below instead use replacement
node `8772628354803788712`, created 16:50:32 UTC, same zone/configuration/disk.
The first direct replacement create failed before READY due to no capacity;
one bounded same-zone Spot queue then allocated successfully. No cross-zone
copy, extra disk, second active node, on-demand switch or disk format occurred.
Both restored package freezes are byte-identical. Checkpoint recovery reused
the 73 prior verified SHA256s only after unchanged size/mtime/ctime checks;
this is explicitly not a fresh full checkpoint rehash.

## Completed isolated/interface gates

- 94 TPU tests pass, no skips: packed FP4/scales, NumPy matmul comparison,
  previous GMM behavior, new EP1/EP4 entry, M1 through M129, duplicate/cancelled
  routes, inactive experts, single-shard routing, loader ownership and padding.
- 70 CPU execution/evidence/profile contracts pass. An additional 96 framework,
  paging and reporting cases pass, with 17 explicitly skipped hardware cases.
- Real layer-3 weights: 127 bitwise comparisons pass, including independent
  NumPy M1, M1/2/4/8/16/32/64/127/128, route adversaries and post-timing outputs.
  Every tuned-entry HLO has zero full scale layout copies; each control has one.
- The real-weight A/B uses 60 samples/variant in rotated order, five untimed
  warmups, original captured activations/routes and all 256 experts over EP4.

| Routed tokens | Existing GMM median | Tuned entry median |
| --- | ---: | ---: |
| M1 | 0.6054 ms | 0.5256 ms |
| M2 | 0.8666 ms | 0.7293 ms |
| M4 | 1.3106 ms | 1.0876 ms |
| M8 | 1.8602 ms | 1.5213 ms |
| M16 | 2.4402 ms | 1.9874 ms |
| M32 | 3.2884 ms | 2.6816 ms |
| M64 | 4.4025 ms | 3.5711 ms |
| M127 | 6.1416 ms | 4.9208 ms |
| M128 | 5.8047 ms | 3.4852 ms |

These are single-layer routed-MoE latencies, NOT end-to-end model throughput.
Private local evidence is in `GCP_login/results/v4-fp4-tuned-entry-20260909-01/`
and identically named folders on the independent data disk.

## Whole-model gate and measurement contract

The short 43-layer ModelWorker gate passes all 14 B1/B2/B4 comparisons,
including unequal chunks, request reorder and slot reuse. Maximum independent
reference NRMSE is 4.8805901e-5, all outputs finite/top-1 equal; the numerical
metrics match the earlier short gate. This is not an assertion that all outputs
are bitwise equal to the independent reference. Actual B4 HLO verifies all
129 tuned FP4 GMM calls and zero full scale layout copies, as well as the
selected mHC/HCA/CSA/FP8 kernels and TP ownership.

Short receipt: `GCP_login/results/v4-fp4-tuned-paged-20260909-01/report.json`,
SHA256 `618a5638871b22bc6dfd49932b2d4400ba369c0b5c2b4efef479f71f446e688c`.
Executed HLO SHA256:
`0704bcb4d9b4a694e23ad3a0daa7853178f6d0369bf47348835d64ebb3bddbaa`.

Cold 43-layer 8K B1/B2/B4 now passes all 2424 checks through position 8191.
Both B1 fixtures have 516 bitwise baseline/late-oracle checks; maximum NRMSE
over all concurrent checks is 1.6184788e-7, all finite/top-1 equal. B2/B4 start
from empty cache and check all 128-token endpoints, then all 256 decode steps
with request order changes. Actual 8K B4 HLO again verifies all 129 tuned GMMs
and zero full scale layout copies.

Native receipt: `GCP_login/results/v4-fp4-tuned-native-20260909-01/report.json`,
SHA256 `1941c84bb87feb68b5e4bb2b6cd21cc1b9f715f4b0c351242908ed8c20491a2f`.
8K HLO SHA256:
`8f156ce86c01f5dce6c3b92f8eb3b84264182a13d15aac6f5867000ee1230813`.

The independent from-empty-cache oracle also passes all 514 full-vocabulary
rows (435 bitwise, maximum NRMSE 1.0527185e-4, all finite/top-1 equal). Both
fixtures at position 8023 are bitwise equal. It recomputes complete prompts
and decode without sharing native hidden states, KV or compressor scratch.
Its numerical metrics are unchanged from the earlier accepted oracle.

Oracle receipt: `GCP_login/results/v4-fp4-tuned-oracle-20260909-01/report.json`,
SHA256 `f784ea05fe17081715276c67c136bc98b1c2acd3bcfbe6603bb9c996e178db19`.
Full profile/Engine/HTTP acceptance is not implied by these numerical receipts.

Both sides of the completed A/B retain attention TP4, MoE EP4, batched CSA decode
and the four previously tested dense-kernel options:

```text
--attention-tp --csa-decode-batch --fp8-backend gmm
--fused-norm --merged-projections --fused-wo-a
```

Only `--moe-backend gmm` versus `--moe-backend gmm_tuned` changes. Run the short
43-layer paged correctness test first, then fresh cold native B1/B2/B4 at 8K
and an independent from-empty-cache reference. The broad reference fingerprint
includes the new adapter file; historical oracle receipts are not relabeled.

`profile_deepseek_v4_fp4_model.py` reuses the existing ModelWorker profile
harness. Its numerical fixture is the new tuned-backend cold native run plus
its fresh independent oracle; this is explicitly recorded separately from
the A/B control configuration. It requires a fresh same-source GMM control
for the profile candidate and exact control-output hashes in addition to the
unchanged finite/top-1/NRMSE gate.
It checks every 128-token prefill endpoint and all 256 teacher-forced decode
positions through 8191, with B1/B2/B4 and alternating request order. Profiles
capture late prefill, interior decode and the 8063 compressor boundary.

Loaded backend receipts and actual HLO must show 129 tuned FP4 GMM calls
(W1/W3/W2 for all 43 layers), with no full expert weight/scale slices or full
scale layout copies. Timings exclude compilation/cache misses, loading,
prefix snapshot/restore, first-three-call warmups and profiler calls. They
include the worker/sampler, completion wait and full-logit host transfer.

Allocator HBM snapshots and high-water are reported separately from compiler
temporary estimates. XProf decomposition is checked against raw TensorCore
events using `audit_deepseek_v4_fp4_profiles.py --layers 43`; collectives include
waiting. No physical HBM/VMEM traffic or utilization counter is inferred.

These ModelWorker runs do not substitute for Engine/HTTP stress acceptance.

## Completed replacement-node A/B (`-02`)

The same-node control and candidate each pass all 3832 full-vocabulary output
checks, all finite/top-1 equal, maximum fixture NRMSE 1.6165744e-7. Their 3832
paired output hashes are all BITWISE identical. This is numerical regression
against the accepted native/oracle fixtures, not an official GPU accuracy
benchmark. The existing from-empty 8K native/oracle receipts remain explicitly
`-01` fixtures; they are not falsely dated as replacement-node runs.

| Workload | Control median / p95 | Tuned median / p95 | Control tokens/s | Tuned tokens/s | Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill fixture 0, chunks of 128 | 396.43 / 400.98 ms | 283.99 / 287.79 ms | 322.90 | 450.98 | 39.66% |
| Prefill fixture 1, chunks of 128 | 391.81 / 395.21 ms | 278.60 / 282.36 ms | 326.81 | 459.42 | 40.58% |
| Decode B1 | 43.84 / 44.75 ms | 39.51 / 40.23 ms | 22.79 | 25.28 | 10.91% |
| Decode B2 | 58.37 / 60.29 ms | 52.57 / 54.09 ms | 34.28 | 38.08 | 11.07% |
| Decode B4 | 63.93 / 65.72 ms | 58.34 / 59.73 ms | 62.58 | 68.57 | 9.58% |

Decode tokens/s is aggregate across the batch, not per-request speed. These
are full 43-layer, 8K-tail ModelWorker measurements, NOT Engine/HTTP throughput.
Each side retains 116/118 prefill and 504/504/503 decode warm calls over two
rounds, excluding first-three-call warmups, compilation/cache misses and
profiler calls; no timing outliers are discarded. Control and candidate run
sequentially on the same node; CPU exports and large transfers occur only
after their respective warm measurements have ended.

All five captures per side pass exact 43-layer GMM/collective coverage. Raw
TensorCore GMM and source-attributed MoE all-reduce durations match XProf
within 1 ns. The same XProf HLO-reconstruction warnings remain in the logs;
they are not suppressed or treated as independent timing evidence. Raw-event
audits are required before accepting the exported timings.

| Device stage (mean ms per TensorCore per captured call) | Prefill control → tuned | B1 control → tuned | B4 control → tuned |
| --- | ---: | ---: | ---: |
| FP4 GMM, including conversion | 205.08 → 119.26 | 7.37 → 5.87 | 13.06 → 10.38 |
| MoE all-reduce, including waiting | 51.39 → 24.42 | 6.02 → 4.79 | 8.11 → 6.51 |
| Full FP4 scale layout copies | 1.35 → 0 | 1.32 → 0 | 1.32 → 0 |
| GMM metadata/zeroing | 4.99 → 7.00 | 1.65 → 1.67 | 3.14 → 3.16 |
| Compressor and paged state | 23.44 → 23.36 | 4.91 → 4.87 | 11.49 → 11.49 |
| FP8 projections, including online dequant | 29.68 → 29.59 | 4.74 → 4.78 | 4.60 → 4.88 |

The actual tuned B4 executable has all 129 tuned GMMs and zero full scale
layout copies; control has 129 existing GMMs and 43 full scale copies. Neither
has full expert weight/scale slices. Lower collective elapsed time includes
less waiting; with unchanged collectives this is consistent with faster,
better-balanced expert work, NOT evidence of higher interconnect bandwidth.
The smaller prefill M tile increases metadata overhead, but its net benefit
is measured above. No standalone dequant latency, HBM/VMEM traffic or hardware
utilization is inferred from these grouped timings.

Tuned B4's largest groups now include compressor/paged state (11.49 ms), FP4
GMM (10.38 ms), MoE collective/wait (6.51 ms) and FP8 projections (4.88 ms).
These are profiled exclusive HLO groups, not additional wall-time components
to add to the warm latency. Core scheduler algorithms were not changed.

Allocator memory per chip (maximum over four chips): loaded use is identical
at 44.44646 GiB. B4 completion is 44.77522 GiB control, 44.77875 GiB tuned;
high-water is 49.34152 / 49.34506 GiB. The high-water includes prefix restore
copies and accumulated compiled shapes in this harness, not serving steady
state. No full BF16 expert collection or extra on-device scale collection is
introduced.

Evidence folders are private/local under `GCP_login/results/` and on the
original disk under `profiles/`, with identical names:

- `v4-fp4-model-control-20260909-02`, report SHA256
  `41f860dd8d77cb820cb29ff233b2dc19a6996828c48e0a26aeb889299e33091a`.
- `v4-fp4-model-tuned-20260909-02`, report SHA256
  `081cf75500c5f4fc3c97b5f6dc89e422774989a6f2222390307589210c2fac8c`.
- Control executed HLO SHA256
  `02ddb24d3ed83fb8872248623adc08a75e3fd103b53ae30d89818c6cf2c1b72b`;
  tuned executed HLO SHA256
  `8f156ce86c01f5dce6c3b92f8eb3b84264182a13d15aac6f5867000ee1230813`.

Each folder retains report, executed HLO, five raw XPlanes, Chrome traces,
full CPU exports/grouping witnesses and `raw_timing_audit.json`. Streaming
test/profile/analysis logs are local too. Production fingerprint remains
`82f6e542105c208a410d4561f1eef6ba360f3911c3643f46570b7eb13cabd363`.
Local/remote manifests match all 110 A/B artifacts (4,359,116,886 bytes),
including ten raw captures, detailed exports and four complete logs. Manifest
validation also recomputes both source fingerprints, HLO/raw-trace digests and
every control/candidate output hash pairing.
No production source changed during this replacement retest. Seven protected
scheduler/runner/shared-GMM/attention/compressor files match the pre-integration
snapshot. No GitHub commit/push was made.

Next acceptance boundary is tuned Engine/HTTP concurrency/stress with the same
explicit options. Do not promote this backend to default or reuse old HTTP
receipts as evidence for it.

## Historical `-01` control completed; candidate interrupted

The fresh GMM control completed all 3832 numerical checks and five captures.
Its complete report, executed B4 HLO, five raw XPlanes and five Chrome traces
are local in `GCP_login/results/v4-fp4-model-control-20260909-01/`.
Report SHA256:
`2677c23745ed5a9b3d672bbafee1f4cfed1e3b7bd2b4c3849836579cebc62b65`.

| Workload | GMM control warm median | GMM control throughput | Tuned candidate throughput |
| --- | ---: | ---: | ---: |
| Prefill fixture 0, 128-token chunks | 396.96 ms | 322.35 tokens/s | 450.89 tokens/s, preliminary |
| Prefill fixture 1, 128-token chunks | 392.35 ms | 326.40 tokens/s | Incomplete |
| Decode B1 | 44.22 ms | 22.61 tokens/s | Not reached |
| Decode B2 | 58.86 ms | 33.97 tokens/s | Not reached |
| Decode B4 | 64.37 ms | 62.21 tokens/s | Not reached |

Decode throughput is aggregate across the batch, not per request. Each decode
control has 503 or 504 unprofiled warm calls; prefill has 116 and 118. The first
candidate prefill fixture completed 116 warm calls, median 284.10 ms and p95
287.93 ms: a preliminary 39.88% throughput increase on the same node. This is
one completed fixture inside an INTERRUPTED run, not a complete A/B acceptance.
Its streamed local log is
`GCP_login/logs/v4-fp4-model-tuned-20260909-01.log`. The second fixture was still
running when SSH closed. Do not infer any tuned decode speed from native
correctness timings or isolated-MoE speedups.

Both backends loaded at 47,724,017,664 allocator bytes per chip (44.446 GiB).
Control B4 completion used at most 48,077,023,744 bytes/chip; the observed
allocator high-water was 52,980,024,832 bytes/chip. High-water includes the
profile harness's prefix copies/restore and accumulated compiled shapes, not
only steady-state serving. Candidate B4 memory has not been measured.

All five control CPU exports completed on the independent disk; their printed
summaries are backed up in the local analysis log. The first raw timing audit
then correctly failed its coverage check because its newly added whole-model
mode counted sampler all-reduces as MoE sums (92 instead of 86 for two B1
calls per TensorCore). This was a reporting bug, not a numerical/kernel error.
The corrected auditor matches exact typed instructions to MoE source or sampler
caller identity, retains sampler timings separately and rejects unknown sources.
Five unit contracts and 13 existing profile-accounting tests pass locally.
It also passes all five archived `v4-dense-enabled-profile-20260909-01` captures
against raw timings; that is tool regression evidence, NOT an audit of the new
control. On recovery, the original control's detailed HLO tables were retrieved;
its corrected raw audit now passes all five actual `-01` captures as well.
The `-01` candidate remains incomplete. The separate fresh replacement A/B is
completed above; its results do not retroactively complete the interrupted run.

Recovery preserved the interrupted `-01` outputs and used distinct `-02`
directories for both fresh control and candidate on the replacement machine.
No machines are mixed in the claimed same-machine A/B; the strict control-output
bitwise gate and all original numerical tolerances were retained.
