# V4 attention parallelism and CSA decode batching: independent gates

## Scope

The accepted starting framework fingerprint is
`9b9ddeca670489a95cbd127f1aa9edead3aee3609cc580057b78a66a92853bd7`.
The existing four-chip model uses EP4, but replicates attention, compressor,
index heads and KV/state. Existing CSA/HCA head-TP wrappers are not its callers.
This work is an isolated operator/single-layer experiment, not an Engine or
full-model rollout. Do not change scheduler, default backend, checkpoint bytes,
KV ownership, reference arithmetic, or accepted numerical tolerances.

## Ordered gates

1. Add a batched decode-only CSA projection entry reusing the unchanged
   `_csa_v4_gemv_kernel`. Check synthetic inputs, an independent NumPy FP32
   reduction, frozen real 7979 inputs, permutation/partition invariance, and
   replicated four-chip execution. Compare the actual adapter boundary and
   sequential GEMVs against the single-grid candidate on identical arrays.
2. Exercise original V4 attention with Q/sink/complete wo_a groups head-sharded.
   Retain all index heads and replicated compressor/KV. Gather wo_a BF16 output
   before the unchanged wo_b, preserving its full-K accumulation. Compare the
   same real layer inputs on one chip, replicated four chips, and head-TP4.
   Check all cache/state leaves, attention output, query/head ownership and
   executed HLO, including the collective boundary.
3. Time only accepted candidates: alternating warmed calls, same operands,
   no checkpoint loading or compilation in timings, separate device captures.
   Report both same-chip batching and TP4 vs replicated4/single-chip results;
   never label these full-model throughput or linear scaling.
4. Only propose production selection after these gates. Full 43-layer cold 8K,
   independent numerical reference, Engine/HTTP and a new model profile are
   required separately before claiming an integrated performance improvement.

Gates 1–3 are complete for the isolated decode fixtures below. The new entries
remain opt-in and are not selected by the production model. Gate 4, including
TP prefill/chunk validation and whole-model/serving rollout, is not complete.

## Independent projection gate

`v4-csa-projection-batch-unit-20260908-02.xml`: **14 passed**, no skips/failures.
The new entry matches sequential accepted GEMVs and an independent NumPy FP32
tree bitwise for main/index widths, B1/B2/B4/B9, zero rows, permutation and
partition, every replica on four physical chips, and the frozen real 7979
projection outputs. It rejects unsupported shapes. Every pre-existing CSA
compressor function has an unchanged AST; this entry only changes the grid.

The preceding `...01.xml` is retained as failed (6 passed, 8 compile failures):
one-row blocks directly inside `[B,K]` violate TPU DMA alignment when B > 1.
A mapped leading token axis in `[B,1,K]` leaves the original `[1,K] kernel
reference unchanged and satisfies that layout constraint. This was a compile
failure, not a numerical discrepancy.

## Attention experiment protocol

`v4-attention-headtp-cpu-20260908-02.xml`: **10 passed**. Guards cover exact
weight-sharding selection, whole output-group ownership, independent B4 pages
and private slots, unchanged source fixtures, strict nonfinite/state gates,
and fresh rather than feedback-fed timing inputs. The preceding 9-test receipt
is retained. Six dependency-free raw-profile classifier tests also pass.

The first attention smoke receipt is retained as failed at the *unmodified
single-chip baseline*. Repeatedly feeding the updated cache back to the same
8023 decode step was not idempotent (output NRMSE 0.0195482). That would be an
invalid timing fixture; it was not a TP-candidate comparison. The corrected
harness prepares independent copies of the same frozen cache outside each
timed interval and prepares the entire input ring before a device capture.
Each measured program donates its own cache copy. Fresh-snapshot repeatability,
traced/untraced outputs and all state leaves must pass before timing.

`v4-attention-headtp-smoke-20260908-02`: the layer-2/B2 single, replicated4,
and head-TP4 paths pass. Executed HLO and all four query shards confirm 16
local heads for TP4. These four timing samples are only a smoke test, not
the performance conclusion; larger paired measurements are a separate gate.

## Real single-layer acceptance

Framework source fingerprint for all successful measurements:
`538191ea3a138fbab26fd5e48a7f313439507f12ae46970949f45ac4b56de4d2`.
The independent reference source is unchanged. Only two production files
change: one additional CSA projection entry and an optional gather boundary
in V4 attention. There is no model-loader, model-selection, scheduler or
allocator change, and no persistent low-bit-to-BF16 weight conversion.

- `v4-attention-headtp-profile-20260908-01`: layers 2/3, B1/B2/B4, **1164
  tensor/state checks pass bitwise**. These are six input cases, not 1164
  independent model examples.
- `v4-attention-headtp-real-20260908-01`: layers 22/23/42/41, the same three
  batch fixtures, **2328 additional checks pass bitwise**. This run has no
  performance samples or profiling.
- `v4-attention-parallel-regression-20260908-01.xml`: **76 passed, zero
  skips/failures**, covering the new projection tests and the existing V4
  CSA/HCA numerical/integration tests. The standalone 14-test set is included
  in these 76; it is not counted twice.

Across the six representative layers, all **3492** comparisons are bitwise,
including full attention output, Q/QR/KV, attention values, CSA scores/top-k,
private compressor state, paged KV/snapshots, every replicated cache copy,
and traced/untraced/repeated fresh-input checks. Existing attention-output
NRMSE limits remain 0.005 and attention-value limits remain 0.0002; this run
does not merely pass those tolerances but produces identical bits.

Actual compiled calls use 64 local heads for the single/replicated baselines
and 16 for TP4. Q/sink and complete wo_a groups are partitioned; wo_b, all 64
index heads, compressor weights, private state and KV remain replicated.
The only added collective gathers the BF16 wo_a groups before full-K wo_b;
there is no new floating-point sum collective. Both Q and output projection
checkpoint scales are partitioned together with their raw FP8 weight blocks.

B4 is explicitly **two historical B2 requests cloned onto independent pages
and slots**, not a separately captured live four-user workload. Historical
8023 hidden/cache arrays are operator fixtures, not a newly independent 8K
model accuracy benchmark. The TP candidate has not yet passed chunked prefill
or advancing decode through a full-model 8K context.

## Paired single-layer attention performance

Forty alternating warmed dispatch-to-ready samples per variant, plus separate
eight-call device captures. Weight loading, compilation and independent cache
copy preparation are outside timing/profiling. Cache donation is retained;
compiler alias bytes match between variants. No Engine/ModelRunner is created.

The device values below are medians of the active-TensorCore **module envelope
per call** (two lanes on one chip, eight on four chips), not averaged HLO
self-time. They must not be subtracted from unprofiled host medians to infer
host idle time.

| Layer / batch | One chip device ms | Replicated4 device ms | Head-TP4 device ms |
| --- | ---: | ---: | ---: |
| CSA 2 / B1 | 0.732793 | 0.766042 | 0.634335 |
| CSA 2 / B2 | 0.955059 | 1.001018 | 0.849626 |
| CSA 2 / B4 clone | 1.166310 | 1.210166 | 1.047151 |
| HCA 3 / B1 | 0.513029 | 0.544822 | 0.412964 |
| HCA 3 / B2 | 0.578286 | 0.620291 | 0.476344 |
| HCA 3 / B4 clone | 0.655842 | 0.694875 | 0.523275 |

At B4, head-TP4 versus replicated4 reduces device latency **13.47% (CSA)**
and **24.70% (HCA)**. Corresponding unprofiled host medians are
**1.932625 → 1.775926 ms (8.11%)** and
**1.426407 → 1.259411 ms (11.71%)**. Single-chip host medians are 1.684418
and 1.222234 ms: four chips do not make these small isolated calls four times
faster, and host and device comparisons are different measurements.

The new raw reader uses `device_duration_ps * Time Scale Multiplier` on
**XLA Ops only**, excludes duplicate Async Ops/control-flow envelopes and
checks active-lane/module/attention-call coverage. Selected-family durations
below are **means per TensorCore per call**, not the module medians above:

| B4 selected work | CSA replicated4 → TP4, ms | HCA replicated4 → TP4, ms |
| --- | ---: | ---: |
| FP8 projection calls, including dequant | 0.194485 → 0.093463 | 0.172551 → 0.076677 |
| Slice instructions (not all normalization work) | 0.114261 → 0.076241 | 0.112876 → 0.074194 |
| Attention core | 0.025260 → 0.015363 | 0.016488 → 0.010150 |
| Added all-gather, **including waiting** | 0 → 0.031984 | 0 → 0.016866 |
| CSA single-row GEMVs | 0.226159 → 0.226261 | — |

Thus TP benefits primarily projection/data processing, while the shared
compressor work is essentially unchanged. Each layer's compiled argument bytes
drop by 48 MiB/chip; compiler temporary estimates drop by about 3.11 MiB for
CSA and 5.63 MiB for HCA. These are compiler estimates, not measured full-model
HBM savings or physical traffic counters.

## Independent CSA projection-boundary performance

`v4-csa-projection-batch-profile-20260908-01`: 16 shape cases, B1/B2/B4/B9,
main/index, one/four chips; **960 repeated/per-replica comparisons pass bitwise**.
Four input buffers repeat/sign-flip the frozen real 7979 activation. The
unchanged V4 adapter is compared with bare sequential GEMVs and the batched
entry, all within comparable compiled boundaries. Forty paired host samples;
separate ten-call device captures for B1/B4.

| B4 / four replicated chips | Existing adapter device ms | Bare sequential device ms | Batched device ms |
| --- | ---: | ---: | ---: |
| Main projection | 0.331500 | 0.257579 | 0.251722 |
| Index projection | 0.138424 | 0.078606 | 0.073221 |

Relative to the adapter, device latency falls **24.07% / 47.10%**; host medians
fall **0.573504 → 0.481723 ms / 0.376620 → 0.301909 ms**. But versus *bare
sequential GEMVs*, device latency only falls **2.27% / 6.85%**. Most of the
boundary benefit removes V4 adapter packing, inert blocks and per-request
invocation overhead; it does not remove most of the FP32 GEMV compute.
Raw main GEMV self-time is 0.198083 ms in the adapter, 0.197741 ms bare,
and 0.195774 ms batched. The previous full-model 4.85 ms projection hotspot
therefore must not be advertised as eliminated.

Compiler temporary estimates at this isolated boundary fall from 17923264
to 131008 bytes (main), and from 4865152 to 131008 (index). These are not
measured HBM/VMEM traffic or integrated model memory savings.

## Receipts and next gate

All successful/failed receipts, HLO and raw traces are retained under the
Git-ignored local `GCP_login/results/`; remote originals are on the independent
model data disk. Logs stream to local `GCP_login/logs/`. Report SHA256 values:

- Attention profile: `10cb1a8a7bb065851591600dba8f14e5265f8c9821f45ff45741a3cbf8964690`.
- Expanded layer acceptance: `cae330b9a0ef80a2b79b8482baebbba47c168007f0839dc2a2e452353c7ea1c8`.
- Projection profile: `11536fdb1fa7c12095ccaa4fb0bee522981234b518d8645d27e3be377444d9c7`.

Next, validate TP on prefill/chunk boundaries, and add an explicitly selected
decode-only CSA adapter fast path without replacing mixed/prefill MXU math.
Only then thinly connect the candidates to an opt-in whole-model path and run
cold 43-layer/8K, independent numerical and Engine/HTTP gates. The two
experiments here were measured separately; their gains are not additive and
there is **no new integrated tokens/s result or default serving change**.
