# V4 integration of the existing CSA / HCA / mHC Pallas kernels

**mHC, HCA and CSA are numerically accepted for the tested Flash/EP4 configuration
on four TPU v5p chips: short/ragged worker, native 43-layer 8K B1/B2/B4 and
independent whole-prefill/decode gates all pass. HCA's two numerical reduction
differences are repaired. [CSA integration](deepseek_v4_csa_integration.md)
also repairs the decode projection accumulation difference that caused its
initial 7984 failure. The repaired source passes 261 V4/template tests, 106
bitwise real-input checks, all 14 short 43-layer checks, all 2424 cold 8K
B1/B2/B4 checks and all 514 independent full-logit comparisons, with actual
compiled coverage of all 21 CSA layers. No tolerance was relaxed and no
performance benchmark was run for this CSA phase.**

Current accepted framework/profile fingerprint:
`cf77664b90f217c3bf1d1bdaa2cb112b59bc70d757a7223992d8920a6c86f7e0`.
Current-source receipts are listed in the [CSA integration record](deepseek_v4_csa_integration.md#fixed-decode-projection-and-accepted-gates).

2026-09-08 follow-up: the same source now has passing Engine/pressure and native
performance results, plus passing HTTP workload/soak and separate lifecycle
evidence. The first full HTTP attempt remains failed on a cold status timeout;
see the [serving/performance report](deepseek_v4_csa_serving_performance.md).
No production code was changed during that test phase.

## Accepted mHC/HCA baseline

Framework fingerprint:
`e467f11dd53fea511920b55d5f7a468d5ca68fb767a1a7a64ad52d22c92e88a7`.
At this historical HCA-only baseline, the model executes original Pallas mHC
and all 20 HCA layers; CSA is still the separate V4 JAX/XLA path. HCA receipts:

- 213 TPU tests passed (201 V4 + 12 original HCA), `v4-original-hca-v4-gate-20260907-04.xml`.
- 42 real-input checks passed, `v4-original-hca-real-20260907-05/report.json`.
- 49 shared/reporting/snapshot CPU tests passed, `v4-original-hca-shared-cpu-20260907-02.xml`.
- 2424 full-logit native 8K checks passed, `v4-original-hca-8k-native-20260907-02/report.json`.
- 514 independently constructed reference rows passed, `v4-original-hca-8k-oracle-20260907-01/report.json`.
- 14 same-source short/ragged B1/B2/B4 worker checks passed, `v4-original-hca-worker-20260907-03/report.json`.

All receipts are under local ignored `GCP_login/results`. The historical
failed runs below are retained to explain the diagnosis; they are not accepted
results. These checks do not certify official GPU/API benchmark accuracy,
new Engine pressure/HTTP behavior, or performance of the migrated kernels.

## Objective and preserved baseline

The goal is to run the existing kernels inside the standard V4 ModelRunner,
not merely to implement the same algorithms in another JAX/XLA path. Keep
the original FP4/FP8/E8M0 checkpoint bytes and online weight dequantization.
Prefer V4-only adapters and additive kernel options; do not redesign the
scheduler, allocator or RadixCache.

Before this work, the passing numerical baseline was framework fingerprint
`510bfa6cad2ea7050280ef67eb3c561a31ee3bd9158083ca811e04ed510e0f3d`.
It has 1910 native 43-layer 8K cold B1/B2/B4 logit checks and 514 independently
constructed reference rows. Preserve its reports, arrays and source snapshot:
`GCP_login/snapshots/20260907T121142.706819Z-push`.
These results do not certify a subsequent kernel replacement.

## Integration order and explicit gaps

1. **mHC (first bounded gate).** Keep existing Pallas pre/Sinkhorn. Select
   existing Pallas post explicitly in the candidate model. Extend the existing
   head-collapse kernel with an opt-in FP32 projection-then-RMS mode: the old
   head mode rounds normalized inputs to BF16, whereas V4's retained reference
   does not. This is also verified in the pinned official checkpoint's
   `inference/model.py:728` (`ParallelHead.hc_head`), which uses
   `F.linear(x.float(), hc_fn) * rsqrt`. Preserve the old default and its NumPy
   regression. Retain an explicit V4 reference backend for A/B tests; never
   silently fall back.
2. **HCA.** Reuse original compressor and paged attention. It already supports
   decode, uniform prefill and ragged cross-chunk execution. Adapt physical
   compressed-page addressing, request-private scratch and prefix snapshots.
   Audit numerical boundaries before integration: original attention has a
   BF16 numerator boundary between SWA and compressed segments; the V4
   reference instead retains the FP32 accumulator across 64-key blocks.
   Original compressor pooling/normalization also needs an audit: the current
   template pools and normalizes in FP32 before its output cast, while the
   checkpoint path has a BF16 pooled-value boundary before normalization.
   K-tiled projection must retain request/order-invariant FP32 scratch; do not
   hide projection differences by rounding persistent state to BF16.
   Establish the required model-specific mode with identical real inputs.
3. **CSA.** Reuse original projection/compressor/indexer/joint attention.
   Add the model's Hadamard + FP4 index path rather than feeding it into the
   old FP8 ABI. Replace static request-length specialization with bucketed
   dynamic metadata, and preserve page/prefix state ownership. Keep packed
   KV storage migration distinct from packed weight storage; current V4 KV
   buffers are BF16 and must not be reported as already packed FP8/FP4.
4. **Full model acceptance.** For each replacement, run independent operator
   oracles, true request-order/chunk invariance, real-checkpoint layer tests,
   and then complete 43-layer 8K cold B1/B2/B4 plus independent full-logit
   reference checks. Include position 8023, compressor boundaries and 8191.
   Only then rerun Engine pressure/prefix/retraction/HTTP gates and profile.

## Evidence required

- Record source fingerprint, checkpoint revision, explicit backend selection
  and finite/full-logit comparisons. Preserve the 0.5% logit NRMSE gate and
  top-1 agreement; do not weaken gates or regenerate old baseline arrays.
- Record actual compiled Pallas calls / executable evidence, not just imports
  or function names. Report which components still use the reference path.
- Run one TPU-owning process at a time. Synchronize edits and receipts locally
  so Spot interruption cannot lose either source or diagnostic results.
- Passing synthetic kernels is not full-model acceptance. Passing mHC does
  not imply CSA/HCA integration is complete. No performance claims until the
  applicable complete numerical gates pass.

## Current status

Baseline preserved. The mHC integration uses the shared existing kernels:
pre/Sinkhorn directly, post with explicit `backend="pallas"`, and head with
the additive `head_rms_mode="fp32_post"` option. Existing head callers keep
the old `bf16_pre` default. No scheduler/allocator/RadixCache changes.

The first real-TPU unit gate passed: 24 tests, including the six earlier BF16
fusion-boundary regressions. New checks include CPU FP64 head arithmetic,
batch/order invariance, explicit backend dispatch and executed compiled
post/head Pallas custom calls at token counts 1/2/4/128.
Receipt: `GCP_login/results/v4-original-mhc-unit-20260907-01.xml`.

Broader TPU regression completed: **182/182 V4 tests passed**, plus 50 shared
legacy mHC tests passed. The three known 7168-wide pre-kernel VMEM failures
recurred, with no additional failures (232 passed / 3 failed total). The
requested node-ID deselection did not match this mixed-root pytest invocation,
so those tests did execute. Their failure receipts are retained, not hidden.
They were already reproduced on the unchanged old source; this is not a claim
that all non-Flash mHC shapes now pass. Receipt:
`GCP_login/results/v4-original-mhc-regression-20260907-01.xml`.
The real-checkpoint short-context ModelWorker gate passed: **14 full-logit
comparisons** across B1/B2/B4 chunked prefill, reordered decode and slot reuse.
All 43 layers execute. Maximum NRMSE is `4.880590131506324e-5` (0.004881%),
all values finite and top-1 equal. Not all rows are bitwise equal.
Receipt: `v4-original-mhc-worker-20260907-01/report.json`.

Its exported actual ModelRunner decode executable contains **86 original
Pallas pre, 86 Sinkhorn, 86 post and one FP32-post-RMS head-collapse custom
calls**. This establishes execution-path identity for mHC, not CSA/HCA.
Framework fingerprint:
`dc8b2bf0bbe898c700cc6cb1b78bdb3b27b18687d1bae465e2630dc5d6f9bd9a`.
HLO SHA256:
`36999db315d7009865609cc4ddc5433103bfe689aec6a41947d2f9d39fc1b6d6`.
Short-context agreement alone is not full-context acceptance; the subsequent
complete 8K results are recorded below.

The final V4-only TPU suite, including an additional strict Pallas-post
batch/order invariance test, passed **183 tests / 0 failures / 0 skips** in
115.98 seconds. Receipt: `v4-original-mhc-v4-gate-20260907-01.xml`.
The complete native 8K run **passed 2424 full-logit comparisons**: 1910 cold
B1/B2/B4 native comparisons plus 514 comparisons against the immutable
pre-migration B1 logits. All values are finite and every top-1 agrees. Maximum
NRMSE is `0.00010527185077080503` (0.010527%), below the unchanged 0.5% gate.
This includes 7936 prompt tokens followed by all 256 decode steps, and
prefill-boundary comparisons during empty-cache concurrent reconstruction.
B4 contains two copies of each of two prompts, not four distinct prompts.
Receipt: `v4-original-mhc-8k-native-20260907-01/report.json`.

At the original B2/request1/position8023, NRMSE is
`1.2395231863138179e-7`, maximum absolute error `1.1444091796875e-5`.
The 8K decode executable again has 86 pre / 86 Sinkhorn / 86 post / 1 head
Pallas calls. Its SHA256 is
`c4d2eb5473213850071dfa27130b20c409768818c2800343b93f1897c49ecd87`.
The final independently constructed whole-prefill/all-decode reference gate
also **passed all 514 full-vocabulary rows**: two empty-cache 7936-token
prefills and every one of their 256 teacher-forced decode steps. Both caches
reach 8192. All values are finite, all top-1 tokens agree, and maximum NRMSE
is `0.00010527185077080503` (0.010527%). Of the 514 rows, 435 are bitwise equal;
the remaining rows pass the unchanged tolerance. The reference source hash is
`633d44de83db5865631ae30ec64ec611fd596fbbeb35868d184b5e1dab5b897f`.
Receipt: `v4-original-mhc-8k-oracle-20260907-01/report.json`.

The reference independently constructs cache/state and uses its retained
execution path; it still shares pre/Sinkhorn and low-bit primitives with the
native model. The separate CPU FP64 and legacy NumPy kernel tests supplement
that comparison. This is not an accuracy benchmark against the official GPU
deployment or an official hosted API.

The shared allocator/RadixCache and updated diagnostic-helper CPU gate also
passed: **40 tests**, 7.15 seconds, with `JAX_PLATFORMS=cpu` (no second TPU
owner). Receipt: `v4-original-mhc-shared-cpu-20260907-01.xml`.
Capture/replay scripts now record and reuse the selected mHC backend; an old
capture without that field explicitly retains its historical reference path.

The native model defaults to Pallas mHC. To run the preserved V4 mHC
arithmetic explicitly through the same model, use the existing server option
`--json-model-override-args '{"v4_mhc_backend":"reference"}'`.
The ModelWorker/8K gate scripts expose `--mhc-backend pallas|reference` and
`--check-kernel-dispatch`. Unknown modes fail; neither path silently falls
back. Original FP4/FP8 weight loading and online dequantization are unchanged.

## HCA integration: implementation, diagnosis and acceptance

The numerical-layout gates below use four TPU v5p chips, Python 3.12.14,
JAX/jaxlib 0.11.1, libtpu 0.0.46.1, Flax 0.12.9 and NumPy 2.2.6. Revalidate
the producer-faithful FP32 tests after a compiler/runtime upgrade rather than
assuming the old reduction layout. Runtime receipt:
`v4-original-hca-runtime-20260907.json`.

The HCA phase preserves the accepted mHC snapshot
`GCP_login/snapshots/20260907T140609.477634Z-push`. The candidate uses a V4-only
adapter to the existing `hca_project_fused_pallas`, boundary snapshot emitter
and paged streaming attention. It does not select the legacy HCABackend's
separate allocator. Existing V4 page/snapshot/private-state ownership stays
unchanged, with no scheduler or allocator edit.

Two explicit additive numerical modes keep old template defaults unchanged:
BF16 pool-before-RMS and normalized-value boundaries in the emitter; fixed
64-key blocks, explicit BF16 probabilities, FP32 numerator across both segments,
and a stable sink denominator in attention. FP8 KV QAT remains in the V4
adapter; stored KV is still BF16, not a newly packed KV representation.

Each physical V4 token page owns one compressed record. The streaming kernel
therefore supports a one-record logical page, gathering aligned eight-record
DMA tiles and selecting the owned row in VMEM. It does not build a request-major
compressed-cache copy. This correctness-first adapter uses the original
per-query streaming program for both prefill and decode; its SWA gather and
eight-row read amplification are not claimed to be optimized.

The existing projection kernel itself is unchanged. Full-K, absolute-position
aligned eight-row launches match retained prefill projection bitwise. They are
also used for decode, so prefill/decode/batch arithmetic is uniform. The old
single-row XLA GEMV has a different FP32 reduction order: measured real-weight
NRMSE `1.5516e-7` in the initial probe. An attempted one-row Pallas dot still
did not reproduce that GEMV exactly and was removed. Tests explicitly separate
cross-backend FP64/FP32 agreement (projection NRMSE < `5e-7`) from the stricter
bitwise batch/chunk/private-state invariant. FP32 persistent state is not rounded
to BF16 and the full-model 0.5% gate is unchanged.

Initial operator gate: **13 passed** on v5p, including FP64 pooling/RMS/RoPE,
noncontiguous physical-page attention, padding, and compiled original custom
calls. Receipt: `v4-original-hca-unit-20260907-03.xml`. Earlier receipts 01/02
preserve unsuccessful compile-adapter attempts; they are not model results.
Extended 8K/state lifecycle tests and complete real-model gates are pending.
The explicit candidate switch is `v4_hca_backend=pallas`; `reference` retains
the old path with no silent fallback. HCA source files now enter the framework
fingerprint. Historical mHC-only reports do not certify this candidate.

The extended operator/legacy suite passed **27 tests** (15 new V4 HCA and
12 original HCA tests), including bitwise scratch/snapshot/compressed-record
invariance under ragged chunking, request reorder, prefix fork and slot reuse;
attention includes positions 8023 and 8191. Receipt:
`v4-original-hca-regression-20260907-01.xml`.

Real-input gate passed **42 comparisons** at early/middle/late HCA layers
3/23/41. Historical captured hidden states at position 8023 are used only as
immutable operator inputs; this is not claimed to replay the new full model.
Identical inputs/cache give bitwise-equal attention values and outputs in B1
and B2. Re-emission of all 62 completed real groups includes checkpoint YaRN
and FP8 QAT; largest final-record NRMSE is `8.884893031790853e-5` (0.008885%).
New FP32 projection-row NRMSE is at most `1.552250807890232e-7`.
Receipt: `v4-original-hca-real-20260907-02/report.json`.

The first 62-group emitter compilation exceeded v5p VMEM by 108 KiB
(16.11 MiB vs 16 MiB). V4's v5p boundary tile was reduced from eight to four,
with aligned singleton-row RoPE metadata. No arithmetic threshold changed;
the shared legacy schedule remains unchanged. A 65-group regression was added.
Shared allocator/RadixCache/diagnostic CPU tests passed **40 tests** with the TPU
backend disabled; receipt: `v4-original-hca-shared-cpu-20260907-01.xml`.

The wider first suite reported 209 passes and two harness failures. One caught
the missing original-HCA directory in the production fingerprint; it is now
included in both fingerprint implementations (the CPU reporting suite passes
all 11 tests). The other was the new 65-group oracle mixing different RoPE
tables: independently generated NumPy vs TPU FP32 phases differ by up to
0.00830078125 in that 8K fixture; pooled/RMS NRMSE was only about `4.06e-5`.
Receipt `v4-original-hca-emitter-20260907-03.xml` preserves that diagnosis.
The independent FP64 emitter test now supplies identical CPU-generated FP32
cos/sin tables to the original Pallas kernel and oracle, matching its input
ABI. Adapter + TPU-generated RoPE tables are separately checked against the
retained path. Neither tolerance nor production RoPE generation was changed.
This does not certify the existing table producer against ideal FP64 phases
or the official GPU implementation; that limitation is explicit.

The final broader suite passes **211 tests / 0 failures / 0 skips**:
199 V4 tests (including 16 new HCA tests) plus the 12 original HCA tests.
Receipt: `v4-original-hca-v4-gate-20260907-02.xml` (166.78 seconds).
The real-input gate was rerun after fixing source attribution: all 42 checks
pass again, now under framework fingerprint
`eea49ccec8c115b05fc9171d9f6f5ed0d5e8f2efc20c38f0826ce9b34367bc72`.
Receipt: `v4-original-hca-real-20260907-03/report.json`.
The four-chip 43-layer ModelWorker gate **failed and rejects this candidate**.
Its first B1 prefill row has NRMSE `0.06643485277891159` (6.64%) and maximum
absolute logit error `1.830026388168335`; finite values and matching top-1 do
not override the unchanged 0.5% gate. Receipt:
`v4-original-hca-worker-20260907-01/report.json`. No HCA 8K or performance
acceptance run was started after that failure.

Layer replay localizes the first difference to layer 3, attention coordinate
`[position=39, head=50, channel=121]`. Layers 0–2 are bitwise equal; whole
132-token prefill and chunk-31 prefill have the same layer-3 differences.
Position 39 precedes the first HCA compressed group, excluding compressor
state inheritance as the first cause. Fifteen BF16 attention elements differ
over this layer's 132-token trace. Capture:
`v4-original-hca-prefill-capture-20260907-01`.

Instrumentation of the **actual original Pallas kernel's scratch buffers**
first reproduces both saved uninstrumented outputs bitwise. Its maximum,
FP32 probabilities and FP32 numerator are then bitwise equal to the retained
path, but the probability sum differs by up to `9.5367431640625e-7` (one FP32
ULP at this magnitude). The different reduction order flips a BF16 rounding
boundary and is amplified downstream. Receipt:
`v4-original-hca-attention-probe-20260907-02/report.json`.
An explicit V4-only reduction now accumulates four eight-key stripes within
each 32-key half, halves the eight lanes, then combines the two halves. This
matches the retained **MXU-produced** probability layout; a plain HBM array
or a batched standalone sum has different XLA layouts and is not that contract.
The diagnostic attempts `v4-original-hca-sum-regression-20260907-01/02.xml`
record that distinction. The corrected producer-faithful test passes 30
random/scale/mask cases with all 1920 FP32 sums bitwise equal, and separately
checks Pallas against an explicit NumPy FP32 order. Receipt:
`v4-original-hca-sum-regression-20260907-03.xml`.

Replaying the captured real 132-token attention with this reduction resolves
**all 15 BF16 differences**, including the first compressed group. Receipt:
`v4-original-hca-prefill-reduction-20260907-01/report.json` (fingerprint
`51dbf551989a82e752c0117de3755e5ef9a73824e70388365e3587bcff9e96dd`).
The legacy mode, independent reference and acceptance tolerances are unchanged.
Complete model acceptance must still be rerun; this isolated repair does not
by itself establish correctness of all HCA layers and cache histories.

The repaired source subsequently passes **all 14 short-context full-logit
checks** across 43-layer B1/B2/B4 packed prefill, reordered decode and slot
reuse. Maximum NRMSE is `4.880590131506324e-5` (0.004881%); every row is finite
and top-1 equal. The previously failing first prefill is now
`7.378265109991844e-8`. Receipt:
`v4-original-hca-worker-20260907-02/report.json`, framework fingerprint
`6e15e8ed229fd1e310de2b99191f788ae5ec387a35961292910e48a1c0a42508`.

Its actual ModelRunner decode executable contains **20 HCA attention and
20 HCA boundary-emission Pallas custom calls**, matching all **20** ratio-128
layers in the official Flash configuration, plus 40 projection instructions.
The latter are compiler instructions including conditional branches, not
40 runtime projections per token. mHC retains 86 pre / 86 Sinkhorn / 86 post /
one head custom calls. HLO SHA256:
`36a75c46b86a5c250cee9ab441967410cbd9d9a1b4cce9009630d211ffcc9f81`.
Only CSA remains on the separate V4 JAX/XLA attention/compressor/indexer path.
This short gate is not 8K acceptance or a performance profile.

The strengthened same-source suite passes **213 tests / zero failures / zero
skips** in 172.00 seconds: 201 V4 tests (18 HCA integration checks) and all 12
original HCA regressions. The five physical-page attention cases now require
BF16 bitwise equality, including the added position-39/65 case; no tolerance
was relaxed. Receipt: `v4-original-hca-v4-gate-20260907-03.xml`.
The real-input gate also passes all 42 checks again under this source;
receipt: `v4-original-hca-real-20260907-04/report.json`. Attention values and
outputs remain bitwise equal at layers 3/23/41, while the already documented
FP32 projection and compressed-record cross-backend differences remain
explicit and require full-context model validation.
The actual-kernel scratch diagnostic was rerun with `--verify-fix`:
maximum, denominator, numerator and final BF16 output are now all bitwise
equal to the retained first-failure query; both instrumented paths reproduce
their uninstrumented values. Receipt:
`v4-original-hca-attention-probe-20260907-03/report.json`.

The subsequent cold 8K run **fails** its first 7936-token B1 prefill against
the immutable mHC-accepted baseline: NRMSE `0.07747571170330048` (7.75%),
maximum absolute error `1.4260661602020264`, finite values and matching top-1.
It stops at this first failure; no long decode, B2/B4, or performance acceptance
is claimed. Receipt: `v4-original-hca-8k-native-20260907-01/report.json`.
The short-context reduction repair remains validated, but is not sufficient
for long-context HCA acceptance. Layer/cache/emitter diagnosis continues.

The long-prefix layer diagnosis subsequently finds the first additional HCA
attention difference at **layer 9 / position 3071 / head 38 / channel 502**.
Layers 0–8 match bitwise on identical independent-reference inputs, with all
7936 tokens processed in 128-token chunks. The shorter 1024-token probe also
matched layers 0–8 and therefore did not preserve this failure condition.
Receipts: `v4-original-hca-8k-layer-capture-20260907-01/02` and the CPU index
comparison `v4-original-hca-8k-layer9-inspect-20260907-01.json`.

Only one of layer 9's completed compressed-record elements differs:
`[group=23, channel=502]`, reference `-0.01153564453125`, candidate
`-0.011474609375`. Its query, ordinary KV and window cache are identical.
An original-Pallas query first reproduces the captured failing value bitwise;
replacing only compressed KV with the reference records then restores the
query bitwise. Receipt: `v4-original-hca-cache-swap-20260907-01/report.json`.
The discrepancy subsequently changes expert selection (first observed at
position 3341 in this layer), explaining why small local errors are not an
adequate whole-model correctness gate.

Actual-emitter instrumentation reproduces both immutable outputs and its
previously recorded FP32 intermediates before diagnosis. Maximum scores,
exp results and cos/sin tables are bitwise identical, but the 128-row softmax
denominator and pooling differ. Pooling crosses a BF16 boundary before RMS
and RoPE; this is **not a RoPE table error** in the failing record.
Receipts: `v4-original-hca-emitter-probe-20260907-01/02/report.json`.

The V4-only emitter now reshapes the packed `[entry,time,4,128]` channel view
to `[entry,time,512]` before softmax and pooling. This restores the retained
time-axis reduction layout. The original template's default branch is
unchanged. On the immutable failing group, denominator, probability, FP32
pooling, BF16 boundaries, normalization, rotation and final FP8-QAT/BF16 record
are now **all bitwise equal**. The instrumented fixed kernel also reproduces
its uninstrumented result. Receipt:
`v4-original-hca-emitter-probe-20260907-03/report.json`; new framework
fingerprint `e467f11dd53fea511920b55d5f7a468d5ca68fb767a1a7a64ad52d22c92e88a7`.
This isolated correction still requires same-source regression and full
43-layer 8K acceptance; old passing receipts do not certify the new source.

The same-source strengthened suite now passes **213 / 0 failed / 0 skipped**
in 180.89 seconds, including bitwise emitter comparisons at 1/4/9/65 groups.
Receipt: `v4-original-hca-v4-gate-20260907-04.xml`.
Layer 9's complete 62 records were re-emitted from its immutable FP32 snapshots
and are all bitwise equal to the reference. The original attention first
reproduces all old captured values, then consumes these repaired records:
all **7936-token attention values now match bitwise**, resolving all **5679**
formerly differing BF16 elements. Receipt:
`v4-original-hca-layer9-fixed-20260907-01/report.json`.
This is a full-layer identical-input check, not yet a new complete-model run.

The current source also passes all **42** real-input checks at layers 3/23/41.
Attention values/outputs and re-emitted real compressed records (including
FP8 QAT) now all match bitwise; the documented tiny FP32 projection difference
against the old single-row GEMV remains explicit. Receipt:
`v4-original-hca-real-20260907-05/report.json`.
The shared allocator/RadixCache, report attribution and snapshot helper CPU
gate passes **49 tests** in 7.39 seconds, including both compressed and raw
bit-preserving diagnostic snapshots. Receipt:
`v4-original-hca-shared-cpu-20260907-02.xml`. This uses `JAX_PLATFORMS=cpu` and
does not start a second TPU owner.

The new-source full 43-layer, 7936-prefill + 256-decode, cold B1/B2/B4 run
**passes all 2424 full-logit comparisons**, with zero numerical failures and
all values finite / top-1 equal. Maximum NRMSE is `1.618478790987865e-7`
(0.0000162%), against the unchanged 0.5% gate. All 514 comparisons with the
immutable accepted mHC B1 logits and both late-reference rows are bitwise
equal (516 total); the 1908 concurrent comparisons have the documented tiny
FP32 differences. B4 is two copies of each of two prompts, not four distinct
prompts. Receipt: `v4-original-hca-8k-native-20260907-02/report.json`.
At B2/request1/position8023, NRMSE is `1.2395231863138179e-7`, maximum absolute
error `1.1444091796875e-5`.

The actual executed four-request decode entry has **40 HCA projection
instructions, 20 boundary-emission calls and 20 attention calls**, covering
all 20 HCA layers. Projection instruction counts include conditional branches,
not runtime invocation counts. mHC retains 86 pre / 86 Sinkhorn / 86 post /
one head call. HLO SHA256:
`c7e3112e084bee8252106cc3bbf3b438e1a0e9514363763132e675737e72f487`.
The current source and diagnostic arrays are preserved locally in snapshot
`GCP_login/snapshots/20260907T173332.295948Z-push` and `GCP_login/results`;
the large immutable failing-layer capture was additionally checksum-verified.

The independent whole-prefill/all-decode reference gate subsequently **passes
all 514 full-vocabulary rows**. It builds its own state from empty caches,
executes each whole 7936-token prompt and all 256 teacher-forced decode steps,
and both caches reach 8192. Every row is finite and top-1 equal; 435 are
bitwise equal. Maximum NRMSE is `0.00010527185077080503` (0.010527%), unchanged
from the accepted mHC baseline. For both prompts, positions 8023, 8063 and
8191 are bitwise equal. Receipt:
`v4-original-hca-8k-oracle-20260907-01/report.json`. Reference fingerprint:
`633d44de83db5865631ae30ec64ec611fd596fbbeb35868d184b5e1dab5b897f`.
As before, retained pre/Sinkhorn and low-bit primitives are shared; independent
CPU FP64/legacy NumPy tests supplement this state-independent model comparison.
It is not an official GPU or hosted-API accuracy benchmark.

The final same-source short/ragged 43-layer worker rerun also **passes all
14 full-logit comparisons**: nonaligned 31/32-token chunks, B1/B2/B4,
reordered decode and reused request slots. Maximum NRMSE is
`4.880590131506324e-5` (0.004881%); every row is finite and top-1 equal.
Receipt: `v4-original-hca-worker-20260907-03/report.json`.
Its actual decode entry again contains 40 HCA projection instructions,
20 HCA boundary-emission and 20 HCA attention calls, plus the accepted mHC
calls. HLO SHA256:
`0b7429eff2bd7b6b4913752fd844f550ed9f7a574f613438931f6793f969e5e5`.

## Rerunning the HCA-only gates

Use fresh output directories and run these sequentially on the four-device
slice. Stop on any failure; the independent gate refuses an incomplete or
different-source native report. Keep the old baseline arrays immutable.
These commands explicitly retain reference CSA on the current source; they
do not recreate the archived HCA source fingerprint. Exact historical receipt
reproduction also requires its preserved source snapshot and toolchain.

```bash
set -e
python scripts/run_deepseek_v4_paged.py \
  --checkpoint /path/to/pinned/checkpoint \
  --mhc-backend pallas --hca-backend pallas --csa-backend reference \
  --check-kernel-dispatch \
  --output /new/hca-worker
python scripts/run_deepseek_v4_8k_native.py \
  --checkpoint /path/to/pinned/checkpoint --cold-prefill \
  --baseline-report /preserved/mhc-native/report.json \
  --mhc-backend pallas --hca-backend pallas --csa-backend reference \
  --check-kernel-dispatch \
  --output /new/hca-8k-native
python scripts/validate_deepseek_v4_8k_oracle.py \
  --native-report /new/hca-8k-native/report.json \
  --output /new/hca-8k-oracle
```

The native V4 model defaults to `v4_hca_backend="pallas"`. For an explicit
A/B run in the normal server, the existing server option
`--json-model-override-args '{"v4_hca_backend":"reference"}'` selects the
retained HCA path. Unknown values fail; no automatic fallback is implemented.

## Remaining work

The mHC, HCA and [CSA integration](deepseek_v4_csa_integration.md) phases are
numerically accepted for the tested Flash/EP4 configuration. All original CSA
projection/emission/indexer/joint-attention components are present in the
executed whole-model entry. Explicit reference backends remain available for
A/B comparisons, without automatic fallback. No CSA performance or official
GPU/API accuracy claim is made.

At the numerical-phase close, Engine/HTTP and performance had not yet been
rerun; the same-source follow-up is now recorded in the
[serving/performance report](deepseek_v4_csa_serving_performance.md), including
its two diagnosed control-plane limitations and split HTTP acceptance scope.
Do not promote historical timing reports to results for these new kernels.
FP4/FP8 weight storage and online dequantization remain
unchanged; KV remains BF16. No scheduler/allocator/RadixCache modification,
new cloud resource, Git commit or GitHub push was made for the HCA/CSA phases.
