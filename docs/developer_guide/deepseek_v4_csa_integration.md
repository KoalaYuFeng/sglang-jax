# DeepSeek V4: original CSA kernel integration

Status: **numerically accepted for the tested Flash/EP4 configuration on four
TPU v5p chips**. Original CSA programs execute in the whole-model ModelRunner
path alongside original mHC/HCA. The initial cold 8K failure at 7984 was traced
to decode projection accumulation and repaired with a Pallas GEMV adapter.
The repaired source passes **261 V4/template tests, 106 bitwise real-input
checks, 14 full 43-layer short/ragged checks, 2424 cold 8K B1/B2/B4 checks and
514 independent full-logit comparisons**, with complete 21-layer CSA dispatch
evidence. No tolerance was relaxed. No performance benchmark or profile was
run in this phase; serving-pressure and official-accuracy scope is below.

2026-09-08 follow-up: same-source Engine/pressure and native performance gates
now pass. HTTP workload/soak and a separate lifecycle subset pass, with the
failed cold-status-timeout receipt preserved. See the
[serving and performance report](deepseek_v4_csa_serving_performance.md) for
scope, HBM, B4 profiles and the expert-weight-slicing hotspot. Historical
"not rerun" statements below describe the numerical integration phase only.

Follow-up: the external overlap channel-selection gathers are now fused into
the original CSA compressor through an additive raw-window entry. Independent
kernel gates precede thin integration, 2424 cold 8K checks, 514 independent
reference rows and a same-workload profile. See the
[compressor fusion report](deepseek_v4_csa_compressor_fusion.md) for scope,
numerical receipts and the measured B4 throughput improvement of 8.27%.

Accepted framework/profile fingerprint (both implementations agree):
`cf77664b90f217c3bf1d1bdaa2cb112b59bc70d757a7223992d8920a6c86f7e0`.
The immutable failing candidate was
`1855b17c25017f6edc4aa8aa1cfde746279a9a5ec3ffb4651ff6bfdbe22f6639`.
Independent reference remains
`633d44de83db5865631ae30ec64ec611fd596fbbeb35868d184b5e1dab5b897f`.

## Preserved acceptance baseline

The four-chip v5p Flash/EP4 baseline uses original Pallas mHC and HCA, with
framework fingerprint
`e467f11dd53fea511920b55d5f7a468d5ca68fb767a1a7a64ad52d22c92e88a7`.
Source snapshot: `GCP_login/snapshots/20260908T013949.511199Z-push`.
Immutable reports under `GCP_login/results`:

- `v4-original-hca-worker-20260907-03`: 14 short/ragged full-logit checks.
- `v4-original-hca-8k-native-20260907-02`: 2424 native cold B1/B2/B4 checks.
- `v4-original-hca-8k-oracle-20260907-01`: 514 independently constructed rows.

These reports certify that source, not the new CSA candidate. Preserve raw
FP4/FP8/E8M0 checkpoint bytes, online weight dequantization, request-private
FP32 scratch and page-owned prefix snapshots. Current KV is BF16 with QAT,
not packed FP4/FP8 storage. Do not redesign scheduler/allocator/RadixCache.

## Plan and numerical contracts

1. **Compressor.** Add projection-only access to the original CSA fused
   projection program, with full-K/eight-row arithmetic and an output-channel
   tile that fits v5p VMEM. Reuse the original overlap pool/normalize/rotate
   routine for framework-selected windows, with opt-in V4 BF16 rounding
   boundaries. Keep previous/current channel-half selection and page/private
   ownership in the existing V4 metadata/state adapter. Persistent scratch
   stays FP32. Single-token projection must match the retained GEMV arithmetic,
   including scratch at page boundaries; see the post-failure adapter below.
2. **Indexer.** Reuse CSA's shared StreamIndex Pallas score kernel. Its BF16
   key-cache support can consume V4's Hadamard + FP4-QAT keys without treating
   them as the old packed-FP8 ABI. Add opt-in BF16 score/mixing/reduction
   boundaries, completed-group causality and exact top-k/tie behavior. Dynamic
   ragged metadata must not introduce static request-length specialization.
3. **Joint attention.** Reuse the original CSA joint attention program with
   an explicit BF16 cache adapter and V4 numerical mode: fixed 64-key blocks,
   observable BF16 probability boundaries, FP32 accumulators and stable sink
   normalization. Preserve the old packed-FP8 ABI and numerical defaults.
   Existing V4 gather/quantization bridges may stage selected BF16 records;
   packed-cache migration and gather/fusion optimization are separate work.
4. **Model wiring.** Add an explicit CSA backend switch and a thin V4 adapter.
   Require compiled original projection/emitter/indexer/attention evidence for
   all 21 CSA layers. Include every changed kernel dependency in the production
   and profiling fingerprints. Keep the retained reference backend available,
   with no automatic fallback; do not change the independent reference.
5. **Acceptance.** Real-TPU primitive/legacy regressions, independent numerical
   oracles, strict batch/order/chunk/state invariance, identical real-layer
   inputs, then the complete 43-layer short/ragged and cold 8K B1/B2/B4 gates.
   Compare the immutable baseline and all 514 independent full-vocabulary
   rows, including positions 8023/8063/8191. Keep finite checks, top-1 agreement
   and the existing 0.5% full-logit NRMSE threshold. Diagnose the first failure
   rather than accepting top-1 alone or widening tolerance.

Run one TPU-owning process at a time. Edit locally, then synchronize code and
receipts through the existing Spot workflow. Do not claim official GPU/API
benchmark accuracy, packed KV storage, or performance acceptance from these
numerical fixtures.

## Implementation and initial gates (2026-09-08)

`kernels/deepseek_v4/csa.py` is the thin model adapter. The default model CSA
backend is `pallas`; `v4_csa_backend=reference` explicitly selects the retained
V4 path for diagnosis, with no automatic fallback. The paged/native regression
drivers expose `--csa-backend`; capture/replay propagates that selection.

- Projection calls the original `_csa_state_step_kernel` in projection-only
  mode, with eight position-aligned rows, full K=4096, N tiles of 512 and FP32
  accumulators. Main and index private state/snapshots remain framework-owned.
- Emission calls the original `_pool_uniform_groups` with selected overlap
  windows and explicit V4 BF16/fixed-tree-RMS boundaries. Its first launch hit
  Mosaic `Lane broadcast`; aligned `[D/128,128]` normalization and `[G,8,128]`
  validity storage fixed the ABI without changing arithmetic.
- CSA `paged_lightning_topk` reuses the original shared StreamIndex score
  program. Its opt-in V4 mode keeps BF16/QAT keys, explicit BF16 rounding,
  completed-group causality and exact top-k. Requests are dynamically packed
  and unpacked around the ragged ABI, including inert decode slots; request
  lengths do not become static compilation arguments.
- Original `_joint_attention_kernel` consumes explicit BF16/QAT selected KV
  in V4 mode, with 64-key blocks and stable sink normalization. One query per
  Pallas program is a tile in a batched device grid, **not host dispatch per
  request/layer**. The existing gather bridge can materialize selected KV in
  HBM; this is correctness integration, not a fused-gather performance claim.
- Legacy CSA packed-FP8 interfaces/math remain the default for legacy callers.
  Original CSA and shared DSA dependencies now participate in both source
  fingerprints. Compiled evidence checks each of the 21 CSA layers for original
  projection, main/index emission, indexer and joint attention instructions.

Receipts under `GCP_login/results`:

- `v4-original-csa-primitives-20260908-01.xml`: **32 passed**. Projection,
  independent FP64 emitter oracle, exact retained BF16 emission, 8K scores/
  top-k/ties/inert requests, fixed-block joint attention and extreme sinks.
- `v4-original-csa-gate-20260908-01.xml`: **36 passed**. Adds main/index
  chunk/batch/order/prefix-fork/dirty-slot invariance and both original CSA
  end-to-end / four-chip tensor-parallel NumPy regressions.
- `v4-original-csa-v4-gate-20260908-01.xml`: **250 passed, zero failed/skipped**,
  including all V4 low-bit/model-sized four-chip tests, mHC/HCA integration,
  shared framework contracts and original HCA/CSA templates. Five existing
  Flax/pytest deprecation/reporting warnings, no numerical failures.
- `v4-original-csa-shared-cpu-20260908-01.xml`: **49 passed**, shared
  allocator/RadixCache, snapshot and reporting tests with TPU initialization
  disabled.
- `v4-original-csa-shared-cpu-20260908-02.xml`: **53 passed**, also checks
  explicit CSA backend propagation through the compressor-instrumented replay
  path for both main/index and both reference/Pallas modes.
- `v4-original-csa-real-20260908-01/report.json`: **78 passed** on identical
  historical real inputs for layers 2/22/42 at position 8023, B1 and B2.
  All 54 index/attention/compressed-KV/emitter checks are bitwise. The other
  24 FP32-state checks differ only at the known GEMV/MXU reduction boundary;
  maximum NRMSE `8.192056100142509e-8` (gate `5e-7`). These are immutable
  operator fixtures, not a current-source model replay or full-model acceptance.
- `v4-original-csa-real-20260908-02/report.json`: strengthened **102 passed**;
  adds 24 checks of newly written FP32 projection values alone, so untouched
  state cannot dilute their errors. Maximum NRMSE `1.5081032245234383e-7`,
  below the unchanged `5e-7` gate; the same 54 downstream checks remain bitwise.
- `v4-original-csa-indexer-dynamic-20260908-01.xml`: strengthened **6 passed**;
  reversing requests changes ragged lengths, positions, pages and request slots
  while retaining the same query bucket. Both results are exact and the outer
  indexer JIT traces only once.

The initial failed compile receipt is retained as
`v4-original-csa-compressor-20260908-01.xml`; the repaired emitter-only gate
is `v4-original-csa-compressor-20260908-02.xml` (8 passed).

## Short-model dispatch-audit correction

`v4-original-csa-worker-20260908-01/report.json` passed all **14** 43-layer
B1/B2/B4 short/ragged/reorder/reuse full-logit comparisons, maximum NRMSE
`4.880590131506324e-5`, all finite and top-1 equal. It remains **incomplete**:
the subsequent layer-coverage audit failed, and the historical report is not
rewritten/promoted.

The saved, actually executed B4 decode HLO contains 42 CSA projection, 21 main
emitter, 21 index emitter, 63 StreamIndex and 21 joint-attention custom-call
instructions. Inlining removes `v4_layer_N` from the emitter/attention calls'
own `op_name`; their outputs still feed the uniquely scoped corresponding
layer. The audit now keeps an SSA consumer-path witness per instruction. It
rejects ambiguous ownership, tuple aggregation and cross-computation name
collisions, and does not infer layers from ordinal call counts or shared
scalar inputs. All five CSA components map to exactly layers 2,4,...,42 in
the saved HLO. Counts include dynamic zero-grid indexer branches and are not
runtime invocation counts.

`v4-original-csa-evidence-cpu-20260908-01.xml`: **5 passed**, including rejection
of names found only in metadata and ambiguous/cross-computation attribution.
The model/kernel source fingerprint is unchanged by this reporting-only fix.
The worker gate is rerun as `v4-original-csa-worker-20260908-02` before the
new cold 8K native and independent-oracle gates. No tolerance was changed.

The rerun `v4-original-csa-worker-20260908-02/report.json` is **complete**:
all 14 full-logit checks pass with the same maximum NRMSE
`4.880590131506324e-5` (0.004881%), all finite and top-1 equal. Its executed B4
decode HLO verifies every required CSA component in all 21 layers, with the
same instruction counts listed above. HLO SHA256:
`e1d4e004e4f3c3317e50aca0c402b98ee7457ca4dc89fee48cc9f11b5b52a1cf`.
Source fingerprint remains `1855b17c25017f6edc4aa8aa1cfde746279a9a5ec3ffb4651ff6bfdbe22f6639`.

## Cold 8K failure: position 7984

`v4-original-csa-8k-native-20260908-01/report.json` stops on
`B1/0/original_decode48` (absolute position 7984), after 50 comparisons.
The first 49 comparisons, including the full 7936-token prefill, stay below
NRMSE `1e-6`. The failing full-vocabulary row has NRMSE
`0.05838952213525772` (5.84%), max absolute error `1.5636708736419678`.
It is finite and top-1 agrees, but this does **not** pass the unchanged 0.5%
gate. All 43 post-step caches plus exact inputs/metadata are saved under
`failure-state` (about 4.8 GiB), and failure logits are preserved. The chained
independent oracle did not run. No performance work has started.

### Faithful diagnosis, unchanged failing source

`v4-original-csa-dual-capture-20260908-01` restarts each explicit CSA backend
from an empty request history in the same ModelRunner, sharing only read-only
checkpoint weights and teacher-forced input IDs. The executed HLO verifies
the actual backend selection. Reference logits reproduce the immutable HCA
baseline; Pallas reproduces the saved failing logits **bitwise**. Before-step
states at 7983, exact inputs/metadata and production logits at 7983/7984 are
saved for both modes.

`v4-original-csa-dual-cache-20260908-01` compares request-logical cache content:
all window KV, all HCA state, CSA index compressed KV, and page snapshots are
equal before 7983. CSA private FP32 projection scratch differs around `1e-7`.
The only compressed-KV difference is layer 10, logical row 1994 (terminal
position 7979), channel 463: one BF16 value, absolute difference `0.000244140625`.

`v4-original-csa-dual-replay-20260908-02` is **faithful**: all four final
full-vocabulary logits are bitwise equal to their uninstrumented captures.
It reuses the 43 saved layer traces from `...dual-replay-20260908-01`, whose
head initially failed to compile because the Pallas head needed an explicit
`shard_map`; that incomplete receipt is preserved. The corrected head-only
resume does not recompute or alter the saved layer traces. Position 7983 has
no hidden-state difference at any layer. At 7984 the first hidden difference
is layer 10. Its input/query/index scores/top-k/window KV are equal; its
attention value differs in 7 of 32768 BF16 elements (NRMSE `7.584040577e-5`),
then output projection, mHC and later layers amplify the perturbation.

`v4-original-csa-attention-cross-20260908-01` crosses both attention kernels
with both captured KV inputs. The selected KV differs in exactly one element,
rank 343/channel 463. **Both kernels produce the same bits on either identical
input**, and swapping KV swaps the reproduced attention result. Thus this
failure is caused by historical compressor output, not a current indexer or
joint-attention discrepancy. This does not yet distinguish projection rounding
from pooling/normalization inside the compressor.

At this stage of the investigation, the remaining diagnostic was to capture
the record's construction at 7979 and cross identical projection windows
through both emitters. The following captures retain the failing source and
unchanged tolerances; the production repair is described separately below.

### Root cause: decode projection accumulation order

`v4-original-csa-7979-capture-20260908-01` moves the host-only capture to 7979
and saves all six steps through 7984. `v4-original-csa-7979-replay-20260908-01`
faithfully reproduces all twelve uninstrumented full-logit rows. The differing
main record is visibly constructed at 7979; subsequent hidden states remain
equal until the established layer-10 attention difference at 7984.

`v4-original-csa-compressor-cross-20260908-01` crosses actual reference/Pallas
projection windows with both emitters. Either emitter gives identical bits on
the same inputs. The Pallas-projected window changes one BF16 pooled value at
channel 462, which becomes the observed RoPE channel 463 difference. Index
emission is bitwise equal in all four combinations. Thus projection FP32
accumulation, rather than the emitter, is the source of this failure.

The initial integration used eight-row MXU projection for decode, while the
retained request-aligned single-token path uses an eight-iteration GEMV map.
`v4-original-csa-gemv-probe-20260908-03` verifies the retained `_project` exactly
reproduces the captured real projection. Ordinary Mosaic sum/split reductions
and one-row MXU do not; even a separately compiled plain GEMV differs for the
main compressor. The bounded CPU tree ranking and Pallas projection probes
are diagnostics only, not passing correctness receipts. The full-model gate
must not be relaxed to accept these small projection errors.

## Fixed decode projection and accepted gates

Framework/profile fingerprint:
`cf77664b90f217c3bf1d1bdaa2cb112b59bc70d757a7223992d8920a6c86f7e0`.
Original-module `_csa_v4_gemv_kernel` now implements the v5p retained addition
order: 1024-element main / 2048-element index chunks, sequential 128-lane
vectors, sixteen sequential eight-lane stripes, an eight-lane halving tree,
then adjacent-pair chunk sums. Exact BF16 cancellation probes establish these
clusters (`v4-original-csa-gemv-clusters-{main,index}-20260908-01`), and real
projection probes validate all FP32 bits at output tiles 8/32/128. The final
128-column ABI writes compact FP32 output directly, not a replicated padded
intermediate (`v4-original-csa-gemv-probe-20260908-05`).

This numerical adapter is validated on four TPU v5p chips with JAX/jaxlib
`0.11.1` and libtpu `0.0.46.1`, not certified for every TPU/compiler version.
Recheck exact projection and full-model gates after changing that stack.

The thin adapter dynamically selects this original Pallas GEMV for a one-token
request and the unchanged original eight-row MXU program for longer queries.
Only the router block's live row is evaluated in GEMV mode. All dispatch remains
inside the whole-model compilation; there is no reference fallback or host
per-request loop. Page ownership, FP32 persistent scratch, BF16/QAT KV and raw
checkpoint FP4/FP8/E8M0 storage are unchanged.

Post-fix receipts:

- `v4-original-csa-fixed-primitives-20260908-02.xml`: **36 passed**. Projection
  is now required to be bitwise, including all eight absolute token lanes,
  inert request slots and mixed singleton/multitoken queries. Every chunk's
  private state, snapshots and compressed KV is compared bitwise against
  independent B1 reference histories with the **same chunk schedule**.
- `v4-original-csa-fixed-real-20260908-01/report.json`: **102 passed, all
  bitwise**, including the previously non-bitwise real FP32 scratch/projection.
- `v4-original-csa-fixed-real-20260908-02/report.json`: **106 passed, all
  bitwise**; additionally gates the four real main/index projection outputs
  from the frozen 7979 failure fixture with zero tolerance.
- `v4-original-csa-fixed-v4-gate-20260908-01.xml`: **261 passed, zero
  failed/skipped**, covering all V4 tests, low-bit/reference/platform paths,
  original mHC/HCA and both original CSA end-to-end / four-chip NumPy tests.
  Five existing Flax/pytest deprecation/report-format warnings remain.
- `v4-original-csa-fixed-worker-20260908-01/report.json`: **14 passed**, full
  43-layer B1/B2/B4 prefill/decode, reorder and slot reuse. All finite/top-1
  checks pass; maximum logits NRMSE `4.880590131506324e-5`. Actual executed B4
  HLO verifies all five CSA components at each of layers 2,4,...,42. It contains
  84 projection instructions (42 MXU plus 42 GEMV conditional branches), 21
  main and 21 index emitters, 63 indexer instructions including dynamic empty
  branches, and 21 joint-attention instructions. These are not runtime launch
  counts. HLO SHA256:
  `e20dc54752e62e06d4dd9a677acb312107b69223b83d93879952966f630a2de7`.
- `v4-original-csa-fixed-8k-native-20260908-01/report.json`: **2424 passed**,
  complete 43-layer, 7936-token prefill plus 256 decode steps through 8191,
  with cold B1/B2/B4 histories and alternating request order. All finite and
  top-1 checks pass, zero numerical failures; maximum NRMSE
  `1.618478790987865e-7`. All 514 immutable HCA-baseline B1 rows and both
  late-reference comparisons are bitwise (516 total); the other 1908
  concurrent comparisons retain the accepted tiny FP32 differences. Both
  prompts are bitwise at 7979/7984/8023/8063/8191 against the HCA baseline.
  Executed B4 HLO covers all five CSA components at all 21 layers, with the
  same CSA instruction counts as the short gate. HLO SHA256:
  `98215f548afdf79ba6563cac38fd7613cfca84ca0f84944ab9e1101d5f439501`.
  B4 contains two copies of each of two prompts, not four distinct prompts.
- `v4-original-csa-fixed-8k-oracle-20260908-01/report.json`: **514 passed**,
  with 435 bitwise rows, all finite and top-1 equal. Maximum NRMSE
  `0.00010527185077080503` (0.010527%) is unchanged from the accepted HCA
  baseline and below the unchanged 0.5% gate. Both prompts are bitwise at
  7979/7984/8023/8063/8191. Each independent history starts with empty caches,
  evaluates the entire 7936-token prompt in one prefill, then executes all
  256 teacher-forced decode steps and reaches cache length 8192. No native
  hidden state, compressor scratch or prefix KV is reused. Reference source
  fingerprint remains `633d44de83db5865631ae30ec64ec611fd596fbbeb35868d184b5e1dab5b897f`.
  Retained pre/Sinkhorn and low-bit primitives are shared; independent CPU
  FP64 and legacy NumPy tests supplement this state-independent comparison.
  It is not an official GPU or hosted-API accuracy benchmark.

The first post-fix primitive receipt (`...fixed-primitives-20260908-01.xml`)
is preserved as failed: its old test incorrectly demanded identical FP32
scratch between whole-prefill MXU and singleton GEMV chunk schedules. The
retained V4 path deliberately distinguishes these numerical modes. The test
now uses equal schedules and checks every chunk against retained B1 histories,
instead of only checking final state against whole-prefill. No tolerance was
increased, and independent whole-prefill/model-logit gates remain required.

## Reproducing the integration gates

Use the tested four-device Flash/EP4 environment, `PYTHONPATH=python`, the
pinned official checkpoint revision
`60d8d70770c6776ff598c94bb586a859a38244f1`, and fresh output directories.
Run sequentially; the independent oracle rejects incomplete native reports
and production source fingerprints that differ from its input report.

```bash
set -e
python scripts/run_deepseek_v4_paged.py \
  --checkpoint /path/to/pinned/checkpoint \
  --mhc-backend pallas --hca-backend pallas --csa-backend pallas \
  --check-kernel-dispatch --output /new/csa-worker
python scripts/run_deepseek_v4_8k_native.py \
  --checkpoint /path/to/pinned/checkpoint \
  --mhc-backend pallas --hca-backend pallas --csa-backend pallas \
  --check-kernel-dispatch --cold-prefill --capture-failure-state \
  --baseline-report /preserved/hca-8k-native/report.json \
  --output /new/csa-8k-native
python scripts/validate_deepseek_v4_8k_oracle.py \
  --native-report /new/csa-8k-native/report.json \
  --output /new/csa-8k-oracle
```

The production model defaults to `v4_csa_backend="pallas"`. For an explicit
CSA-only A/B comparison, use
`--json-model-override-args '{"v4_csa_backend":"reference"}'` in the server,
or `--csa-backend reference` in these regression drivers. Unknown backend
names fail; there is no automatic fallback. Reference comparisons are not a
substitute for the original-kernel dispatch audit.

## Scope and remaining gates

This phase integrates original CSA projection/emission, StreamIndex and joint
attention into the single whole-model compiled entry, alongside original mHC
and HCA. Scheduler and the memory-cache/allocator/RadixCache source trees are
unchanged relative to the preserved pre-CSA snapshot. Added V4 numerical
options leave legacy CSA/DSA defaults intact. Code and diagnostic receipts
are synchronized back to the ignored local `GCP_login` workflow; no GitHub
push is part of this phase.

The fixtures cover two prompts, cold B1/B2/B4 histories, request reordering,
short ragged chunks and slot reuse. They are not broad official GPU/API
accuracy evaluation or a new Engine/HTTP pressure, prefix-eviction or
retraction acceptance run. Those serving gates have not been rerun on the
CSA source. No performance benchmark or profile was run; historical kernel
timings must not be attributed to this implementation. Selected BF16/QAT KV
still uses the existing gather bridge; packed KV and gather/dequant fusion
remain separate optimization work, not prerequisites to this numerical gate.
