# Position 8023 numerical failure: diagnosis and repair

Status: **PASS — the fixed 43-layer, 8K numerical regression**, 2026-09-07.
The original position-8023 failure is reproduced, causally localized and fixed.
Stronger checks also exposed and repaired a fused mHC BF16 rounding boundary
and a phase-dependent reference pooling/lowering discrepancy.

| Final same-source gate | Result |
| --- | --- |
| V4/framework/low-bit tests on the four-device v5p runner | 164 passed, no skips |
| Shared allocator/radix/input-length CPU tests | 35 passed |
| Native B1/B2/B4, cold 7936-token prefixes + 256 decode steps | 1910 full-logit checks passed |
| Independent empty-cache whole prefill + all decode, two fixtures | 514 full-logit rows **bitwise equal** |

Native worst NRMSE is `5.8791898482013494e-5`, below the unchanged `0.005`
gate; the original failing B2/request-1/position-8023 NRMSE is now
`1.2395231863138179e-7` instead of `0.10763054341077805`. All checked logits
are finite and top-1 agrees. Both independent caches reach exactly 8192.

This certifies the fixed numerical fixtures, **not** official GPU/API accuracy
parity, arbitrary prompts, Engine pressure correctness or performance. The
independent implementation retains its own cache construction; its pooling
arithmetic correction and independent CPU/FP64 evidence are explicitly
documented below. Performance remains paused. Earlier candidates and failed
reports remain diagnostic history, not accepted baselines or benchmarks.

## Reproduction and trustworthy layer comparison

The unchanged production graph with source fingerprint
`c7ec7e1357f41b4710f5505fa76ed95fc458693998df6ebab4ec5b0fc39ce983`
reproduces the original first failure: B2 request 1 at zero-based cache position
8023 (decode index 87), full-vocabulary NRMSE `0.10763054341077805`, maximum
absolute error `3.1207332611083984`. Both B1 baselines match the old goldens
bitwise. B2 request 0's NRMSE at that position is `1.3686926081390993e-07`.

The capture saves complete physical caches, including request-private FP32
compressor scratch and page-owned snapshots, before and after the ordinary
ModelWorker call. It adds no outputs to the production JIT. BF16 is serialized
as its original 16-bit payload. The subsequent layer replay matches **every
post-step cache buffer and all three final production logit arrays bitwise**.
Thus the observed intermediate divergence is not an instrumentation artifact.

Before the target step, layers 0 and 1 agree completely. Starting at layer 2,
the private FP32 compressor state differs by a few ulps, while valid window
and compressed KV in layers 2–5 still agree. The first hidden-state difference
is in layer 6 (the seventh layer), request 1:

| Point in layer 6 | B1 | B2 |
| --- | ---: | ---: |
| Input, normalized input, Q/KV, index scores/top-k | Bitwise equal | Bitwise equal |
| Main compressor pooled BF16, channel 468 | -0.01007080078125 | -0.0101318359375 |
| Normalized compressed value, channel 468 | -0.0751953125 | -0.07568359375 |
| After RoPE, compressed vector 2005/channel 469 | -1.1953125 | -1.203125 |

At layer 6's output, 6602/16384 hidden elements differ, with NRMSE
`0.004925544839352369`. Subsequent layers amplify this discrepancy. The initial
index score/top-k and logical cache addresses are identical: this is not a
cross-request page write or top-k ownership error. Later index/routing/cache
differences are downstream effects.

Position 8023 is the first **full-logit acceptance failure**, not the first
bit-level internal difference. The all-layer pre-step comparison also finds
one earlier layer-40 compressed-KV difference at vector 2003/channel 471
(group ending at position 8015), magnitude `0.00048828125`, without an earlier
full-logit gate failure. This is another reason to test persistent state
strictly instead of relying only on final-token/logit thresholds.

## Cause and minimal fix

The BF16-input, FP32-output compressor dot is sensitive to the packed leading
dimension/token lane on v5p. Tiny projection differences remain in FP32 scratch
across decode calls. At the target compression boundary they change the BF16
pooling result. One changed compressed-KV element then perturbs attention and
the following low-bit/routed layers.

An isolated real-weight compressor run first matches both captured production
caches exactly. Replacing either B2's historical scratch or only its current
projection with B1's removes the visible difference; replacing both also does.
A naive one-row `lax.map` does **not** remove the FP32 projection differences.
The existing absolute-position-aligned eight-row layout does remove them.

The first production change is in
`python/sgl_jax/srt/kernels/deepseek_v4/compressor.py`: project each token in the
existing request/absolute-position-aligned eight-row blocks. The metadata is
already provided for V4 routing, so no shared scheduler, allocator or model
interface changes are needed. Both main CSA/HCA and index CSA use this helper.
Checkpoint FP4/FP8/block-scale storage, online dequant, and FP32 scratch remain
unchanged. There is no extra BF16 rounding or loosened acceptance tolerance.

First-candidate framework fingerprint (not numerically accepted):
`ca6ac09d091f781d8b2c06fb4f9bd39ba4c72a37f1afd3344644562f3afdeb03`.

### Second candidate: preserve single-request arithmetic

The follow-up projection sweep found that fixed blocks containing **eight
single-row GEMVs**, rather than one eight-row GEMM, reproduce the original B1
FP32 projections bitwise while remaining batch invariant. A naive map over
only the packed tokens does not have this property. Explicit FP32 inputs,
HIGHEST dot precision, and a fixed reduction tree alone do not reproduce B1.

The new candidate uses each request's query length: one-token queries execute
the fixed eight-iteration GEMV path; multi-token queries retain GEMM. Each
existing router block belongs to exactly one request, so the decision also
works for mixed query lengths without adding framework metadata. Both
branches preserve FP32 outputs/scratch. The real layer-6/8023 hybrid probe
matches original B1 and B1/B2 bitwise for KV, scores, pooled, normalized and
final compressed values.

The synthetic 8K reference comparison is now strict for FP32 scratch, too;
candidate 1 fails it. The full-model runner additionally supports
`--baseline-report`, checking every B1 prefill/decode result against the frozen
original B1 independently of the newly generated goldens. Candidate 2 passes
all **154 tests** (including the newly strict independent FP32 state check) on
the v5p runner, in 82.29 seconds. Its full-model regression passes with the
original B1 checks enabled; the oracle and tolerance are unchanged.

Candidate 2 framework fingerprint:
`e0ecbe85d59f136c9d53a135229d2902444905ba5e98b8abd2a2bf2e7a700834`.

### Candidate 2 full-model result

`v4-8023-native-fixed-20260907-02/report.json` is finished and complete:

- Both B1 prefill endpoints and every decode row (514 comparisons) are
  **bitwise equal** to the frozen original B1 baseline.
- Both independent late single-step checks at position 8191 pass bitwise.
- B2: all 512 full-vocabulary decode rows pass.
- B4: all 1024 full-vocabulary decode rows pass.
- All 2052 checks are finite and top-1 agrees. Maximum B2/B4 NRMSE is
  `0.0000655393669148907` (0.006554%), below the unchanged 0.5% threshold.
- At the original failing request/position 8023, NRMSE drops from
  `0.10763054341077805` to `1.284537631818239e-07`; maximum absolute error
  drops from `3.1207332611083984` to `0.000011444091796875`.

These candidate-2 results did not substitute for independently constructing
the entire prefix cache or cold packed-prefill comparisons. The stronger gates
and additional failures/repairs are recorded below.

### Independent whole-prefill failure

`v4-8023-oracle-full-20260907-01` recomputes all 7936 prompt tokens in one
independent reference call from empty cache. Case 0 fails its first endpoint
comparison: position 7935, NRMSE `0.13016195595264435`, maximum absolute error
`2.536099910736084`, all finite, top-1 equal. No native KV/scratch was imported.
Because the native B1 equals the original B1 bitwise, this is also a gap in
the original baseline, not proof that the new B1/B2 fix alone certifies the
complete model. Next: localize whole versus chunked prefill layer arithmetic
and state without changing the oracle/tolerance to force agreement.

### Additional cause: an elided BF16 boundary in fused mHC post

The same independent reference, whole versus 128-token chunked, first differs
at layer 0 FFN post (no compressor in this layer). All preceding traced
attention/FFN operators and routing match bitwise; the output NRMSE is
`0.002356972312554717`, maximum absolute error `0.0078125`.

The real first-token FFN output and residual, replayed in isolation with either
XLA or Pallas post, agree bitwise with a NumPy FP64 formula and the recorded
whole-prefill output. Post/comb gates agree between lengths 128 and 7936.
However, integrating an FP32 producer -> BF16 activation -> XLA post in one
JIT reproduces the error even though the materialized activation trace is
correct. The same integrated computation using Pallas preserves the boundary
and matches NumPy. This is not evidence of a faulty Pallas post matmul.

A seeded CPU-oracle regression catches the original XLA path (2 failed,
4 passed across backend/input variants; failing NRMSE `0.0025595245`). The
candidate fix uses the existing bit-explicit `round_bf16` helper on BF16
inputs of the XLA post path only, preserving the FP32 API and the Pallas path.
No reference formula or numerical threshold is changed. Full-model acceptance
must be rerun; preserving a demonstrably incorrect old fused result cannot
replace the CPU and independently constructed whole-prefill gates.

All six new fused-rounding variants pass after this change. The broader mHC
suite has 56 passes and three compile-time VMEM OOMs in **pre**, for non-Flash
hidden size 7168 (hc=4/8). An isolated replay using the unmodified HEAD mHC
source reproduces all three pre OOMs with the same allocation sizes
(`v4-8023-mhc-wide-baseline-20260907-01.xml`). These are existing wider-shape
schedule failures, not new post regressions. Flash uses hidden 4096/hc=4 and
its mHC cases pass. Do not describe the entire broader suite as passing or
silently skip those failures.

With the rounding repair, the real 7936-token fixture passes **all 43 layers
bitwise** in whole versus 128-token chunked independent-reference replay
(`v4-8023-whole-chunk-reference-20260907-02.json`). This includes every token's
hidden output, not merely the endpoint or top-1. Native B1/B2/B4 cold-cache
regression and the independent complete 8192-boundary oracle were still
required at this stage; their final passing receipts appear below.

Intermediate framework fingerprint (compressor plus mHC repair):
`da637c28b006f6db573a93d11621ae3b286485889aa3ccc65319292975d5ee7a`.
Both fresh B1 runs reach 8192 and pass their late single-step oracle. All 514
top-1 IDs remain equal to the original fixture, so the decode input trajectory
still includes the original position-8023 case. Logits deliberately differ
from the old, incorrectly rounded fused baseline: prefill NRMSE relative to
old B1 is 12.4834% / 7.6706%, and maximum full-sequence NRMSE is
41.7005% / 54.7810% for the two fixtures. This is diagnostic comparison, not
an accepted substitute for the then-pending independent from-empty-cache oracle.

That cold-cache B2 run passes all 124 prefill-boundary and 512 decode
comparisons. At the original failing request/position 8023, NRMSE is now
`1.2395231863138179e-07`, maximum absolute error `1.1444091796875e-05`, all
finite and top-1 equal. This recomputes history rather than restoring B1
prefix pages. B4 also passes; the independent whole-prefix oracle subsequently
exposes the position-7939 discrepancy described below.

An additional immutable-oracle check already passes bitwise: the new native
case-0 prefill endpoint equals the **previously saved, pre-fix independent
whole-prefill expected logits** in
`v4-8023-oracle-full-20260907-01/failure.npz` (NRMSE/max-abs both zero).
Those expected values were not regenerated after the fix. Thus the repair
does not merely make two newly changed paths agree with one another. The
complete independent two-case/decode rerun is recorded in the final result.

### Intermediate full-model cold-cache result

`v4-8023-native-cold-fixed-20260907-03/report.json` is finished and complete,
with **1910 full-vocabulary comparisons, zero failures**, all finite and top-1
equal. B2 and B4 build every cache from empty state, alternate request order,
then decode all the way to the 8192 boundary:

| Check | Rows | Maximum NRMSE |
| --- | ---: | ---: |
| B1 late independent decode | 2 | 0 (bitwise) |
| B2 cold prefill, every 128-token boundary | 124 | 1.61848e-7 |
| B2 decode | 512 | 5.87919e-5 |
| B4 cold prefill, every 128-token boundary | 248 | 1.61848e-7 |
| B4 decode | 1024 | 5.87919e-5 |

Worst NRMSE is **0.0058792%**, versus the unchanged **0.5%** gate. B2/B4
results are within tolerance, not all bitwise equal; do not conflate this gate
with the bitwise whole/chunked layer replay. The independent from-empty-cache
two-case full-prefill/decode oracle was the remaining full-model prerequisite.

### Independent position 7939: pooling arithmetic and reference repair

`v4-8023-oracle-full-20260907-02` passes the entire prefill endpoint and decode
positions 7936–7938 **bitwise**, then fails at 7939: NRMSE
`0.0447622612118721`, maximum absolute error `1.2386021614074707`. The top-1
still matches. This gate starts with independent empty caches, not imported
native prefix state.

`v4-7939-replay-20260907-01` reproduces both frozen native and reference
logits bitwise. Its 43-layer native replay also matches every production
physical post-cache buffer and the final logits bitwise. First hidden
divergence: layer 2, NRMSE `0.0031107019`. All incoming logical KV, CSA/index
scratch, hidden states, current projections, index scores/top-k agree. The
first cache difference is just main compressed vector 1984, channel 465,
magnitude `0.00390625`; after-step FP32 scratch and index cache still agree.

Frozen-input CPU and TPU probes isolate the reason:

- The native vectorized pooling matches a NumPy FP64 softmax/weighted-sum
  rounded to BF16 **bitwise for all 512 channels**.
- The reference's hand-written serial FP32 sum differs by one BF16 ULP at
  pooling channel 39. Official PyTorch **CPU FP32** pooling produces that same
  value. This is a real floating-point reduction-order sensitivity, not proof
  that the CPU implementation is mathematically corrupt or a certification
  of official GPU bitwise equivalence.
- Replaying the serial reduction reproduces the frozen reference compressed
  value bitwise; vectorized pooling reproduces native pooled, normalized and
  final compressed values bitwise. Changing division order or adding a
  probability barrier does not fix the serial/vector difference.

The reference repair keeps the official vectorized `[batch, rows, channels]`
structure in both phases. A leading batch axis **alone is insufficient**:
inside the real dynamic scratch-update path, sliced concatenation still
mis-lowers on v5p (the new integrated unit test gives -0.0922852 instead of
-0.118164). Explicit per-row channel gathers remove that separate compiler
lowering hazard. This does not import native KV or call the native compressor;
reference cache construction remains independent.

The new test reconstructs the exact real FP32 channel using BF16 input/weight
projections, then compares whole prefill with token-by-token decode and NumPy
FP64. It fails against the original serial reference (-0.118652 versus
-0.118164), and passes with vectorized pooling plus gathers. Frozen failed
oracle values are retained. The reference arithmetic contract is explicitly
changed for phase consistency; we do not pretend the oracle was unchanged.
Neither the finite/top-1 checks nor the 0.005 full-logit NRMSE threshold changes.
CSA, index CSA and HCA phase/state tests also pass. These use exact dyadic
projection inputs to isolate state/pooling from the known GEMM-versus-GEMV
FP32 accumulation-order distinction; the separate batch-invariance tests
still use real-sized random inputs and unchanged strict FP32 checks.

`v4-7939-pool-order-20260907-03` runs the repaired reference compressor on
the frozen real hidden input and pre-cache, with original checkpoint weights.
Both main and index FP32 KV/scores and **all valid compressed vectors** match
the saved native post-cache bitwise (six state checks). It imports the frozen
state only for this isolated reproducer, not for the full independent oracle.
The same-source combined suite passes **164 tests**, zero failures or skips
(`v4-8023-kernels-final-20260907-04.xml`, 79.38 seconds).
The fresh complete native and independent 8K gates below complete this repair.

Current framework fingerprint (compressor, mHC and reference phase repair):
`510bfa6cad2ea7050280ef67eb3c561a31ee3bd9158083ca811e04ed510e0f3d`.
The fresh native run is `v4-8023-native-cold-fixed-20260907-04`; its dependent
independent run is `v4-8023-oracle-full-20260907-03`. Native code is unchanged
since the prior 1910-pass run, but the fingerprint includes reference source;
therefore the current receipts are regenerated without bypassing that guard.
The regenerated native report is **finished and complete: 1910 comparisons,
zero failures**, every row finite and top-1 equal. Its maximum NRMSE remains
`0.000058791898482013494`; original B2/request-1/position-8023 NRMSE remains
`1.2395231863138179e-7` (maximum absolute error `1.1444091796875e-5`). The
independent whole-prefix/decode report is also **finished and complete**:
both cases pass all **514 full-vocabulary rows bitwise**, with NRMSE and
maximum absolute error both zero, all values finite and top-1 equal.
Each case comprises one complete prefill and 256 decode steps; both independent
caches reach exactly 8192. No native prefix KV, scratch or hidden state is
imported by this final gate.
The new B1 goldens contain 514 rows of 129280 logits. Every element is bitwise
equal to the native goldens saved **before** the reference pooling change;
all rows are finite and all original pre-repair teacher-forcing token IDs
remain unchanged. Case-0 prefill also still matches the immutable pre-mHC-fix
whole-prefill expected array bitwise. See the local
`v4-8023-baseline-integrity-20260907-01.json` audit.

Repeat the full-model numerical gates sequentially, with only one TPU owner:

```bash
python scripts/run_deepseek_v4_8k_native.py \
  --checkpoint /path/to/original-checkpoint --cold-prefill --output /new/native
python scripts/validate_deepseek_v4_8k_oracle.py \
  --native-report /new/native/report.json --output /new/independent-oracle
```

Do not enable `--continue-on-numerical-failure` for acceptance. No profile is
requested here. The independent script refuses incomplete, failing or
different-source native goldens.

## Validation status

- New strict FP32 state-isolation tests: old code **3 failed** (CSA, index CSA,
  HCA); fixed code passes all three. Unlike the older synthetic allclose check,
  these require scratch to remain bitwise invariant when a request is added or
  reordered in the packed batch.
- Fixed-kernel first suite: **137 passed**, including the new regression and
  snapshot tests. Only existing non-failing Flax/JUnit warnings remain.
- Reporting/attribution regression: **17 passed** on CPU; together these cover
  all 146 earlier tests plus eight new invariance/snapshot tests (154 passed).
- Real position-8023 input, with identical historical logical state in B1/B2:
  compressed KV, FP32 KV/scores and page snapshots all match bitwise. This
  isolates the corrected operation; it is not a replacement for recomputing
  the full decode history with the fix.
- Candidate 1 history, not the current result: its fresh 43-layer B1/B2/B4
  regression was **interrupted after failing the old-B1 comparison**.
  Both B1 last-step independent checks passed bitwise, but those checks import
  the candidate's prefix cache and cannot certify its history. Both oracle and
  actual logits must be finite; top-1 must agree and NRMSE must remain <=0.005.
- Comparing the candidate's full B1 sequences with the original B1 reveals
  164 failing rows for case 0 (first position 8028, max NRMSE 0.4095113) and
  169 for case 1 (first position 8023, max NRMSE 0.4719146). Greedy tokens still
  agree. Rebuilding self-consistent goldens does not make this pass.
- Independent full prompt from empty cache plus all 256 decode steps:
  **failed before the repairs; now 514/514 bitwise passes**. This check imports
  no native prefix KV, scratch or hidden state.
- B2/B4 cold packed prefill from empty cache, comparing every 128-token boundary
  and subsequent decode against B1: **passed**, using `--cold-prefill` in the
  native runner. This closes the coverage gap left by restored B1 prefix pages.
- Current combined V4/framework/low-bit/numerical suite: **164 passed**, zero
  failures/skips. Shared paged allocator, radix cache and input-length suite:
  **35 passed** on CPU (`v4-8023-shared-cpu-final-20260907-01.xml`, 6.86 seconds).
- Serving lifecycle/prefix/pressure revalidation: **not rerun on this fixed
  source**; it is follow-up work, not established by these numerical fixtures.

No performance result is accepted on the strength of matching greedy tokens.
The old failed reports remain diagnostic history, not passing benchmarks.

## Local-only evidence

The following receipts and arrays are preserved under ignored
`GCP_login/results/` and on the existing TPU's results directory:

- `v4-8023-capture-20260907-01`: unchanged production before/after states.
- `v4-8023-replay-20260907-01`: 43-layer hidden/stage/cache comparisons and
  bitwise production fidelity checks.
- `v4-8023-compressor-probe-20260907-01`: causal state/projection swaps.
- `v4-8023-state-invariance-before-20260907-01.xml`: three expected pre-fix
  regression failures.
- `v4-8023-kernels-after-20260907-01.xml`: 137 passing fixed-kernel tests.
- `v4-8023-compressor-fixed-20260907-01`: bitwise real-state isolation check.
- `v4-8023-native-fixed-20260907-01`: interrupted/rejected first candidate.
- `v4-8023-native-fixed-20260907-02`: compressor-only fix, 2052 passing checks,
  including compatibility with old B1; not sufficient for whole-prefix acceptance.
- `v4-8023-mhc-post-probe-20260907-02`: real-input CPU/backend/fusion isolation.
- `v4-8023-mhc-rounding-before-20260907-01.xml`: two pre-fix rounding failures.
- `v4-8023-mhc-rounding-after-20260907-01.xml`: six rounding tests pass;
  broader suite has 56 passes and three existing non-Flash pre OOMs.
- `v4-8023-mhc-wide-baseline-20260907-01.xml`: same three OOMs with unmodified mHC.
- `v4-8023-whole-chunk-reference-20260907-02.json`: all 43 layers bitwise equal.
- `v4-8023-native-cold-fixed-20260907-03`: 1910 passing cold-cache comparisons.
- `v4-8023-oracle-full-20260907-02`: independent prefill passes, 7939 fails.
- `v4-7939-replay-20260907-01`: faithful native/reference 43-layer replay.
- `v4-7939-cpu-pool-20260907-01`: official CPU FP32 and NumPy FP64 adjudication.
- `v4-7939-pool-order-20260907-01`: frozen real TPU reduction-order sweep.
- `v4-7939-pooling-before-20260907-02.xml`: intended serial-reduction failure.
- `v4-7939-pooling-after-20260907-01.xml`: rejected vector-only lowering failure.
- `v4-7939-pooling-after-20260907-02.xml`: vector/gather real-fixture test passes.
- `v4-7939-pooling-after-20260907-03.xml`: all four pooling/phase tests pass.
- `v4-7939-pool-order-20260907-03`: repaired real reference state matches frozen native.
- `v4-8023-kernels-final-20260907-04.xml`: 164 same-source tests pass.
- `v4-8023-shared-cpu-final-20260907-01.xml`: 35 shared-framework CPU tests pass.
- `v4-8023-native-cold-fixed-20260907-04`: final-source native 1910/1910 pass.
- `v4-8023-oracle-full-20260907-03`: final-source independent 514/514 bitwise pass.
- `v4-8023-baseline-integrity-20260907-01.json`: native goldens unchanged by
  reference repair; original teacher forcing and immutable whole-prefill oracle retained.

The independent gate is `scripts/validate_deepseek_v4_8k_oracle.py`. No Git
commit or GitHub push is made; the existing unrelated dirty worktree is kept.
