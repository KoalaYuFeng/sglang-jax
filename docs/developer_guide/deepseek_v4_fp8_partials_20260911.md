# FP8 split-K regressions and held-out projection controls — 2026-09-11

## Conclusion

The two new K=128 compensated-summation discrepancies are caused by rounding
inside individual partial dot products, not a broken compensation formula.
At both coordinates the affected partial is itself correctly rounded to FP32.
Once that partial has lost its low bits, even an exact sum of partials cannot
recover the original product sum. This is a finite-precision trade-off, not
proof of an erroneous BF16 cast or incorrect matrix multiplication semantics.

K=64 compensation corrects both coordinates and the original position-108
coordinate, but has one other new discrepancy in that input chunk. A held-out
chunk improves without new discrepancies at K=64. K=32 is not uniformly better.
**No candidate is promoted.** No scheduler, runtime kernel, quantizer, oracle,
or numerical gate changes. Historical acceptance stays **15/18, 3 failed
records**; no new 43-layer model, HTTP, or performance acceptance is claimed.

## Experimental setting

- Existing v5p-8 Spot VM, four physical v5p chips, JAX/jaxlib 0.11.1 and
  libtpu 0.0.46.1; official checkpoint revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`.
- Frozen native layer-0 `wq_b` inputs; original FP8 checkpoint tile decoder and
  activation FP8 roundtrip; K=1024, N=32768. This is a projection-only experiment.
- Attribution: positions 64–126 plus a zero row, M=64, tile M=32. Native and
  K=128-candidate FP32 replay both match last experiment bitwise, all outputs.
- Held-out projection inputs: positions 0–63, previously unused in split-K
  diagnosis. Its baseline BF16 output matches frozen native outputs bitwise.
  Also test its first eight rows with M tile 8; these rows are a shape control,
  not an independent additional input set.
- Four-device control explicitly places one 8192-output-channel shard per
  physical device and assembles the result. This is the independent column
  partition of this linear projection, **not** a full-model TP4/EP4 execution
  or a test of collectives, KV state, or request scheduling.
- Unchanged runtime fingerprint:
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.

## Attribution of the two regressions

Export each of eight K=128 partial dot products through the existing GMM using
a diagnostic adapter. Recombine in NumPy with explicit FP32 TwoSum-style
operations; separately sum those exported values in FP64.

Both reconstructions match the TPU compensated candidate FP32 output bitwise
over all 64 × 32768 elements. Thus in this replay there is no evidence that
compiler reassociation broke the compensation result. This tests exported
partials and full-result replay, not internal tracing of the unchanged binary.

| Position/channel | Partial K range (zero-based) | Partial rounding error | Consequence |
|---|---|---:|---|
| 65 / 12873 | [512,640) | +2^-30 | Final BF16 differs despite exact summation of the rounded partials |
| 102 / 20773 | [0,128) | -2^-31 | Final BF16 differs despite exact summation of the rounded partials |
| 108 / 7656 | All eight partials exact | 0 | Compensation recovers the original exact sum |

For the first two rows, the other seven partials are exact. The affected
partial in each case rounds correctly to FP32 (nearest-even); it lies on a
FP32 midpoint. Exact rational sums verify all eight partials at each of these
three coordinates. Frozen CPU counterexamples test that exact summation of
these correctly rounded partials really changes final BF16.

Across all exported partials (including the zero padding row), 394,772 differ
from exact FP64 partial sums, of which 116,557 differ even from correctly
rounded FP32 partial sums. These are element-level arithmetic comparisons,
not failed model tokens or violations of a model tolerance.

## Candidate and held-out comparisons

Counts use independent **direct FP64 → BF16** rounding, avoiding the diagnostic
double-rounding issue documented in the preceding experiment. They do not
replace the official CPU oracle or model acceptance criteria.

| Valid positions | Variant | Different BF16 elements | Baseline differences removed | Newly different elements |
|---|---|---:|---:|---:|
| 64–126 (2,064,384 elements) | Original | 47 | — | — |
| 64–126 | Compensated K=128 | 14 | 35 | 2 |
| 64–126 | Compensated K=64 | 7 | 41 | 1 |
| 64–126 | Compensated K=32 | 7 | 41 | 1 |
| 0–63 (2,097,152 elements) | Original | 38 | — | — |
| 0–63 | Compensated K=128 | 20 | 19 | 1 |
| 0–63 | Compensated K=64 | 13 | 25 | 0 |
| 0–63 | Compensated K=32 | 15 | 24 | 1 |

For positions 64–126, five differences remain even for a correctly rounded
FP32 sum followed by BF16 conversion. K=64/32 have two additional differences
against that staged reference: position 74/channel 22505 (already different
in baseline), and position 107/channel 6431 (new). The latter's baseline FP32
is exact; the candidate is lower by 2^-30. Neither shrinking K nor compensation
alone guarantees uniformly improved rounding.

In the held-out chunk, differences versus correctly rounded FP32 then BF16
are respectively 30, 12, 5, 7 for original/K128/K64/K32. For the first-eight-row
shape control, direct-reference differences are respectively 4, 1, 1, 2.
All variants' eight-row FP32 results match the corresponding rows of their
64-row results bitwise. All four variants' four-device assembled FP32 results
match their single-device full projections bitwise.

The held-out reference uses FP64 dot products and the independently tested
direct BF16 rounding helper. Exact-rational product verification in this wave
is scoped to the three attribution coordinates above, not every held-out
coordinate.

## Verification and evidence

- Local regression suite: **91 passed, 5 skipped**, including two new frozen
  partial-rounding mechanism controls. Ruff and `git diff --check` pass.
- Remote runs:
  `profiles/v4-fp8-partials-20260911-01` and
  `profiles/v4-fp8-split-matrix-20260911-01`.
- Scripts: `scripts/debug_deepseek_v4_fp8_partials.py` and
  `scripts/validate_deepseek_v4_fp8_split_matrix.py`; new tests:
  `python/sgl_jax/test/test_deepseek_v4_fp8_partial_rounding.py`.
- Evidence: `profiles/v4-fp8-partials-20260911-evidence.tar.gz`; local copy,
  SHA and verification receipts in `GCP_login/results/v4-fp8-partials-20260911/`.
  Archives retain full partial arrays and matrix results, verify frozen runtime
  and input hashes, and are verified locally before handoff.

Next investigate the remaining K=64 regressions on isolated partials and
evaluate any candidate against a broader projection matrix before considering
integration. Do not select a candidate solely because it matches one official
CPU shape or one rounding-sensitive coordinate. Sparse-attention/softmax
residuals and full-trajectory validation remain separate outstanding work.
