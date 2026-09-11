# FP8 partial precision and retained final residual — 2026-09-11

## Outcome

The K=64 regression at position 107 is localized to K indices [320,384): its
partial dot loses 2^-30 before compensated summation. A smaller partial plus
residual-aware final BF16 rounding corrects this coordinate. On the two frozen
layer-0 `wq_b` input chunks, the K=8 diagnostic candidate reduces direct
high-precision-reference differences from **85 to 2 out of 4,161,536 valid
BF16 outputs**, with **no newly different outputs relative to baseline**.

This is a numerical diagnostic milestone, **not a production optimization or
model-correctness pass**. Runtime kernels, scheduler, official CPU oracle, and
acceptance thresholds are unchanged. Original model gate remains **15/18,
3 failed records**. No candidate is enabled in the serving model, and no new
43-layer forward/decode, HTTP, accuracy benchmark, or latency result is claimed.

## Setting and attribution

- Existing v5p-8 Spot VM / four physical v5p chips; JAX/jaxlib 0.11.1,
  libtpu 0.0.46.1. Official checkpoint revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`.
- Same original FP8/block-scale weights and activation FP8 roundtrip. Layer-0
  `wq_b`: K=1024, N=32768. Inputs are native positions 0–63 and 64–126; the
  latter has one zero padding row. No new independent prompt or layer here.
- Trace sixteen K=64 partials for the two implicated output-channel tiles,
  using the unchanged shared GMM and a diagnostic RHS adapter. Selected-column
  candidate replay and NumPy compensated reconstruction match the preceding
  full candidate FP32 outputs bitwise.

| Position/channel | Erroneous partial, zero-based K | Error | Interpretation |
|---|---|---:|---|
| 107 / 6431 | [320,384) | -2^-30 | Exact partial is FP32-representable; TPU partial differs by one FP32 ULP |
| 74 / 22505 | [384,448) | -2^-30 | Correctly rounded FP32 midpoint; nevertheless loses information |

All other fifteen partials at each coordinate are exact. Exact rational
summation verifies all 32 partials. These distinguish ordinary partial-output
rounding from accumulation error inside a partial; neither is repaired by
merely applying compensation after the partial has been rounded.

## Diagnostic candidate

`scripts/deepseek_v4_fp8_residual_candidate.py` adds an isolated adapter with:

1. Smaller dot products and compensated FP32 addition, retaining `high` and
   `low` components.
2. An optional final BF16 rounding carrier. If `high + low` rounds to an FP32
   number exactly on a BF16 midpoint, retain the sign of its lost residual by
   stepping the carrier one FP32 ULP in that direction before BF16 conversion.

The carrier is **only for subsequent BF16 conversion**. It is not a correctly
rounded FP32 sum and must not be used as a general FP32 linear result. The
method requires finite, non-overflowing inputs/intermediates. It does not
recover errors already present in the partials or guarantee exact accumulation.

For K=64 and K=16, exported high/low components reproduce the actual Pallas
candidate BF16 outputs with independent CPU rounding, over both complete
chunks. The non-residual K=64 form replays the previous FP32 candidate bitwise.
Standalone TPU casting matches the host conversion. K=16 matrix reruns also
match this diagnostic's carriers and BF16 outputs bitwise.

## Results

Different BF16 elements versus **direct FP64-to-BF16** arithmetic reference:

| Candidate | Positions 0–63 | Positions 64–126 | Newly different vs baseline, first/second chunk |
|---|---:|---:|---|
| Original | 38 | 47 | — |
| K=64 compensated, ordinary final conversion | 13 | 7 | 0 / 1 |
| K=64 compensated, retained final residual | 4 | 3 | 0 / 1 |
| K=16 compensated, ordinary final conversion | 11 | 6 | 0 / 0 |
| K=16 compensated, retained final residual | 2 | 1 | 0 / 0 |
| K=8 compensated, retained final residual | 1 | 1 | 0 / 0 |

K=16 residual-aware remaining coordinates are 19/11338, 61/10345, and
88/26881. The artifact `matrix-verification.json` records K=8's remaining
coordinates and exact sums. Every remaining K=16/K=8 coordinate is checked
using independent exact rational products/summation against the FP64 result.
The first diagnostic additionally rational-checks every disputed output in
its K=64/K=16 comparisons, including the original baseline discrepancies.

Matching direct FP64/BF16 is stricter than ordinary FP32 accumulation then
BF16. Consequently an improved direct result can disagree with the latter
staged reference more often. These element counts are **not** failed model
tokens, model tolerances, or a replacement for the official reference gate.

## Shape and physical-device controls

For both K=16 and K=8 residual-aware candidates, on both input chunks:

- First-eight-row M=8 results match corresponding M=64 carrier values bitwise;
  these subsets have zero direct-reference BF16 differences.
- Four explicit 8192-output-channel shards, one on each physical TPU, assemble
  bitwise to the single-device full projection (including carrier values).
- Baseline BF16 projections replay the historical native valid outputs bitwise.

These are independent output-channel projection shards. They do not exercise
full-model TP4/EP4, collectives, KV cache, scheduler, or concurrent requests.
The zero padding row is excluded from reported valid-output totals.

K=16 and K=8 respectively request 64 and 128 small dot products per K=1024
projection tile, versus one full-K dot in the original adapter, plus compensation.
Compiler lowering may transform these operations; no latency has been measured.
Do not infer speedup or acceptability for serving from numerical improvement.

## Verification and handoff

- Local suite: **98 passed, 5 skipped**; seven new residual-rounding tests cover
  midpoint offsets in both directions/signs and 16,384 random finite pairs.
  Ruff and `git diff --check` pass.
- New diagnostic/verification drivers:
  `scripts/debug_deepseek_v4_fp8_residuals.py`,
  `scripts/validate_deepseek_v4_fp8_residual_matrix.py`,
  `scripts/analyze_deepseek_v4_fp8_residual_matrix.py`.
- Runtime fingerprint remains
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`;
  archive verifies all 731 frozen source hashes and upstream inputs/helpers.
- Remote folders: `profiles/v4-fp8-residuals-20260911-01` and
  `profiles/v4-fp8-residual-matrix-20260911-01`, `-02`.
- Archive: `profiles/v4-fp8-residuals-20260911-evidence.tar.gz`; local copy,
  checksums and complete member/NPZ verification receipts:
  `GCP_login/results/v4-fp8-residuals-20260911/`.

Next isolate the two residual coordinates and broaden projection-level
validation beyond this layer. Use the higher-precision path as a diagnostic
control for downstream hidden-state/routing attribution before considering
serving integration. Sparse-attention/softmax differences remain separate.
