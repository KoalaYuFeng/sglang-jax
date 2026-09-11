# V4 position-108 FP8 projection accumulator diagnosis — 2026-09-11

## Outcome and limits

The dominant layer-0 `wq_b` discrepancy at position 108, channel 7656 is
localized to FP32 dot-product accumulation, not the final FP32-to-BF16 cast.
An isolated compensated split-K prototype corrects that coordinate, but
introduces discrepancies elsewhere; it is **not enabled in serving**.
These are finite-precision arithmetic differences, not evidence of corrupt
weights, an incorrect BF16 cast, or a CSA/HCA state defect.

No production runtime, official CPU oracle, scheduler, acceptance threshold,
or historical report was changed. Original acceptance remains **15/18 with
3 failed records**. This experiment does not establish complete-model accuracy,
TP4 correctness, performance, or throughput.

## Setting and replay controls

- Existing `sglang-jax-v5p8-spot`, v5p-8 / four physical chips; JAX/jaxlib
  0.11.1, libtpu 0.0.46.1. The isolated full projection uses default single-device
  placement on that VM, not a new four-chip distributed forward.
- Full official checkpoint revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`; layer-0 FP8 `wq_b` weights
  remain in original checkpoint storage until the existing tile decoder.
- Frozen native Q-latent inputs from positions 64–126, padded with one zero
  row: M=64, K=1024, N=32768. Compare only 63 valid rows / 2,064,384 outputs.
- Production baseline calls existing `fp8_linear`. Trace calls the same shared
  GMM, adapter, activation FP8 roundtrip, and `(32,1024,128)` tile, changing
  only its output type from BF16 to FP32 to expose the final accumulator.
  This is a replay-validated trace, not instrumentation of the original binary.
- Baseline projection matches all frozen native outputs for these positions
  bitwise. Casting the FP32 trace to BF16 reproduces all 64 rows of baseline
  bitwise. Standalone TPU and CPU FP32-to-BF16 casts match bitwise.
- Independent NumPy activation quantization matches the TPU helper bitwise.
  Independent weight decoding matches the TPU helper for both implicated
  128-channel tiles (16 and 59).
- Frozen runtime fingerprint:
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.

## Actual accumulator values

| Channel at position 108 | Traced FP32 | Independent FP64 sum | Interpretation |
|---|---:|---:|---|
| 2108 | -0.000003645196557044983 | -0.000003645196557044983 | Native is exact here; its BF16 rounding is supported |
| 7656 | 0.02716064453125 | 0.02716064639389515 | FP32 accumulation is lower by 2^-29 |

At channel 7656, `0.02716064453125` is exactly the midpoint between BF16
`0.027099609375` and `0.0272216796875`. The traced FP32 accumulator therefore
correctly rounds to the lower, even neighbor; the exact sum is one FP32 ULP
above the midpoint and rounds to the upper neighbor. Changing only the final
cast cannot recover information already lost in accumulation.

## Reference audit: avoid FP64 → FP32 → BF16 double rounding

The initial diagnostic used Torch's FP64-to-BF16 conversion. A midpoint
control showed that this path can round through FP32. Five output coordinates
in this dataset consequently differ from direct FP64-to-BF16 rounding.
Raw initial reports are retained, but their counts (42, 9, 17) are superseded
for high-precision adjudication by `direct-rounding/report.json` below.
The two position-108 coordinates and the baseline replay conclusions do not
change.

Added an independent power-of-two/significand rounding helper, tested below,
without altering any existing official CPU reference. For all 51 coordinates
where any tested variant disagrees, exact rational summation verifies that the
FP64 dot result is exactly representable and correct. Exact distances to the
adjacent BF16 values verify nearest-even rounding at every such coordinate.

## Isolated candidate comparison

Counts compare BF16 outputs with the independently adjudicated direct result;
they are not model acceptance failures or percentages of bad model tokens.

| Variant | Different BF16 elements | Baseline differences removed | Newly different elements |
|---|---:|---:|---:|
| Original M=8,16,32,64,128 (each) | 47 | 0 | 0 |
| K=128 split, plain accumulation | 47 | 0 | 0 |
| K=128 split, compensated accumulation | 14 | 35 | 2 |
| K=256 split, compensated accumulation | 22 | 28 | 3 |

Every original M tile and the plain split prototype reproduce baseline FP32
values bitwise. Both compensated variants recover the exact FP32 value at
position 108/channel 7656, while keeping channel 2108 correct.

Of the 47 baseline differences, 42 involve an incorrectly rounded FP32 sum;
5 arise even with a correctly rounded FP32 sum followed by BF16 conversion
(double rounding). The corresponding decomposition is 9+5 for K=128
compensated and 17+5 for K=256 compensated. Thus matching direct FP64/BF16
everywhere is stricter than ordinary FP32 accumulation followed by BF16;
these counts must not be presented as proof that every differing element is
an implementation bug.

The prototype uses short dot products and TwoSum-style residual additions
inside a diagnostic adapter. The first attempt failed because Pallas TPU
does not lower `optimization_barrier`; that failure and source are retained.
The successful run omits the unsupported barrier. We judge its measured
outputs, not assume compiler-preserved exact TwoSum semantics. No latency
measurement or claim of an efficient production implementation is made.

## Verification, artifacts, next step

- Local tests: **89 passed, 5 skipped**. Seven new tests cover both signs of
  every adjacent finite positive BF16 midpoint, each midpoint's FP64 neighbors,
  exact representable values, signed zero, and invalid/out-of-range inputs.
- Ruff and `git diff --check` pass for this wave.
- Preserve both remote experiment folders:
  `profiles/v4-fp8-accumulator-20260911-01` (partial compile failure) and
  `profiles/v4-fp8-accumulator-20260911-02` (eight completed variants and
  authoritative `direct-rounding` re-adjudication).
- Runners: `scripts/debug_deepseek_v4_fp8_accumulator.py`,
  `scripts/analyze_deepseek_v4_fp8_accumulator.py`; diagnostic reference:
  `scripts/deepseek_v4_bf16_reference.py`.
- Evidence archive: `profiles/v4-fp8-accumulator-20260911-evidence.tar.gz`;
  local copy and SHA/verification receipts:
  `GCP_login/results/v4-fp8-accumulator-20260911/`.
  Archive verifies all 731 frozen runtime source files and both upstream input
  hashes; local verification checks every member hash and NPZ integrity.

Next isolate the new compensated-accumulation regressions and establish a
broader same-input projection matrix before considering a runtime change.
The independently observed sparse-attention/softmax residual remains a
separate issue. Existing Qnorm/exp candidates remain unpromoted. Do not force
CPU router choices or infer that correcting this one Q coordinate resolves
the multi-layer trajectory or the original 43-layer numerical gate.
