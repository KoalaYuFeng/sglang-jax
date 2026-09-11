# V4 CSA index pooling integration, 2026-09-11

Follow-up to [the RoPE recovery regression](deepseek_v4_numerical_recovery_20260911.md).
This change connects the independently tested pool primitive inside the original
CSA Pallas emitter. Full-model accuracy and performance are not certified by
these operator tests.

## Implementation boundary

- `kernels/csa/compressor.py::_pool_uniform_groups` dispatches only V4
  128-channel index windows to `csa/numerics.py::index_pool_v4`.
- Both selected-window and raw-overlap emitters use this branch. No outer
  framework gather, fallback, scheduler modification, or new state owner.
- CSA main KV (512 channels), HCA, and legacy numerical modes retain their
  existing pooling arithmetic. BF16 rounding, RMSNorm, RoPE, Hadamard and FP4
  quantization remain in their existing stages.
- Pooling remains device FP32: range-reduced exponential, unnormalized weighted
  terms, compensated eight-term reduction, and one final division. No FP64,
  host callback, PyTorch dependency, or unsupported Pallas optimization barrier
  in the runtime primitive.
- Compensation adds arithmetic. Its latency has **not** been measured here;
  this is a correctness change, not a demonstrated performance improvement.

## Integration uncovered an additional cancellation case

The first integration attempt passed 17/18 selected-emitter/state tests. Its
only failure compared the corrected index pool against the old JAX softmax
algorithm bitwise: six BF16 output elements differed in the 65-group fixture.

For index pooling, the test's expected pooled BF16 input was changed to the
already available, independently computed NumPy FP64 formula, followed by the
same retained JAX normalization/RoPE. The main-KV expectation was untouched;
the separate end-to-end FP64 tolerance remained 2e-4. The frozen official-Python
reference and its thresholds were **not changed**.

That independent expectation still failed on one element. The retained failed
run and stage probe isolate group 34/channel 41:

| Pooling result, before BF16 | Value |
| --- | ---: |
| Old TPU softmax-weighted sum | -0.0016657114028930664 |
| Polynomial exp + ordinary sum | -0.0016670170007273555 |
| Independent same-input FP64 | -0.001667116420160878 |

The latter two round to different BF16 values. TPU Pallas and XLA reproduce
the same pooled value; normalization agrees bitwise when supplied identical
pooled BF16 values. Thus this fixture does not implicate normalization, page
ownership, or tile padding.

An isolated four-way experiment compared the original polynomial primitive,
compensated summation, a corrected score subtraction, and both corrections.
Compensated summation alone removes the boundary failure without adding the
score-subtraction change. It also preserves same-input FP64 BF16 equality for
the four frozen real-checkpoint windows and three independent random sets.
The implementation therefore adds only compensation to the existing primitive.

The entire 65-group fixture is now a standalone regression at both tile sizes
1 and 4, including a partial final tile. This does not claim correctly rounded
BF16 results for every possible FP32 input; exponential/product/input rounding
still exists. Randomized error bounds and frozen numerical gates remain separate.

## Validation results

Runtime fingerprint:
`1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.
Branch `integration/deepseek-v4`, base `270dfcbd`, retaining the existing EP and
RoPE fixes. One v5p-8 Spot, four physical chips; exact restored dependency freeze.

- Local CPU pool/RoPE tests: **28 passed**.
- TPU-environment tests: **74 passed**: 46 CSA integration tests, 12 standalone
  pool tests, and 16 RoPE tests. The CSA selection includes state/chunk/batch,
  fork/reuse, raw-versus-selected ABI, traced/untraced execution, projection,
  index scores/top-k at 8K and 8320, and joint attention. This is not a
  full-model top-k or logits test.
- Frozen official-Python/CPU matrix: **15/18 PASS, 8436 metric records,
  3 failed records**. The previous RoPE-only result was **16/18 and 2 failed
  records**. The integration does **not** pass the original acceptance gate.
- All 12 CSA-main/HCA cases pass and their final candidate caches remain
  bitwise equal to the RoPE-only run. All live-state arrays, reference arrays,
  input/checkpoint hashes, chunk boundaries, and limits remain unchanged.
  All nine same-input B1/B4 cache pairs are bitwise equal.

| Case | Status | Worst compressed-vector NRMSE |
| --- | --- | ---: |
| CSA main, layers 2/22/42, B1/B4 | 6/6 PASS | Within unchanged 2.5% gate |
| HCA main, layers 3/23/41, B1/B4 | 6/6 PASS | Within unchanged 2.5% gate |
| CSA index, layer 42, B1/B4 | 2/2 PASS | 0% |
| CSA index, layer 22, B1 | PASS | 2.56410% |
| CSA index, layer 2, B1 | NUMERICAL_DIFFERENCE | 3.77695% |
| CSA index, layer 2, B4 | NUMERICAL_DIFFERENCE | 4.18030% |
| CSA index, layer 22, B4 | NUMERICAL_DIFFERENCE | 4.69841% |

The index limit remains **3.5%**, for both aggregate and per-vector NRMSE.
Aggregate improvement is not allowed to conceal a failing vector.

## Real-window replay and adjudication

| Layer / request / terminal position | Before → after vector NRMSE | Finding |
| --- | ---: | --- |
| 2 / 2 / 6575 | 3.64662% → 0% | Original pool-sensitive failure corrected |
| 2 / 3 / 7715 | 4.62250% → 0% | Original pool-sensitive failure corrected |
| 2 / 0 / 2571 | 0% → 3.77695% | Newly visible CPU-reference pooling boundary |
| 2 / 3 / 3427 | 4.18030% → 4.18030% | Existing projection/reference sensitivity |
| 22 / 2 / 1091 | 4.69841% → 4.69841% | Existing projection/reference sensitivity |

The new layer-2/request-0 vector appears in both B1 and B4, not only a batched
execution. A traced replay of the entire layer-2/B4 case reproduces every final
array of the untraced integrated run byte for byte, with **no phase override**.

At position 2571/channel 6, independent FP64 pooling of the native projection
window is `3.789062713746396`. The native FP32 pool rounds to BF16 `3.796875`;
the CPU reference had rounded to `3.78125`. Independent FP64 pooling of the
**CPU reference's own projection inputs**, FP64 full projection + pooling,
and FP64 projection rounded to FP32 before pooling all support `3.796875`.
Their downstream final vector reproduces the same 3.77695% discrepancy against
the CPU reference. A standalone regression now preserves this exact channel's
eight values/scores at tile sizes 1 and 4, without requiring a model download.

For all four replayed layer-2 windows (groups 642/856/1643/1928), every native
pooled BF16 element equals independent same-input FP64. Given the same pooled
BF16 input, retained normalization equals the emitter bitwise. The existing
layer-22 projection adjudication remains applicable because its captured state
and offending final vector are unchanged. These results support the corrected
arithmetic in these windows; they do **not** prove official GPU equivalence.

## Remaining acceptance boundary

This matrix uses official Python control flow with independent **CPU** shims,
embedding-derived BF16 activations, and four-way **replicated compressors**.
It is not a full 43-layer TP4/EP4 inference trajectory. The three failed metric
records are retained as failed, including the newly failing B1 case.

Do not introduce CPU-rounding special cases or loosen the gate to hide this
discrepancy. The next accuracy step needs independently obtained official
execution outputs and real-layer/index-top-k/logits comparisons to distinguish
deployment equivalence from this CPU reference's rounding artifacts. Full
43-layer 8K forward/decode acceptance and performance remain unverified for this
source. No inference server has been deployed with this change.

## Evidence and reproducibility

Directories under `/mnt/disks/deepseek-models/profiles/`:

- `v4-index-pool-integration-20260911-01`: old-algorithm bitwise test failure.
- `v4-index-pool-integration-20260911-02`: independent-FP64 test failure and
  exported pool/normalization stages.
- `v4-index-pool-compensation-20260911-01`: isolated arithmetic alternatives.
- `v4-index-pool-integration-20260911-03`: compensated integration, frozen
  18-case matrix, unit logs and audits.
- `v4-index-pool-residual-20260911-01`: faithful replay, raw/stage arrays,
  same-input pool comparison, and full-projection FP64 adjudication.

Original frozen midpoint evidence SHA-256:
`521582e7815418122b1aa78abc888b79d41fc72c29d8e9f42a20cd37103deac3`.
The original checkpoint is unchanged; FP4 here describes compressed **index
activations**, not expert-weight loading. No HTTP server or benchmark was run.
