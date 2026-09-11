# V4 standalone attention candidate validation — 2026-09-11

## Outcome

The compensated Q-normalization sum passed the expanded shape and directed
rounding checks. The more precise softmax exp is **not uniformly better at the
attention output**: 14 of 384 diagnostic cases have higher NRMSE against an
independent FP64 adjudicator. Neither candidate is enabled in serving code.

Historical acceptance remains **15/18 cases with 3 failed records**. These
experiments do not replace that gate, relax its thresholds, or establish
43-layer model accuracy. No performance or HTTP benchmark was run.

This extends the [attention attribution report](deepseek_v4_cpu_attention_debug_20260911.md).

## Experimental setting

- Checkpoint: official DeepSeek-V4-Flash revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`, original low-bit storage.
- Hardware: one v5p-8 Spot VM, four physical TPU v5p chips; CPU-only reference.
- Code: `integration/deepseek-v4`, base `270dfcbd`, preserving prior local fixes.
- Runtime fingerprint unchanged:
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.
- Environment: JAX/jaxlib 0.11.1, libtpu 0.0.46.1, PyTorch 2.14.0+cpu,
  NumPy 2.2.6.
- Inputs: captured native hidden states from layers 0–3 of the previous B1
  trajectory, 127 prefill tokens plus five teacher-forced decode tokens. These
  are real checkpoint activations generated from fixed random vocabulary IDs,
  not natural-language benchmark examples or four independent requests.
- Production mHC/pre-normalization and FP8 projections generate raw Q and KV
  with the original 64+63 padded-prefill and decode-one geometry. Layer 0's
  first 127 raw Q/KV outputs reproduce the prior actual capture bitwise.

The first matrix attempt failed in the diagnostic harness because Pallas did
not support its automatic partitioning. Explicit `shard_map` corrected the
harness; the failed attempt is retained. No production kernel was changed.

## Q-normalization

The candidate retains the Pallas implementation and BF16 rounding boundaries,
replacing only the FP32 pairwise sum with a compensated pairwise sum. A private
function clone in `scripts/deepseek_v4_attention_candidates.py` isolates it;
other normalization calls and the serving module are not patched.

Four layers × two position offsets produce eight reference cases. Each case
also tests chunk sizes 1, 4, 8, 16, 32, 64, 127 and 128, both arithmetic
variants, and replicated 64-head versus TP4 head-sharded execution (16 local
heads). **256/256 reassembled outputs are bitwise identical to the matching
variant's canonical output.** There are no structural failures.

Position offsets are 0 and 8060; the latter exercises rotary positions
8060–8191 on the same short-trajectory inputs. This is not an 8K forward,
8K KV-cache test, or scheduler/chunk-state regression.

Against the independently FP64-summed, BF16-rounded non-rotary normalization
reference, the candidate fixes 416 elements in layer 0, with no new
discrepancies. Layers 1–3 were already exact in these non-rotary channels and
remain so. The repeated offset uses the same non-rotary inputs and must not be
counted as another 416 unique fixes. Full Q retains small rotary differences.

Full Q versus CPU, offset 0 (NRMSE, percent):

| Layer | Baseline | Compensated sum |
| --- | ---: | ---: |
| 0 | 0.00679727% | 0.00077467% |
| 1 | 0.00046029% | 0.00046029% |
| 2 | 0.00021010% | 0.00021010% |
| 3 | 0.00002626% | 0.00002626% |

A separate directed stress test uses the real layer-0 token-109/head-3 vector
and changes each of its 512 channels by ±1 BF16 magnitude ULP: **1,025
vectors**, including the original. In the 448 non-rotary channels, baseline
has 22 discrepant vectors; the candidate has **zero**, bitwise matching the
FP64-rounded reference, with zero new discrepancies. These are synthetic
perturbations of one real vector, not 1,025 independently captured tokens.

## Softmax exp: improvements and retained counterexamples

The standalone candidate preserves the 64-key online-softmax structure and
substitutes the existing range-reduced `_exp_nonpositive` for exp. It accepts
nonempty slot vectors and pads incomplete blocks with masked indices.

The 384 cases cover four layers, 1 or 4 queries, slot counts
1/63/64/65/127/128/129/256, causal-prefix/holes/all-masked patterns, and the
learned sink or a +80 sink stress. Both variants receive the same baseline Q
and KV, isolating exp from the Q-normalization candidate. All-masked outputs
are exactly zero. The 4-query cases are operator shapes, not Engine batch 4.

These are generic block-attention tests using real Q/KV from those layers,
**not** a test of actual compressed CSA/HCA routing, long-context state,
distributed softmax, or scheduler behavior. CPU reference is official Python
arithmetic through the existing CPU adaptation, not an official GPU run.

| Output NRMSE comparison | Candidate lower | Equal | Candidate higher |
| --- | ---: | ---: | ---: |
| Versus unchanged CPU reference | 214 | 151 | 19 |
| Versus independent FP64 adjudicator | 219 | 151 | 14 |

These are relative error comparisons, **not accuracy pass/fail counts**.
The FP64 adjudicator uses independent NumPy dots, exp and accumulators while
retaining FP32 score boundaries, 64-key blocks, BF16 probability/output
boundaries and the sink. It intentionally differs from official execution
arithmetic and is not a replacement acceptance oracle. All 384 CPU metric
records were exactly reproduced during the second adjudication run.

Among the 192 unshifted-sink cases, the worst aggregate CPU NRMSE falls from
0.028324% to 0.014815%, and worst head-vector NRMSE from 0.299326% to 0.139136%.
That worst-case improvement does not eliminate individual regressions.

### Reproduced counterexample

Layer 0, query position 131, 64 slots with holes, original learned sink:

- CPU output is bitwise equal to the FP64 adjudicator output in this case.
- Baseline output NRMSE: **0.010862%**; exp candidate: **0.014815%**.
- Maximum absolute error is 0.00390625 for both variants.
- Instrumented baseline and candidate outputs reproduce their saved outputs
  bitwise; QK scores are identical between variants.
- Given those actual QK scores, baseline has two BF16 probability discrepancies
  against FP64 exp; the candidate has zero.
- Against probabilities formed from independently FP64-computed QK scores,
  each variant still has one BF16 probability discrepancy. There are 1,316
  live score elements differing from the FP64-dot/FP32-round scores.

Thus more accurate local exp does not guarantee a closer final BF16 attention
output when QK rounding, probability casting and value accumulation interact.
The trace establishes this observation, not a unique attribution of the
final error to numerator, denominator, or cancellation. All 14 counterexamples
are retained; no threshold was selected to hide them.

## Tests, evidence and next gate

Local CPU suite: **82 passed, 5 TPU-only skips**. This includes 18 new
candidate/padding/mask tests and 13 independent FP64-adjudicator tests, plus
existing attention-diagnostic, numerical-acceptance, CSA-pool and RoPE tests.
The four new public helper/test files pass Ruff. This wave modifies only
standalone diagnostics, their tests and this report, not the serving runtime.

Remote evidence under `/mnt/disks/deepseek-models/profiles/`:

- `v4-attention-candidate-matrix-20260911-01` (retained harness failure).
- `v4-attention-candidate-matrix-20260911-02` (completed matrix and raw Q/KV).
- `v4-attention-matrix-fp64-20260911-01` (all cases and 14 counterexamples).
- `v4-qnorm-midpoint-stress-20260911-01` (all directed vectors/outputs).
- `v4-exp-regression-20260911-01` (counterexample intermediate traces).

Archive: `v4-attention-candidate-matrix-20260911-evidence.tar.gz`, with local
copy, checksum receipt, JUnit and verification result under
`GCP_login/results/v4-attention-candidate-matrix-20260911/`. The archive
preserves reports, arrays, logs, diagnostic source, tests and the unchanged
731-file runtime manifest. Its checksum is recorded in the receipt and local
`GCP_login/CURRENT_TPU.md`, outside this self-archived report.

Upstream trajectory evidence checksum:
`cac91c1cfc2c8d1cd6b39af1748ca93c7ce3ea056de3b6e21c93303b57469b92`.
Upstream attention attribution evidence checksum:
`1f7318f550de5cf745c5027cab9562f25868fe7727d64bd7fae05f960f2aa81c`.

Next, test **compensated Q-normalization alone** through the isolated full
attention and independent layer/logits trajectories before promotion. Keep
the exp candidate separate until its counterexamples and downstream effects
are resolved. Complete 43-layer, 8K, multi-request and performance regressions
remain open; neither candidate is described as a completed model fix.
