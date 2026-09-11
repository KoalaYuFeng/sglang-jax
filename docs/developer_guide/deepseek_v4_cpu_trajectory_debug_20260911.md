# V4 CPU/TPU numerical trajectory diagnosis — 2026-09-11

## Status and scope

Historical acceptance remains **15/18 cases, 3 failed records**. This investigation
does not relabel those failures, change thresholds, or change serving code.
It adds independent pre-quantization measurements and real production-layer
CPU/TPU comparisons. There is no full-model accuracy acceptance yet.

- Checkpoint: official DeepSeek-V4-Flash revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`.
- Runtime: SGLang-JAX `integration/deepseek-v4`, base `270dfcbd` with the existing
  EP/RoPE/compensated-index-pool changes preserved. Runtime fingerprint:
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.
- Hardware: one v5p-8 Spot, four physical chips; EP4/DP1. Head TP4 for CSA/HCA;
  the first two SWA layers retain the production replicated-head path.
- Production `DeepseekV4DecoderLayer`, Pallas mHC/CSA/HCA, `gmm_tuned` FP4 MoE,
  FP8 GMM, fused norm, merged projections, fused inverse-RoPE/wo_a. Compact
  FP4 scales use the tuned backend's transposed layout. Original packed weights
  remain packed; no whole BF16 expert collection is created on TPU.
- Reference: unchanged official Python with independent CPU-only GPU-kernel
  shims, torch 2.14.0+cpu, eight CPU threads; **not an official GPU execution**.

The trajectory is batch 1, layers 0–3, 127 prefill tokens plus 5 teacher-forced
decode inputs (positions 127–131), TPU chunks 64+63 (one padding row in the last
chunk), CPU one-shot prefill. The 132 fixed random vocabulary IDs use seed
20260912 and real checkpoint embeddings. These are not natural-language
benchmark prompts. Each side independently propagates its own hidden state.
Execution is causal and layer-major, staging one layer at a time; it is not a
whole-model compiled/Engine/HTTP test or performance measurement.

An initial harness run stopped before computation because it supplied the
untransposed scale layout to `gmm_tuned`. The second run corrected only that
harness setting. Both records are retained.

## Frozen FP4 residual attribution

An independent NumPy E2M1/UE8M0 quantizer checks each side's output against its
own input, including ties-to-even and signed zero. Attribution separately
reports upstream error, scale changes, adjacent-bin crossings, incorrect
quantization, and unexplained changes. It deliberately has no `passed` result.

NRMSE below is per failed 128-channel vector; positions and layers are zero-based.

| Layer / request / position | Before FP4 | After FP4 | Changed elements |
| --- | ---: | ---: | ---: |
| 2 / 0 / 2571 | 0.17550% | 3.77695% | 2 |
| 2 / 3 / 3427 | 0.23928% | 4.18030% | 1 |
| 22 / 2 / 1091 | 0.07780% | 4.69841% | 1 |

Across five frozen windows / twenty request vectors, both sides' quantizers
match the independent quantizer bitwise. All four changed output elements are
same-scale, adjacent-bin boundary crossings; there are zero scale changes and
zero unexplained changes. The previously repaired positions 6575 and 7715
have identical final FP4 outputs. Raw projection NRMSE across these windows is
approximately 1.57e-7–1.89e-7, before BF16 and FP4 boundary amplification.

This supports the earlier [CPU projection-shape diagnosis](deepseek_v4_cpu_reference_shape_debug_20260911.md).
It explains these local residuals, **not their downstream harmlessness**.

## Independently propagated production-layer results

| Layer | Type | All-token hidden NRMSE | Maximum hidden-vector NRMSE | Decode hidden NRMSE |
| --- | --- | ---: | ---: | ---: |
| 0 | SWA | 1.0027% | 3.3687% | 1.0414% |
| 1 | SWA | 1.9259% | 5.3486% | 1.7238% |
| 2 | CSA | 2.5428% | 7.1917% | 1.9894% |
| 3 | HCA | 4.1767% | 32.8773% | 2.7247% |

All tensors are finite, but this is not a pass decision. The real checkpoint
head, applied diagnostically after just four layers, gives logits NRMSE 1.5586%
and top-1 agreement 6/6. **These are truncated-model logits, not the deployed
43-layer model's logits or an accuracy score.**

## Where the amplification occurs

Instrumentation calls the actual production class with temporary in-process
hooks, without editing runtime files. For both inspected layers, instrumented
outputs reproduce the original uninstrumented prefill outputs bitwise; CPU
hooked outputs also reproduce the original CPU prefill bitwise.

| Layer / stage | Independent trajectory NRMSE | Same-input local CPU comparison |
| --- | ---: | ---: |
| 0 attention | 0.1830% | 0.1830% |
| 0 MoE input after norm | 0.1957% | — |
| 0 MoE output | 1.1482% | 0.01525% |
| 3 attention | 4.3030% | 0.18110% |
| 3 MoE input after norm | 3.1596% | — |
| 3 MoE output | 9.9258% | 0.03865% |

Same-input MoE maximum vector NRMSE is 0.1186% in layer 0 and 0.2402% in layer 3.
The same-input expert selections agree in both layers; routing-weight NRMSE is
3.87e-7 and 5.11e-7 respectively. Same-input norm and mHC checks are also saved;
their residual errors are much smaller than the independently propagated MoE
output difference. No new local acceptance threshold was inferred from these
observations.

Layer 0 hash routing agrees even in independent trajectories. In layer 3,
15/127 prefill positions choose different expert sets after propagated upstream
perturbations. At position 108, the first five experts agree; the sixth is 214
on TPU versus 199 on CPU. This position has the largest layer-output difference
and occurs **before** the first HCA compressed vector at position 127. It cannot
be attributed to a prior HCA compressor-state update in this short fixture.

Thus the large end-to-end MoE error is not reproduced as a large same-input
MoE-kernel error. Small upstream errors are amplified by low-bit activation
processing and later discrete routing changes. The current evidence does not
establish whether every upstream arithmetic difference is acceptable; in
particular, the approximately 0.18% same-input attention difference still needs
projection/QK/softmax/output-projection attribution before accepting the model.

## CPU-only prefill/decode split control

The unchanged CPU reference was run again on the same 132 IDs and checkpoint,
but with 128 prefill + 4 decode instead of 127 + 5. Its hidden states propagate
independently; neither TPU values nor a revised oracle are injected. This
changes projection/grouped-expert shapes and moves the first compression
boundary from decode into prefill. It is a diagnostic A/B, not a substitute
reference or a new acceptance limit.

After four layers, CPU–CPU hidden NRMSE is **3.5365%**, maximum vector NRMSE
**32.9154%**; TPU versus this alternate CPU trajectory is **4.2212%**. The two
CPU truncated-head logits differ by **1.4344%** NRMSE, with 6/6 top-1 agreement.
The CPU reference therefore exhibits nontrivial trajectory sensitivity itself.
This prevents attributing every large propagated hidden difference uniquely to
a TPU kernel bug, but does not prove all TPU differences harmless or justify
using the CPU–CPU error as a tolerance.

## Acceptance implications

1. Keep the original strict matrix and failed records unchanged.
2. Require a correct same-input quantizer; a boundary label alone cannot pass
   a case with large upstream error, changed scales, or downstream divergence.
3. Separate independently propagated trajectories from identical-input local
   operator controls. Expert routing changes need explicit reporting.
4. Do not extrapolate four random-token layers or six truncated-head positions
   to 43-layer accuracy, 8K, batching, or official benchmark equivalence.

## Reproduction and evidence

New public diagnostics:

- `scripts/validate_deepseek_v4_cpu_trajectory.py`
- `scripts/debug_deepseek_v4_cpu_layer_stages.py`
- `scripts/deepseek_v4_numerical_acceptance.py`
- `python/sgl_jax/test/test_deepseek_v4_numerical_acceptance.py`

Successful trajectory: `profiles/v4-cpu-trajectory-20260911-02` under the
independent data disk. Frozen attribution: `v4-frozen-boundaries-20260911-01`.
Layer 3/0 instrumentation: `v4-cpu-stages-20260911-01` / `-02`; same-input
attention/mHC/norm controls: `v4-cpu-local-stages-20260911-01` / `-02`.
CPU-only split control: `v4-cpu-trajectory-split-20260911-01`.

Local CPU tests: **35 passed, 5 skipped** across the new boundary-analysis
tests and existing index-pool/RoPE tests; the skips require a TPU path. Ruff
passes for the three new public diagnostic scripts and boundary tests.
No fresh full 43-layer, 8K, multi-request, or performance gate was run.

Evidence archive: `profiles/v4-cpu-trajectory-20260911-evidence.tar.gz`, with
per-file SHA-256 manifest, original runtime source manifest, test XML, logs,
input IDs, hidden/logits arrays, instrumentation captures, CPU controls,
diagnostic scripts and this report. The local copy and verification receipt
are under `GCP_login/results/v4-cpu-trajectory-20260911/` (Git-ignored).
