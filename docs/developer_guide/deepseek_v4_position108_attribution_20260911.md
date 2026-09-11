# Position 108: route-flip amplification — 2026-09-11

## Conclusion

The dominant layer-3 discrepancy at position 108 is amplified by a change in
the sixth selected expert: **214 on the native trajectory, 199 on the original
CPU trajectory**. This is supported by bidirectional route interventions,
not only a correlation between routing and output error.

For identical native MoE input, official CPU MoE reproduces the actual native
output at this position **bitwise**. FP64 gate calculations support each
trajectory's own selected experts. The alternate CPU-only trajectory also
selects 214 and reproduces its saved outputs bitwise. These observations do
not support treating this position as a same-input MoE/router kernel defect.

The upstream input perturbation is **not fully resolved**. It exists by layer
0 and accumulates before the layer-3 router. This experiment localizes its
dominant amplification, not the first arithmetic cause or full-model safety.
No runtime changes, forced production routes, new thresholds, reference
replacement or accuracy acceptance are introduced. Historical acceptance
remains **15/18 cases, three failed records**.

## Scope and reproducibility

- Official checkpoint revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`.
- Frozen native production trajectory from four physical v5p chips, layers
  0–3, B1, 127 prefill tokens (64+63 native chunks) plus five decode inputs.
  TP4 attention for CSA/HCA, EP4/DP1, tuned GMM MoE and original low-bit weights.
- This wave executes **CPU-only** controls on the existing VM. It uses the
  unchanged official Python MoE/mHC control flow and existing CPU kernel
  adaptations, not an official GPU execution or a new TPU benchmark.
- Full 127-token prefill shapes are retained for MoE replay. Only the route
  of zero-based position 108 is overridden in the counterfactual cases.
- Runtime fingerprint remains
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`;
  branch `integration/deepseek-v4`, base `270dfcbd`, prior changes preserved.

Both ordinary CPU MoE runs reproduce their frozen full-prefill results
bitwise: original CPU input versus original CPU MoE, and native input versus
the earlier same-input CPU MoE capture. CPU residual/mHC reconstruction also
reproduces the frozen CPU layer output bitwise. Native-input CPU MoE followed
by the native-side residual/post/comb data reproduces the actual native final
output bitwise **at position 108**; this last claim is not extended to every
prefill position.

The previous [Q-only experiment](deepseek_v4_qnorm_trajectory_20260911.md)
changed only positions 109 onward. Position 108 is therefore unaffected by
that candidate and is studied here using the original frozen execution.

## Where the error grows

NRMSE against the original independent CPU trajectory, percent. A layer token
flattens four residual streams; a MoE/attention vector has 4096 channels.

| Position-108 stage | NRMSE |
| --- | ---: |
| Initial embedding/residual input | 0%, bitwise |
| Layer 0 output | 1.318420% |
| Layer 1 output | 1.793300% |
| Layer 2 output / layer 3 input | 2.102687% |
| Layer 3 normalized attention input | 2.911876% |
| Layer 3 attention output | 3.975133% |
| Layer 3 normalized MoE input | 2.505667% |
| Layer 3 MoE output | 39.732625% |
| Layer 3 final output, four streams together | 13.914685% |

The worst individual residual-stream NRMSE in the final row is 32.877314%.
Do not confuse it with the 13.914685% flattened-token measure.

Local controls at the same position:

- Attention with exactly the native attention input: **0.040709%** NRMSE.
- MoE with exactly the native MoE input: **0%, bitwise**.

Layer 3 is HCA with compression ratio 128. Position 108 precedes the first
completed compressed vector at position 127; a previously emitted HCA
compressed vector cannot explain this position's error in this short fixture.
This is not a general proof that every HCA/cache path is correct.

## Gate scores: different inputs, not a true top-k tie

The common first five experts are `37, 80, 114, 173, 117`.
Define the decision margin as adjusted score(214) minus adjusted score(199),
including gate bias, which affects selection but not routing weights.

| Input | CPU FP32 margin | Independent FP64 margin | Sixth expert |
| --- | ---: | ---: | ---: |
| Captured native MoE input | +0.000579833984 | +0.000580129831 | 214 |
| Captured original CPU MoE input | −0.000068664551 | −0.000068620199 | 199 |

CPU gating on the native input reproduces all captured prefill expert IDs;
CPU gating on the CPU input reproduces its captured IDs. At position 108,
independent FP64 dot products and softplus/sqrt agree with both complete top-6
orders. Consequently a tie-breaking change or merely higher-precision router
arithmetic would not make these two distinct inputs choose the same expert.

Native-input CPU routing weights differ slightly from captured native weights
(NRMSE 5.68e-7), yet the position's final MoE output remains bitwise equal.
This does not establish bitwise agreement of every routing-weight element.

## Bidirectional intervention

All counterfactuals execute the same official CPU MoE. Unchanged rows retain
their normal routes. When selecting the other trajectory's expert IDs,
weights are normally recomputed from the **current input's** gate scores and
the unchanged normalization/route scale. An additional donor-weight control
copies the other trajectory's routing weights for that one row.

All errors below are versus the original CPU trajectory at position 108:

| Input and route | MoE output NRMSE | Final layer-token NRMSE | Worst stream NRMSE |
| --- | ---: | ---: | ---: |
| Native input, original native route | 39.732625% | 13.914685% | 32.877314% |
| Native input, CPU expert IDs, recomputed weights | 5.844346% | 3.000942% | 4.804079% |
| Native input, CPU expert IDs and donor weights | 5.099991% | 2.827944% | 4.189266% |
| CPU input, native expert IDs, recomputed weights | 39.493077% | 13.642170% | 32.671528% |

The first intervention largely removes the large amplification while keeping
the perturbed native input. The reverse intervention produces a similarly
large error without any native-input perturbation. This identifies the
expert-selection discontinuity as the dominant amplifier here.

These are diagnostic interventions, **not fixes or substitute references**.
Changing a route also changes routed-expert dispatch counts/shapes, so these
measurements include that execution consequence. They are not an additive
partition of error or a claim that all remaining error comes from one source.

## CPU-only alternate-trajectory control

The previously frozen independent CPU trajectory used 128 prefill + 4 decode
instead of 127 + 5. This wave replays its entire layer 3 using its own saved
layer-2 outputs, and reproduces the alternate layer-3 output bitwise.

At position 108:

- Original CPU route: `37, 80, 114, 173, 117, 199`.
- Alternate CPU route: `37, 80, 114, 173, 117, 214`, identical to the native set.
- FP64 supports the alternate route; margin(214−199) is +0.000662297458.
- Alternate versus original CPU MoE input NRMSE: **1.963085%**.
- Alternate versus original CPU final layer-token NRMSE: **13.856668%**;
  worst individual stream: **32.915431%**.
- Alternate CPU versus native final layer-token NRMSE: **2.778525%**.

Thus this same route flip and large local discrepancy can arise within CPU
execution alone when earlier execution shapes/trajectories change. This
prevents attributing the flip uniquely to TPU arithmetic. It does not prove
that all native differences are harmless or justify choosing the more
favorable CPU trajectory as the acceptance oracle.

## Evidence, checks and next work

Public runner: `scripts/debug_deepseek_v4_position108.py`.
Remote data: `/mnt/disks/deepseek-models/profiles/v4-position108-20260911-01`.
It retains full-prefill intervention MoE/layer outputs, gate scores, FP64
position-108 scores, CPU alternate replay and JSON measurements.

Evidence archive: `v4-position108-20260911-evidence.tar.gz`; local copy,
checksum receipt and verification under `GCP_login/results/v4-position108-20260911/`.
Source inputs are checked against the previous trajectory evidence manifest.
That upstream archive, already retained locally, has SHA-256
`cac91c1cfc2c8d1cd6b39af1748ca93c7ce3ea056de3b6e21c93303b57469b92`.
The new archive checksum is in its receipt and `GCP_login/CURRENT_TPU.md`.

Local CPU tests: **82 passed, 5 TPU-only skips**. Ruff and diff checks pass.
All 731 frozen runtime source hashes are verified unchanged at archive time.
No serving code or scheduler changes, no commit/push, no HTTP/model server,
no new HBM/performance measurement, and no diagnostic process left running.

Next: trace the original upstream perturbation at position 108, starting with
layer 0 and then layers 1–2, separating same-input projection/normalization/
activation-quantization errors from CPU batch-shape effects. Do not modify
top-k semantics or force common experts to conceal the discrepancy. The
43-layer/8K/multi-request/model-accuracy acceptance remains open.
