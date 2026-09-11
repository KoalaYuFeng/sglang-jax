# Q-normalization-only layer and logits validation — 2026-09-11

## Conclusion and deployment status

The compensated Q sum repairs the identified layer-0 token-109 rounding
boundary, and modestly reduces this fixture's aggregate hidden/logit error.
It **does not fix the dominant trajectory discrepancy**: layer-3 position 108
is unchanged, and the decode subset's hidden-state error slightly increases.
The candidate therefore remains diagnostic-only; no serving default changes.

Historical acceptance is still **15/18 with three failed records**. No new
thresholds, reference replacements, full-model accuracy claims, or performance
results are introduced here. The exp candidate remains disabled.

This follows the [standalone candidate matrix](deepseek_v4_attention_candidate_matrix_20260911.md).

## Setting and comparison controls

- Official checkpoint revision:
  `60d8d70770c6776ff598c94bb586a859a38244f1`, original low-bit weights.
- One v5p-8 Spot VM, four physical TPU v5p chips. Production DecoderLayer,
  attention TP4 for CSA/HCA and replicated heads for the first two SWA layers,
  EP4/DP1, tuned GMM MoE, Pallas mHC/CSA/HCA, merged FP8 projections.
- Branch `integration/deepseek-v4`, base `270dfcbd`, with prior fixes preserved.
  Runtime fingerprint remains
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.
- B1, fixed random token IDs, 127 prefill tokens in native chunks 64+63
  (one masked padding token), then five teacher-forced decode tokens.
- Layers 0–3 only: SWA/SWA/CSA/HCA. Capacity 256, not an 8K regression.
- Six diagnostic head outputs: positions 126–131. Applying the head after
  four layers does **not** produce the deployed 43-layer model's logits.

Three native paths are compared at every layer:

1. Original arithmetic and the frozen native layer input.
2. Compensated Q sum and the same frozen native layer input.
3. Compensated Q sum with independently propagated candidate hidden states.

Baseline outputs from all four instrumented production layers and the head
reproduce the previously frozen native results **bitwise**. Paths 1 and 2
have bitwise-identical normalized attention inputs and final KV/compressor
cache arrays at every layer. Only query normalization changes in that local
control; downstream states may differ in path 3 because its inputs differ.

CPU layer/logit reference is the unchanged, independently propagated CPU
trajectory from the previous experiment, not teacher forcing with candidate
hidden states. The script hashes all frozen inputs and official model source.
It also freshly executes official CPU attention on exactly the native
normalized attention input for each layer, including all five decode steps.
Official CPU uses one-shot 127-token prefill followed by decode-one.

The replacement is scoped to tracing in an isolated diagnostic process and
restored afterwards. Runtime files, other normalization functions, exp, the
CPU oracle, scheduler and acceptance checks are not patched.

## Same-input attention output

NRMSE against the fresh same-input CPU attention reference, percent, across
all 132 tokens:

| Layer | Attention type | Baseline | Compensated Q sum |
| --- | --- | ---: | ---: |
| 0 | SWA | 0.182047% | 0.178633% |
| 1 | SWA | 0.168358% | 0.168358% |
| 2 | CSA | 0.410878% | 0.410878% |
| 3 | HCA | 0.183713% | 0.183713% |

Layer 0's prefill-only result reproduces the previous standalone Q-only A/B:
0.183016% → 0.179499%. Layers 1–3 have no direct output change for their frozen
native inputs, consistent with the earlier raw-Q matrix. This does not imply
that every possible propagated input has identical Q-normalization results.

## Independently propagated hidden states

NRMSE against the frozen independent CPU trajectory, percent:

| Layer | Baseline, all tokens | Candidate, all tokens | Baseline, decode | Candidate, decode |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.002733% | 0.998433% | 1.041390% | 1.041390% |
| 1 | 1.925940% | 1.922514% | 1.723794% | 1.715447% |
| 2 | 2.542840% | 2.539071% | 1.989404% | 2.000100% |
| 3 | 4.176673% | 4.138942% | 2.724737% | 2.772155% |

The last layer's prefill-only error falls from 4.213350% to 4.173838%, while
decode error increases. Aggregate improvement must not hide the latter.

Token-level analysis flattens the four residual streams for each token; these
counts are relative error changes, not pass/fail counts:

| Layer | Changed output tokens | First changed position | Lower / equal / higher token NRMSE |
| --- | ---: | ---: | --- |
| 0 | 1 | 109 | 1 / 131 / 0 |
| 1 | 23 | 109 | 16 / 109 / 7 |
| 2 | 23 | 109 | 14 / 109 / 9 |
| 3 | 23 | 109 | 11 / 109 / 12 |

At layer 3, the largest per-residual-stream error remains 32.877314%, with
maximum absolute error 0.21435546875. The worst flattened-token error remains
at **position 108**. Its output and every earlier position are unchanged by
the candidate. Thus the identified position-109 Q midpoint cannot explain or
repair the original position-108 discrepancy in this trajectory.

## Diagnostic head

| Measurement | Baseline | Compensated Q sum |
| --- | ---: | ---: |
| NRMSE against CPU | 1.558638% | 1.531705% |
| Worst output-vector NRMSE | 2.062186% | 1.942861% |
| Maximum absolute logit error | 0.617507 | 0.345764 |
| Top-1 agreement | 6/6 | 6/6 |

Both produce the same top-1 IDs as the CPU diagnostic head:
`15854, 60846, 64432, 11653, 117160, 108635`.
These six truncated-head comparisons are not a language benchmark or a
full-model accuracy acceptance result.

## CPU-only attention split control

Fresh official CPU attention instances receive exactly the same captured
native normalized inputs. The original 127-prefill + 5-decode execution is
repeated and reproduces the reference **bitwise in all four layers**. A
separate execution uses 128-prefill + 4-decode, retaining all 132 token inputs,
positions and output comparisons. Neither reference receives candidate input.
This uses the official initial-prefill/decode-one API, not an unsupported
multi-token call with nonzero start position.

NRMSE, percent:

| Layer | CPU alternate vs original | TPU baseline vs original CPU | TPU baseline vs alternate CPU |
| --- | ---: | ---: | ---: |
| 0 | 0.026182% | 0.182047% | 0.183094% |
| 1 | 0.003672% | 0.168358% | 0.168359% |
| 2 | 0.398554% | 0.410878% | 0.210222% |
| 3 | 0.015844% | 0.183713% | 0.183569% |

Layer 2 is strongly reference-shape-sensitive. This control prevents
attributing its entire 0.410878% residual to a TPU implementation defect. It
does not establish that the TPU is correct, select the more favorable CPU
reference, or uniquely identify the arithmetic substage responsible. NRMSEs
cannot be subtracted to assign independent causal contributions. The original
reference remains the reported comparison and historical gates are unchanged.
Both CPU outputs are retained for every layer.

## Evidence and remaining work

Runner: `scripts/validate_deepseek_v4_qnorm_trajectory.py`.
Remote experiment: `/mnt/disks/deepseek-models/profiles/v4-qnorm-trajectory-20260911-01`.
It retains all three native paths, independent CPU outputs, normalized
attention inputs/outputs, final caches, logits and per-token error arrays.
Cache NPZs preserve BF16 payloads as raw two-byte NumPy storage where needed;
interpret those fields as BF16 when doing numerical analysis.

Local CPU regression: **82 passed, 5 TPU-only skips**. Runner Ruff and
`git diff --check` pass. The unchanged 731-file runtime manifest is rechecked
at archive time. No HTTP server, full-model run or performance measurement
was started. No commit or push was performed.

Archive: `v4-qnorm-trajectory-20260911-evidence.tar.gz`; local copy, checksum
receipt, JUnit and verification under `GCP_login/results/v4-qnorm-trajectory-20260911/`.
Upstream frozen trajectory archive SHA-256:
`cac91c1cfc2c8d1cd6b39af1748ca93c7ce3ea056de3b6e21c93303b57469b92`.
Archive checksum is recorded outside this self-archived report in its receipt
and `GCP_login/CURRENT_TPU.md`.

Do not promote the candidate solely on these aggregate improvements. The next
target is the unchanged position-108 error and the larger same-input CSA
attention residual, retaining CPU shape controls and the decode regressions.
Full 43-layer, 8K, multi-request and performance gates remain open.
