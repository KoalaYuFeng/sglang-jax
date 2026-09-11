# DeepSeek-V4-Flash full-model accuracy validation — 2026-09-11

## Scope and experimental setting

This wave evaluates the existing deployment, without enabling the experimental
Q-normalization, exponential, FP8 split-accumulator or FP64 diagnostic candidates.
It does not modify production kernels or scheduler behavior.

| Item | Setting |
| --- | --- |
| Model | Complete 43-layer DeepSeek-V4-Flash |
| Official checkpoint revision | `60d8d70770c6776ff598c94bb586a859a38244f1` |
| Hardware | One v5p-8 Spot VM, four physical TPU v5p chips |
| Parallelism | TP4 / EP4 / DP1 |
| Source | `integration/deepseek-v4`, base `270dfcbd`, existing uncommitted fixes preserved |
| Runtime fingerprint | `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3` |
| Serving | SGLang-JAX HTTP, context 8192, chunked prefill 128, page size 128, max running requests 4 |
| Kernels | Pallas mHC/HCA/CSA, TP attention, batched CSA decode, tuned GMM MoE, FP8 GMM, fused norm/merged projections/fused wo_a |
| Production precision | Original packed FP4/FP8/block scales with online dequantization; major activations/outputs BF16, accumulation FP32 |
| Request protocol | Greedy, temperature 0, top-p 1, normal EOS handling, concurrency 1 |

### Full-model reference and probability comparison

The reference runs the checkpoint's official Python model with the existing
independent CPU replacements for GPU-only kernels. It is **not** the official
GPU runtime or an officially supplied native CPU kernel. The original CPU
reference is unchanged: low-bit GEMM uses FP32 computation and BF16 output;
the official output head uses FP32 matrix multiplication. Existing scale
calculation helpers are retained, not replaced by new high-precision operators.

A fixed natural-language prompt contains twelve background notes followed by
`Compute 17 + 25. State the calculation in one sentence.` Official non-thinking
chat encoding yields 164 prompt tokens, crossing the 128-token compressor
boundary. Before either runtime runs, the continuation is fixed to
`17 + 25 = 42.` (8 tokens).

The CPU reference advances its own hidden states and caches through all 43
layers: one 164-token prefill followed by eight single-token teacher-forced
steps. Layer-wise staging saves host memory; no TPU states are injected.
The output head produces 9 prediction rows over all 129,280 vocabulary tokens.

The normal HTTP API returns full-vocabulary **log probabilities**, not raw
logits. For each of the same nine fixed contexts it performs a one-token
request using the existing prefix-cache/chunked-prefill path. Thus this tests
the deployed HTTP execution shapes against the CPU prefill/decode reference;
it is not an identical-shape kernel comparison or a single persistent
teacher-forced HTTP decode session.

CPU logits and HTTP log probabilities are normalized before comparing top-1,
top-5 overlap, KL divergence, total variation and fixed-continuation NLL.
FP64 is used only for descriptive metric arithmetic. No FP64 equality gate
or newly relaxed numerical tolerance is applied.

Two additional HTTP requests freely generate up to 128 tokens from the same
prompt. These test production generation and repeatability, **not** independent
CPU free-generation equality.

### Frozen MMLU-Pro generation regression

Exactly the previously evaluated 56 questions are replayed: four per category,
five-shot, dataset revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`,
official non-thinking chat encoding, max 2048 generated tokens, concurrency 1.
The original prompt IDs, selection, sampling and answer extractor are reused
unchanged and hash-verified. Failed, truncated or invalid answers count wrong;
there are no score-dependent retries. Prompt lengths are 899–2295 tokens.

The historical baseline is 46/56 (82.14%) on runtime fingerprint
`2543e8b0e98fc56658912cd2d14f05e4d4ae4e00cc961f07bc150744070e0c0f`.
The current comparison includes the intervening existing numerical fixes;
it is not an isolated A/B of any one kernel change.

This is a previously seen regression sample, not a held-out or full-suite
accuracy estimate and not a matched official GPU/API benchmark.

## Results

Both evaluations completed. The full-model path works on this fixture, and the
frozen public-task sample has no aggregate score drop. **This does not establish
that all remaining numerical differences are harmless:** the first prediction
has a material probability shift, and one previously correct task becomes wrong.

### Full-model probability and generation results

| Metric | Result |
| --- | --- |
| Reference layers completed | 43/43, all saved hidden states finite |
| Top-1 agreement, same fixed contexts | 9/9 |
| Mean top-5 set overlap | 95.56% |
| Mean KL(CPU \|\| TPU) | 0.01294635 nats |
| Maximum total variation | 0.20040976, first prediction |
| Fixed 8-token continuation mean NLL, CPU | 0.22679183 nats/token |
| Fixed 8-token continuation mean NLL, TPU | 0.21828036 nats/token |
| Free HTTP generation, both repeats | `17 + 25 equals 42.` |

The first prediction's chosen token is the same, but its probability is
**57.14% CPU vs 77.18% TPU** (about 20.04 percentage points apart), with
KL 0.10594707 nats. Top-1 equality must not hide this discrepancy. At the fifth
prediction, top-1 probability is 70.42% vs 76.29%, KL 0.00904773 nats. The CPU
and TPU both prefer the token for ` equals` there, while the deliberately fixed
teacher continuation uses ` =`; this is not a generation failure.

These shifts can affect sampling even though the nine greedy choices match.
The slightly lower TPU teacher NLL on this one fixed arithmetic continuation is
not evidence of better model quality. It is not representative perplexity.

### MMLU-Pro same-question replay

| Metric | Historical deployment | Current deployment |
| --- | --- | --- |
| Correct | 46/56 | 47/56 |
| Accuracy on this sample | 82.14% | 83.93% |
| Failed / truncated / invalid | 0 / 0 / 0 | 0 / 0 / 0 |

Answer choices are unchanged on 52/56 questions. Two answers change from wrong
to correct, one from correct to wrong, and one from one wrong answer to another.
Only 3/56 full generated explanation strings are byte-identical; explanation
differences do not themselves mean that answers are incorrect.

| Question ID | Category | Gold | Historical answer | Current answer | Outcome |
| --- | --- | --- | --- | --- | --- |
| 6150 | health | J | A | J | wrong → correct |
| 4932 | history | B | B | F | correct → wrong |
| 1597 | law | E | G | E | wrong → correct |
| 5461 | other | C | B | A | wrong → wrong |

The net +1 correct answer does not erase question 4932's regression or establish
a statistically meaningful quality improvement. These old/new changes have
not been causally attributed to a particular floating-point operation or fix.

### Conclusion and next target

This provides positive, limited evidence for functional full-model deployment,
but **not** an unconditional numerical/accuracy acceptance. Continue at the
model level: broaden same-context probability comparisons to natural prompts,
especially ambiguous first-token decisions, and independently reference the
changed-answer cases beginning with question 4932. Keep the current production
kernels unchanged until an isolated, reproducible defect is demonstrated;
do not resume FP64 bitwise matching as the model-level acceptance criterion.

## Interpretation boundaries

The historical **15/18** result belongs to an isolated compressor numerical
matrix, not full-model accuracy or a 43-layer acceptance gate. Its three failed
records remain retained; this wave does not relabel or waive them.

One short natural prompt is insufficient to establish broad logits equivalence.
The MMLU-Pro replay does not establish official-model accuracy equivalence.
Neither test establishes full 8K/multi-request numerical correctness or a new
performance result. Startup/first-request compilation is present in timings;
this wave is not a throughput benchmark.

## Reproduction and evidence

Drivers:

- `scripts/evaluate_deepseek_v4_cpu_full_model.py`
- `scripts/evaluate_deepseek_v4_http_accuracy.py`
- `scripts/analyze_deepseek_v4_endtoend_accuracy.py`
- `scripts/deepseek_v4_endtoend_metrics.py`

CPU metric unit tests: `python/sgl_jax/test/test_deepseek_v4_endtoend_metrics.py`.
Run directory on the attached model disk:
`profiles/v4-endtoend-accuracy-20260911-01`.
Raw layer outputs, raw CPU logits, all HTTP log probabilities, generated texts,
per-question responses, protocol hashes and server settings are retained.

Metric tests: 9 passed locally; Ruff checks and formatting checks passed for all
five new public Python files. Runtime/source and archive verification are
recorded in the private evidence receipts. No commit or push in this wave.
The loopback HTTP service is left running with the recorded original settings;
CPU reference and evaluation client processes have completed.
