# DeepSeek-V4-Flash GPQA Diamond Non-Think — 2026-09-11

Status: complete; remote and local re-scoring agree. Experiment
`v4-gpqa-nonthink-20260911-01`. The cancelled full MMLU-Pro job remains stopped.

## Result

**143 / 198 correct = 72.22%**, single-sample accuracy, all Diamond questions
included in the denominator. The official Flash Instruct Non-Think reference
is 71.2% (source and protocol caveat below), a numerical difference of +1.02
percentage points. This is a similar aggregate score, **not evidence that the
TPU deployment outperforms or exactly reproduces the official runtime**:
the official prompt/repetition protocol is not fully matched, this is one
stochastic run, and no paired official-runtime evaluation was performed.

| Domain | Correct / total | Accuracy |
| --- | ---: | ---: |
| Biology | 9 / 19 | 47.37% |
| Chemistry | 59 / 93 | 63.44% |
| Physics | 75 / 86 | 87.21% |
| Overall | 143 / 198 | 72.22% |

- HTTP/validation failures: **0**. Truncated responses: **1**. Invalid final
  answers: **1**, the same truncated response, not a second failed case.
  Conflicting explicit answer letters: **0**. No re-generation or retry.
- The truncated response consumed its 8090-token output budget with a
  100-token prompt and ended with `finish_reason.type="length"`; there was
  no extractable final option. It remains wrong in the 198-question score.
  It took 261.30 seconds, consistent with ongoing decode, not a stalled
  scheduler. The actual question and response remain private.
- Run wall time: **63.77 minutes**, completed at **2026-09-11 12:41:34 UTC**
  (20:41:34 Singapore). Generated 114,311 tokens; output length min/median/max
  98 / 442.5 / 8090. These are accuracy-run diagnostics, not a controlled
  throughput benchmark.
- All 198 raw result records and attempts are on the persistent disk and
  local private backup. Remote and downloaded frozen local scorers agree;
  both the score report and all-result hash manifest match byte-for-byte.
  The evaluation client and local mirror exited; the existing model service
  is preserved. No MMLU-Pro restart, production changes, commit or push.

This supports task-level viability on this benchmark, but does not settle
the previously observed CPU/TPU logit-distribution differences or establish
full numerical, long-context, multi-request or cross-benchmark equivalence.

## Setting

- Complete official Instruct checkpoint, revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`, all 43 layers.
- Existing SGLang-JAX HTTP service, v5p-8 / four physical chips, TP4/EP4/DP1.
  Original FP4/FP8/scales, online dequantization, unchanged production kernels,
  scheduler, 128-token chunked prefill and 8192-token context.
- Runtime fingerprint:
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.
  All 731 frozen runtime file hashes verified before launch.
- GPQA Diamond: all 198 rows from the authors' public archive at repository
  revision `56686c06f5e19865c153de0fdb11be3890014df7`. Only the standard
  `Question`, `Correct Answer`, and three `Incorrect Answer` fields are used;
  no pre-revision or extra-revision substitution, label repair or filtering.
- Zero-shot fixed instruction requests an explanation and a final `Answer: X`
  line. No dataset explanation or validation metadata appears in the prompt.
- Official V4 encoder and pinned tokenizer; `thinking_mode="chat"`, with
  `</think>` closing the thinking block in the input. No Think Max prefix.
- Temperature 1.0, top-p 1.0, normal EOS; one sample per question, concurrency 1.
  Option order uses Python `Random(20260911 + row_index)` and is frozen before
  requests. Server seed/settings are recorded; deterministic per-request
  sampling is not enabled and is not claimed.
- Prompt lengths: 81–2563 tokens. Output budget for each question is
  `8192 - prompt_tokens - 2`, preserving the total 8K limit without a separate
  2048-token generation cap. No context overflow or shortened question.

The [official technical report](https://arxiv.org/html/2606.19348v1#S5.SS3.SSS1)
specifies temperature 1.0 and 8K/128K/384K evaluation context for
Non-Think/High/Max. The [official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/README.md#comparison-across-modes)
reports Flash GPQA Diamond pass@1 of 71.2/87.4/88.1 for those modes.
The exact official GPQA prompt, option ordering and repeat count have not been
established here. Therefore this is a deployment measurement with some aligned
settings, **not** strict reproduction of the official 71.2% result.

## Reasoning modes

These modes use the same released Instruct checkpoint, not different FP4/FP8
precision settings. Non-Think generates the answer directly and can still
include explanations. High uses explicit thinking tokens before the final
answer. Max adds the official special reasoning instruction and targets more
extensive reasoning. Larger reasoning/context budgets increase potential
latency and cache usage; they do not guarantee improvement on every question.
Our 8K setup matches the reported Non-Think context, not the reported High/Max
budgets. The context numbers are evaluation settings, not fixed output lengths.

## Scoring and persistence

- Frozen extractor: last explicit answer line, then boxed answer, then bare
  letter. Conflicting explicit answer letters are counted separately. No random
  guessing or answer-text inference after seeing the gold label.
- Failed, invalid and truncated completions count wrong; denominator 198.
  No score-dependent retry. Interrupted attempts stay recorded; automatic
  retry is disabled. Three consecutive errors or a four-hour lease stop the
  client, without restarting the server or allocating cloud resources.
- Atomic, fsynced per-question results on the persistent disk. A separate
  read-only local backup copies new files every minute while the laptop and
  process are available, with a five-hour lease and no automatic job restart.
- At completion, the remote driver and downloaded frozen local driver both
  re-score every response and hash the result files. Partial progress is not
  a completed accuracy score.
- Dataset examples, options, explanations and model responses remain private
  under ignored `GCP_login/results/` and the attached disk. Do not include
  them in public reports, commits or tool commentary; publish aggregates only,
  respecting the [dataset authors' request](https://huggingface.co/datasets/Idavidrein/gpqa).

## Files and validation

- Driver: `scripts/evaluate_deepseek_v4_gpqa.py`.
- Shared atomic JSON helpers: `scripts/evaluate_deepseek_v4_mmlu_full.py`;
  importing these helpers does not start MMLU-Pro.
- Synthetic tests: `python/sgl_jax/test/test_deepseek_v4_gpqa.py`, 12 passed.
- Independent, CPU-only input audit passed all 198 cases: downloaded archive
  member equals the saved CSV byte-for-byte; standard question/options rebuild
  every prompt exactly through the pinned official encoder and tokenizer;
  shuffled gold labels, input IDs and 8K output budgets all match. This audit
  makes no inference requests and does not change the frozen scoring protocol.
- Remote directory: `profiles/v4-gpqa-nonthink-20260911-01/` on the model disk.
- Local mirror: `GCP_login/results/v4-gpqa-nonthink-20260911-01/`.
  `progress.json` and `mirror-status.json` both report completion;
  `audit.json` contains the final score after all 198 records were re-scored.

Matching remote/local score report SHA-256:
`b8dea455e03d17b0291583fc30808644a7c837505ad0f95c27ca5ff3cf8b290e`.
Matching remote/local all-result hash manifest SHA-256:
`af023e0a183c2261f0b4be9611b739ea13820a269f9b22d94a5a4305122f30d7`.

Protocol SHA-256:
`b24f1724c17d111e1518d5294b813c29f1bb5212f3fc74c114b16d927bcad2c9`.
Runner SHA-256:
`eb7055d9e49a508f4736d78872c46783fdf4e2c8218cb049bebbb5f32d58b75c`.
GPQA Diamond CSV SHA-256:
`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
No production change, commit or push.
