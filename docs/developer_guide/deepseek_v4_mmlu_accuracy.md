# DeepSeek-V4-Flash: public MMLU-Pro generation accuracy

Experiment: `v4-mmlu-accuracy-20260911-01`. Status: complete; inference,
remote/local scoring audits and verified evidence archival all finished.

Result: **46/56 = 82.14%**, with zero request failures, zero truncated
completions and zero invalid answer extractions. This is not evidence of
official-runtime accuracy equivalence.

## Why this benchmark

[MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) is a public,
14-domain multiple-choice knowledge/reasoning benchmark. The pinned
[DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/README.md)
reports 83.0% for Non-Think mode. This is contextual information, **not** a
same-protocol acceptance threshold for the small experiment below.

Other relevant public datasets include
[GPQA](https://github.com/idavidrein/gpqa),
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), and
[GSM8K](https://github.com/openai/grade-school-math).
GSM8K's published score in this model card is for the separate Base model;
it must not be used as an equivalent official score for our Instruct model.

## Frozen protocol

- Full 43-layer checkpoint `60d8d70770c6776ff598c94bb586a859a38244f1`,
  four physical TPU v5p chips, TP4/EP4/DP1, original FP4/FP8/block-scale
  storage with the previously accepted execution options.
- Same SGLang-JAX HTTP service on loopback port 30126. No server restart,
  model/kernel change, scheduler change or context-limit override.
- Runtime source fingerprint:
  `2543e8b0e98fc56658912cd2d14f05e4d4ae4e00cc961f07bc150744070e0c0f`.
- Dataset `TIGER-Lab/MMLU-Pro`, revision
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`, containing 12032 test questions.
- 56 test questions: four per each of fourteen categories; sort by
  SHA256(`20260911:category:question_id`) and take the first four. No filtering
  by answer, correctness, prompt length or past performance.
- Five examples from the **validation** split of the same category, using
  the public benchmark CoT prompt structure. Test solutions/labels are never
  included in prompts. Full selected rows and prompts are frozen before
  generation; selected questions do not occur among validation examples.
- Official V4 Python encoder, `thinking_mode='chat'` (Non-Think), serialized
  to token IDs with the same tokenizer previously checked against serving.
- `temperature=0`, `top_p=1`, `max_new_tokens=2048`, normal EOS, concurrency1.
  Prompt lengths 899–2295 tokens; every complete prompt plus output budget
  fits the unchanged 8192 total-context limit. No shortening/replacement.
- Actual generated text is scored, not restricted-choice logits or a
  one-token forced answer. The public API harness's three extraction regex
  tiers are recorded. Missing/out-of-range answers, failed requests and
  truncated generations count incorrect; no random guessing or score-based
  retries. Denominator remains56.
- The same cache revision's Arrow files are hashed and all selected raw
  test/validation rows checked against them, because offline `load_dataset`
  can otherwise fall back to a latest cached configuration.
- Five lightweight protocol tests cover extraction tiers, invalid answers,
  absence of the test solution from prompts and interval calculation.

The evaluation is a small equal-category stratified sample, not full-test
natural category weighting. Greedy decoding and2048 output tokens also do
not reproduce DeepSeek's private full-suite settings. No official GPU/API
same-input A/B is performed. Any Wilson interval is only a descriptive
binomial approximation, not exact stratified-design uncertainty.

## Results and audit

| Category | Correct / sampled |
| --- | --- |
| Biology | 4 / 4 |
| Business | 4 / 4 |
| Chemistry | 4 / 4 |
| Computer science | 4 / 4 |
| Economics | 4 / 4 |
| Engineering | 2 / 4 |
| Health | 3 / 4 |
| History | 3 / 4 |
| Law | 3 / 4 |
| Math | 4 / 4 |
| Other | 2 / 4 |
| Philosophy | 3 / 4 |
| Physics | 4 / 4 |
| Psychology | 2 / 4 |
| **Total** | **46 / 56 (82.14%)** |

Four questions per category cannot establish reliable subject rankings.
The descriptive Wilson 95% interval is approximately 70.16%–90.00%; because
sampling is stratified, this is not a formal full-dataset accuracy interval.
The numerical proximity to the published 83.0% is not an equivalence test.

The audit re-extracted answers from all saved response texts and reproduced
every score. All 56 used the first, explicit-answer extraction tier; no
conflicting explicit answer letters were found. All ten mismatches retain
their original dataset gold and full generated text in `audit.json`.
There were no score-dependent retries, label corrections or sample changes.
This identifies answer-key mismatches, not their cause: model knowledge,
dataset issues and deployment numerical behavior are not separated by this
single-runtime experiment.

The HTTP service remained on its original settings. After completion it was
ready and idle, all four request slots and 33280 KV tokens were free, and the
accepted runtime fingerprint remained unchanged. No model/kernel source,
historical benchmark or production context guard was modified.

## Evidence

The experiment directory on the persistent model disk contains `protocol.json`,
`selected.json`, `validation-examples.json`, `dataset-cache-audit.json`,
`responses.jsonl`, per-question progress and server receipts. Corresponding
local artifacts live under ignored `GCP_login/results/`.

Archive: `profiles/v4-mmlu-accuracy-20260911-01-evidence.tar.gz` on the
persistent disk; matching local copy
`GCP_login/results/v4-mmlu-accuracy-20260911-01/evidence.tar.gz`.
Size: 206328 bytes. SHA-256:
`15643c08056f36db18f1c10bfd2f2a6e1fc71b656c0fcbc56aa98c7a62ed8465`.
The local archive was verified/extracted and `audit.py` rerun locally,
reproducing all 56 scores and the same 46 correct answers. This is scoring
verification from saved responses, not a second inference run. The final
Markdown report is also saved separately beside the archive's experiment
files. No commit or Git push was performed.

This run does not overwrite the earlier 20-question, prompt-length-conditioned
MMLU-Pro numerical fixture or its 70% one-token result: that historical gate
used a different selection and answer protocol and is not a before/after A/B.
