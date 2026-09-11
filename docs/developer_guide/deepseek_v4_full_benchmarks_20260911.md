# DeepSeek-V4-Flash: full benchmark coverage — 2026-09-11–12

This run replaces the GSM8K-128 smoke test as the intended formal GSM8K
measurement. Every question in the named test split is included, with one
generation per question. Complete coverage does not imply an identical
prompt/sampling protocol to a model vendor's reported score.

## Status and model identity

Both new full-test runs and their remote/local frozen-score audits are complete.
GPQA Diamond's previously completed full 198-question result is retained, not
rerun. Every row covers its entire named test split, not a random sample.

| Benchmark | Full coverage | Protocol | Correct | Score | Elapsed |
| --- | ---: | --- | ---: | ---: | ---: |
| GPQA Diamond | 198 | Zero-shot Non-Think, temperature 1, one sample | 143/198 | 72.22% | 63.77 min |
| GSM8K | 1,319 | Zero-shot Non-Think, greedy, no tools | 1,280/1,319 | 97.04% | 77.47 min |
| HumanEval | 164 | Zero-shot Non-Think, greedy pass@1, original tests | 149/164 | 90.85% | 19.37 min |

## Official Instruct comparison

The [DeepSeek release's Instruct mode table](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/commit/a7aaed80dd2df27620eb534454253ea25eb11c7a)
is the reference below, checked September 12. Compare our Non-Think runs with
**Flash Non-Think**, not Base, Pro, Think High or Think Max.

| Benchmark | Official Flash Instruct Non-Think | This v5p deployment | Interpretation |
| --- | ---: | ---: | --- |
| GPQA Diamond | 71.2% | 72.22% (143/198) | Similar aggregate accuracy; prompt/repetition protocol not fully matched |
| GSM8K | No corresponding score found in the official Instruct table | 97.04% (1280/1319) | Full-test deployment baseline, not a reproduced vendor score |
| HumanEval | No corresponding score found in the official Instruct table | 90.85% (149/164) | Full-test deployment baseline, not a reproduced vendor score |

The GPQA difference is **+1.02 percentage points**, not evidence of superiority
or sample-by-sample runtime equivalence. The official 90.8% GSM8K (8-shot) and
69.5% HumanEval figures belong to **Flash-Base** and are not valid Instruct
baselines. Benchmark coverage and matching the vendor's evaluation protocol
are separate requirements.

No additional long benchmark is being launched: MMLU-Pro remains cancelled
and incomplete; LiveCodeBench has not been run. Neither has a full-test score
for this deployment.

## Run settings and result details

GSM8K: **4,648.04 seconds**, zero request failures, zero truncations, one
invalid final-answer format (counted wrong), zero conflicting final numbers.
HumanEval: **1,162.35 seconds**, zero request/infrastructure failures, zero
truncations or execution timeouts; **five invalid code formats** and **ten
official-test failures**, all counted wrong in denominator 164. The five
format-invalid outputs each contain a single unmatched code-fence marker;
their functional correctness is not established. No parser repair or selective
retry was used to increase the score. This extraction policy is a reported
limitation, not evidence that all five functions are algorithmically wrong.

Singapore times: GSM8K launch **2026-09-11 23:02:22**, completed **2026-09-12
00:19:50**; HumanEval inference/evaluation **2026-09-12 00:20:09–00:39:31**.
HumanEval elapsed includes per-sample container grading, excludes its queue
wait and preflight. Directory names identify the September 11 preparation date.
Evaluation clients and backup process exited after completion; the original
model service was preserved, verified healthy and idle by its final snapshot.

- Deployment code published at `1ea97e16`, with no runtime/kernel changes in
  this evaluation work; only evaluation scripts and synthetic tests added.
- Complete 43-layer **Instruct** checkpoint `deepseek-ai/DeepSeek-V4-Flash`,
  revision `60d8d70770c6776ff598c94bb586a859a38244f1`.
- One v5p-8 VM / four physical chips, TP4 / EP4 / DP1; original FP4/FP8/scales
  and online dequantization. Same running service and explicit backend options
  as the [release report](deepseek_v4_v5p_release_20260911.md).
- Runtime fingerprint:
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.
- Context 8192, global prefill chunk 128, maximum running requests 4, evaluation
  concurrency **1**. GSM8K and HumanEval inference run serially, not together.
- Both new runs: zero-shot official Non-Think chat encoding, temperature 0,
  top-p 1, EOS enabled, maximum 2048 generated tokens, no tools/calculator.
  Prompt lengths: GSM8K 59–222, HumanEval 80–434. No input overflow.
- Infrastructure failures, malformed outputs and length-truncated responses
  count wrong; no answer-dependent retries, filtering or parser changes.

## GSM8K: all 1,319 test questions

[Original authors' repository](https://github.com/openai/grade-school-math),
revision `3101c7d5072418e28b9008a6636bde82a006892c`, original `test.jsonl`:
SHA-256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.

All indices `0..1318`, in source order. This is a fresh full run: the previous
128 answers are not spliced into it. The previous custom zero-shot instruction
is retained: brief calculation followed by a separate `#### <number>` line.
Exact Decimal equality, with decimal/comma normalization and the last explicit
final marker. Invalid final formats remain wrong. This is **not** the original
few-shot GSM8K evaluation protocol or proof of equivalence to a vendor score.

Preparation protocol SHA-256:
`387ea26c1718352770455a80d382ed42ff06dbd77cb970436b5d3f8aed9b28b5`.
Score-report SHA-256:
`b7d5447d2853cff79dcf7c292a1436e5e032f1688a063c3ea31243209b654b27`.
All-result manifest SHA-256:
`cc09c02590f0a2b18817e27227768e68da4ae0f56d9946a4680bacc138a76d2f`.
All 1,319 gold numbers and prompt questions were independently rechecked
against source JSONL before completion. The full-coverage audit is tested with
1,319 synthetic records, including four-digit record filenames.
All 128 questions overlapping the prior smoke test have identical input IDs,
generated text, parsed predictions and correctness grades; they were freshly
generated, not reused. The 96.09% smoke score is not the formal full-test score.

Evidence directory: `profiles/v4-gsm8k-full-20260911-02/`. The first `-01`
preparation stopped before inference because the earlier file manifest still
contained the old paging **test** bytes. The known, published test-only change
was recorded explicitly in `-02`; serving source and fingerprint did not change.

## HumanEval: all 164 tasks

[Official repository](https://github.com/openai/human-eval), revision
`6d43fb980f9fee3c892a914eda09951f772ad10d`. All `HumanEval/0..163`, sorted by
numeric task ID; original tests unchanged. One greedy sample per task yields
greedy **pass@1**, not pass@10/pass@100 or repeated-sampling estimates.

- Dataset `HumanEval.jsonl.gz` SHA-256:
  `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.
- Unmodified official `execution.py` SHA-256:
  `79901c6f5b59701c465b164aed290fa53abc35abc80a5378fc10f4dac1f9a84c`.
- Prompt requests the complete Python definition, required imports/helpers,
  and no tools. Only the official function prompt is shown, not hidden tests
  or canonical solutions.
- Frozen extraction: one Python code fence or plain Python; require a valid
  top-level entry-point function and preserve imports/helpers. Do not repair
  code based on tests. Append the complete generated module after the original
  docstring-only function prefix, then run the original official tests.
- Official per-task execution timeout: 3 seconds. Each generated solution runs
  in a fresh Docker container: network disabled, UID 65534, read-only root,
  all capabilities dropped, no-new-privileges, default seccomp, one CPU,
  512 MiB memory, 64-process limit, temporary filesystem only. Only trusted
  evaluator files are mounted read-only; no credentials, model disk, repository
  or Docker socket is mounted. No generated code executes on the host.
- Pinned evaluation image ID:
  `sha256:8e79d4490b856f2e823cb4d20b5a8061c867f1b79951e9a3fb7571185187ea06`.
  Its supplied Dockerfile starts from Python image digest
  `9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534`
  and installs the official harness dependencies: NumPy 2.2.6, fire 0.7.0,
  tqdm 4.67.1 and transitive dependency termcolor 3.1.0. NumPy/BLAS uses one
  thread. An exact image export is retained on the persistent disk and locally.
- Preflight: **164/164 official canonical solutions passed**. A synthetic wrong
  answer failed, and a synthetic infinite loop timed out. Isolation assertions
  for UID, read-only root, networking, capabilities and seccomp passed.
  A second all-164 canonical check through the exact complete-definition
  extraction/append adapter also passed. These are harness controls, not model
  accuracy measurements, and were repeated for the final dependency-complete image.

Evidence directory: `profiles/v4-humaneval-full-20260911-02/`. The initial `-01`
queue was stopped with **zero model requests** to align the container with
the official harness dependencies instead of using a stdlib-only environment.
Its preparation and preflight evidence are retained, not scored. No model
response was inspected to choose the environment or modify extraction.

Final protocol SHA-256:
`831411f5abcdc4670333db62daf60b79e65236eb957877e343af1e0a92e7c76e`.
Score-report SHA-256:
`4051b91d0efd95911d173a7d6f8623ee3ef6b168a24de0a773999b78237cb6fd`.
All-result manifest SHA-256:
`5569a3171d1e9fbe90a6b99e56dc0b93bc378160edf39d13d195aac35b1b72d3`.
Both score-report and all-result manifest hashes match the remote evidence.

## Evidence and reproduction

The measured evaluator code is published at **`7c0c67105136236ca3c0026ce3fe654dd50dd745`**.
The September 12 publication cleanup only reformats the GSM8K/HumanEval
evaluators and their synthetic tests; it does not change prompts, sampling,
answer extraction, scoring, sandbox policy, or model execution. File hashes
nevertheless change. To audit the existing frozen evidence, use the evaluator
files from that commit or the archived snapshot. Do not rewrite old protocol
hashes or bypass their checks to accommodate newer files; prepare a new
protocol for any future run with changed files.

Cleanup verification: the two evaluator scripts have identical Python ASTs
before/after formatting. The local CPU-only GSM8K, HumanEval, GPQA, MMLU-full
protocol and release-benchmark audit suites report **70 passed**. Black, isort,
Ruff and `git diff --check` pass for the cleanup scope. These are synthetic
protocol/metric checks, not new model accuracy or TPU execution tests.

Local mirrors are under ignored `GCP_login/results/`; each response is saved
atomically to the independent model disk and backed up incrementally. Raw
questions, solutions, responses and access credentials are not published.
Frozen evaluation/scoring code, protocol hashes, original source hashes,
per-question records, server snapshots and score manifests are retained.

- GSM8K preparer: `scripts/prepare_deepseek_v4_gsm8k_data.py`.
- Full GSM8K: `scripts/evaluate_deepseek_v4_gsm8k.py prepare --full-test ...`,
  then `run` and `audit`. Legacy subset mode remains available for smoke tests.
- HumanEval: `scripts/evaluate_deepseek_v4_humaneval.py` modes `prepare`,
  `preflight`, `run`, `audit`. Its container entry point is
  `scripts/deepseek_v4_humaneval_sandbox.py`; never execute generated samples
  outside the restricted container. Build `scripts/deepseek_v4_humaneval.Dockerfile`
  with an empty context, then pass the resulting immutable local image ID via
  `prepare --image sha256:...` before freezing a new protocol.
- Synthetic protocol/extraction/isolation/audit tests: **31 passed** locally.

Known CPU/TPU numerical differences, current-source long-context/concurrency
acceptance gaps and performance limitations remain as recorded in the release
report; benchmark accuracy does not erase those limitations.
Do not compare these **Instruct** measurements directly to the earlier **Base**
GSM8K 90.8 / HumanEval 69.5 figures, or claim exact vendor-protocol equivalence.

## Performance reference

Use the [same-runtime HTTP measurements and exact settings](deepseek_v4_v5p_release_20260911.md#performance-provenance)
for performance alongside these accuracy results. That report retains the
128/1024-token input, 32-token output, concurrency 1/4 measurements and defines
the timing boundaries. Its output tokens/s includes prefill and whole-batch
wall time; it is not isolated decode throughput.

The [historical 4K/8K ModelWorker matrix](deepseek_v4_8320_validation_profile.md)
uses an older numerical runtime and a different measurement path. It remains
available for historical comparison, not as a new performance measurement of
this accuracy-validated baseline. This publication cleanup reruns neither
TPU performance nor model accuracy and changes no scheduler or kernel code.
