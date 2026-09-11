# DeepSeek-V4-Flash full MMLU-Pro evaluation — 2026-09-11

Status: **stopped at the user's request**, not a completed full-test result.
Experiment: `v4-mmlu-full-20260911-02`. Evaluation and local periodic backup
processes are stopped; no automatic continuation. 47 completed responses are
retained (40 correct); one unfinished request was cancelled. This small prefix
must not be reported as full MMLU-Pro accuracy. The TPU and model server remain
running; the server was verified ready and idle after cancellation.

## Purpose

Measure practical deployment accuracy over the full public test split instead
of using FP64/ULP equality as the primary model acceptance criterion. This does
not declare the earlier full-model probability shifts harmless or prove all
kernels correct. No production kernel, scheduler or server setting is changed.

## Fixed experimental setting

| Item | Setting |
| --- | --- |
| Model | Full 43-layer official DeepSeek-V4-Flash checkpoint |
| Checkpoint revision | `60d8d70770c6776ff598c94bb586a859a38244f1` |
| Hardware / parallelism | v5p-8, 4 physical chips; TP4 / EP4 / DP1 |
| Runtime | Existing SGLang-JAX HTTP service, `integration/deepseek-v4`, base `270dfcbd` with preserved fixes |
| Runtime fingerprint | `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3` |
| Precision | Original FP4/FP8/block-scale storage, online dequantization; existing BF16/FP32 compute path |
| Dataset | `TIGER-Lab/MMLU-Pro`, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` |
| Coverage | All 12,032 test questions, 14 categories; no answer/length filtering |
| Examples | Five validation examples from the same category; no test solutions in prompts |
| Encoding | Pinned official V4 encoder, Non-Think `chat`, pinned tokenizer |
| Sampling | Greedy, temperature 0, top-p 1, max new tokens 2048, normal EOS |
| Concurrency | 1, preserving the previous accuracy protocol |
| Serving | Context 8192, chunked prefill 128, page size 128, original max running requests 4 |
| Prompt lengths | 825–2575 tokens; zero context-limit cases including full output budget |
| Processing order | SHA256(`20260911:full:category:question_id`); all questions, score-independent |

The pinned Arrow files are loaded directly after SHA-256 verification, avoiding
offline cache revision fallback. The historical 56 prompts are reproduced
exactly. Full test coverage uses natural category counts rather than the old
four-per-category sampling weights.

Full `cases.jsonl` SHA-256:
`cdfc1b7b49d80a63f121a7b913f94c7063c6afdc2307dfd55ee7e41ae2be9257`.
Frozen runner SHA-256:
`2005a55b4f43b6d33f5c968787a05fe27800412a41bce0bc59bbaa25540f4950`.

Protocol reference: the [MMLU-Pro benchmark repository](https://github.com/TIGER-AI-Lab/MMLU-Pro).
This follows our frozen 5-shot generation/extraction protocol, not DeepSeek's
private official evaluation settings. A close score is not proof of official
runtime equivalence; the earlier 47/56 is a small regression sample, not a
like-for-like overall baseline for this full-test score.

## Scoring, interruption and backup policy

- Score actual generated text using the unchanged three-tier answer extractor.
  Invalid, truncated, failed and context-limit cases count wrong. The final
  denominator is always 12,032; no score-based retry or random guessing.
- Save each result atomically with `fsync` on the persistent model disk, with
  raw response, answer, failure category, timing and protocol hash.
- Resume only unattempted question IDs. A recorded in-flight attempt without
  a durable result is retained as interrupted/failed, not silently retried.
- Stop on three consecutive request failures, changed runtime/settings or a
  72-hour run lease. No automatic TPU recreation, server restart or resource
  expansion. Recovery requires explicitly starting a compatible continuation.
- The local backup process incrementally copies immutable evidence every five
  minutes, with a separate progress snapshot. It reports both locally copied
  and remotely completed counts. It never deletes remote or local results.
- If the laptop sleeps or loses connectivity, remote disk checkpoints continue;
  local backups catch up when the process runs again. The local backup has its
  own 72-hour lease and stops after final/error-state synchronization. On normal
  completion it re-scores and hashes the downloaded records automatically.

`progress.json` distinguishes accuracy among completed records from a
full-denominator lower bound. Neither is a final score while incomplete.
Final audit reports micro accuracy, category counts/scores and category-macro
accuracy, plus failed/truncated/invalid/context-limit counts.

## Harness preflight failure retained

Attempt `v4-mmlu-full-20260911-01` completed prompt preparation but failed before
any inference request: Python `str.splitlines()` split Unicode line separators
inside valid JSON strings. The original log, frozen source and protocol remain
on disk. The second attempt reads physical JSONL records via file iteration;
the complete cases hash is unchanged. This is an evaluation I/O fix, not a model
or numerical change. No model answer was retried because of this failure.

## Reproduction and current status

Public runner: `scripts/evaluate_deepseek_v4_mmlu_full.py`.
Offline audit: `scripts/audit_deepseek_v4_mmlu_full.py`.
CPU-only tests: `python/sgl_jax/test/test_deepseek_v4_mmlu_full.py` — 16 passed,
including Unicode JSONL, completed/interrupted resume, failed-case denominator,
truncated correct-letter scoring, and atomic record replacement.

Remote evidence under the persistent disk:
`profiles/v4-mmlu-full-20260911-02/`.
Ignored local mirror:
`GCP_login/results/v4-mmlu-full-20260911-02/`.
See `progress.json`, `mirror-status.json` and `results/*.json` for live status;
`audit.json` is produced locally only after complete synchronization and scoring.
No commit or push in this wave.
