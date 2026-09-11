# DeepSeek-V4-Flash: integration snapshot and experiment ledger

Updated on 2026-09-12 (Singapore). This is the entry point for this branch's V4 deployment source,
experiment settings and results. The historical reports linked below retain
their original stage, source identity, failures and acceptance scope; their
older uses of "current" do not supersede this ledger.

## Latest numerical and task-accuracy baseline — 2026-09-11

The [full benchmark report](deepseek_v4_full_benchmarks_20260911.md), completed
on September 12, is the current task-accuracy summary: **GPQA Diamond 143/198
(72.22%)**, **GSM8K 1,280/1,319 (97.04%)**, **HumanEval 149/164 (90.85% greedy
pass@1)**. Every named test split is complete. GSM8K took 77.47 minutes and
HumanEval 19.37 minutes; no new inference failures or truncations. Format-invalid
outputs remain wrong (one GSM8K, five HumanEval); no parser changes or retries.
The serving runtime is unchanged. The earlier GSM8K-128 result below is a
historical smoke test, not the formal benchmark score.

See the [v5p release/evaluation report](deepseek_v4_v5p_release_20260911.md)
for the published EP reduction, RoPE and CSA index-pooling fixes, full
GPQA Diamond **143/198 (72.22%)**, fixed GSM8K-128 **123/128 (96.09%)**,
and the current-source short HTTP performance check. It distinguishes the
older 4K/8K ModelWorker matrix from the latest runtime and retains numerical
limitations and failed-before-fix test receipts. This is not full production
or official-runtime equivalence certification.

## Historical accepted scope — 2026-09-10

The full 43-layer model runs through SGLang-JAX ModelWorker with paged state,
chunked prefill, KV updates and the output head in the whole-model compiled
path. Original Pallas mHC/CSA/HCA implementations and shared GMM scheduling are
connected through V4-specific adaptations. Checkpoint FP4/FP8/E8M0 storage and
online dequantization remain intact; no full BF16 expert collection is kept.
The tuned adapter permutes only compact FP4 scales at loading to avoid repeated
device scale-layout copies. Scheduler algorithms are not replaced.

`gmm_tuned` is still an explicit experimental choice, not the default. Its
numerical and ModelWorker profile gates pass. Engine/HTTP stress for this
latest combination is **deferred by user choice**, not completed or required
for this source-publication step. Earlier Engine/HTTP passes refer to their
own older backend configuration. No official GPU/API accuracy equivalence is
claimed.

## Experiment E01: same-node FP4 adapter A/B

Machine-readable source of exact values:
[20260909-v5p4-fp4-model-ab.json](../../benchmark/deepseek_v4/experiments/20260909-v5p4-fp4-model-ab.json).
Detailed evidence interpretation:
[FP4 integration report](deepseek_v4_fp4_integration.md).

| Setting | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash`, full 43 layers |
| Checkpoint revision | `60d8d70770c6776ff598c94bb586a859a38244f1` |
| Checkpoint size | 73 files; 159,630,041,626 bytes |
| Hardware | One `v5p-8` Spot: four physical chips / eight TensorCores; both variants on the same node |
| Runtime | `v5p-ubuntu-2204`, `us-east5-a`; original independent 500 GB data disk |
| Software | Python 3.12.14; JAX/jaxlib 0.11.1; libtpu 0.0.46.1; NumPy 2.2.6; CPU XProf 2.23.1 in a separate venv |
| Parallelism | `tp_size=4`, `ep_size=4`, `dp_size=1`; 41 CSA/HCA attention layers head-sharded; first two SWA layers and compressor/KV replicated |
| State capacity | Context 8192; four requests; 33,280 physical token slots; page size 128 |
| Workload | Two deterministic 7936-token prompts; 128-token chunks; 256 teacher-forced decode steps through position 8191; reordered B1/B2/B4 |
| Framework settings | V4 paged backend; `moe_backend=epmoe`; mixed chunks off; overlap off in this ModelWorker harness; precompile disabled, whole-model JIT still enabled |
| Sampling / timing | Seed 42; two rounds; exclude first three calls per round, compilation/cache misses and profiler calls; no outlier trimming |

Exact dependency snapshots are in the
[experiment directory](../../benchmark/deepseek_v4/experiments/README.md).

Both sides explicitly select Pallas mHC/HCA/CSA, attention TP, batched CSA
decode, FP8 GMM, fused normalization, merged projections and fused inverse-RoPE
plus `wo_a`. **Only V4 MoE changes from `gmm` to `gmm_tuned`.** The framework's
`moe_backend=epmoe` alone does not select either V4 implementation.

Candidate model overrides (passed via `json_model_override_args`):

```json
{"v4_mhc_backend":"pallas","v4_hca_backend":"pallas","v4_csa_backend":"pallas","v4_moe_backend":"gmm_tuned","v4_attention_tp":true,"v4_csa_decode_batch":true,"v4_fp8_backend":"gmm","v4_fused_norm":true,"v4_merged_projections":true,"v4_fused_wo_a":true}
```

| Workload | GMM tokens/s | Tuned tokens/s | Gain | GMM median / p95 ms | Tuned median / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill fixture 0 | 322.90 | 450.98 | 39.66% | 396.43 / 400.98 | 283.99 / 287.79 |
| Prefill fixture 1 | 326.81 | 459.42 | 40.58% | 391.81 / 395.21 | 278.60 / 282.36 |
| Decode B1 | 22.79 | 25.28 | 10.91% | 43.84 / 44.75 | 39.51 / 40.23 |
| Decode B2 aggregate | 34.28 | 38.08 | 11.07% | 58.37 / 60.29 | 52.57 / 54.09 |
| Decode B4 aggregate | 62.58 | 68.57 | 9.58% | 63.93 / 65.72 | 58.34 / 59.73 |

Each side retains 116/118 prefill and 504/504/503 decode warm calls. Timers
include worker/sampler execution, completion wait, full-logit host transfer
and finite/argmax checks; exclude prefix preparation/restore and oracle
comparisons. These are ModelWorker measurements, not HTTP serving throughput.
Device captures are separate; do not subtract them from warm medians and label
the difference scheduler idle time.

Allocator memory (maximum over four chips): loaded 44.44646 GiB/chip for both;
B4 completion 44.77522 GiB control / 44.77875 GiB tuned; observed high-water
49.34152 / 49.34506 GiB. High-water includes harness prefix restoration and
accumulated compiled shapes, not just steady-state serving memory.

Numerical / execution acceptance:

- All 3832 checks on each side pass; all 3832 paired output hashes are bitwise
  equal. Maximum native-fixture NRMSE is 1.6165744e-7; all finite/top-1 equal.
- Prior same-source cold 8K native: 2424 checks pass. Independent from-empty
  oracle: 514 rows pass, 435 bitwise, maximum NRMSE 1.0527185e-4. Both oracle
  fixtures at position 8023 are bitwise equal. These `-01` fixtures retain their
  actual provenance and are not relabeled as new-node runs.
- Actual tuned B4 HLO contains 129 tuned GMM calls and zero full scale-layout
  copies; control has 129 original GMM calls and 43 scale copies. All selected
  kernel/TP ownership gates pass.
- Five captures on each side pass coverage and raw timing audits. GMM and MoE
  collective durations match raw TensorCore events within 1 ns. Collective
  time includes waiting; grouped GMM time includes conversion. Neither is an
  isolated dequant or hardware-bandwidth measurement.

Residual hotspots are recorded, not claimed solved: tuned B4 compressor/paged
state 11.49 ms, FP4 GMM 10.38 ms, MoE collective/wait 6.51 ms and FP8 projection
4.88 ms, averaged per TensorCore per profiled call. The broad compressor bucket
must be split into kernel work and state handling before deciding on fusion.
Prefill still has 119.26 ms of FP4 GMM work. No further tuning is performed in
this publication snapshot.

## Source and evidence identity

The containing Git commit snapshots the deployment work previously uncommitted
on top of `d5e58ee6a9720eb5bba1749c54fe471e1d437ce4`. That base commit alone is
**not** the measured implementation. The measured source fingerprints are:

```text
framework: 82f6e542105c208a410d4561f1eef6ba360f3911c3643f46570b7eb13cabd363
reference: e040be404de999f234c9563112e1e16bc4a7f75e86667d80ccae8880d2348eb4
```

Control/candidate report hashes, HLO hashes, native/oracle fixture hashes and
the exact warm metrics are in the JSON record. Both private result copies
were verified across all 110 A/B artifacts (4,359,116,886 bytes), including
raw traces, detailed exports and logs. Credential files, account/project
configuration, checkpoint binaries, raw logits and traces are excluded from
Git. `GCP_login/` is ignored at repository level as well as locally.

Publication verification repeats CPU execution/receipt/framework/paging tests:
188 passed, seven hardware tests explicitly skipped. The initial lightweight
local attempt had 79 passes and one missing-`psutil` dependency failure; the
complete pinned environment resolves that check without changing source or
tolerances. The latest earlier replacement-machine hardware receipt is 94
TPU tests passed with no skips. Publication is not a new TPU performance run.

Source-preservation checks compare the published Python files with the measured
local snapshot and recompute the model/reference fingerprints. The implementation,
test and benchmark Ruff check passes with `UP042` excluded: its two enum warnings
already exist in the base commit. An all-scripts lint pass additionally reports
13 existing closure/unused-import advisories in three historical diagnostic
scripts; these scripts are retained unchanged, not claimed lint-clean. No broad
formatting or runtime-source changes are bundled with publication.
The staged whitespace check also reports one preserved extra blank line at EOF
in `scripts/run_deepseek_v4_8k_engine.py`; that historical file is retained
byte-for-byte. These style warnings are not numerical test failures.

## Reproducing the A/B

Use a matching four-chip TPU environment and one TPU-owning process at a time.
All output directories must be new. The following Bash commands are a recipe,
not commands automatically run by reading this document. Set paths explicitly;
never place checkpoints or credentials in tracked directories.

```bash
export JAX_PLATFORMS=tpu
export JAX_COMPILATION_CACHE_DIR=/path/to/persistent-jax-cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
V4_CHECKPOINT=/path/to/original-checkpoint
V4_RESULTS=/path/to/new-experiment
V4_BASELINE=/path/to/accepted-earlier-native/report.json
V4_OPTIONS=(--mhc-backend pallas --hca-backend pallas --csa-backend pallas
  --attention-tp --csa-decode-batch --fp8-backend gmm
  --fused-norm --merged-projections --fused-wo-a)

python scripts/run_deepseek_v4_paged.py \
  --checkpoint "$V4_CHECKPOINT" --output "$V4_RESULTS/short" \
  --moe-backend gmm_tuned "${V4_OPTIONS[@]}"
python scripts/run_deepseek_v4_8k_native.py \
  --checkpoint "$V4_CHECKPOINT" --output "$V4_RESULTS/native" \
  --moe-backend gmm_tuned "${V4_OPTIONS[@]}" --cold-prefill \
  --check-kernel-dispatch --baseline-report "$V4_BASELINE"
python scripts/validate_deepseek_v4_8k_oracle.py \
  --native-report "$V4_RESULTS/native/report.json" --output "$V4_RESULTS/oracle"

python scripts/profile_deepseek_v4_fp4_model.py \
  --checkpoint "$V4_CHECKPOINT" --native-report "$V4_RESULTS/native/report.json" \
  --oracle-report "$V4_RESULTS/oracle/report.json" --output "$V4_RESULTS/control" \
  --rounds 2 --moe-backend gmm "${V4_OPTIONS[@]}"
python scripts/profile_deepseek_v4_fp4_model.py \
  --checkpoint "$V4_CHECKPOINT" --native-report "$V4_RESULTS/native/report.json" \
  --oracle-report "$V4_RESULTS/oracle/report.json" --output "$V4_RESULTS/tuned" \
  --control-report "$V4_RESULTS/control/report.json" \
  --rounds 2 --moe-backend gmm_tuned "${V4_OPTIONS[@]}"
```

The earlier native fixture and its referenced files are private artifacts. If
they are unavailable, the native driver permits omitting `--baseline-report`,
but that is a new baseline and does not establish historical compatibility;
still complete the independent from-empty oracle before profiling. On the
published run, both profile sides explicitly use the same accepted tuned
native/oracle fixture, separately from the `gmm` A/B control.

After both warm runs exit, export each output with the separate XProf venv:

```bash
JAX_PLATFORMS=cpu /path/to/profile-venv/bin/python \
  scripts/analyze_deepseek_v4_fp4_moe.py --profile "$V4_RESULTS/tuned" --export
JAX_PLATFORMS=cpu /path/to/inference-venv/bin/python \
  scripts/audit_deepseek_v4_fp4_profiles.py --profile "$V4_RESULTS/tuned" --layers 43
```

Repeat for `control`. Keep raw captures, warnings, grouping witnesses and
audits. Do not run CPU exports or large backup transfers during warm timings.

## Historical experiment index

| Stage | Record | Interpretation |
| --- | --- | --- |
| Position-8023 repair | [Correctness diagnosis](deepseek_v4_8023_correctness_report.md) | Earlier numerical failures stay failed; repair and regression are explicit |
| Original kernels / CSA repair | [Kernel integration](deepseek_v4_original_kernel_integration.md), [CSA integration](deepseek_v4_csa_integration.md), [compressor fusion](deepseek_v4_csa_compressor_fusion.md) | Separate isolated-kernel and integrated model gates |
| TP4 and batched CSA | [Four-chip acceptance](deepseek_v4_four_chip_acceptance.md), [profile](deepseek_v4_four_chip_profile.md) | Historical serving acceptance for its exact options, not tuned FP4 acceptance |
| FP8 / normalization / projection kernels | [Dense kernels](deepseek_v4_dense_kernels.md), [dense profile](deepseek_v4_dense_profile.md) | Earlier A/B; do not sum isolated speedups into model throughput |
| FP4 isolated tuning | [Tuning](deepseek_v4_fp4_tuning.md), [MoE breakdown](deepseek_v4_fp4_moe_profile.md) | Single-layer timings, not full-model throughput |
| FP4 full-model integration | [Latest integration and A/B](deepseek_v4_fp4_integration.md) | E01 above; interrupted `-01` candidate is not used |

For future comparisons append a new immutable JSON record and ledger row with
the Git commit, fingerprints, checkpoint/runtime, actual selections, workload,
timer/sample policy, correctness status and artifact hashes. Use the same
machine and otherwise identical options for an A/B. Keep serving and
ModelWorker results in separate categories.
