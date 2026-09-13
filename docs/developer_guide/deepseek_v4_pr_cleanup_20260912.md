# V4 PR cleanup — 2026-09-12

Branch: `integration/deepseek-v4`. Base commit:
`e8e1eb8896f49c138cffb377f848f15be8dff5c1`.
This report describes local source cleanup, not a new TPU performance or
model-accuracy result. It supersedes the temporary compatibility policy in
the [stage-one refactor](deepseek_v4_refactor_20260912.md).

Subsequent verification: the user authorized a Spot rebuild and the
[four-case TPU A/B](deepseek_v4_refactor_ab_20260912.md) has now completed.
Captured logits, generations and logical state hashes match; warm timing
changes are below 1% and allocator snapshots match. The local-only actions
and pending-device statements below describe the earlier cleanup phase;
the linked report records the later, bounded device acceptance scope.

## Two implementation boundaries

| Ownership | Current location and scope |
| --- | --- |
| Kernel implementation | `srt/kernels/{csa,hca,mhc}/`, `srt/kernels/low_bit/`, and V4 normalization/projection fusion in `srt/kernels/deepseek_v4/` |
| Model adaptation | `srt/layers/deepseek_v4/`, standard model/loader/config entries, V4 paged backend and pool; composition, checkpoint semantics, TP/EP rules and request state |

GMM scheduling, tensor/expert parallelism, chunked prefill, paging and scheduler
algorithms are existing framework facilities. Their V4 adapters are not
described as newly developed framework features. Tile-local dequantization
is kernel work; selecting that kernel, packing checkpoint scales and connecting
its inputs to the model are model adaptation.

## Implemented cleanup

- Migrated repository runtime, tests and diagnostic imports to canonical
  owners and removed all 12 transitional `kernels/deepseek_v4` alias modules.
  They are not retained as an alternative serving implementation.
- Combined `DenseKernels` selection and `merged_linear` in
  `layers/deepseek_v4/linear.py`. Host checkpoint packing belongs exclusively
  to `model_loader/deepseek_v4_packing.py`; tensor fusion belongs to
  `kernels/deepseek_v4/projection_kernels.py`, without Linear re-exports.
- Combined routing, legacy expert dispatch and GMM expert dispatch in
  `layers/deepseek_v4/moe.py`. Backend choices and the explicit `legacy`
  default are preserved. Selected-backend errors still propagate, with no
  silent fallback.
- Combined the original and tuned checkpoint FP4 adapters in
  `kernels/low_bit/fp4.py`, removing `gmm.py` and `fp4_tuning.py` from that
  directory. Renamed `CandidateCheckpointFP4Rhs` to `TiledCheckpointFP4Rhs`;
  tile policy and arithmetic are preserved. The compiled adapter marker now
  says `checkpoint_fp4_tiled`; evidence readers accept both the historical
  candidate marker and the new marker without dropping execution/layout gates.
- Moved the historical B1 backend, pool and validator into
  `sgl_jax/test/deepseek_v4_legacy.py`. Their contract tests remain. Production
  uses the existing V4 paged backend/pool and serving validator and does not
  depend on this fixture.
- Exposed `streaming_attention_pallas` and `hca_emit_selected_pallas` as narrow
  public HCA entries. They alias the existing implementations, with no extra
  computation or scheduler ownership. V4 adapters and diagnostic monkeypatch
  targets now use those entries.
- Updated runtime/profile fingerprints and source attribution. Merged MoE
  attribution uses function ranges, including nested frames; Linear
  normalization is not classified as FP8 projection glue. Tests cover these
  boundaries and reject old imports across repository Python code.
- Final follow-up: extracted defaults and execution-option validation into
  the standard-library-only `configs/deepseek_v4_execution.py`. Model
  construction, Linear selection and CLI acceptance plumbing share this
  policy. Model/whole-forward defaults retain their original values;
  independent kernel input guards remain local.
- Final follow-up: execution and profile tools import the same
  `scripts/deepseek_v4_source.py` fingerprint function and manifest. It covers
  the shared execution policy and manifest source itself. The independent
  bring-up reference fingerprint remains unchanged and is checked for
  agreement with the helper's reference digest.
- Final follow-up: added a classified diagnostic archive index, preserving
  all 30 original script paths and shared-helper imports. Coverage tests
  reject missing or duplicate index entries; this is not a claim that the
  diagnostics were physically moved or all rerun.

The removed tracked sources remain recoverable at the base commit. Historical
reports and raw evidence were not deleted or rewritten. Private archived
diagnostics using old imports must use their matching frozen source revision.

## Validation before the final follow-up

Local environment: macOS, Python 3.12, JAX/jaxlib 0.11.1, Flax 0.12.9,
NumPy 2.2.6, pytest 8.4.2. Commands use `PYTHONPATH=python`,
`JAX_PLATFORMS=cpu` and
`XLA_FLAGS=--xla_force_host_platform_device_count=4`. These are **four logical
CPU devices**, not TPU emulation.

| Selected regression group | Result |
| --- | --- |
| Module boundaries, MoE/GMM, FP4 tuning, framework contracts, execution evidence, FP4 profile/raw audit, collectives, RoPE, GPQA/GSM8K/HumanEval/MMLU protocol checks, release/dense/8K reporting | 295 passed, 1 skipped |
| Projection kernels, paged state, HCA/CSA/mHC/FP4/parallel integration and compressor invariance | 111 passed, 100 skipped, **1 failed** |
| Total, disjoint groups | **406 passed, 101 skipped, 1 failed** |

The failure remains
`test_deepseek_v4_projection_kernels.py::test_qnorm_and_grouped_wo_a_tp4`:
2/294912 BF16 values differ by one raw-bit increment in the CPU QNorm
assertion. The unmodified base commit was reproduced in a separate worktree
and the same environment during stage one, with the same two coordinates
and bits; see the [baseline receipt](deepseek_v4_refactor_20260912.md#local-results).
This stage reproduces the two-value failure. The test stops at QNorm, so it
does not certify the later grouped-output assertion. No tolerance, expected
value or skip condition was changed to obtain a pass.

Before the execution-policy follow-up, one-off AST checks against the base
matched all 47 moved top-level definitions
after normalizing imports and API namespaces. All seven FP4 definitions also
matched after class-name/import/text normalization. These are structural
checks, not evidence of identical TPU lowering. Changed Python files pass
undefined-name/duplicate-definition checks (`ruff --select F821,F811`).

Local JUnit receipts, not publication artifacts:
`/tmp/v4-publication-check.EATkcd/pr-cleanup-core-reporting-final.xml` and
`/tmp/v4-publication-check.EATkcd/pr-cleanup-execution-final.xml`.

## Final follow-up validation

The same local four-logical-CPU environment was used after consolidating
execution policy and provenance. No TPU or checkpoint benchmark was launched.

| Disjoint regression group | Result |
| --- | --- |
| Shared policy/provenance/index tests, execution options, module boundaries and parallel integration | 82 passed, 10 skipped |
| Remaining MoE, FP4, framework, paging, CSA/HCA/mHC, projection, numerical and reporting/protocol checks | 372 passed, 91 skipped, **1 failed** |
| Total | **454 passed, 101 skipped, 1 failed** |

The only failure is the same two-value CPU QNorm BF16 difference described
above. It remains visible in the test result, with no relaxed assertion or
new skip. The user accepts the known tail-bit difference for this refactor;
it is not a reason to claim the strict numerical test passed.

Additional checks cover all 1,536 combinations of the current backend and
boolean choices, malformed boolean overrides, consumption by model
initialization before allocation, source-fingerprint sensitivity, all 30
diagnostic index entries, and offline imports with Python `-S` (no JAX,
Flax, Transformers or PyTorch). The new policy/provenance/support modules
pass Ruff. `git diff --check` passes.

Receipts: `/tmp/v4-publication-check.EATkcd/pr-final-support.xml` and
`/tmp/v4-publication-check.EATkcd/pr-final-regression.xml`. The fast support
command is recorded in the benchmark index. These local test receipts are
not new model-accuracy or throughput evidence.

## PR handoff and remaining gates

Review kernel ownership and model adaptation as two logical groups, with
their call-site migrations and tests included. Do not include credentials,
checkpoint files, raw prompts/answers or private `GCP_login` artifacts.
The [benchmark entry index](../../benchmark/deepseek_v4/README.md) distinguishes
maintained acceptance entry points from historical diagnostics.

This pass does not bulk-move or delete historical diagnostic scripts: many
share helpers and encode source-relative evidence paths. The new
[diagnostic archive index](../../scripts/diagnostics/deepseek_v4/README.md)
organizes them without breaking those paths. Further physical relocation can
be a separate change after checking those dependencies. This pass also
does not remove independent reference arithmetic or force optimized backends
to become defaults.

Before mainline acceptance, rerun the affected kernels on v5p, full 43-layer
8K logits and compressor/KV state comparisons, and same-input TTFT/TPOT/HBM
checks. CPU tests and historical benchmark tables do not replace these gates.
No cloud machine was recreated and no model benchmark, commit, push or PR
creation was performed as part of this cleanup.
