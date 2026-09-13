# DeepSeek-V4 validation and result entry points

Use this index to review model acceptance without treating every historical
debug script as a serving component. Paths below are relative to the repo root.
Existing scripts remain in place to preserve their helper imports and evidence
paths; this index does not imply they were all rerun after a source change.

| Purpose | Entry points |
| --- | --- |
| Source ownership and cleanup acceptance | `docs/developer_guide/deepseek_v4_pr_cleanup_20260912.md`, `python/sgl_jax/test/test_deepseek_v4_module_boundaries.py` |
| Shared execution options and source identity | `python/sgl_jax/srt/configs/deepseek_v4_execution.py`, `scripts/deepseek_v4_source.py` |
| Runtime and checkpoint settings / published results | `docs/developer_guide/deepseek_v4_flash_experiments.md` |
| Completed same-node refactor A/B | `docs/developer_guide/deepseek_v4_refactor_ab_20260912.md` (four cases; not HTTP or a public-task score) |
| Framework / paged ModelWorker checks | `scripts/run_deepseek_v4_framework.py`, `scripts/run_deepseek_v4_paged.py` |
| Full named accuracy splits | `scripts/evaluate_deepseek_v4_gpqa.py`, `scripts/evaluate_deepseek_v4_gsm8k.py`, `scripts/evaluate_deepseek_v4_humaneval.py` |
| Finite-batch HTTP performance and evidence audit | `scripts/benchmark_deepseek_v4_release.py`, `scripts/audit_deepseek_v4_release_benchmark.py` |
| Native / FP4 MoE profile attribution and raw audit | `scripts/analyze_deepseek_v4_native_profile.py`, `scripts/analyze_deepseek_v4_fp4_moe.py`, `scripts/audit_deepseek_v4_fp4_profiles.py` |

For exact benchmark options, use each evaluator's `--help` and the protocol in
its linked experiment report. Preserve original seeds, splits, generation
budgets, scorers and sandboxing. HumanEval executes generated code and must
use its configured sandbox, not an unrestricted host process.

`debug_*`, `probe_*`, `replay_*`, `isolate_*` and numerical-candidate scripts
are targeted diagnostics, not extra runtime backends. Microbenchmarks are not
end-to-end TTFT/TPOT. A cancelled MMLU run is not a full-dataset accuracy result.
The [diagnostic archive index](../../scripts/diagnostics/deepseek_v4/README.md)
classifies all 30 named diagnostic entry points while preserving their paths.

## Fast source-refactor checks

From the repository root, using the local test environment:

```sh
PYTHONPATH=python JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
python -m pytest -q \
  python/sgl_jax/test/test_deepseek_v4_execution_options.py \
  python/sgl_jax/test/test_deepseek_v4_refactor_support.py \
  python/sgl_jax/test/test_deepseek_v4_module_boundaries.py \
  python/sgl_jax/test/test_deepseek_v4_parallel_integration.py
```

These are CPU policy/boundary checks, not full TPU acceptance. The module
boundary tests also check runtime/profile fingerprint agreement and coverage;
the support tests enumerate every current backend/boolean combination and
check diagnostic-index coverage. Use `PYTHONPATH=python` for CLI tools that
import the shared execution policy; the policy itself needs no JAX or serving
dependencies. The standalone profile analyzer keeps its existing invocation.

The `execution_helper_sha256` receipt now binds both the CLI adapter and shared
policy bytes (adapter bytes, a NUL separator, then policy bytes). The framework
fingerprint includes the shared policy and its own manifest implementation.
Historical receipts retain their original hash definition and source snapshot.

Keep raw answers, checkpoints and account data outside tracked publication
artifacts. A source refactor changes fingerprints: generate new evidence for
new runs, and analyze old source-line-based traces with the matching archived
analyzer. Do not relabel old accuracy/performance tables as measurements of
the reorganized code.
