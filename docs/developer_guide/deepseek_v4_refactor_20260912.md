# V4 kernel / model-adaptation boundary refactor — 2026-09-12

Historical stage-one record. The subsequent
[PR cleanup](deepseek_v4_pr_cleanup_20260912.md) removes the temporary aliases,
combines adapters and relocates B1 test support. That report describes the
current working tree; the compatibility policy and counts below describe
the earlier stage only.

Baseline: `e8e1eb8896f49c138cffb377f848f15be8dff5c1` on
`integration/deepseek-v4`. This is a source-organization refactor, not a new
performance optimization or a claim of fresh TPU/model accuracy acceptance.

## Ownership

| Responsibility | Canonical location |
| --- | --- |
| CSA/HCA/mHC computation and hardware schedules | `srt/kernels/{csa,hca,mhc}/` (unchanged in this refactor) |
| FP4 formats, tile decoding and shared-GMM adapter | `srt/kernels/low_bit/` (existing arithmetic retained) |
| FP8 tile decoding and Linear | `srt/kernels/low_bit/fp8.py` |
| V4 tensor-only normalization and fused inverse-RoPE/output projection | `srt/kernels/deepseek_v4/{normalization,projection_kernels}.py` |
| Attention/MoE composition, backend selection, TP/EP numerical rules | `srt/layers/deepseek_v4/` |
| Compressor request scratch, page snapshots and restoration | `srt/layers/deepseek_v4/compressor.py` |
| V4 configuration, checkpoint-key-aware fallback arithmetic and rounding rules | `srt/layers/deepseek_v4/numerics.py` |
| Host-only checkpoint projection packing | `srt/model_loader/deepseek_v4_packing.py` |
| Model, loader, paged backend and memory-pool entry points | Existing standard framework locations retained |

Dependency direction: model adaptation calls kernels. Tensor kernel
implementations must not import V4 layers, loaders, request managers or cache
ownership code. A V4-specific fused kernel need not pretend to be universal.

The former `kernels/deepseek_v4/projections.py` mixed three responsibilities:
host checkpoint packing, model-key dispatch and a Pallas fusion. These now
live in the loader, model-adaptation and kernel modules respectively.

No scheduler, allocator, RadixCache, GMM driver, TPU schedule, sharding policy,
default backend, numerical formula or frozen benchmark scorer is changed.
The existing GMM adapter and precompile hook remain model-enabling interfaces,
not a separately claimed framework feature.

## Compatibility and evidence

Twelve former `kernels/deepseek_v4` modules remain as compatibility aliases.
They contain no computation and refer to the same module objects as their
canonical replacements, preserving private diagnostic imports and monkeypatch
behavior. Production V4 entry points import canonical paths directly. These
aliases are a temporary migration exception to the directory dependency rule,
not a second implementation; removal belongs in a later explicit API cleanup.

Existing diagnostic scripts/tests may keep the compatibility imports. Historical
reports retain their original paths and source hashes; use their original Git
revision or archived source when reproducing them. New runs must generate new
protocols and source fingerprints. Never rewrite old hashes to accept changed
files or claim the old accuracy/performance tables were measured on this tree.

Both runtime and profile fingerprints now include the new model-adaptation
directory and packing module. Profile function ranges follow canonical source;
FP8/projection grouping recognizes the relocated paths. Analyze historical
line-number-based traces with the matching archived analyzer, not current ranges.

## Validation and remaining acceptance

- Source comparison against the baseline: all 47 moved top-level functions /
  classes retain identical ASTs after ignoring imports. Module paths and code
  locations change; this is not proof of identical XLA lowering.
- Boundary tests cover canonical/legacy module identity, private names and
  monkeypatches, one-way kernel dependencies, production imports, fingerprint
  coverage/agreement and relocated profile attribution.
- Local CPU regression results are recorded below. The four-device tests use
  four logical CPU devices, not TPU emulation or a TPU performance result.
- TPU kernel tests, full 43-layer / 8K logits and persistent-state comparisons,
  and same-input TTFT/TPOT/HBM checks still require a live v5p slice. This task
  does not recreate a cloud machine or launch long benchmarks.

Before a mainline merge, complete those device gates without relaxing numerical
tolerances. Public kernel API cleanup and compatibility-alias removal can then
be reviewed separately rather than mixed with arithmetic changes.

### Local results

Isolated macOS Python 3.12 environment: JAX/jaxlib 0.11.1, Flax 0.12.9,
NumPy 2.2.6, PyTorch 2.14.0, pytest 8.4.2. Tests use `PYTHONPATH=python` and
`JAX_PLATFORMS=cpu`; four-device runs additionally use
`XLA_FLAGS=--xla_force_host_platform_device_count=4`.

| Suite | Result |
| --- | --- |
| Boundaries, collectives, RoPE, evaluation protocols, profile/report checks (four CPU devices) | 182 passed, 1 skipped |
| MoE, projection kernels, paging, HCA/CSA/mHC/FP4 integration (four CPU devices) | 128 passed, 90 skipped, **1 failed** |
| Framework contracts, parallel adaptation, compressor invariance and EP reference (four CPU devices) | 59 passed, 10 skipped |
| Earlier single-device boundary/MoE/projection/paging check (overlaps rows above) | 97 passed, 20 skipped |

The failure is
`test_deepseek_v4_projection_kernels.py::test_qnorm_and_grouped_wo_a_tp4`.
On four logical CPU devices its QNorm assertion differs from the unsharded
reference at 2/294912 BF16 values. The unmodified baseline commit was checked
out in a separate temporary worktree and rerun in the **same environment**:
it fails at exactly the same positions and raw BF16 values as the refactor.

| Coordinate | Actual uint16 bits | Reference uint16 bits |
| --- | ---: | ---: |
| `[6, 10, 505]` | 14427 | 14426 |
| `[8, 17, 488]` | 15178 | 15179 |

This establishes a pre-existing CPU failure, not a new refactor regression.
It remains failed: no tolerance, skip condition, expected value or computation
was changed. The test stops at QNorm, so this rerun does not certify its later
grouped-output assertion. Device-specific tests remain skipped on CPU.

Model-class import, Black/isort on new canonical modules, Ruff on the moved
implementation/boundary-test scope and `git diff --check` pass. Initial broader
test collection required installing missing dependencies in the isolated local
environment; no serving environment or repository dependency pins were changed.
