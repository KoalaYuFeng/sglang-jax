# v5p recovery and numerical regression, 2026-09-11

Later update: [CSA index pooling integration and residual reference
adjudication](deepseek_v4_index_pool_integration_20260911.md). The results below
remain the historical **RoPE-only** run, before pool integration.

**Recovery completed. RoPE fix validated on TPU; full numerical acceptance
remains incomplete. A new pool primitive is independently tested, not enabled
in the serving path. No performance or full-model accuracy claim.**

## Restored environment

- One user-authorized v5p-8 Spot, four physical TPU chips, us-east5-a.
- Original 500 GB independent disk retained and mounted by its existing UUID;
  no formatting, disk copy, checkpoint download, or additional TPU allocation.
- Checkpoint `60d8d70770c6776ff598c94bb586a859a38244f1`: 73 files, including all
  46 safetensors shards. Prior SHA-256 verification was reused only after
  matching size and unchanged mtime/ctime; this was not a new full-file hash scan.
- Branch `integration/deepseek-v4`, base `270dfcbd`, preserving local uncommitted
  EP fixes and the RoPE correction. Initial 1195-file source manifest verified.
- Python 3.12.14, JAX/jaxlib 0.11.1, libtpu 0.0.46.1, NumPy 2.2.6,
  torch 2.14.0+cpu. Complete dependency freeze matches the prior environment
  byte for byte. Four-chip BF16 smoke test passed.

## RoPE fix: actual TPU results

Source fingerprint during the complete matrix:
`ae9d2727105a86f8e017a4928de60edc336785e177865b6b4c3ef7c5b8777d97`.
This predates adding the independently tested, unused pool module below.

- RoPE coefficient/position-219 regression: **16 passed** on TPU.
- Existing Q-normalization/KV-normalization/projection selection:
  **10 passed, 25 deselected** on TPU; no skipped tests in this selection.
- Frozen real-checkpoint operator matrix: **16/18 PASS**, 8436 metric records,
  **2 failed records**, down from the original 13/18 PASS and 16 failed records.

| Operator | Passing cases |
| --- | ---: |
| CSA main, layers 2/22/42, B1/B4 | 6/6 |
| HCA main, layers 3/23/41, B1/B4 | 6/6 |
| CSA index, layers 2/22/42, B1/B4 | 4/6 |

The two remaining cases are layer-2/index/B4/8192 and
layer-22/index/B4/4096. Their worst compressed-vector NRMSE is 4.62250% and
4.69841%, respectively, still above the unchanged 3.5% index gate.

Audit confirms unchanged input/checkpoint hashes, chunk boundaries, numerical
limits, live-state arrays, and official CPU reference arrays. All nine same-input
B1/B4 cache pairs remain bitwise equal. All six index final caches are bitwise
equal to the earlier diagnostic phase-only A/B: this is now a result from the
actual fixed runtime source, without coefficient overrides.

These are isolated four-way-replicated compressor tests with embedding-derived
activations, not complete-model TP/EP inference. The reference still uses
official Python control flow with independent CPU shims, not official GPU kernels.

## Residual projection adjudication

Reconstructed the four residual eight-token windows from the original embedding
IDs and checkpoint weights. Compared FP64 projection + FP64 pooling, projection
rounded to FP32 before FP64 pooling, and the captured CPU/native projection states.

- Layer 2/group 856/request 3 (terminal position 3427): the high-precision
  projection/pooling result rounds to the **native** BF16 pooled value `-1.25`,
  not the CPU reference's `-1.2578125`.
- Layer 22/group 272/request 2 (terminal position 1091): the high-precision
  result likewise supports the native `0.07861328125`, not CPU `0.0791015625`.

Both continue to differ after FP4 quantization. This is evidence of CPU FP32
reference sensitivity, not grounds to force a less accurate projection merely
to match that reference. It also does not establish official GPU/full-model
equivalence. Their existing failed numerical records are retained.

## Standalone pooling correction

New, **not yet wired into the serving emitter**:
`python/sgl_jax/srt/kernels/csa/numerics.py::index_pool_v4`.

It uses range-reduced FP32 exponential evaluation and accumulates unnormalized
weighted values before a single normalization division. Integral powers of two
are constructed exactly from FP32 exponent bits. It preserves FP32 inputs and
outputs; BF16 rounding, RMSNorm, RoPE, Hadamard, and FP4 remain caller-owned.
Only [groups,8,128] index windows are supported; no framework/scheduler changes.

An isolated v5p probe first reproduced the original pooled FP32 results bitwise
in all four captured windows. Changing only reduction order was insufficient.
The polynomial-exponential variant removed both pool-sensitive BF16 crossings
(layer 2/groups 1643 and 1928) against **same-input independent FP64** pooling.

The committed-to-worktree primitive then passed **9 CPU tests and 9 TPU tests**:
three randomized window sets, empty/single-live windows, 16384 exponential
inputs over [-87,0], and all four frozen real-checkpoint windows. In the frozen
windows, every pooled BF16 element equals same-input FP64 rounded to BF16.
This does not eliminate projection differences already present in those inputs.

Experiment failures are retained, not counted as model regressions or passes:

- First diagnostic attempted `optimization_barrier` inside Pallas; this JAX TPU
  lowering does not implement that primitive. The retry used supported pairwise
  reductions and retained the failed log.
- Initial standalone CPU exponential-range test exposed underflow/error in
  `exp2` lowering at integral exponents. Exact exponent-bit construction fixed
  it, without changing the test's 1e-6 relative-error limit.

## Remaining acceptance gate

The measured 18-case matrix above uses the RoPE fix and **original pool**.
Do not attribute those results to the new pool primitive. Next, connect the
independently validated primitive through the V4 index emitter and rerun the
unchanged matrix plus state/chunk/batch regressions. Preserve and separately
adjudicate CPU-reference-sensitive differences. Only then proceed to real
index top-k/full-model logits and the 43-layer 8K forward/decode gate.

No full-model HTTP server or performance run was started in this recovery.
The node is retained for continued testing. Local connection receipts and
commands are in ignored `GCP_login/CURRENT_TPU.md`; source, raw arrays, tests,
and audit reports are retained on the independent disk and backed up locally.
