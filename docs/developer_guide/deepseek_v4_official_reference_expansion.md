# Expanded official-Python compressor comparison

Status: **testing and diagnosis complete, not all numerical gates pass**,
2026-09-11. Experiment
`v4-official-expand-20260911-04`.

Attempt `-01` encountered a diagnostic logging argument collision before any
numerical check. It is retained as a harness error, not a model result; its
controller restored the original HTTP service. Attempt `-02` adds a regression
test for that collision (six comparison/logging self-tests pass), but passed
the decode-only batching flag to prefill; the kernel rejected that invalid
call before producing numerical comparisons. Attempt `-03` selects the flag
only for the one-token decode calls, matching the deployed model's phase
selection. Attempt `-03` then exposed the missing explicit `shard_map` wrapper
for four-device Pallas calls; no numerical comparison ran. Attempt `-04` adds
that wrapper, matching the native model's manual SPMD execution boundary.
Earlier attempts are retained as harness errors, not model numerical failures,
and each restores the original service.

This is an independent **operator-level** check, not official GPU inference
or a full 43-layer logits comparison. No production model, kernel, scheduler,
checkpoint, or numerical tolerance is modified for this experiment.

## Reference and candidate

- Checkpoint: `deepseek-ai/DeepSeek-V4-Flash`, revision
  `60d8d70770c6776ff598c94bb586a859a38244f1`.
- Official `inference/model.py` `Compressor` control flow is unmodified.
  The existing independent PyTorch CPU oracle replaces GPU-only quantization
  entry points and CUDA Hadamard rotation. Its implementation does not call
  production JAX low-bit routines. This does **not** execute official TileLang
  GPU kernels or establish their bitwise equivalence.
- Candidate: native paged `compress`, explicit Pallas CSA/HCA backends and
  `csa_decode_batch=True`, source fingerprint
  `2543e8b0e98fc56658912cd2d14f05e4d4ae4e00cc961f07bc150744070e0c0f`.
- Four physical v5p chips, replicated compressor state/weights. This isolated
  operator does not test tensor-parallel attention or expert-parallel MoE.

The compressor projection weights themselves are checkpoint BF16 matrices;
the FP4/FP8 operations tested here are activation quantize-dequantize paths for
compressed KV/index values. This matrix does not exercise FP4 expert GEMMs.

## Frozen matrix and comparisons

The matrix selects early, middle, and last CSA/HCA layers from the checkpoint
configuration, testing CSA main, CSA index, and HCA main compressors at B1/B4.
Early layers use an 8192-token trajectory; middle layers 4096; last layers 256.
This is 18 cases, not every combination of layer and length.

Inputs are genuine checkpoint embedding rows selected by fixed per-request
seeds `20260911 + request`, independently RMS-normalized in CPU FP32 and rounded
once to BF16. Both implementations receive identical arrays with a recorded
checksum. These are standalone operator stimuli, **not** hidden activations
captured from a complete model forward pass.

The native path prefills in 128-token chunks with a 127-token tail. The official
Python path prefills the same prefix in one call from empty state. Both then
consume identical one-token activations for 129 steps (short case) or 257 steps
(4K/8K), reaching the final context position. Thus the early 8K case includes
position 8023 and both CSA/HCA emission boundaries. Native request order changes
between calls; slots are permuted and physical pages are non-contiguous.

Checks compare every new compressed vector after FP4/FP8 activation
quantize-dequantize and the live FP32 projection state after every decode step.
Unused or stale physical state rows are not compared as logical state. Existing
real-checkpoint output tolerances are 0.025 NRMSE for main and 0.035 for index;
this test additionally applies them per vector to prevent error dilution.
Projection-state tolerance is 5e-7, including a per-vector check. A mismatch is
recorded for diagnosis rather than converted into a pass by changing limits.

Raw final arrays, first-failure arrays, input IDs, per-step metrics, source and
weight hashes, and process receipts are saved on the model data disk. The
controller restores the original 8192-context/B4 HTTP service after testing.

## Results

All 18 cases executed: **13 PASS, 5 NUMERICAL_DIFFERENCE**, zero execution errors
in the final matrix. There are 8436 metric records, of which 16 exceed the
predeclared gate. A record can cover several vectors, so this is not a count
of 16 individually failing vectors. All failures concern CSA **index activation
FP4 quantize-dequantize outputs**, not FP4 expert weight loading.

| Operator | Cases passing | Worst compressed-vector NRMSE |
| --- | ---: | ---: |
| CSA main (layers 2/22/42) | 6/6 | 0.82704% |
| HCA main (layers 3/23/41) | 6/6 | 0.04030% |
| CSA index (layers 2/22/42) | 1/6 | 9.73297% |

All live FP32 state checks pass; worst per-vector NRMSE is
`2.464556490990983e-7`, below `5e-7`. For all nine paired fixtures, request 0's
entire final compressed cache is bitwise equal between native B1 and B4. This
is evidence against a batch-dependent error for these fixtures, not general
distributed inference certification.

The failing cases are layer 42/index/B4/256; layer 2/index/B1 and B4/8192;
layer 22/index/B1 and B4/4096. Even these cases' final whole-cache NRMSE is only
0.275–0.599%; the additional per-vector criterion catches sparse differences
that an aggregate cache norm hides. Their statuses remain failed under the
new test gate; no threshold was relaxed.

### Faithful replay of the first short-case difference

Diagnostic: `v4-official-index-diagnose-20260911-02`, layer 42/index/B4,
zero-based position 219, request 3, channel 75. Attempt `-01` lacked a CPU
backend needed for `jax.debug.callback`; it is retained as a diagnostic error.
The retry enables `JAX_PLATFORMS=tpu,cpu` only for the diagnostic child.

- All final cached output/state arrays in the traced replay reproduce the
  frozen untraced fixture bitwise.
- Feeding **identical pre-quantization inputs** to the independent CPU FP4
  quantizer reproduces the native FP4 outputs bitwise.
- A difference is already observable in the compressor's post-RoPE output:
  worst-vector NRMSE `0.0006226849719225129`, max absolute `0.00390625`.
- After Hadamard, worst-vector NRMSE is `0.001628009518802104`.
- After FP4 QAT, worst-vector NRMSE is `0.039276134381716975` (3.93%).

| Channel 75, request 3 | Official Python/CPU shim | Native TPU |
| --- | ---: | ---: |
| Before FP4 QAT | 0.625 | 0.62890625 |
| After FP4 QAT | 0.5 | 0.75 |

The two pre-QAT values differ by one BF16 spacing. The exact midpoint goes to
0.5, while the slightly larger input goes to 0.75. This replay establishes
**upstream numerical differences amplified by an FP4 threshold**, not a faulty
FP4 conversion for identical inputs. It does not yet distinguish projection,
pooling, norm, and RoPE as the original cause, because the fused Pallas emitter
does not export those internal stages. It also does not certify the other four
failing cases have the same cause, or that downstream top-k/logits are unaffected.

## Remaining scope

Even a pass does not establish end-to-end numerical equivalence. Full-model
official-reference teacher-forced logits, actual intermediate hidden states,
attention/index top-k effects, TP/EP collectives, and free generation require
separate checks. Existing independent EP and batch-consistency regressions must
not be relabeled as an official full-model oracle.

## Service and evidence

The original HTTP service is restored and verified ready/idle: PID `304237`,
context 8192, four request slots, 33280 KV tokens free, TP4/EP4/DP1,
chunk/max-prefill 128, overlap enabled, mixed chunks disabled. The source
fingerprint is unchanged. No production code or checkpoint was modified;
no commit or push was performed.

Evidence roots:

- Data disk: `/mnt/disks/deepseek-models/profiles/v4-official-expand-20260911-04`.
- Target replay: sibling `v4-official-index-diagnose-20260911-02`.
- Local: `GCP_login/results/v4-official-expand-20260911-04`.
- Matrix audit: `audit.json`; final service receipt: `final-service-check.json`.
- Archive receipt/hash: `archive-result.json`; local verification receipt:
  `local-verification.json` (written after transfer).

The evidence archive is retained on the data disk and locally: 50,241,636 bytes,
SHA256 `139d3cf628ca40e0c72248d8820ab8323089c8cdbd3d9e47f7c79f2ac7540889`.
Local verification passes the archive hash, 8436 recorded gate decisions,
27 NPZ integrity checks, and 727 locally present source files against the
728-file remote source snapshot. This is provenance/integrity verification;
numerical array recomputation and the independent-quantizer replay ran remotely.
The one remote-only file is the installation-generated
`python/sgl_jax/_version.py`, which is included in the archive.
