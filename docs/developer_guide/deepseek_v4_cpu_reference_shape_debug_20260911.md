# V4 CPU-reference shape diagnosis, 2026-09-11

**CPU remains the agreed reference; no GPU is required. Original acceptance
remains 15/18, with three failed records. No serving kernel, CPU oracle source,
checkpoint, scheduler, or acceptance threshold was changed in this follow-up.**

This follows [index pool integration](deepseek_v4_index_pool_integration_20260911.md).
The additional experiments diagnose the remaining failures; they are not a
replacement reference that turns the original failures into passes.

## Main finding: the CPU reproduces the three failures without a TPU

For each captured window, reconstruct identical BF16-normalized embedding
inputs and load the same real FP32-converted BF16 compressor weights. Execute
the CPU projection with 1, 8, 32, 128, 512, or the original large matrix's
number of rows. Keep the CPU softmax, normalization, RoPE, Hadamard, and FP4
operations unchanged after projection.

| Layer / request / terminal position | CPU M=8 vs original CPU vector NRMSE | CPU M=8 vs TPU final vector |
| --- | ---: | --- |
| 2 / 0 / 2571 | 3.77695% | Bitwise equal |
| 2 / 3 / 3427 | 4.18030% | Bitwise equal |
| 22 / 2 / 1091 | 4.69841% | Bitwise equal |

The original CPU shape is M=31740 for layer 2 (4 requests × 7935 prefill tokens)
and M=15356 for layer 22 (4 × 3839). To isolate shape, unused rows are zero
and the eight selected rows per request retain their exact original values
and row positions. Restoring the original large shape reproduces **both raw
projection arrays and every final vector** of the captured CPU reference
bitwise. Thus this is a controlled shape probe, not a full-model benchmark.

The CPU-only M=8 results are bitwise equal to the corresponding failing TPU
vectors, not merely equal in their reported error norm. The full-shape control
and exact target-vector comparisons are asserted in `final-audit.json`.

These observations establish shape-dependent CPU projection rounding followed
by amplification at BF16/FP4 boundaries for these three vectors. They do not
establish correctness of every TPU operator or full-model inference. They also
show that the original 3.5% per-vector FP4 gate can reject an alternative CPU
execution of the same mathematical computation.

## Pool-order probe: not promoted to runtime

The pinned PyTorch commit is
`08187d9e0fba026dc8217405802ab5381dc88d90` (2.14.0+cpu), using MKL 2024.2,
AVX512 and eight CPU threads. Its non-last-dimension softmax performs an
exponential sum followed by elementwise division; the relevant outer reduction
uses an eight-element running sum. See the pinned
[softmax source](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L696)
and [sum source](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/cpu/SumKernel.cpp#L317).

Evaluated four isolated Pallas variants on five captured windows, each with
native and CPU projection inputs: current compensated pooling, native exp with
CPU-style operation order, polynomial exp with that order, and supplied CPU
exponentials with that order. This is 40 diagnostic evaluations, not 40 passing
acceptance tests.

Polynomial exp with normalized-probability/running-sum order matches the CPU
pooled BF16 results for the five **native-input** fixtures, but still differs
on one **CPU-input** fixture. Supplying CPU exponentials also does not guarantee
bitwise equality. Merely translating the source-level reduction order is not
an established portable CPU numerical contract. No variant was enabled in the
serving path, and no FP64 test expectation was edited to accept one.

## Six-case diagnostic CPU-shape A/B

As a separate intervention, change only the diagnostic CPU index wkv/wgate
**prefill** projection calls to fixed M=8; keep CPU decode, all other CPU
operations, TPU source, inputs, checkpoint weights and numerical limits fixed.
The official CPU oracle file on disk remains untouched; the override exists
only in diagnostic child processes and is recorded in their protocol.

| Case | Diagnostic status | Worst compressed-vector NRMSE |
| --- | --- | ---: |
| Layer 42 / B1 / 256 | PASS | 0% |
| Layer 42 / B4 / 256 | NUMERICAL_DIFFERENCE | 4.32945% |
| Layer 2 / B1 / 8192 | NUMERICAL_DIFFERENCE | 3.51256% |
| Layer 2 / B4 / 8192 | NUMERICAL_DIFFERENCE | 4.34680% |
| Layer 22 / B1 / 4096 | PASS | 2.17649% |
| Layer 22 / B4 / 4096 | PASS | 2.43541% |

This remains **3/6 with three failed records**, over 2916 comparisons. Layer
22/B4 improves, but layer 42/B4 becomes a new failing case. Changing CPU
projection shape therefore does not by itself solve numerical acceptance.

Audit confirms that all six TPU candidate caches and state arrays are bitwise
unchanged from the integrated run. Input/weight hashes, chunk boundaries,
comparison labels, shapes and limits are unchanged. CPU reference results
change, as intended by this diagnostic. The old layer-22 CPU B1/B4 request-0
cache pair was not bitwise equal; all three such pairs are equal in this A/B.
This is further evidence of CPU-reference execution sensitivity, not permission
to replace the original acceptance result.

## Independent downstream checks

On the five frozen windows (layer-2 groups 642/856/1643/1928 and layer-22 group
272), supply identical captured TPU inputs to the independent CPU Hadamard and
FP4 implementations. Both stages match the TPU outputs bitwise in all five
windows. This checks **index activation quantization**, not FP4 expert-weight
loading or full-model accuracy.

## Acceptance decision still needed

Keep using the official Python + independent CPU implementation. The original
15/18 result and its 3.5% gate remain unchanged and unpassed. Emulating a
particular CPU GEMM shape is not an established fix; the diagnostic A/B also
fails. No performance run or inference server was started.

Proposed next protocol, requiring agreement before it changes pass/fail rules:
retain the original baseline, add stagewise pre-quantization CPU comparisons,
and validate actual layer/index-top-k/logits behavior after quantization.
Report boundary-induced code changes separately rather than silently increasing
the existing threshold. These proposed checks have **not** yet been used to
certify the current deployment.

## Evidence

Runtime fingerprint remains
`1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.
The exact source is in the preceding integration archive, SHA-256
`9e6b06db8ad57db9460919c6cd938917b650cd36f7d8585066c6bdc28c3f773d`.

Under `/mnt/disks/deepseek-models/profiles/`:

- `v4-cpu-contract-debug-20260911-01`: Pallas pool-order variants and inputs.
- `v4-cpu-projection-shapes-20260911-01`: 30 CPU projection-shape evaluations,
  raw arrays, exact counterexamples, downstream checks and final audit.
- `v4-cpu-shape-ab-20260911-01`: six-case reference-shape diagnostic,
  explicit override protocol, logs and final arrays.

The source, original CPU reference and previous archives are preserved.
