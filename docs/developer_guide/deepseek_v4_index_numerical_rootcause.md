# CSA index numerical divergence: causal diagnosis

2026-09-11. **Diagnosis complete; production fix not implemented.** This follows
[the expanded official-Python comparison](deepseek_v4_official_reference_expansion.md).
No runtime source, checkpoint, scheduler setting, or error tolerance was changed.
All coefficient overrides below exist only inside diagnostic child processes.

Subsequent implementation status:
[local RoPE fix and validation](deepseek_v4_rope_numerical_fix.md).
The diagnosis and A/B numbers below describe the original, unchanged source,
not a TPU validation of that follow-up fix.

## Scope and evidence

- Four physical TPU v5p chips; isolated, replicated compressor tests, not a
  full-model TP4/EP4 accuracy certification.
- Official checkpoint revision: `60d8d70770c6776ff598c94bb586a859a38244f1`.
- Runtime source fingerprint:
  `2543e8b0e98fc56658912cd2d14f05e4d4ae4e00cc961f07bc150744070e0c0f`.
- Reference: official Python/PyTorch compressor control flow with independent
  CPU replacements for GPU-only operations. This does **not** establish
  bitwise equivalence to the actual official GPU runtime.
- Frozen inputs are checkpoint embedding rows, CPU-normalized and rounded to
  BF16, not hidden activations captured from a complete 43-layer forward.
- These FP4 operations quantize compressed **index activations**. They do not
  load or multiply FP4 expert weights.

Evidence directories under `/mnt/disks/deepseek-models/profiles/`:

- `v4-index-rootcause-20260911-01`: causal phase swap, independent CPU/FP64
  decomposition, audits, scripts, and final service verification.
- `v4-index-phase-ab-20260911-01`: six frozen CSA-index cases, phase-only A/B.
- `v4-index-residual-20260911-01`: faithful captures of the remaining windows.
- `v4-index-pool-probe-20260911-01`: isolated Pallas pooling-stage exports.

## 1. Position 219: RoPE angle generation is causal

The original layer-42/B4 fixture, request 3, group start 216, terminal position
219, reproduces bitwise with the unchanged Pallas emitter. Its pre-FP4 channel
75 is `0.62890625` rather than the reference `0.625`; FP4 rounds these to
`0.75` and `0.5`, respectively.

Controlled interventions retain the exact same projection windows, emitter,
weights, and downstream routines:

1. Identity RoPE coefficients expose the native normalized vector. It equals
   the official normalized vector bitwise.
2. Replacing only the cosine/sine inputs with the official CPU-generated
   coefficients makes post-RoPE, post-Hadamard, and final FP4 outputs all equal
   to the reference bitwise.
3. With identical pre-quantization inputs, native and independent CPU FP4
   outputs are bitwise equal.

The maximum native/reference angle difference at position 216 is
`4.291534423828125e-5`. Against an independent FP64 formula, the native maximum
angle error is `4.2290989911819565e-5`, versus `5.927374530756424e-6` for the
official FP32 recipe. CPU and TPU sine/cosine on **identical native angles**
differ by at most `5.960464477539063e-8`. The dominant discrepancy therefore
already exists in frequency/angle generation, not the trigonometric call.

The relevant path is `deepseek_v4/numerics.py::rope_angles`: inverse-frequency
generation, YaRN interpolation, and multiplication by position. This experiment
does not identify a particular compiler instruction as the cause. The helper
also serves HCA and attention paths, so a production fix must cover its V4
consumers consistently, rather than special-case this one index fixture.

## 2. Phase-only A/B: most failures disappear, not all

The frozen six-case index matrix was rerun with CPU reference coefficients
supplied to the unchanged Pallas emitter. Inputs, checkpoint hashes, chunk
boundaries, CPU reference arrays, state arrays, and numerical limits remain
unchanged. All 2916 metric decisions were audited.

| Case | Failed metric records before → after | Worst vector NRMSE after |
| --- | ---: | ---: |
| Layer 42, B1, 256 tokens | 0 → 0 | 0% |
| Layer 42, B4, 256 tokens | 1 → 0 | 0% |
| Layer 2, B1, 8192 tokens | 3 → 0 | 1.67062% |
| Layer 2, B4, 8192 tokens | 5 → 1 | 4.62250% |
| Layer 22, B1, 4096 tokens | 3 → 0 | 2.56410% |
| Layer 22, B4, 4096 tokens | 4 → 1 | 4.69841% |

Passing cases improve from **1/6 to 4/6**; failed metric records fall from
**16 to 2**. The two remaining records cover **four** over-limit vectors.
The unchanged index gate is 3.5% NRMSE, including a per-vector requirement.
Only the two layer-42 cases become completely bitwise equal to the reference.
All three B1/B4 paired request-0 caches remain bitwise equal.

## 3. Remaining failures: projection/pooling rounding before RMSNorm

Traced residual replays reproduce untraced A/B final arrays byte for byte.
An isolated diagnostic Pallas program exports pooling and normalization stages;
its normalized BF16 outputs reproduce the original emitter exactly.

| Layer | Request (zero-based) | Group | Terminal position | Localized source |
| --- | ---: | ---: | ---: | --- |
| 2 | 3 | 856 | 3427 | Projection-state rounding |
| 2 | 2 | 1643 | 6575 | Pooling arithmetic |
| 2 | 3 | 1928 | 7715 | Pooling arithmetic |
| 22 | 2 | 272 | 1091 | Projection-state rounding |

Projection-state differences are only a few FP32 ulps and pass the original
state gate, but can put weighted pooled values on opposite sides of a BF16
midpoint. Replaying CPU and FP64 pooling on the captured native projection
state is sufficient to reproduce the first and fourth rows' native rounding.
This is not evidence of wrong page ownership or a wrong projection formula;
TPU and CPU accumulation orders need not agree bitwise.

For the pooling-sensitive windows, differences persist even with identical
raw projection inputs. Two concrete midpoint examples:

- Layer 2/group 1643/request 2/channel 58: native pooled FP32 is
  `0.16552734375`, exactly a BF16 midpoint. Independent FP64 pooling of the
  same inputs gives `0.16552713031496702`. Native rounds to `0.166015625`,
  while the reference rounds to `0.1650390625`.
- Layer 2/group 1928/request 3/channel 38: native pooled FP32 is
  `-2.132812976837158`; FP64 on the same inputs gives `-2.1328124228662384`.
  These straddle midpoint `-2.1328125`, producing BF16 `-2.140625` and `-2.125`.

Given identical **pooled BF16** inputs, both CPU and FP64 RMSNorm reproduce the
native normalized BF16 output for all four captured windows. Given identical
post-RoPE inputs, independent CPU Hadamard reproduces native output; given
identical pre-FP4 inputs, FP4 outputs are bitwise equal in all four windows.
Thus those stages propagate/amplify the differences but are not their source
in these fixtures. For example, layer 22/group 272 has pre-FP4 `2.5` versus
`2.515625`, becoming final `2.0` versus `3.0`.

The pooling expression is the softmax-weighted sum in
`kernels/csa/compressor.py` preceding the V4 BF16 rounding and fixed-tree
RMSNorm. This diagnosis does not justify blindly changing the RMSNorm sum tree.

## Next implementation and validation boundary

1. Define and validate one V4 FP32 RoPE coefficient recipe, potentially generated
   at initialization, and use it consistently in compressor and attention paths.
   Merely computing an arbitrary NumPy/FP64 table is not a guarantee of matching
   the official FP32 recipe.
2. Preserve the four residual midpoint windows as standalone regression fixtures.
   Evaluate projection/pooling precision separately before changing production
   kernels. Distinguish a numerical contract from mandatory CPU bitwise parity.
3. Keep the existing failed gates visible. Then validate real index top-k,
   full-model logits, prefill/chunk/decode, and long-context/batched behavior.
   No claim of official full-model accuracy follows from this operator study.

The original HTTP service was restored and verified ready/idle after diagnostics,
with the same source fingerprint and settings: port 30126, context 8192, B4,
TP4/EP4/DP1, prefill chunk 128, overlap enabled, mixed chunk disabled. Its
diagnostic receipt records PID 328801; this is a historical observation, not
a permanent process identifier. No test coefficient override is deployed.
