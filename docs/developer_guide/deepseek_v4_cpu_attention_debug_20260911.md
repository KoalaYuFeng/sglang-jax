# V4 attention numerical attribution — 2026-09-11

## Scope and status

The frozen layer-0 attention residual now has independently reproduced
arithmetic causes. **No serving code or acceptance thresholds changed.**
Historical compressor acceptance remains **15/18 cases, 3 failed records**.
Candidate arithmetic below is diagnostic-only, not deployed or performance-tested.

This follows the [four-layer trajectory diagnosis](deepseek_v4_cpu_trajectory_debug_20260911.md).
Checkpoint revision is `60d8d70770c6776ff598c94bb586a859a38244f1`; source is
`integration/deepseek-v4`, base `270dfcbd` with preserved earlier changes.
Runtime fingerprint remains
`1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.

Input: the actual layer-0 normalized attention input from the prior production
trajectory, B1, 127 fixed random-token embeddings, native chunks 64+63. This is
the replicated-head SWA layer, not a CSA/HCA distributed-attention test. Main
replay uses four physical v5p chips; isolated arithmetic prototypes also run on
one chip because this layer has no head collective. No MoE is involved here.
The oracle remains official Python with independent CPU-only kernel shims,
not an official GPU execution.

## Structural and same-input checks

The newly instrumented production attention output reproduces the previously
saved actual DecoderLayer output **bitwise**. The separate QK/softmax trace and
full post-QKV arithmetic baseline also reproduce their frozen outputs bitwise.

The first instrumentation run stopped at a diagnostic shape assertion: native
indices have 128 slots versus official prefill's 127. After appending only one
trailing `-1` mask to the CPU indices, all key identities, order and masks agree
strictly. The original numerical reference was not padded or modified. Tests
ensure this alignment cannot hide live-tail, ordering, or mask differences.

NRMSE measurements (percent); stage controls use the exact native input to that
suboperator. These are measurements, not newly selected pass limits.

| Comparison | NRMSE |
| --- | ---: |
| Initial FP8 Q projection, identical layer input | 0.000942% |
| Initial FP8 KV projection, identical layer input | 0.000526% |
| Q latent RMSNorm, identical projection input | 0%, bitwise |
| FP8 `wq_b`, identical normalized input | 0.000608% |
| Q normalization + RoPE, identical raw Q | 0.006930% |
| KV norm + RoPE + non-RoPE FP8 QAT, identical raw KV | 0%, bitwise |
| Sparse attention, identical Q/KV/indices/sink | 0.010022% |
| Inverse RoPE + `wo_a`, identical attention values | 0.005914% |
| FP8 `wo_b`, identical projected input | 0.000914% |
| Entire attention, identical layer input | 0.183016% |

These numbers cannot be added as independent percentage contributions.

## Q normalization: an identified BF16 midpoint

The only differing non-RoPE Q token/head under identical raw Q is zero-based
position 109, head 3. A Pallas intermediate trace reproduces all frozen
non-RoPE outputs bitwise; squared BF16 inputs match CPU bitwise as well.

| Quantity | Value |
| --- | ---: |
| Exact FP64 mean of the already-BF16-rounded squares | 0.0020828248601674204 |
| Native FP32 adjacent-tree mean | 0.00208282470703125 |
| Independent CPU replay of that FP32 tree | 0.00208282470703125 |
| Native subsequent BF16 variance | 0.0020751953125 |
| Official CPU / FP64-rounded BF16 mean | 0.0020904541015625 |
| Native BF16 inverse square root | 22.0 |
| CPU / FP64-rounded BF16 inverse square root | 21.875 |

The native FP32 mean lands exactly on the BF16 midpoint and rounds down, while
the exact mean lies slightly above it. This is reproducible from the explicit
FP32 tree on CPU: **not a distributed gather/cache error or a v5p slicing bug**.

A standalone compensated pairwise-sum candidate retains each addition's
rounding residual. Across all 127×64×448 non-RoPE Q elements in this fixture,
the 416 baseline discrepancies become **zero**, matching CPU and FP64-rounded
values bitwise, with no new differing token/heads. Full Q still has small
rotary-channel differences; full Q bitwise equality is not claimed.

## Softmax and FP8 activation amplification

QK score NRMSE against CPU is approximately 1.24e-7–1.29e-7; against an
independent FP64-dot-then-FP32-round control, native score error is smaller than
CPU's in both 64-key blocks. Exact summation here adjudicates arithmetic;
it does not replace official execution semantics.

For identical stored FP32 score differences, the native softmax exponentials
cross **65 + 22 = 87 BF16 probability boundaries** relative to the FP64 control.
CPU exponentials match the FP64-rounded BF16 probabilities in this fixture.
Replaying the recorded differences through standalone TPU `exp` reproduces
the frozen FP32 exponentials and BF16 probabilities bitwise in both blocks.
Reusing the repo's existing range-reduced `_exp_nonpositive` in an isolated
prototype reduces those frozen BF16 discrepancies to **zero**. This only
validates these inputs; it is not a general proof of correctly rounded exp.

Immediately before final `wo_b`, projected-input NRMSE is **0.042863%**.
After FP8 activation quantization it is **0.175822%**: 2,658 of 1,040,384 values
change codes between the two trajectories, with **zero block-scale changes**.
The actual TPU FP8 roundtrip primitive and the unchanged CPU shim both match
a new independent NumPy E4M3FN nearest-even quantizer bitwise on **both** inputs.
This supports amplification of small input perturbations, not an FP8 format
or quantizer implementation failure on these tensors.

CPU `wo_b` fed the native projected input reproduces essentially the same
0.183015% deviation from the original CPU output. The identical-input `wo_b`
kernel error itself is only 0.000914%.

## Standalone arithmetic candidates

The baseline below reproduces frozen attention value, `wo_a` projection and
final output bitwise. Only Q normalization summation and/or SWA exp are varied;
weights, QKV projections, indices, masks, output projections and oracle stay
unchanged. The compensated qnorm function is cloned with private diagnostic
globals; the production normalization module is not patched.

| Candidate | Attention output NRMSE | Maximum per-token output NRMSE |
| --- | ---: | ---: |
| Original | 0.183016% | 0.442544% |
| Compensated Q sum only | 0.179499% | 0.402742% |
| Range-reduced exp only | 0.134737% | 0.439329% |
| Both | 0.129572% | 0.365673% |

The combined candidate reduces aggregate error by about **29.2%** on this
fixture, but retains residual differences. It is **not** a speedup, an accepted
full-model fix, or proof of official accuracy. No candidate has been enabled.

Next work is broader standalone Q-normalization/softmax boundary testing,
including real layer inputs, different token/head shapes and decode, before
considering integration and rerunning independently propagated layer/logit
trajectories. The 43-layer, 8K, multi-request and performance gates remain open.

## Tests and evidence

Local CPU tests: **51 passed, 5 skipped** (the skipped paths require TPU or
optional frozen evidence). Ruff and `git diff --check` pass. Live TPU checks
cover instrumented-output reproduction, strict index/mask agreement, the
normalization midpoint trace, candidate A/B baselines and primitive replay.

New reusable diagnostics are `scripts/debug_deepseek_v4_cpu_attention_stages.py`,
`scripts/deepseek_v4_attention_diagnostics.py`, and
`python/sgl_jax/test/test_deepseek_v4_attention_diagnostics.py`.

Raw evidence under `/mnt/disks/deepseek-models/profiles/`:

- `v4-cpu-attention-stages-20260911-01` (shape-assertion diagnostic) and `-02`;
- `v4-attention-boundaries-20260911-01`;
- `v4-softmax-stages-20260911-01`;
- `v4-qnorm-midpoint-20260911-01`;
- `v4-attention-candidates-20260911-01`;
- `v4-attention-primitive-audit-20260911-01`;
- `v4-exp-replay-20260911-01`.

The evidence archive is `v4-cpu-attention-debug-20260911-evidence.tar.gz`, with
SHA-256 manifest, arrays, logs, diagnostic scripts, test XML, runtime source
manifest and this report. Local backup and verification receipt live under
`GCP_login/results/v4-cpu-attention-debug-20260911/` (Git-ignored).
