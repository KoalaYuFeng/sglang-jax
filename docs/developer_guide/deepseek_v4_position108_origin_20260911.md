# Position 108: layer-0 origin and quantization amplification — 2026-09-11

## Outcome

The layer-0 origin is narrowed to small query-projection and attention
arithmetic differences, subsequently amplified by **FP8 activation
quantization with FP4 expert weights**. This is not FP4 activation quantization
on the MoE path; the two formats must not be conflated.

At position 108, normalized attention input is bitwise identical. The first
observed difference in this token's projection sequence is two `wq_b` output
coordinates. Independent FP64 supports the native value for one coordinate
and the original CPU value for the other. Correcting the latter alone reduces
attention output NRMSE from **0.402742% to 0.314802%**, but does not eliminate
the residual. This is an isolated counterfactual, not a production kernel fix.

No serving changes, threshold changes, forced routing or reference replacement.
Historical acceptance remains **15/18 with three failed records**. Complete
43-layer/8K/multi-request accuracy and performance acceptance remain open.

## Setting and controls

Same official checkpoint revision
`60d8d70770c6776ff598c94bb586a859a38244f1`; branch
`integration/deepseek-v4`, base `270dfcbd` with earlier changes preserved.
Runtime fingerprint unchanged:
`1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.

Inputs come from the frozen B1 production trajectory: 127 prefill tokens,
native chunks 64+63, and five decode tokens. This wave examines **layer 0,
prefill position 108**. It uses fresh official CPU MoE runs at the complete
127-token shape, existing independently captured attention traces, CPU
projection-shape controls, and standalone TPU helper/post-QKV replays on the
existing four-chip v5p VM. It is not a new full-model or Engine execution.

The instrumented CPU MoE reproduces both full-prefill frozen CPU outputs
bitwise: original CPU input and captured native input. At position 108,
same-input CPU MoE also equals actual native MoE output bitwise. Existing
same-input mHC/pre/post and normalization controls are bitwise at this token.
The post-QKV TPU baseline reproduces all stored Q/value/projected/output
arrays bitwise before the single-element intervention is interpreted.

## Error sequence at position 108

NRMSE versus the original CPU result, percent:

| Stage | NRMSE |
| --- | ---: |
| Normalized attention input | 0%, bitwise |
| Initial Q-latent projection `qa`, raw KV, normalized latent `qr` | 0%, bitwise |
| FP8-weight query projection `wq_b` | 0.001827% |
| Query after normalization/RoPE | 0.002225% |
| Attention value | 0.038727% |
| Inverse RoPE / `wo_a` output | 0.109883% |
| Attention output after `wo_b` | 0.402742% |
| Normalized MoE input | 0.295901% |
| MoE output | 1.704832% |

Same-input local controls at this token:

- Q normalization/RoPE and KV normalization/RoPE/QAT: bitwise.
- Sparse attention: 0.027653% NRMSE.
- Inverse RoPE / `wo_a`: 0.00004155% NRMSE.
- `wo_b`: bitwise.
- MoE: bitwise.

Final KV arrays agree over **all 127 positions**, and the earlier trace verified
strict effective key/index/mask equality. Earlier raw projections can differ
at other positions, but those differences do not survive into this fixture's
stored KV. Consequently this is not evidence of a divergent historical KV
input at position 108. Layer 0 is SWA, with no CSA/HCA compressor involved.

## Query projection: two disagreements, opposite adjudications

The full CPU projection on the native latent input reproduces the stored
same-input CPU result bitwise. Of 32,768 output coordinates at position 108,
two differ from native. FP64 uses independently decoded E4M3/E8M0 weight rows
and independently quantized/dequantized FP8 input, then rounds to BF16.

| Channel | Native | Original CPU (M=127) | FP64 sum | FP64 rounded to BF16 |
| --- | ---: | ---: | ---: | ---: |
| 2108 | −0.000003650784492 | −0.000003635883331 | −0.000003645196557 | native value |
| 7656 | 0.027099609375 | 0.0272216796875 | 0.02716064639389515 | CPU value |

At channel 7656, the BF16 midpoint is 0.02716064453125; the FP64 sum is just
above it. Native rounds to the lower neighbor. This is consistent with FP32
dot accumulation error affecting the final BF16 boundary; the actual native
FP32 accumulator was not separately captured in this wave.

CPU M=64 (the relevant native chunk, padded with one masked row) selects the
native value at 7656 but the original CPU value at 2108. CPU M=1 selects the
FP64-rounded value at both. These shape controls demonstrate sensitivity;
they do not license choosing a favorable shape as the official reference.
Adjudication here covers the two disagreements, not every agreeing coordinate
or all checkpoint projections.

### Single-element downstream intervention

Only native raw-Q coordinate `[108, 7656]` is replaced by the FP64-rounded
value. The other coordinate already agrees with FP64 and is unchanged. Q
normalization, exp, KV, indices and output projections retain their original
implementations. Other tokens' attention outputs remain bitwise unchanged.

| Position-108 output | Original NRMSE | Single-element intervention |
| --- | ---: | ---: |
| Normalized/rotated Q | 0.002225% | 0.0000002635% |
| Attention value | 0.038727% | 0.027653% |
| `wo_a` projected value | 0.109883% | 0.088994% |
| Final attention output | 0.402742% | 0.314802% |

The previous exp-only candidate, on its frozen original inputs, gives
0.315768% attention-output NRMSE at this token. It was **not enabled** here;
its broader regressions remain in the earlier candidate report. These
individual effects are not additive and no combined new candidate was tested.

## FP8 activation boundaries amplify the remaining differences

Before `wo_b`, the 8,192-channel input has 0.109883% NRMSE. Independent FP8
roundtrip makes it **0.452172%**, with 107 changed dequantized elements and
zero changed 128-channel scale blocks. Given identical projected inputs,
native and CPU `wo_b` output are bitwise equal at position 108.

Layer-0 routing is hash-based. Both trajectories select the same six experts:
`84, 114, 38, 100, 168, 113`. There is no route flip in this layer.

For W1/W3, the common 4,096-channel MoE input difference of **0.295901%** becomes
**1.139837%** after FP8 roundtrip. There are **243 changed FP8 codes**, zero
changed scales. This same input is reused by all six experts and both
projections; do not count those copies as additional unique crossings.

For W2, after SwiGLU and pre-down-projection routing-weight multiplication:

| Expert | Before FP8 NRMSE | After FP8 NRMSE | Changed codes / 2048 | Changed scale blocks |
| --- | ---: | ---: | ---: | ---: |
| 84 | 1.603744% | 3.622027% | 756 | 0 |
| 114 | 1.481450% | 2.315661% | 730 | 0 |
| 38 | 1.265144% | 2.401736% | 835 | 1 |
| 100 | 1.481165% | 2.666297% | 829 | 1 |
| 168 | 1.552680% | 3.395137% | 756 | 0 |
| 113 | 1.526000% | 2.827147% | 751 | 0 |

All 36 captured input cases (two trajectories × six experts × W1/W3/W2)
match independent NumPy quantization bitwise. The standalone TPU activation
helper also matches both CPU and NumPy bitwise for all 36. This validates the
helper on these real inputs, not a new full GMM execution or every possible
format/range. Scale changes in the W2 rows are legitimate consequences of
different inputs, not evidence of a scale-loading error.

## Interpretation and next step

The measured sequence is: small FP8 query-projection/attention arithmetic
differences → output-projection FP8 boundary crossings → MoE FP8 activation
amplification → later expert-selection discontinuity documented in the
[layer-3 position-108 report](deepseek_v4_position108_attribution_20260911.md).
It is not evidence that mHC/compressor state or FP4 weight decoding is the
source of this layer-0 discrepancy. Neither does it prove model accuracy or
that all observed floating-point differences are harmless.

The next bounded work is to inspect/replay the **actual FP32 query-projection
accumulator** near the identified BF16 boundary, broaden that check before
designing an accumulation change, and retain a separate softmax-error track.
Any revised accumulation/exp kernel needs same-input FP64 controls, shape and
TP checks, then independent layer/logit and full-model regression. Do not
force CPU codes or change routes in serving to conceal the discrepancy.

## Evidence

Public runner: `scripts/debug_deepseek_v4_position108_origin.py`.
Remote run: `/mnt/disks/deepseek-models/profiles/v4-position108-origin-20260911-01`.
Local evidence and checksum receipts:
`GCP_login/results/v4-position108-origin-20260911/`.
Archive: `v4-position108-origin-20260911-evidence.tar.gz`; checksum recorded in
its receipt and local `GCP_login/CURRENT_TPU.md` outside this self-archived file.
Original arrays are checked against the retained trajectory and attention
evidence manifests; all 731 frozen runtime source hashes remain unchanged.

Local CPU regression: **82 passed, 5 TPU-only skips**; runner Ruff/diff checks
pass. No full-model/HTTP/performance run, no commit/push, and no serving
numerical candidate enabled. Completed diagnostic jobs leave the VM running.
