# Cause of the two remaining FP8 projection differences — 2026-09-11

## Conclusion

Both remaining K=8 residual-aware coordinates lose **half an FP32 ULP in one
partial dot product**, with the partial itself correctly rounded nearest-even.
All other 127 partials are exact. Compensated addition introduces no further
error; the final BF16 conversion correctly rounds its input. The missing bits
are already absent before compensation begins, so the final residual carrier
cannot recover them.

This is a finite-precision arithmetic difference, not evidence of incorrect
checkpoint decoding, a broken compensation formula, a wrong BF16 conversion,
or a scheduler/distributed-state bug. It does not establish that the downstream
effect is harmless, nor resolve the historical model gate (15/18, 3 failures).
This wave diagnoses only; no production or candidate arithmetic was modified.

## Exact locations

All positions refer to the frozen native layer-0 `wq_b` input sequence, K=1024,
N=32768, official checkpoint revision
`60d8d70770c6776ff598c94bb586a859a38244f1`.

| Position / channel | Partial index (zero-based) | K indices | Lost amount |
|---|---:|---|---:|
| 61 / 10345 | 101 | [808,816) | 2^-33 = 1.1641532182693481e-10 |
| 88 / 26881 | 81 | [648,656) | 2^-34 = 5.820766091346741e-11 |

Both errors are negative: the rounded partial is lower than the exact partial.

| Quantity | Position 61 | Position 88 |
|---|---:|---:|
| Exact partial dot | 0.0027255081804469228 | -0.0017786139505915344 |
| Exported FP32 partial | 0.002725508064031601 | -0.0017786140087991953 |
| Exact full dot | -7.375958375632763e-6 | -1.7843558453023434e-6 |
| Exact sum of rounded partials | -7.37607479095459e-6 | -1.7844140529632568e-6 |
| Compensation error | 0 | 0 |
| Final `low` component | 0 | 0 |
| BF16 of exact full dot | -7.361173629760742e-6 | -1.780688762664795e-6 |
| Candidate BF16 | -7.3909759521484375e-6 | -1.7881393432617188e-6 |

The two exact full sums lie respectively 2^-33 and 2^-34 above a BF16 midpoint.
The partial-rounding loss moves each sum exactly onto that midpoint, where
ties-to-even selects the other BF16 neighbor. The final BF16 values therefore
differ by one BF16 ULP even though the earlier loss is extremely small.

The partial products contain substantially different magnitudes; for example,
position 61's affected group includes terms about 1.37e-3 and 2.62e-8. Subsequent
positive/negative cancellation makes the full dot much smaller than some
partials. Exact products and all partials are retained in the diagnostic.

## Evidence and controls

On the existing v5p-8 Spot VM (four physical chips), this is a single-device
projection-tile trace, not a new distributed or full-model inference test.

- Replay the complete 64-row × 128-channel tile for each target. Carrier values
  match the preceding full-projection candidate bitwise.
- Independent CPU FP8 activation quantization and checkpoint weight decoding
  match the TPU helpers bitwise for both selected tiles.
- Export all 128 K=8 partial dot results using the unchanged shared GMM and the
  existing diagnostic partial adapter. Independently exported high/low outputs
  match explicit NumPy FP32 reconstruction over both complete tiles bitwise.
- Exact rational multiplication/summation verifies all 1024 products and all
  128 partials at each target coordinate. Exactly one partial differs per target,
  and both differing partials match correctly rounded FP32 exact values.
- Exact rational sums show zero compensation error. No rounding loss occurs
  while adding error terms into `low` at either target; final `low` is zero.
- Independent BF16 rounding of the exported high/low pair matches the candidate
  over each entire tile. No reference threshold or expected value is changed.

These are replay-validated exported values, not instrumentation of an unchanged
production binary's internal registers. All serving sources remain unchanged.

## Implication

Changing only the final cast or adding compensation after a partial is rounded
cannot recover these bits. Eliminating these exact-reference discrepancies
requires retaining more information **inside/before the partial output** (for
example, a more precise partial representation or a different decomposition).
This report does not implement or benchmark such a change. A further numerical
candidate still requires broad checks and cost measurement; exact FP64/BF16
agreement is stricter than standard FP32-accumulation semantics.

The remaining two projection differences must not be conflated with the three
historical model-gate failures. Downstream hidden-state/routing impact and the
separate attention/softmax residual still require their own attribution.

## Reproduction and archive

- Driver: `scripts/debug_deepseek_v4_fp8_last_coordinates.py`.
- Run: `profiles/v4-fp8-last-coordinates-20260911-01` (`report.json`,
  `position61.npz`, `position88.npz`).
- Existing CPU regression suite rerun: **98 passed, 5 skipped**; no new tests
  or runtime fixes in this wave. Ruff and `git diff --check` pass.
- Unchanged runtime fingerprint:
  `1a83e8ac9731d885187c80ecf25834b6782528c3a002f4b5a994570dfdcd60b3`.
- Evidence: `profiles/v4-fp8-last-coordinates-20260911-evidence.tar.gz`;
  locally verified copy and receipts:
  `GCP_login/results/v4-fp8-last-coordinates-20260911/`.
  Archive verifies 731 frozen runtime files, upstream inputs and reused helpers.
