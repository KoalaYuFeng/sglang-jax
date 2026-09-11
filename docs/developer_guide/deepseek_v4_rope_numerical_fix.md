# V4 RoPE numerical fix — local validation, TPU validation pending

Update: [TPU recovery and validation completed](deepseek_v4_numerical_recovery_20260911.md).
The local-only status below is the earlier checkpoint; the follow-up records
the actual 18-case TPU result and remaining numerical differences.

2026-09-11. Follow-up to
[the causal CSA-index diagnosis](deepseek_v4_index_numerical_rootcause.md).
**Partial fix: RoPE source changed and CPU-tested; residual projection/pooling
differences are not fixed, and no changed code has been deployed.**

## Change

`python/sgl_jax/srt/kernels/deepseek_v4/numerics.py` now constructs and caches
the small inverse-frequency vector on the host, outside JAX tracing:

- Evaluate scalar powers, round to FP32, then take the FP32 reciprocal.
- Preserve the official FP32 YaRN operation order, including both subtractions
  in `smooth = 1 - ramp` and `1 - smooth`. Replacing the latter with `ramp`
  is not generally rounding-equivalent.
- Pass immutable FP32 frequencies as constants into the existing shared
  `rope_angles` interface. Dynamic positions still multiply on device;
  trigonometric evaluation remains on device.

This removes device power approximation and compiler reassociation of frequency
construction. All existing consumers (CSA, HCA, attention, index queries, and
inverse output rotation) retain the same interface and share the fix. There is
no scheduler change, PyTorch runtime dependency, host callback, checkpoint
conversion, full-context RoPE table, or layer-specific workaround.

The cache key contains only rotary parameters, not layer, batch, or context;
each Flash vector has 32 FP32 entries. The cache is populated on the first
host call/trace, not on each token.
This is not a claim that arbitrary libm implementations reproduce every
PyTorch/GPU power result bitwise; the pinned Flash coefficients and the actual
captured failure are explicitly checked.

## Local validation

Environment: macOS ARM64 CPU, Python 3.12.13, JAX/jaxlib 0.11.1, NumPy 2.2.6,
PyTorch 2.14.0. Dependencies are isolated under ignored
`GCP_login/v4-numerical-fix-venv`; production dependencies are unchanged.

New tests: `python/sgl_jax/test/test_deepseek_v4_rope_numerics.py`:

- **16 passed**, including all optional frozen-evidence and PyTorch checks.
- Exact Flash frequency bits and an independent PyTorch FP32 formula.
- Dynamic positions across the 0–8191 range, shapes 1/4/8/16/32/127/128/8192.
- Immutable cache reuse and invalid dimension checks.
- JAX tracing contains no power, division, frequency construction, or callbacks.
- Position-219 replay from captured normalized vectors: post-RoPE,
  post-Hadamard, and final FP4 outputs all equal the frozen reference bitwise.
  This replay intentionally isolates the corrected path; it does not rerun
  projection/pooling or the full model.

Existing projection tests selected with `-k 'qnorm or kv_norm'`:
**5 passed, 5 skipped, 25 deselected**. The skipped tests require real TPU
lowering/four devices and are not counted as validation passes.

Replay command from the repository root:

```sh
PYTHONPATH=python JAX_PLATFORMS=cpu \
SGL_V4_NUMERICAL_EVIDENCE=GCP_login/results/v4-index-rootcause-20260911-01/evidence.tar.gz \
GCP_login/v4-numerical-fix-venv/bin/python -m pytest -q \
python/sgl_jax/test/test_deepseek_v4_rope_numerics.py
```

The test verifies archive SHA-256
`521582e7815418122b1aa78abc888b79d41fc72c29d8e9f42a20cd37103deac3`
before reading it; no cloud access or checkpoint download is required.

## Outstanding work and machine state

The exact prior TPU node lookup returned `NOT_FOUND`; the subsequent us-east5-a
TPU VM list was empty. This does not establish who deleted the VM or whether
Spot preemption caused it. No TPU was rebuilt and no remote source was changed.

Consequently, the former diagnostic phase-table A/B result (**16 → 2 failed
records**) is **not** a measured result for this new production implementation.
Its remaining four over-limit vectors (projection-sensitive positions 3427 and
1091; pooling-sensitive positions 6575 and 7715) remain unresolved. No tolerance
was relaxed and no failures were converted into expected passes.

After an authorized TPU restoration, run the new coefficient and captured
position-219 tests on v5p, then the unchanged six-case index matrix. Continue
isolated projection/pooling precision work on the saved midpoint fixtures,
and only after the numerical gates pass proceed to full-model top-k/logits,
43-layer 8K prefill/chunk/decode regression, and performance measurements.
