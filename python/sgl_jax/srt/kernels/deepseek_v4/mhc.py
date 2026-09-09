"""Thin V4 adapters for the existing mHC kernels, with explicit A/B selection.

The reference option retains the previously accepted numerical path. It is
never an automatic fallback for a failed or unsupported Pallas execution.
Pre/Sinkhorn already call the original Pallas kernels directly in the model.
"""

import jax

from sgl_jax.srt.kernels.deepseek_v4.numerics import official_head_collapse
from sgl_jax.srt.kernels.mhc import mhc_head_collapse_fused, mhc_post_fused


def validate_backend(backend):
    if backend not in ("pallas", "reference"):
        raise ValueError("v4_mhc_backend must be 'pallas' or 'reference'; no automatic fallback")
    return backend


def post(x, residual, post_gate, comb, *, backend="pallas"):
    validate_backend(backend)
    return mhc_post_fused(
        x,
        residual,
        post_gate,
        comb,
        backend="pallas" if backend == "pallas" else "auto",
        precision=jax.lax.Precision.HIGHEST,
    )


def head_collapse(streams, fn, scale, base, *, eps=1e-6, hc_eps=1e-6, backend="pallas"):
    validate_backend(backend)
    if backend == "reference":
        return official_head_collapse(streams, fn, scale, base, eps=eps, hc_eps=hc_eps)
    return mhc_head_collapse_fused(
        streams,
        fn,
        scale,
        base,
        hc_mult=streams.shape[-2],
        norm_eps=eps,
        hc_eps=hc_eps,
        dot_precision=jax.lax.Precision.HIGHEST,
        head_rms_mode="fp32_post",
    )
