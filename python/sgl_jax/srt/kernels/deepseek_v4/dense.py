"""Static V4-only kernel selection; no scheduler or framework policy changes."""

from dataclasses import dataclass

from sgl_jax.srt.kernels.deepseek_v4 import normalization, numerics
from sgl_jax.srt.kernels.deepseek_v4.fp8 import fp8_linear


@dataclass(frozen=True)
class DenseKernels:
    fp8_backend: str = "legacy"
    fused_norm: bool = False
    merged_projections: bool = False
    fused_wo_a: bool = False

    def __post_init__(self):
        if self.fp8_backend not in ("legacy", "gmm"):
            raise ValueError("V4 FP8 backend must be legacy or gmm; no automatic fallback")
        for name in ("fused_norm", "merged_projections", "fused_wo_a"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.merged_projections and self.fp8_backend != "gmm":
            raise ValueError("merged V4 projections require the checkpoint FP8 GMM backend")

    def linear(self, x, weights, prefix, *, quantize=True):
        if self.fp8_backend == "gmm" and prefix + ".scale" in weights:
            return fp8_linear(
                x,
                weights[prefix + ".weight"],
                weights[prefix + ".scale"],
                quantize_activation=quantize,
            )
        return numerics.linear(x, weights, prefix, quantize=quantize)

    def norm(self, x, weight, eps):
        return (normalization.rms_norm if self.fused_norm else numerics.rms_norm)(x, weight, eps)
