"""Static V4-only kernel selection; no scheduler or framework policy changes."""

from dataclasses import dataclass

from sgl_jax.srt.configs.deepseek_v4_execution import DEFAULTS, validate_options
from sgl_jax.srt.kernels.deepseek_v4 import normalization
from sgl_jax.srt.kernels.low_bit.fp8 import fp8_linear
from sgl_jax.srt.layers.deepseek_v4 import numerics


@dataclass(frozen=True)
class DenseKernels:
    fp8_backend: str = DEFAULTS["fp8_backend"]
    fused_norm: bool = DEFAULTS["fused_norm"]
    merged_projections: bool = DEFAULTS["merged_projections"]
    fused_wo_a: bool = DEFAULTS["fused_wo_a"]

    def __post_init__(self):
        validate_options(
            {
                **DEFAULTS,
                "fp8_backend": self.fp8_backend,
                "fused_norm": self.fused_norm,
                "merged_projections": self.merged_projections,
                "fused_wo_a": self.fused_wo_a,
            }
        )

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
        return (normalization.rms_norm if self.fused_norm else numerics.rms_norm)(
            x, weight, eps
        )


def merged_linear(x, weights, prefix, split, *, block_m=None):
    result = fp8_linear(
        x, weights[prefix + ".weight"], weights[prefix + ".scale"], block_m=block_m
    )
    if not 0 < split < result.shape[1] or split % 128:
        raise ValueError(
            "merged projection split must be an interior scale-block boundary"
        )
    return result[:, :split], result[:, split:]
