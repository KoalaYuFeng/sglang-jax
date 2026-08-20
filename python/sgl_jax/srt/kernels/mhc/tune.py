"""Platform parameters and block-size selection for the mHC kernels.

Block sizes come from an explicit VMEM budget rather than a hardcoded tile, so a
new TPU generation only needs its numbers added to ``_PLATFORMS``.

The model is calibrated against measured compile outcomes on v6e: at hc_mult=4
and hidden=4096 a 128-token collapse block fits the 32 MiB scoped default while
256 needs ~56 MiB, and at hc_mult=8 the compiler reports 33.11 MiB for a 64-token
block where this predicts 33.1.

It also explains why collapse and Sinkhorn remain separate.  Collapse's working
set scales with the full [bt, hc*d] activation and stays near 128 tokens; gates
only hold the small [bt, hc, hc] state and prefer 2048.  Forced onto one block,
the pair measured 0.80x.
"""

from __future__ import annotations

from dataclasses import dataclass

# Mosaic reserves scratch beyond the buffers a caller can enumerate.  A 1.14
# factor reproduces the compiler's reported requirement across hc_mult 4 and 8
# (33.11 MiB reported versus 33.1 MiB predicted).
_SCRATCH_HEADROOM = 1.14


@dataclass(frozen=True)
class MHCPlatform:
    name: str
    device_markers: tuple[str, ...]
    # Scoped VMEM the compiler grants without an explicit override.
    vmem_bytes: int
    collapse_blocks: tuple[int, ...]
    gates_blocks: tuple[int, ...]
    post_blocks: tuple[int, ...]


_PLATFORMS = (
    MHCPlatform(
        name="TPU v6e",
        device_markers=("v6e", "v6 lite", "tpu v6"),
        vmem_bytes=32 * 1024 * 1024,
        collapse_blocks=(8, 16, 32, 64, 128, 256),
        gates_blocks=(512, 1024, 2048),
        post_blocks=(8, 16, 32, 64, 128, 256),
    ),
)


@dataclass(frozen=True)
class MHCKernelSchedule:
    """Static launch geometry for one (platform, model shape) pair."""

    platform: str
    collapse_block_tokens: int
    gates_block_tokens: int
    post_block_tokens: int

    def __post_init__(self) -> None:
        for name in ("collapse_block_tokens", "gates_block_tokens", "post_block_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def _platform_parameters(device_kind: str) -> MHCPlatform:
    normalized = (device_kind or "").lower()
    for platform in _PLATFORMS:
        if any(marker in normalized for marker in platform.device_markers):
            return platform
    supported = ", ".join(platform.name for platform in _PLATFORMS)
    raise ValueError(f"mHC has no schedule for {device_kind!r}; supported: {supported}")


def collapse_vmem_bytes(block_tokens: int, *, hc_mult: int, hidden: int, rows: int) -> int:
    """Peak VMEM of one collapse program, counting every buffer allocated."""
    bt, k = block_tokens, hc_mult * hidden
    activation = 2 * bt * k * 2  # [bt, hc, d] bf16, double buffered across steps
    projection_operand = bt * k * 4  # FP32 view fed to the dot
    gated_sum = 2 * bt * hidden * 4  # one stream and the accumulator
    weights = rows * k * 4  # resident, shared by every grid step
    y_out = 2 * bt * hidden * 2
    mixes_out = 2 * bt * rows * 4
    total = activation + projection_operand + gated_sum + weights + y_out + mixes_out
    return int(total * _SCRATCH_HEADROOM)


def gates_vmem_bytes(block_tokens: int, *, hc_mult: int, mix_hc: int) -> int:
    bt = block_tokens
    mixes_in = 2 * mix_hc * bt * 4
    state = hc_mult * hc_mult * bt * 4  # the [hc, hc, bt] Sinkhorn state
    outputs = 2 * (hc_mult + hc_mult + hc_mult * hc_mult) * bt * 4
    return int((mixes_in + state + outputs) * _SCRATCH_HEADROOM)


def post_vmem_bytes(block_tokens: int, *, hc_mult: int, hidden: int) -> int:
    bt = block_tokens
    residual = 2 * bt * hc_mult * hidden * 2
    block_output = 2 * bt * hidden * 2
    gates = 2 * bt * (hc_mult + hc_mult * hc_mult) * 4
    widened = bt * hc_mult * hidden * 4  # FP32 accumulation before the BF16 cast
    y_out = 2 * bt * hc_mult * hidden * 2
    return int((residual + block_output + gates + widened + y_out) * _SCRATCH_HEADROOM)


def _largest_fitting(blocks, budget: int, cost) -> int:
    """Largest block that fits, or the smallest when none does.

    A wide enough model exceeds the budget at every candidate; the compiler then
    reports the real limit.
    """
    fitting = [block for block in blocks if cost(block) <= budget]
    return max(fitting) if fitting else min(blocks)


def kernel_schedule(
    device_kind: str,
    *,
    hc_mult: int,
    hidden: int,
) -> MHCKernelSchedule:
    """Select block sizes for one platform and model shape."""
    if min(hc_mult, hidden) <= 0:
        raise ValueError("hc_mult and hidden must be positive")
    platform = _platform_parameters(device_kind)
    budget = platform.vmem_bytes
    mix_hc = (2 + hc_mult) * hc_mult

    return MHCKernelSchedule(
        platform=platform.name,
        collapse_block_tokens=_largest_fitting(
            platform.collapse_blocks,
            budget,
            lambda bt: collapse_vmem_bytes(bt, hc_mult=hc_mult, hidden=hidden, rows=mix_hc),
        ),
        gates_block_tokens=_largest_fitting(
            platform.gates_blocks,
            budget,
            lambda bt: gates_vmem_bytes(bt, hc_mult=hc_mult, mix_hc=mix_hc),
        ),
        post_block_tokens=_largest_fitting(
            platform.post_blocks,
            budget,
            lambda bt: post_vmem_bytes(bt, hc_mult=hc_mult, hidden=hidden),
        ),
    )


__all__ = [
    "MHCKernelSchedule",
    "MHCPlatform",
    "collapse_vmem_bytes",
    "gates_vmem_bytes",
    "kernel_schedule",
    "post_vmem_bytes",
]
