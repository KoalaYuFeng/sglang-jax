"""DeepSeek checkpoint formats, without a native FP4 arithmetic requirement.

FP4 E2M1: two values per byte, low nibble first, one E8M0 scale per 32
columns and output row. FP8 E4M3FN: one byte per value and one E8M0 scale
per 128x128 weight block. Raw byte storage is intentional: converting an
I8 checkpoint container numerically to FP8 would corrupt packed FP4 values.

These functions also work inside a Pallas kernel. Calling the dequantizers
as ordinary JAX functions materializes their output in HBM; the online
matmul must instead call them *inside* its tiled Pallas program.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def round_bf16(x):
    """Observable BF16 RNE boundary, even across a fused quantization kernel."""
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    rounded = bits + jnp.uint32(0x7FFF) + ((bits >> jnp.uint32(16)) & jnp.uint32(1))
    rounded = rounded & jnp.uint32(0xFFFF0000)
    absolute = bits & jnp.uint32(0x7FFFFFFF)
    rounded = jnp.where(absolute > jnp.uint32(0x7F800000), bits | jnp.uint32(0x00400000), rounded)
    return jax.lax.bitcast_convert_type(rounded, jnp.float32).astype(jnp.bfloat16)


def decode_e8m0(raw):
    """Decode unsigned exponent bytes to FP32, including code 0 and NaN."""
    exponent = raw.astype(jnp.uint32)
    bits = exponent << jnp.uint32(23)
    bits = jnp.where(exponent == 0, jnp.uint32(0x00400000), bits)
    bits = jnp.where(exponent == 255, jnp.uint32(0x7FC00000), bits)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def decode_fp4_codes(codes):
    """E2M1 -> BF16 using exact bit construction (preserves negative zero)."""
    codes = codes.astype(jnp.uint16)
    magnitude = codes & jnp.uint16(7)
    exponent = magnitude >> jnp.uint16(1)
    mantissa = magnitude & jnp.uint16(1)
    normal = ((exponent + jnp.uint16(126)) << jnp.uint16(7)) | (mantissa << jnp.uint16(6))
    bits = jnp.where(exponent == 0, mantissa * jnp.uint16(0x3F00), normal)
    bits = bits | ((codes & jnp.uint16(8)) << jnp.uint16(12))
    return jax.lax.bitcast_convert_type(bits, jnp.bfloat16)


def unpack_fp4(packed):
    """Unpack [..., K/2] bytes to [..., K] exact BF16 values, without scales."""
    packed = packed.astype(jnp.uint8)
    low = decode_fp4_codes(packed & jnp.uint8(15))
    high = decode_fp4_codes(packed >> jnp.uint8(4))
    return jnp.stack((low, high), axis=-1).reshape(*packed.shape[:-1], 2 * packed.shape[-1])


def decode_fp8(raw):
    return jax.lax.bitcast_convert_type(raw.astype(jnp.uint8), jnp.float8_e4m3fn).astype(
        jnp.float32
    )


def encode_fp8(x):
    """Saturating round-to-nearest-even E4M3FN encoding, as explicit bytes.

    Do not rely on BF16 -> FP8 -> BF16 casts for quantization simulation:
    TPU compilation can eliminate that round trip. Integer rounding keeps
    the quantization boundary observable even when this function is fused.
    """
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    sign = (bits >> jnp.uint32(24)) & jnp.uint32(128)
    absolute = bits & jnp.uint32(0x7FFFFFFF)
    # Reduce 23 mantissa bits to 3, with ties to an even final mantissa.
    rounded = absolute + jnp.uint32(0x7FFFF) + ((absolute >> jnp.uint32(20)) & 1)
    normal = (rounded >> jnp.uint32(20)).astype(jnp.int32) - 120 * 8
    magnitude = jax.lax.bitcast_convert_type(absolute, jnp.float32)
    subnormal = jnp.rint(magnitude * jnp.float32(512)).astype(jnp.int32)
    encoded = jnp.where(absolute < jnp.uint32(0x3C800000), subnormal, normal)
    encoded = jnp.clip(encoded, 0, 126).astype(jnp.uint32)
    encoded = jnp.where(absolute > jnp.uint32(0x7F800000), jnp.uint32(127), encoded)
    return (encoded | sign).astype(jnp.uint8)


def dequantize_fp4(packed, scale):
    values = unpack_fp4(packed).astype(jnp.float32)
    expanded = jnp.repeat(decode_e8m0(scale), 32, axis=-1)
    return (values * expanded[..., : values.shape[-1]]).astype(jnp.bfloat16)


def dequantize_fp8(raw, scale):
    values = decode_fp8(raw)
    expanded = jnp.repeat(jnp.repeat(decode_e8m0(scale), 128, axis=-2), 128, axis=-1)
    return (values * expanded[..., : values.shape[-2], : values.shape[-1]]).astype(jnp.bfloat16)


def quantize_activation_block_fp8(x):
    """Quantize one last-axis block; shape-preserving for Pallas VMEM."""
    x = x.astype(jnp.float32)
    amax = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True), jnp.float32(1e-4))
    scaled = amax * jnp.float32(1.0 / 448.0)
    bits = jax.lax.bitcast_convert_type(scaled, jnp.uint32)
    exponent = (bits >> jnp.uint32(23)) + ((bits & jnp.uint32(0x7FFFFF)) != 0)
    scale = decode_e8m0(exponent.astype(jnp.uint8))
    raw = encode_fp8(jnp.clip(x / scale, -448.0, 448.0))
    return raw, exponent.astype(jnp.uint8)


def quantize_activation_fp8(x):
    """Official dynamic per-token/per-128 E4M3 + power-of-two scale recipe.

    Return raw FP8 bytes and compact E8M0 scales. This is *activation*
    quantization; checkpoint weights are never requantized by this function.
    """
    if x.shape[-1] % 128:
        raise ValueError("FP8 activation blocks require K divisible by 128")
    pairs = [quantize_activation_block_fp8(x[..., i : i + 128]) for i in range(0, x.shape[-1], 128)]
    return jnp.concatenate([p[0] for p in pairs], axis=-1), jnp.concatenate(
        [p[1] for p in pairs], axis=-1
    )


def activation_fp8_roundtrip(x, block_size=128):
    if x.shape[-1] % block_size:
        raise ValueError("FP8 activation width must be divisible by the block size")
    tiles = []
    for i in range(0, x.shape[-1], block_size):
        raw, scale = quantize_activation_block_fp8(x[..., i : i + block_size])
        tiles.append((decode_fp8(raw) * decode_e8m0(scale)).astype(jnp.bfloat16))
    return jnp.concatenate(tiles, axis=-1)


def encode_fp4_codes(x):
    """Saturating E2M1 round-to-nearest-even encoding (one nibble per byte)."""
    x = x.astype(jnp.float32)
    absolute = jnp.abs(x)
    code = jnp.zeros(x.shape, jnp.uint8)
    for index, boundary in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        crossed = absolute >= boundary if index % 2 else absolute > boundary
        code = code + crossed.astype(jnp.uint8)
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    sign = ((bits >> jnp.uint32(28)) & jnp.uint32(8)).astype(jnp.uint8)
    return code | sign


def activation_fp4_roundtrip(x, block_size=32):
    """Official indexer QAT simulation; output BF16, not FP4 model weights."""
    if x.shape[-1] % block_size:
        raise ValueError("FP4 activation width must divide into complete blocks")
    grouped = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, block_size)
    amax = jnp.maximum(
        jnp.max(jnp.abs(grouped), axis=-1, keepdims=True), jnp.float32(6 * 2.0**-126)
    )
    scaled = amax * jnp.float32(1.0 / 6.0)
    bits = jax.lax.bitcast_convert_type(scaled, jnp.uint32)
    exponent = (bits >> jnp.uint32(23)) + ((bits & jnp.uint32(0x7FFFFF)) != 0)
    scale = decode_e8m0(exponent.astype(jnp.uint8))
    codes = encode_fp4_codes(jnp.clip(grouped / scale, -6.0, 6.0))
    return (
        (decode_fp4_codes(codes).astype(jnp.float32) * scale).astype(jnp.bfloat16).reshape(x.shape)
    )
