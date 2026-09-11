"""V4 RoPE coefficient regression, independent of scheduler/model loading.

Optional frozen real-checkpoint replay:
SGL_V4_NUMERICAL_EVIDENCE=/path/to/v4-index-rootcause-evidence.tar.gz pytest ...
The archive contains operator captures, not an official GPU execution oracle.
"""

import hashlib
import io
import json
import math
import os
import tarfile
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.deepseek_v4.numerics import (
    V4LayerConfig,
    _rope_frequencies,
    config_for_layer,
    hadamard_rotate,
    rope,
    rope_angles,
)
from sgl_jax.srt.kernels.low_bit.formats import activation_fp4_roundtrip

FLASH = V4LayerConfig(rope_base=160000.0, original_seq_len=65536, max_context=8192)
# FP32 official-Python recipe from the frozen 2026-09-11 CSA-index diagnosis.
# Integer bits avoid decimal parsing or using the candidate to build its oracle.
FLASH_FREQUENCY_BITS = np.array(
    [
        1065353216,
        1060112954,
        1056054303,
        1051098369,
        1046804782,
        1042117749,
        1037601972,
        1033169251,
        1028443341,
        1024251131,
        1019326489,
        1015361742,
        1010249151,
        1006366088,
        1001209182,
        997081655,
        991197921,
        985076960,
        979287478,
        973418702,
        966505391,
        959469944,
        952012877,
        943627103,
        933484300,
        917783504,
        913828774,
        908753896,
        904558591,
        899759068,
        895336240,
        890797133,
    ],
    dtype=np.uint32,
)


def frequencies(config):
    return _rope_frequencies(
        config.rope_dim,
        config.rope_base,
        config.original_seq_len,
        config.rope_factor,
        config.beta_fast,
        config.beta_slow,
    )


def test_flash_frequencies_match_frozen_official_fp32_bits():
    actual = frequencies(FLASH)
    np.testing.assert_array_equal(actual.view(np.uint32), FLASH_FREQUENCY_BITS)
    assert not actual.flags.writeable
    assert actual is frequencies(replace(FLASH, max_context=256, ratio=128))
    with pytest.raises(ValueError):
        actual[0] = 0


@pytest.mark.parametrize("base,original", [(10000.0, 0), (160000.0, 65536)])
def test_host_coefficients_against_independent_torch_fp32_recipe(base, original):
    torch = pytest.importorskip("torch")
    config = replace(FLASH, rope_base=base, original_seq_len=original)
    rd = config.rope_dim
    expected = 1 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    if original:

        def correction(rotations):
            return rd * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(correction(config.beta_fast)), 0)
        high = min(math.ceil(correction(config.beta_slow)), rd - 1)
        ramp = (
            (torch.arange(rd // 2, dtype=torch.float32) - low)
            / (high - low if high != low else 0.001)
        ).clamp(0, 1)
        smooth = 1 - ramp
        expected = expected / config.rope_factor * (1 - smooth) + expected * smooth
    np.testing.assert_array_equal(frequencies(config), expected.numpy())


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32, 127, 128, 8192])
def test_jitted_phase_matches_frozen_recipe_across_shapes(batch):
    positions = (np.arange(batch, dtype=np.int32) * 127 + 216) % 8192
    expected = positions.astype(np.float32)[:, None] * FLASH_FREQUENCY_BITS.view(np.float32)
    run = jax.jit(lambda p: rope_angles(p, FLASH))
    actual = np.asarray(run(jnp.asarray(positions)))
    np.testing.assert_array_equal(actual, expected)
    # Also exercise new dynamic positions through the same compiled shape.
    positions = 8191 - positions
    expected = positions.astype(np.float32)[:, None] * FLASH_FREQUENCY_BITS.view(np.float32)
    np.testing.assert_array_equal(run(jnp.asarray(positions)), expected)


def test_compiled_phase_does_not_construct_frequencies_on_device():
    expression = jax.make_jaxpr(lambda p: rope_angles(p, FLASH))(jnp.zeros(4, jnp.int32))
    primitives = {equation.primitive.name for equation in expression.jaxpr.eqns}
    assert "mul" in primitives
    assert not primitives & {"pow", "div", "exp", "log", "iota", "pure_callback", "io_callback"}


@pytest.mark.parametrize("dim", [0, -2, 3])
def test_invalid_rope_dimensions(dim):
    with pytest.raises(ValueError, match="positive and even"):
        frequencies(replace(FLASH, rope_dim=dim))


def test_frozen_position_219_rope_hadamard_fp4_replay():
    path = os.environ.get("SGL_V4_NUMERICAL_EVIDENCE")
    if not path:
        pytest.skip("set SGL_V4_NUMERICAL_EVIDENCE to the frozen diagnostic archive")
    path = Path(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "521582e7815418122b1aa78abc888b79d41fc72c29d8e9f42a20cd37103deac3"
    )
    with tarfile.open(path, "r:gz") as archive:
        config = config_for_layer(json.load(archive.extractfile("official/config.json")), 42, 256)
        name = "v4-official-index-diagnose-20260911-02/official-target.npz"
        with np.load(io.BytesIO(archive.extractfile(name).read())) as data:
            expected = {key: data[key].copy() for key in data.files}

    @jax.jit
    def run(normalized):
        rotated = rope(normalized, jnp.full((4,), 216, jnp.int32), config)
        pre_fp4 = hadamard_rotate(rotated)
        return rotated, pre_fp4, activation_fp4_roundtrip(pre_fp4)

    actual = run(jnp.asarray(expected["normalized"][:, 0], jnp.bfloat16))
    for value, key in zip(actual, ("after_rope", "pre_fp4", "final"), strict=True):
        np.testing.assert_array_equal(np.asarray(value, np.float32), expected[key][:, 0])
