"""Independent coefficient-oracle checks; no model checkpoint needed."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.deepseek_v4.numerics import rope
from sgl_jax.test.deepseek_v4_reference_rope import official_frequencies, official_recipe_rope
from sgl_jax.test.test_deepseek_v4_rope_numerics import FLASH, FLASH_FREQUENCY_BITS


def test_reference_coefficients_against_frozen_official_bits():
    pytest.importorskip("torch")
    np.testing.assert_array_equal(official_frequencies(FLASH).view(np.uint32), FLASH_FREQUENCY_BITS)
    assert not official_frequencies(FLASH).flags.writeable


@pytest.mark.parametrize("original", [0, 65536])
@pytest.mark.parametrize("inverse", [False, True])
def test_paging_oracle_against_native_without_sharing_coefficient_builder(original, inverse):
    pytest.importorskip("torch")
    cfg = replace(FLASH, original_seq_len=original, rope_base=10000 if not original else 160000)
    rng = np.random.default_rng(911)
    x = jnp.asarray(rng.normal(size=(7, 2, 128)), jnp.bfloat16)
    positions = jnp.array([0, 127, 219, 2571, 8023, 8191, 8319], jnp.int32)
    expected = jax.jit(lambda x, p: official_recipe_rope(x, p, cfg, inverse=inverse))(x, positions)
    actual = jax.jit(lambda x, p: rope(x, p, cfg, inverse=inverse))(x, positions)
    np.testing.assert_array_equal(actual, expected)
