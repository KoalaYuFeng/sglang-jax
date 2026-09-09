"""Regression coverage for scratch allocations that fill the compiler VMEM budget."""

import jax
import jax.numpy as jnp
import pytest

from sgl_jax.srt.kernels.update_kv_cache.update_kv_cache import (
    get_num_slices_per_block,
)


@pytest.mark.parametrize("combined_heads,expected_slices", [(16, 255), (4, 1023)])
def test_page64_scratch_leaves_room_for_compiler(combined_heads, expected_slices):
    # These MHA/GQA shapes previously requested exactly 64 MiB of scratch and
    # failed to compile on v5p: XLA's usable limit is 64 MiB minus 64 KiB.
    new_kv = jax.ShapeDtypeStruct((2048, 1, combined_heads // 2, 2, 128), jnp.bfloat16)
    kv_cache = jax.ShapeDtypeStruct((4096, 64, combined_heads // 2, 2, 128), jnp.bfloat16)

    slices = get_num_slices_per_block(new_kv, kv_cache, page_size=64)

    assert slices == expected_slices
    scratch_bytes = slices * 64 * combined_heads * 128 * 2
    assert scratch_bytes <= 64 * 1024 * 1024 - 64 * 1024


def test_small_update_keeps_its_token_count():
    new_kv = jax.ShapeDtypeStruct((3, 1, 8, 2, 128), jnp.bfloat16)
    kv_cache = jax.ShapeDtypeStruct((4096, 64, 8, 2, 128), jnp.bfloat16)

    assert get_num_slices_per_block(new_kv, kv_cache, page_size=64) == 3
