"""Standalone candidate contract tests; no serving default changes."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[3] / "scripts/deepseek_v4_attention_candidates.py"
SPEC = importlib.util.spec_from_file_location("attention_candidates", PATH)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


@pytest.mark.parametrize("slots", [1, 63, 64, 65, 127, 128, 129, 256])
@pytest.mark.parametrize("masked", [False, True])
def test_key_blocks_and_sink(slots, masked):
    q = jnp.zeros((2, 128), jnp.bfloat16)
    kv = jnp.ones((slots, 128), jnp.bfloat16)
    ids = jnp.full(slots, -1, jnp.int32) if masked else jnp.arange(slots, dtype=jnp.int32)
    result = jax.jit(candidate.precise_single_query_attention)(q, kv, ids, jnp.zeros(2), 128**-0.5)
    expected = jnp.asarray(0 if masked else slots / (slots + 1), jnp.bfloat16)
    np.testing.assert_array_equal(result, jnp.full_like(q, expected))


def test_padding_cannot_change_valid_keys():
    q = jnp.zeros((2, 128), jnp.bfloat16)
    kv = jnp.arange(65, dtype=jnp.bfloat16)[:, None] * jnp.ones((1, 128), jnp.bfloat16)
    run = jax.jit(candidate.precise_single_query_attention)
    ids = jnp.arange(65, dtype=jnp.int32)
    a = run(q, kv, ids, jnp.zeros(2), 1.0)
    b = run(q, kv, jnp.pad(ids, (0, 63), constant_values=-1), jnp.zeros(2), 1.0)
    np.testing.assert_array_equal(a, b)


def test_empty_key_slots_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        candidate.precise_single_query_attention(
            jnp.zeros((2, 128)), jnp.zeros((1, 128)), jnp.zeros(0, jnp.int32), jnp.zeros(2), 1.0
        )
