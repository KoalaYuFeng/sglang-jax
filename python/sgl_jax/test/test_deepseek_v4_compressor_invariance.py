"""FP32 scratch must not depend on unrelated requests in the packed batch.

The real 8023 failure showed that a few FP32 ulps can cross the compressor's
BF16 rounding boundary and grow into a 10.76% full-logit discrepancy. A loose
FP32 allclose assertion is therefore not an adequate batch-isolation gate.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.deepseek_v4.compressor import compress
from sgl_jax.srt.kernels.deepseek_v4.numerics import V4LayerConfig
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.test.kernels.test_deepseek_v4_reference import _compressor_weights
from sgl_jax.test.test_deepseek_v4_paged import make_batch, make_cache


@pytest.mark.parametrize("ratio,index", [(4, False), (4, True), (128, False)])
def test_compressor_fp32_state_is_bitwise_batch_invariant(ratio, index):
    rng = np.random.default_rng(8023)
    config = V4LayerConfig(hidden=4096, max_context=384, ratio=ratio)
    weights = _compressor_weights(rng, config, index=index)
    prefix = "index" if index else "main"
    single, packed = make_cache(config), make_cache(config)
    for suffix in ("kv", "score"):
        key = f"{prefix}.{suffix}"
        primary = jnp.asarray(rng.normal(scale=0.1, size=single[key].shape[1:]), jnp.float32)
        other = jnp.asarray(rng.normal(scale=0.1, size=single[key].shape[1:]), jnp.float32)
        single[key] = single[key].at[1].set(primary)
        packed[key] = packed[key].at[3].set(primary).at[1].set(other)
    inputs = jnp.asarray(rng.normal(size=(2, config.hidden)), jnp.bfloat16)
    backend = V4PagedBackend(max_context=384)
    single_meta = backend.get_forward_metadata(make_batch([131], [1], decode=True))
    packed_meta = backend.get_forward_metadata(
        make_batch([127, 131], [1, 1], slots=[0, 2], pages=[[4, 6, 5], [7, 9, 8]], decode=True)
    )

    @jax.jit
    def run(x, weights, cache, meta):
        return compress(x, weights, dict(cache), config, meta, index=index)

    expected = run(inputs[1:], weights, single, single_meta)
    actual = run(inputs, weights, packed, packed_meta)
    for suffix in ("kv", "score"):
        np.testing.assert_array_equal(
            expected[f"{prefix}.{suffix}"][1],
            actual[f"{prefix}.{suffix}"][3],
            err_msg=f"{prefix}.{suffix}: adding/reordering another request changed FP32 state",
        )
    # Compare the complete emitted group at the same logical position; physical
    # page addresses deliberately differ between the two fixtures.
    if ratio == 4:
        np.testing.assert_array_equal(
            expected[f"{prefix}.compressed"][3 * 128 // ratio],
            actual[f"{prefix}.compressed"][9 * 128 // ratio],
        )
