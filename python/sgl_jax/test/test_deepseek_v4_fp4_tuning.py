"""Independent FP4 tile/layout gates before any model integration."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental import pallas as pl

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm import gmm
from sgl_jax.srt.kernels.low_bit.fp4_tuning import (
    CandidateCheckpointFP4Rhs,
    dequantize_packed_pairs,
    transpose_compact_scales,
)
from sgl_jax.srt.kernels.low_bit.formats import decode_e8m0
from sgl_jax.srt.kernels.low_bit.gmm import CheckpointFP4Rhs, grouped_fp4_matmul
from sgl_jax.srt.kernels.low_bit.matmul import _expand_columns, _unpack_fp4_vmem
from sgl_jax.test.kernels.test_deepseek_v4_low_bit import (
    _assert_close,
    _bf16,
    _fixture_weight,
    _reference_activation,
    _reference_weight,
)


@pytest.mark.parametrize("tile_m", [8, 16, 32])
@pytest.mark.parametrize("tile_n", [128, 256])
@pytest.mark.parametrize("hidden", [128, 512])
@pytest.mark.parametrize("packed_scale", [False, True])
def test_transposed_scale_candidate_matches_original_and_numpy(
    tile_m, tile_n, hidden, packed_scale
):
    x = np.random.default_rng(21).normal(size=(32, hidden)).astype(ml_dtypes.bfloat16)
    pairs = [_fixture_weight("fp4", 256, hidden, 310 + e) for e in range(4)]
    weights = jnp.asarray(np.stack([w for w, _ in pairs]))
    scales = jnp.asarray(np.stack([s for _, s in pairs]))
    # Partial M tiles, empty groups, non-zero EP offset and non-local sentinels.
    sizes = jnp.array([0, 1, 3, 17, 0, 11], jnp.int32)
    control = grouped_fp4_matmul(
        jnp.asarray(x),
        weights,
        scales,
        sizes,
        first_expert=jnp.int32(1),
        interpret=jax.default_backend() != "tpu",
    )
    actual = gmm(
        jnp.asarray(x),
        weights,
        sizes,
        rhs_scale=transpose_compact_scales(scales),
        group_offset=jnp.int32(1),
        preferred_element_type=jnp.bfloat16,
        tiling=(tile_m, hidden, tile_n),
        rhs_adapter=CandidateCheckpointFP4Rhs(
            transpose_scales=True, tile_m=tile_m, tile_n=tile_n, packed_scale=packed_scale
        ),
        interpret=jax.default_backend() != "tpu",
    )
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16), np.asarray(control).view(np.uint16)
    )
    expected = np.zeros((32, 256), np.float32)
    offset = 0
    for group, count in enumerate(np.asarray(sizes)):
        if 1 <= group < 5 and count:
            w, s = pairs[group - 1]
            expected[offset : offset + count] = _bf16(
                _reference_activation(x[offset : offset + count]) @ _reference_weight(w, s, "fp4").T
            )
        offset += count
    _assert_close(actual, expected)


def test_transposition_preserves_every_scale_code_and_storage_size():
    scales = jnp.asarray(np.arange(1024, dtype=np.uint8).reshape(2, 128, 4))
    transposed = transpose_compact_scales(scales)
    assert transposed.nbytes == scales.nbytes
    np.testing.assert_array_equal(np.asarray(transposed).swapaxes(1, 2), scales)


def test_packed_scale_dequant_all_fp4_codes_and_e8m0_exponents():
    # Low nibbles cycle through all 16 codes inside EVERY scale block. Across
    # rows, all 256 exponent bytes occur, including zeros/subnormals/NaNs.
    packed = jnp.asarray(np.broadcast_to(np.arange(128, dtype=np.uint8), (128, 128)))
    scales = jnp.asarray(np.arange(1024, dtype=np.uint8).reshape(128, 8))

    def run(candidate):
        def kernel(w, s, out):
            if candidate:
                out[...] = dequantize_packed_pairs(w[...], s[...])
            else:
                out[...] = (
                    _unpack_fp4_vmem(w[...]).astype(jnp.float32)
                    * _expand_columns(decode_e8m0(s[...]), 32)
                ).astype(jnp.bfloat16)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((128, 256), jnp.bfloat16),
            interpret=jax.default_backend() != "tpu",
        )(packed, scales)

    np.testing.assert_array_equal(
        np.asarray(run(True)).view(np.uint16), np.asarray(run(False)).view(np.uint16)
    )


def test_candidate_never_accepts_split_k_or_changes_baseline_tile_contract():
    candidate = CandidateCheckpointFP4Rhs(tile_m=32, tile_n=256)
    candidate.validate_tiling(tm=32, tk=4096, tn=256, k=4096)
    with pytest.raises(ValueError, match="full-K"):
        candidate.validate_tiling(tm=32, tk=2048, tn=256, k=4096)
    with pytest.raises(ValueError, match="8/full-K/128"):
        CheckpointFP4Rhs().validate_tiling(tm=32, tk=4096, tn=256, k=4096)


@pytest.mark.parametrize("value", [jnp.zeros((2, 128), jnp.uint8), jnp.zeros((2, 128, 4))])
def test_scale_transpose_rejects_invalid_storage(value):
    with pytest.raises(ValueError, match="uint8"):
        transpose_compact_scales(value)
