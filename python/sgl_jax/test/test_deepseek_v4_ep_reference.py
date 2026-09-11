"""Check independent rank-passing and candidate reductions against NumPy."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.deepseek_v4.collectives import ordered_ep_sum

SOURCE = Path(__file__).resolve().parents[3] / "scripts/deepseek_v4_ep_reference.py"
spec = importlib.util.spec_from_file_location("independent_ep_reference", SOURCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.mark.parametrize("ranks", [1, 2, 4])
@pytest.mark.parametrize("tokens", [1, 16, 32, 128])
def test_rank_passing_candidate_and_numpy_bits(ranks, tokens):
    if jax.device_count() < ranks:
        pytest.skip(f"requires {ranks} devices")
    rng = np.random.default_rng(882)
    partials = rng.normal(size=(ranks, tokens, 128)).astype(np.float32)
    partials *= np.exp2(rng.integers(-20, 20, partials.shape)).astype(np.float32)
    shared = rng.normal(size=(tokens, 128)).astype(ml_dtypes.bfloat16)
    # Three faithful prefill counterexamples plus the original decode case.
    cases = [
        ([-0.0242919921875, -0.0279540978372097, 0, 0.1051025390625], -0.09521484375),
        ([0.058837890625, 0.1020507887005806, -0.056396484375, 0.0172119140625], -0.07666015625),
        ([0.2204599827528, 0.054931640625, -0.02734375, 0], -0.248046875),
        ([-0.1513671875, 2**-27, 0.042236328125, 0], 0.1904296875),
    ]
    if ranks == 4:
        for column, (parts, s) in enumerate(cases):
            partials[:, :, column] = np.asarray(parts, np.float32)[:, None]
            shared[:, column] = s
    mesh = Mesh(np.asarray(jax.devices()[:ranks]), ("tensor",))
    with jax.set_mesh(mesh):
        x = jax.device_put(partials, NamedSharding(mesh, P("tensor", None, None)))
        s = jax.device_put(shared, NamedSharding(mesh, P()))
        expected = reference.numpy_ep_sum(partials)
        expected_bf16 = (expected + shared.astype(np.float32)).astype(ml_dtypes.bfloat16)
        for fn in (reference.ring_ep_sum_reference, ordered_ep_sum):
            def run(value, shared, fn=fn):
                total = fn(value[0])
                return total, (total + shared.astype(jnp.float32)).astype(jnp.bfloat16)

            execute = jax.jit(jax.shard_map(
                run, mesh=mesh, in_specs=(P("tensor", None, None), P()),
                out_specs=(P(), P()), check_vma=False,
            ))
            actual, rounded = execute(x, s)
            for a, e, dtype in ((actual, expected, np.uint32), (rounded, expected_bf16, np.uint16)):
                for shard in a.addressable_shards:
                    np.testing.assert_array_equal(np.asarray(shard.data).view(dtype), e.view(dtype))


def test_reference_is_separate_from_candidate_and_rejects_wrong_dtype():
    source = SOURCE.read_text()
    assert "ordered_ep_sum" not in source and "all_gather" not in source
    with pytest.raises(ValueError, match="FP32"):
        reference.numpy_ep_sum(np.zeros((4, 1, 128), np.float16))
    with pytest.raises(ValueError, match="FP32"):
        reference.ring_ep_sum_reference(jnp.zeros((1, 128), jnp.bfloat16))
