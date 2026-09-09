"""CPU guards for the isolated TP experiment and its fixture construction."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import benchmark_deepseek_v4_attention_tp as experiment

from sgl_jax.srt.kernels.deepseek_v4.numerics import V4LayerConfig
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.test.test_deepseek_v4_paged import make_batch


def test_only_query_sink_and_complete_output_groups_are_head_sharded():
    names = experiment.HEAD_WEIGHTS | {
        "attn.indexer.wq_b.weight",
        "attn.wo_b.weight",
        "attn.compressor.wkv.weight",
    }
    weights = {
        key: np.zeros((64,) if key.endswith("sink") else (128, 128), np.uint8) for key in names
    }
    specs = experiment.weight_specs(weights, True)
    for key, spec in specs.items():
        expected = (
            P("tensor", *([None] * (weights[key].ndim - 1)))
            if key in experiment.HEAD_WEIGHTS
            else P()
        )
        assert spec == expected
    assert all(spec == P() for spec in experiment.weight_specs(weights, False).values())


def test_head_tp_rejects_split_output_groups():
    with pytest.raises(ValueError, match="complete wo_a groups"):
        experiment.build(SimpleNamespace(size=3), V4LayerConfig(), {}, tp=True, trace=False)


def test_bitwise_comparison_handles_strided_and_broadcast_host_views():
    for expected in (
        np.arange(24, dtype=np.float32).reshape(4, 6).T,
        np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2],
        np.broadcast_to(np.float32(-np.inf), (4, 3)),
    ):
        actual = np.array(expected, order="C", copy=True)
        result = experiment.metrics(expected, actual, tolerance=0, exact=True)
        assert result["passed"] and result["bitwise_equal"]
    different = np.array([[0.0, -0.0]], np.float32)
    assert not experiment.metrics(different, np.zeros_like(different), tolerance=0, exact=True)[
        "passed"
    ]


@pytest.mark.parametrize("ratio", [4, 128])
def test_cloned_requests_have_independent_pages_slots_and_identical_prefix_data(ratio):
    config = V4LayerConfig(ratio=ratio, max_context=256)
    batch = make_batch([129, 131], [1, 1], slots=[1, 3], pages=[[2, 1], [5, 4]], decode=True)
    metadata = V4PagedBackend(max_context=256).get_forward_metadata(batch)
    rng = np.random.default_rng(8023)
    width = 2
    shapes = {
        "window": (9 * 128, width),
        "main.compressed": (9 * 128 // ratio, width),
        "main.kv": (5, ratio, width),
        "main.score": (5, ratio, width),
        "main.snapshot_kv": (9, ratio, width),
        "main.snapshot_score": (9, ratio, width),
    }
    cache = {key: rng.normal(size=shape).astype(np.float32) for key, shape in shapes.items()}
    old_digests = {key: experiment.array_digest(value) for key, value in cache.items()}
    hidden = rng.normal(size=(2, 4096)).astype(np.float32)
    inputs = {"positions": batch.positions, "locations": batch.out_cache_loc}
    x, inputs4, new, md = experiment.clone_b4(hidden, inputs, cache, metadata, config)
    np.testing.assert_array_equal(x, hidden[[0, 1, 0, 1]])
    np.testing.assert_array_equal(md.req_slots, [1, 2, 3, 4])
    assert len(np.unique(inputs4["locations"])) == 4
    assert len(np.unique(md.page_table)) == 8
    for dest, source in enumerate((0, 1, 0, 1)):
        src_pages, dst_pages = metadata.page_table[source], md.page_table[dest]
        for key in cache:
            if key.endswith((".kv", ".score")):
                np.testing.assert_array_equal(
                    new[key][dest + 1], cache[key][metadata.req_slots[source]]
                )
            elif ".snapshot_" in key:
                np.testing.assert_array_equal(new[key][dst_pages], cache[key][src_pages])
            else:
                per_page = 128 if key == "window" else 128 // ratio
                np.testing.assert_array_equal(
                    new[key].reshape(9, per_page, width)[dst_pages],
                    cache[key].reshape(9, per_page, width)[src_pages],
                )
    assert old_digests == {key: experiment.array_digest(value) for key, value in cache.items()}


@pytest.mark.parametrize(
    "actual",
    [
        np.array([np.nan, 1], np.float32),
        np.array([np.inf, 1], np.float32),
        np.array([-np.inf, 2], np.float32),
    ],
)
def test_exact_state_gate_does_not_hide_nonfinite_or_value_mismatches(actual):
    assert not experiment.metrics(
        np.array([-np.inf, 1], np.float32), actual, tolerance=0, exact=True
    )["passed"]


def test_exact_state_gate_accepts_matching_negative_infinity():
    x = np.array([-np.inf, 1], np.float32)
    assert experiment.metrics(x, x, tolerance=0, exact=True)["passed"]


def test_comparison_rejects_broadcast_or_dtype_changes():
    with pytest.raises(AssertionError, match="shape/dtype"):
        experiment.metrics(
            np.ones(2, np.float32), np.ones(2, np.float64), tolerance=0.005, exact=False
        )


def test_timing_inputs_are_prepared_independently_not_updated_state_feedback():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tensor",))
    frozen = {"kv": np.zeros(2, np.float32)}

    def compiled(x, positions, weights, cache, metadata, locations):
        return x, {"kv": cache["kv"] + 1}

    steps = experiment.IndependentSteps(compiled, (None,) * 6, frozen, mesh)
    with pytest.raises(ValueError, match="prepare frozen"):
        steps()
    steps.prepare(2)
    with pytest.raises(ValueError, match="unconsumed"):
        steps.prepare(1)
    first, second = steps(), steps()
    np.testing.assert_array_equal(first[1]["kv"], [1, 1])
    np.testing.assert_array_equal(second[1]["kv"], [1, 1])
    np.testing.assert_array_equal(frozen["kv"], [0, 0])
    with pytest.raises(ValueError, match="prepare frozen"):
        steps()
