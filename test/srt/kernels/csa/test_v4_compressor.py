"""Standalone V4 compressor gates: no Engine, scheduler, or model allocation."""

import os
from dataclasses import replace

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.csa.compressor import (
    _select_overlap_channels,
    csa_emit_overlap_pallas,
    csa_emit_selected_pallas,
)
from sgl_jax.test.kernels.csa_compressor_cases import (
    make_case,
    oracle_check,
    real_cases,
    select_channels,
)

TPU = pytest.mark.skipif(jax.default_backend() != "tpu", reason="real Mosaic lowering required")


def overlap_emit(case, tile=1):
    return np.asarray(
        csa_emit_overlap_pallas(*(jnp.asarray(a) for a in case.inputs()), tile_groups=tile)
    )


def selected_emit(case):
    values = select_channels(case.values)
    scores = np.where(case.valid[:, None, None], select_channels(case.scores), 0)
    return np.asarray(
        csa_emit_selected_pallas(
            jnp.asarray(values),
            jnp.asarray(scores),
            jnp.asarray(case.norm),
            jnp.asarray(case.phase),
            jnp.asarray(case.valid),
        )
    )


def assert_oracle(case, actual):
    check = oracle_check(case, actual)
    assert check["passed"], check


@TPU
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("groups", [1, 2, 5, 9, 65])
def test_selected_baseline_independent_oracle(dim, groups):
    case = make_case(dim, groups)
    assert_oracle(case, selected_emit(case))


@TPU
@pytest.mark.parametrize("dim", [128, 512])
def test_selected_baseline_group_order_and_partition(dim):
    case = make_case(dim, 9)
    baseline = selected_emit(case)
    order = np.array([8, 3, 0, 5, 2, 7, 1, 4, 6])
    reordered = replace(
        case,
        values=case.values[order],
        scores=case.scores[order],
        phase=case.phase[order],
        valid=case.valid[order],
    )
    np.testing.assert_array_equal(
        selected_emit(reordered).view(np.uint16), baseline[order].view(np.uint16)
    )
    chunks = []
    for start, end in ((0, 1), (1, 6), (6, 9)):
        chunk = replace(
            case,
            values=case.values[start:end],
            scores=case.scores[start:end],
            phase=case.phase[start:end],
            valid=case.valid[start:end],
        )
        chunks.append(selected_emit(chunk))
    np.testing.assert_array_equal(np.concatenate(chunks).view(np.uint16), baseline.view(np.uint16))


@TPU
def test_selected_baseline_real_7979():
    root = os.environ.get("CSA_V4_REAL_FIXTURE_ROOT")
    if not root:
        pytest.skip("set CSA_V4_REAL_FIXTURE_ROOT to immutable historical receipts")
    cases = list(real_cases(root))
    assert len(cases) == 4
    for case in cases:
        assert_oracle(case, selected_emit(case))


@TPU
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("groups", [1, 2, 5, 9, 65])
def test_overlap_emitter_bitwise_and_oracle(dim, groups):
    case = make_case(dim, groups)
    actual = overlap_emit(case)
    np.testing.assert_array_equal(actual.view(np.uint16), selected_emit(case).view(np.uint16))
    assert_oracle(case, actual)


@TPU
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("tile", [2, 4, 8])
def test_overlap_emitter_alternative_tiles(dim, tile):
    case = make_case(dim, 9)
    np.testing.assert_array_equal(
        overlap_emit(case, tile).view(np.uint16), selected_emit(case).view(np.uint16)
    )


@TPU
@pytest.mark.parametrize("dim", [128, 512])
def test_overlap_channel_selection_in_vmem_with_poisoned_other_halves(dim):
    case = make_case(dim, 4)
    raw = case.values.copy()
    raw[:, :4, dim:] = np.nan
    raw[:, 4:, :dim] = np.nan

    def kernel(value_ref, out_ref):
        out_ref[...] = _select_overlap_channels(value_ref[...], head_dim=dim)

    actual = pl.pallas_call(
        kernel,
        in_specs=(pl.BlockSpec(raw.shape, lambda: (0, 0, 0)),),
        out_specs=pl.BlockSpec((4, 8, dim), lambda: (0, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((4, 8, dim), jnp.float32),
        name=f"csa-v4-overlap-selection-probe-d{dim}",
    )(jnp.asarray(raw))
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), select_channels(raw).view(np.uint32)
    )
    poisoned = replace(case, values=raw)
    np.testing.assert_array_equal(
        overlap_emit(poisoned).view(np.uint16), selected_emit(case).view(np.uint16)
    )


@TPU
@pytest.mark.parametrize("dim", [128, 512])
def test_overlap_emitter_order_partition_and_no_outer_gather(dim):
    case = make_case(dim, 9)
    expected = selected_emit(case)
    order = np.array([8, 3, 0, 5, 2, 7, 1, 4, 6])
    shuffled = replace(
        case,
        values=case.values[order],
        scores=case.scores[order],
        phase=case.phase[order],
        valid=case.valid[order],
    )
    np.testing.assert_array_equal(
        overlap_emit(shuffled).view(np.uint16), expected[order].view(np.uint16)
    )
    pieces = []
    for start, end in ((0, 1), (1, 6), (6, 9)):
        part = replace(
            case,
            values=case.values[start:end],
            scores=case.scores[start:end],
            phase=case.phase[start:end],
            valid=case.valid[start:end],
        )
        pieces.append(overlap_emit(part))
    np.testing.assert_array_equal(np.concatenate(pieces).view(np.uint16), expected.view(np.uint16))
    hlo = (
        jax.jit(csa_emit_overlap_pallas)
        .lower(*(jnp.asarray(a) for a in case.inputs()))
        .compile()
        .as_text()
    )
    instructions = [line.partition("backend_config=")[0] for line in hlo.splitlines()]
    assert any("custom-call(" in line and "v4-overlap" in line for line in instructions)
    assert not any(" gather(" in line for line in instructions)


@TPU
def test_overlap_emitter_real_7979():
    root = os.environ.get("CSA_V4_REAL_FIXTURE_ROOT")
    if not root:
        pytest.skip("set CSA_V4_REAL_FIXTURE_ROOT to immutable historical receipts")
    for case in real_cases(root):
        actual = overlap_emit(case)
        np.testing.assert_array_equal(actual.view(np.uint16), selected_emit(case).view(np.uint16))
        assert_oracle(case, actual)


@TPU
@pytest.mark.parametrize("dim", [128, 512])
def test_overlap_emitter_four_chip_replicas(dim):
    if jax.device_count() != 4:
        pytest.skip("requires the serving configuration's four v5p chips")
    case = make_case(dim, 5)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices(), object), ("tensor",))
    spec = jax.sharding.PartitionSpec()
    sharding = jax.sharding.NamedSharding(mesh, spec)
    call = jax.jit(
        jax.shard_map(
            csa_emit_overlap_pallas,
            mesh=mesh,
            in_specs=(spec,) * 5,
            out_specs=spec,
            check_vma=False,
        )
    )
    actual = call(*(jax.device_put(a, sharding) for a in case.inputs()))
    expected = selected_emit(case).view(np.uint16)
    assert len(actual.addressable_shards) == 4
    for shard in actual.addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data).view(np.uint16), expected)
