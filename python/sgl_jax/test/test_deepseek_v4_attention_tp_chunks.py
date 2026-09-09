"""CPU guards for the advancing TP acceptance fixture, not model accuracy."""

import sys
from pathlib import Path

import numpy as np
import pytest

from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from validate_deepseek_v4_attention_tp_chunks import cold_plan, make_frame, near8k_plan


def operands():
    pages = [[i * 3 + 1, i * 3 + 3, i * 3 + 2] for i in range(4)]
    inputs = [np.full((384, 16), i + 1, np.float32) for i in range(4)]
    return pages, inputs


def test_cold_plan_stays_in_framework_bucket_and_covers_mixed_and_pages():
    previous = np.zeros(4, np.int32)
    modes, batches = set(), set()
    pages, inputs = operands()
    for ends in cold_plan():
        counts = np.asarray(ends) - previous
        assert np.all(counts >= 0) and 0 < counts.sum() <= 128
        _, batch, metadata = make_frame(
            previous, ends, [3, 1, 0, 2], pages, inputs, decode=np.all(counts <= 1)
        )
        modes.add(batch.forward_mode)
        batches.add(batch.real_bs)
        assert metadata.query_lens.sum() == counts.sum()
        assert np.count_nonzero(metadata.token_valid) == counts.sum()
        previous = np.asarray(ends)
    assert modes == {ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.DECODE}
    assert batches == {1, 2, 4}
    assert np.all(previous > 256)


def test_decode_inert_rows_stay_slot_aligned_after_permutation():
    pages, inputs = operands()
    hidden, batch, metadata = make_frame(
        [7, 128, 4, 0], [8, 128, 5, 0], [1, 2, 3, 0], pages, inputs, decode=True
    )
    np.testing.assert_array_equal(metadata.token_valid, [False, True, False, True])
    np.testing.assert_array_equal(metadata.req_slots, [0, 3, 0, 1])
    np.testing.assert_array_equal(batch.positions, [0, 4, 0, 7])
    np.testing.assert_array_equal(hidden[:, 0], [0, 3, 0, 1])
    assert batch.out_cache_loc[0] == batch.out_cache_loc[2] == -1


def test_fork_targets_dirty_slot_three_and_immutable_prefix_pages():
    pages, inputs = operands()
    pages[3], inputs[3] = pages[0][:2] + [13], inputs[0]
    hidden, _, metadata = make_frame([0, 0, 0, 256], [0, 0, 0, 263], [3, 1, 0, 2], pages, inputs)
    np.testing.assert_array_equal(metadata.req_slots, [4, 0, 0, 0])
    np.testing.assert_array_equal(metadata.page_table[0, :3], [1, 3, 13])
    np.testing.assert_array_equal(hidden[:7, 0], np.ones(7))
    assert not np.any(metadata.token_valid[7:])


@pytest.mark.parametrize("previous", [[8023] * 4, [8019, 8023, 8019, 8023]])
def test_near8k_plan_advances_to_last_position_with_bounded_calls(previous):
    cursor = np.asarray(previous)
    seen = set()
    for ends in near8k_plan(previous):
        counts = np.asarray(ends) - cursor
        assert np.all(counts >= 0) and 0 < counts.sum() <= 128
        for begin, end in zip(cursor, ends, strict=True):
            seen.update(range(int(begin), int(end)))
        cursor = np.asarray(ends)
    np.testing.assert_array_equal(cursor, [8192] * 4)
    assert {8023, 8063, 8064, 8191} <= seen


def test_frame_rejects_oversized_or_nonadvancing_calls():
    pages, inputs = operands()
    with pytest.raises(ValueError, match="bucket"):
        make_frame([0] * 4, [33] * 4, list(range(4)), pages, inputs)
    with pytest.raises(ValueError, match="live query"):
        make_frame([2] * 4, [2] * 4, list(range(4)), pages, inputs)
    with pytest.raises(ValueError, match="advancing"):
        make_frame([3] * 4, [1] * 4, list(range(4)), pages, inputs)
