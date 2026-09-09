"""CPU guards for chunk sweeps; these are not full-model TPU correctness tests."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from profile_deepseek_v4_prefill import (
    CHUNKS,
    TRACE_PREFIXES,
    check_logits,
    chunk_plan,
    same_state,
    state_record,
    validate_fixture,
)
from run_deepseek_v4_paged import PagedWorkerSession, paged_test_token_bucket


@pytest.mark.parametrize("chunk", CHUNKS)
def test_chunks_cover_same_prompt_and_reference_boundaries(chunk):
    plan = chunk_plan(chunk)
    assert plan[0][0] == 0 and plan[-1][1] == 7936
    assert all(a[1] == b[0] for a, b in zip(plan, plan[1:]))
    assert sum(end - begin for begin, end in plan) == 7936
    assert all(end % 128 == 0 and 0 < end - begin <= chunk for begin, end in plan)
    assert set(TRACE_PREFIXES) <= {begin for begin, _ in plan}
    if chunk == 512:
        assert plan[-1] == (7680, 7936)  # a real padded tail, never extra input tokens


@pytest.mark.parametrize("chunk,length", [(64, 7936), (1024, 7936), (128, 0), (256, 7935)])
def test_chunk_plan_rejects_unsupported_or_unaligned_experiments(chunk, length):
    with pytest.raises(ValueError):
        chunk_plan(chunk, length)


def test_existing_native_bucket_defaults_do_not_change():
    assert paged_test_token_bucket([128], decode=False, bucket=1) == 128
    assert paged_test_token_bucket([64, 64], decode=False, bucket=2) == 128
    assert paged_test_token_bucket([1, 1], decode=True, bucket=4) == 4


@pytest.mark.parametrize("tokens", [128, 256, 512])
def test_explicit_prefill_capacity_and_padding(tokens):
    assert paged_test_token_bucket([tokens], decode=False, bucket=1, token_bucket=tokens) == tokens
    assert (
        paged_test_token_bucket([tokens // 2], decode=False, bucket=1, token_bucket=tokens)
        == tokens
    )


@pytest.mark.parametrize(
    "counts,decode,bucket,tokens",
    [
        ([129], False, 1, None),
        ([513], False, 1, 512),
        ([128, 128], False, 1, 256),
        ([0], False, 1, 128),
        ([], False, 1, 128),
        ([1], True, 1, 128),
        ([2], True, 2, None),
        ([1], False, 1, True),
        ([1], False, 1, 0),
    ],
)
def test_invalid_bucket_is_rejected_before_page_allocation(counts, decode, bucket, tokens):
    allocator = Mock()
    worker = SimpleNamespace(model_runner=SimpleNamespace(token_to_kv_pool_allocator=allocator))
    session = PagedWorkerSession(worker)
    items = [(i, [1] * count) for i, count in enumerate(counts)]
    with pytest.raises(ValueError, match="invalid test bucket"):
        session.step(items, decode=decode, bucket=bucket, token_bucket=tokens)
    allocator.alloc_decode.assert_not_called()
    allocator.alloc_extend.assert_not_called()


@pytest.mark.parametrize("dtype", [np.float32, jnp.bfloat16])
def test_state_hash_preserves_bits_dtype_shape_and_noncontiguous_views(dtype):
    a = np.arange(24).astype(dtype).reshape(4, 6)[:, ::2]
    assert same_state(a, np.ascontiguousarray(a))
    assert not same_state(a, a.reshape(-1))
    b = np.ascontiguousarray(a).copy()
    b[0, 0] = 1
    assert not same_state(a, b)
    assert len(state_record("test", a)["sha256"]) == 64
    a = np.array([0.0, -np.inf], dtype=dtype)
    b = np.array([-0.0, -np.inf], dtype=dtype)
    assert not same_state(a, b)


def test_logits_retain_original_tolerance_and_require_finite_top1():
    a = np.array([0.0, 2.0], np.float32)
    assert check_logits(a, a)["passed"]
    assert not check_logits(a, np.array([2.0, 0.0]))["passed"]
    assert not check_logits(a, np.array([0.0, 3.0]))["passed"]
    assert not check_logits(a, np.array([0.0, np.inf]))["passed"]


def fixtures(tmp_path):
    checkpoint, native_path = tmp_path / "checkpoint", tmp_path / "native.json"
    native = {
        "complete": True,
        "finished": True,
        "checkpoint": str(checkpoint),
        "framework_source_fingerprint": "source",
        "source_fingerprint": "reference",
        "prompt_tokens": 7936,
        "generation_tokens": 256,
        "server_args": {"json_model_override_args": "{}"},
    }
    oracle = {
        "complete": True,
        "finished": True,
        "checkpoint": str(checkpoint),
        "framework_source_fingerprint": "source",
        "reference_source_fingerprint": "reference",
        "native_report": str(native_path),
    }
    return (
        native,
        oracle,
        dict(
            checkpoint=checkpoint, native_path=native_path, source="source", reference="reference"
        ),
    )


def test_complete_same_source_oracle_is_required(tmp_path):
    native, oracle, options = fixtures(tmp_path)
    assert validate_fixture(native, oracle, **options)["moe_backend"] == "legacy"
    for owner, field, value in (
        ("native", "complete", False),
        ("oracle", "finished", False),
        ("native", "framework_source_fingerprint", "changed"),
        ("oracle", "reference_source_fingerprint", "changed"),
        ("oracle", "checkpoint", "/other/checkpoint"),
        ("oracle", "native_report", "/other/native.json"),
        ("native", "generation_tokens", 128),
    ):
        n, o = copy.deepcopy(native), copy.deepcopy(oracle)
        (n if owner == "native" else o)[field] = value
        with pytest.raises(ValueError):
            validate_fixture(n, o, **options)
