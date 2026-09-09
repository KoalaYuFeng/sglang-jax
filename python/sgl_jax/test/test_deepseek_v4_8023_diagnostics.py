"""Diagnostic snapshots preserve bits; comparisons normalize page ownership."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.deepseek_v4.numerics import V4LayerConfig
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedMetadata

SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
with patch.object(sys, "path", [SCRIPTS, *sys.path]):
    import replay_deepseek_v4_8023 as replay
    from debug_deepseek_v4_8023 import load_arrays, save_arrays
    from replay_deepseek_v4_8023 import logical_attention_indices, logical_cache, request_trace


@pytest.mark.parametrize("compressed", [True, False])
def test_snapshot_preserves_bf16_bits_including_signed_zero_and_nonfinite(tmp_path, compressed):
    bits = np.asarray([0, 0x8000, 0x3F80, 0x7F80, 0xFF80, 0x7FC1], np.uint16)
    values = {"bf16": bits.view(jnp.bfloat16), "slots": np.asarray([2, 1], np.int32)}
    save_arrays(tmp_path / "snapshot", values, compressed=compressed)
    restored = load_arrays(tmp_path / "snapshot")
    np.testing.assert_array_equal(restored["bf16"].view(np.uint16), bits)
    np.testing.assert_array_equal(restored["slots"], values["slots"])


def test_snapshot_rejects_wrong_shape_manifest(tmp_path):
    path = tmp_path / "snapshot"
    save_arrays(path, {"x": np.asarray([1], np.float32)})
    schema = json.loads(path.with_suffix(".json").read_text())
    schema["x"]["shape"] = [2]
    path.with_suffix(".json").write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="invalid diagnostic array"):
        load_arrays(path)


def test_logical_cache_ignores_free_pages_and_selects_private_slot():
    config = V4LayerConfig(max_context=256, ratio=4)
    meta = V4PagedMetadata(
        req_slots=np.asarray([2]),
        prefix_lens=np.asarray([131]),
        seq_lens=np.asarray([132]),
        page_table=np.asarray([[1, 3]]),
    )
    physical = np.r_[np.arange(128, 256), np.arange(384, 512)]
    window = np.full((512, 1), -100, np.float32)
    window[physical, 0] = np.arange(256)
    compressed = np.full((128, 1), -200, np.float32)
    compressed[physical[3::4] // 4, 0] = np.arange(64)
    cache = {
        "window": window,
        "main.compressed": compressed,
        "main.kv": np.arange(5, dtype=np.float32)[:, None, None],
        "main.snapshot_kv": np.arange(4, dtype=np.float32)[:, None, None],
    }
    before = logical_cache(cache, meta, 0, config, after=False)
    after = logical_cache(cache, meta, 0, config, after=True)
    np.testing.assert_array_equal(before["window"].ravel(), np.arange(131))
    np.testing.assert_array_equal(after["main.compressed"].ravel(), np.arange(33))
    assert before["main.compressed"].shape == (32, 1)
    assert before["main.kv"].item() == 2
    assert before["main.snapshot_kv"].item() == 1


def test_attention_indices_compare_logical_addresses_not_physical_pages():
    config = V4LayerConfig(max_context=256, ratio=4)
    meta = V4PagedMetadata(page_table=np.asarray([[3, 1]]))
    actual = logical_attention_indices(np.asarray([386, 512 + 96, -1]), meta, 0, config, 512)
    np.testing.assert_array_equal(actual, [2, 256, -1])


def test_compressor_traces_select_request_groups_not_padded_rows():
    config = V4LayerConfig(ratio=4)
    meta = V4PagedMetadata(
        group4_requests=np.asarray([0, 1, 0]), group4_starts=np.asarray([8020, 8020, -1])
    )
    trace = {
        "compressor.main.kv": np.asarray([[1], [2]]),
        "compressor.main.pooled": np.asarray([[3], [4], [99]]),
        "compressor.main.destination": np.asarray([5, 6, 999]),
    }
    selected = request_trace(trace, meta, 1, config, 512)
    np.testing.assert_array_equal(selected["compressor.main.kv"], [2])
    np.testing.assert_array_equal(selected["compressor.main.pooled"], [[4]])
    assert "compressor.main.destination" not in selected


@pytest.mark.parametrize("backend", ["pallas", "reference"])
@pytest.mark.parametrize("index", [False, True])
def test_replay_propagates_explicit_csa_backend_into_compressor(backend, index):
    seen = []

    def compression(*args, index, trace, hca_backend, csa_backend):
        seen.append((index, hca_backend, csa_backend))
        trace["csa_emitted"] = np.ones((1, 128), np.float32)
        return {}

    def native(*args, config, mhc_backend, hca_backend, csa_backend):
        cache = replay.ATTENTION.compress(
            *args, index=index, hca_backend=hca_backend, csa_backend=csa_backend
        )
        return None, cache, {}

    with patch.object(replay, "COMPRESS", compression), patch.object(replay, "native_step", native):
        _, _, trace = replay.traced_step(
            None, config=None, hca_backend="pallas", csa_backend=backend
        )
    assert seen == [(index, "pallas", backend)]
    prefix = "index" if index else "main"
    assert f"compressor.{prefix}.csa_emitted" in trace
