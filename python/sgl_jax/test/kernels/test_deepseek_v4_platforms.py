"""Hardware schedule and CSA mesh-contract regressions (no TPU needed)."""

from types import SimpleNamespace

import pytest

from sgl_jax.srt.kernels.csa.sharding import validate_csa_mesh
from sgl_jax.srt.kernels.csa.tune import (
    get_csa_attention_schedule,
    get_csa_compressor_projection_k_tile,
)
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.kernels.mhc.tune import (
    select_collapse_block_tokens,
    select_gates_block_tokens,
    select_post_backend,
    select_post_block_tokens,
)


@pytest.mark.parametrize("device", ("TPU v5", "TPU v5p", "v5p", " TPU V5 "))
def test_v5p_schedules(device):
    schedule = get_hca_kernel_schedule(
        device, page_size=128, max_compressed_entries=129, local_heads=16, head_dim=512
    )
    assert schedule.platform == "TPU v5p"
    assert schedule.projection_k_tile == 512
    assert schedule.prefill_entries_per_step % schedule.sublanes == 0
    assert schedule.compressed_tile == 256
    assert schedule.query_block_size % schedule.query_compute_block_size == 0
    assert (
        select_collapse_block_tokens(device, tokens=133, hc_mult=4, hidden=4096, activation_bytes=2)
        % 8
        == 0
    )
    assert select_gates_block_tokens(device, tokens=133, hc_mult=4) == 256
    assert (
        select_post_block_tokens(
            device, tokens=133, hc_mult=4, hidden=4096, x_bytes=2, residual_bytes=2
        )
        == 64
    )


@pytest.mark.parametrize("device", ("TPU v5 lite", "TPU v5e", "TPU v4", "cpu"))
def test_unsupported_devices_are_not_misidentified_as_v5p(device):
    with pytest.raises(ValueError, match="no schedule"):
        get_hca_kernel_schedule(
            device, page_size=128, max_compressed_entries=128, local_heads=64, head_dim=512
        )
    with pytest.raises(ValueError, match="no schedule"):
        select_gates_block_tokens(device, tokens=128, hc_mult=4)


def test_v5p_post_uses_its_own_vmem_budget():
    # 72 MiB live set: spills on the 64 MiB v5p budget, not the v6e budget.
    kwargs = dict(tokens=1536, hc_mult=4, hidden=4096, activation_bytes=2)
    assert select_post_backend("TPU v5", **kwargs) == "pallas"
    assert select_post_backend("TPU v6 lite", **kwargs) == "xla"


def test_v5p_csa_uses_smaller_scoped_vmem_tiles():
    assert get_csa_attention_schedule(128, shared_window=True, device_kind="TPU v5") == (256, 4)
    assert get_csa_attention_schedule(1, device_kind="TPU v5") == (256, 1)
    assert get_csa_attention_schedule(128, shared_window=True, device_kind="TPU v6 lite") == (
        512,
        8,
    )
    assert get_csa_compressor_projection_k_tile(4096, 128, device_kind="TPU v5") == 512
    assert get_csa_compressor_projection_k_tile(4096, 128, device_kind="TPU v6 lite") == 4096
    assert get_csa_compressor_projection_k_tile(128, 1, device_kind="TPU v5") == 128


@pytest.mark.parametrize("tp_size", (1, 2, 4, 8))
def test_csa_tensor_mesh(tp_size):
    mesh = SimpleNamespace(
        size=tp_size, axis_names=("data", "tensor"), shape={"data": 1, "tensor": tp_size}
    )
    assert validate_csa_mesh(mesh) == tp_size


@pytest.mark.parametrize(
    "shape", ({"data": 4, "tensor": 1}, {"data": 2, "tensor": 2}, {"devices": 4}, {"tensor": 16})
)
def test_csa_rejects_unsupported_meshes(shape):
    size = 1
    for value in shape.values():
        size *= value
    with pytest.raises(ValueError, match="CSA"):
        validate_csa_mesh(SimpleNamespace(size=size, axis_names=tuple(shape), shape=shape))
