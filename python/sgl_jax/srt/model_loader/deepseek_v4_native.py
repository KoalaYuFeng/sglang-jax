"""Bounded checkpoint-native staging for the production V4 model."""

import gc

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


HEAD_SHARDED_WEIGHTS = frozenset(
    {
        "attn.wq_b.weight",
        "attn.wq_b.scale",
        "attn.wo_a.weight",
        "attn.wo_a.scale",
        "attn.attn_sink",
    }
)


def _weight_spec(key, ndim, attention_tp):
    if key.startswith("experts."):
        return P("tensor", None, None)
    if attention_tp and key in HEAD_SHARDED_WEIGHTS:
        return P("tensor", *([None] * (ndim - 1)))
    return P()


def weight_specs(weights, *, attention_tp=False):
    return {key: _weight_spec(key, value.ndim, attention_tp) for key, value in weights.items()}


def load_layer(
    checkpoint,
    layer_id,
    mesh,
    *,
    include_experts=True,
    attention_tp=False,
    merged_projections=False,
    fp4_scale_transposed=False,
):
    """Bounded host staging; never allocate a full BF16 layer/expert collection."""
    if type(fp4_scale_transposed) is not bool:
        raise ValueError("fp4_scale_transposed must be a boolean")
    prefix = f"layers.{layer_id}."
    weights = {}
    pending = {}
    from sgl_jax.srt.kernels.deepseek_v4.projections import MERGED_PROJECTIONS, pack_merged_weights

    merged_keys = (
        {
            source + suffix
            for sources in MERGED_PROJECTIONS.values()
            for source in sources
            for suffix in (".weight", ".scale")
        }
        if merged_projections
        else set()
    )
    for name in checkpoint.weight_map:
        if not name.startswith(prefix) or ".ffn.experts." in name:
            continue
        key = name[len(prefix) :]
        info = checkpoint.tensor_info(name)
        value = checkpoint.read_tensor(name)
        if key in merged_keys:
            pending[key] = value
            continue
        if info["dtype"] == "I64":
            if (
                not key.endswith("tid2eid")
                or np.any(value < 0)
                or np.any(value >= checkpoint.config["n_routed_experts"])
            ):
                raise ValueError(f"unexpected/out-of-range integer checkpoint field: {name}")
            value = value.astype(np.int32)  # Lossless routing-index conversion, not weights.
        weights[key] = jax.device_put(
            value, NamedSharding(mesh, _weight_spec(key, value.ndim, attention_tp))
        )
    if merged_projections:
        for key, value in pack_merged_weights(pending).items():
            weights[key] = jax.device_put(value, NamedSharding(mesh, P()))
        del pending
    if include_experts:
        total = checkpoint.config["n_routed_experts"]
        devices = list(mesh.devices.flat)
        if total % len(devices):
            raise ValueError("expert count must divide the device mesh")
        local_count = total // len(devices)
        for projection in (1, 3, 2):
            arrays, scale_arrays = [], []
            for shard, device in enumerate(devices):
                experts = [
                    checkpoint.load_linear(f"layers.{layer_id}.ffn.experts.{expert}.w{projection}")
                    for expert in range(shard * local_count, (shard + 1) * local_count)
                ]
                data = np.stack([expert.data for expert in experts])
                scales = np.stack([expert.scales for expert in experts])
                if fp4_scale_transposed:
                    scales = np.ascontiguousarray(scales.swapaxes(1, 2))
                arrays.append(jax.device_put(data, device).block_until_ready())
                scale_arrays.append(jax.device_put(scales, device).block_until_ready())
                del experts, data, scales
            sharding = NamedSharding(mesh, P("tensor", None, None))
            for key, parts in ((f"w{projection}", arrays), (f"s{projection}", scale_arrays)):
                shape = (total, *parts[0].shape[1:])
                weights["experts." + key] = jax.make_array_from_single_device_arrays(
                    shape, sharding, parts
                )
    jax.block_until_ready(weights)
    gc.collect()
    return weights


def original_fp4_scale_view(weights, *, transposed=False):
    """Read-only lossless view for numerical diagnostics, outside timed inference.

    Expert ownership and raw uint8 storage are retained; no BF16 weights are
    materialized. The caller already supplies the model's explicit layout flag.
    """
    if type(transposed) is not bool:
        raise ValueError("transposed must be a boolean")
    result = dict(weights)
    if transposed:
        for projection in (1, 3, 2):
            key = f"experts.s{projection}"
            result[key] = result[key].swapaxes(1, 2)
    return result
