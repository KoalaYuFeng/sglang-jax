"""V4 checkpoint-name-aware, byte-preserving projection packing."""

MERGED_PROJECTIONS = {
    "attn.wqkv_a": ("attn.wq_a", "attn.wkv"),
    "ffn.shared_experts.gate_up": ("ffn.shared_experts.w1", "ffn.shared_experts.w3"),
}


def pack_merged_weights(weights):
    """Host-only, byte-preserving packing; replace (not duplicate) originals.

    The V4 components all end on 128-output-channel scale boundaries. Reject
    other layouts rather than silently assigning the wrong E8M0 block scale.
    """
    import numpy as np

    result = dict(weights)
    for target, sources in MERGED_PROJECTIONS.items():
        raw = [result[name + ".weight"] for name in sources]
        scales = [result[name + ".scale"] for name in sources]
        if any(not isinstance(value, np.ndarray) for value in (*raw, *scales)):
            raise TypeError(
                "merged packing must run on host checkpoint arrays at load time"
            )
        k = raw[0].shape[-1]
        if target == "ffn.shared_experts.gate_up" and raw[0].shape != raw[1].shape:
            raise ValueError("V4 shared expert gate/up must have identical shapes")
        for w, s in zip(raw, scales, strict=True):
            if (
                w.ndim != 2
                or w.dtype != np.uint8
                or w.shape[1] != k
                or w.shape[0] % 128
            ):
                raise ValueError(
                    "merged FP8 components require raw bytes and aligned N/equal K"
                )
            if (
                k % 128
                or s.dtype != np.uint8
                or s.shape != (w.shape[0] // 128, k // 128)
            ):
                raise ValueError("merged components require compact E8M0 block scales")
        for suffix, arrays in ((".weight", raw), (".scale", scales)):
            result[target + suffix] = np.concatenate(arrays, axis=0)
            for source in sources:
                del result[source + suffix]
    return result


def unpack_merged_weights(weights):
    """Read-only raw-byte views for diagnostics using the unchanged oracle.

    Never used in the production decode path. No dequantization/requantization.
    """
    result = dict(weights)
    for target, sources in MERGED_PROJECTIONS.items():
        if target + ".weight" not in result:
            continue
        split = (
            result["attn.q_norm.weight"].shape[0]
            if target == "attn.wqkv_a"
            else result[target + ".weight"].shape[0] // 2
        )
        for suffix, boundary in ((".weight", split), (".scale", split // 128)):
            value = result.pop(target + suffix)
            result[sources[0] + suffix], result[sources[1] + suffix] = (
                value[:boundary],
                value[boundary:],
            )
    return result
