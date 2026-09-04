"""Independent NumPy oracle for the reduced DeepSeek-V4 vertical slice."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import ml_dtypes
import numpy as np

from test.srt.kernels.csa import ref as csa_ref
from test.srt.kernels.hca import oracle as hca_ref
from test.srt.kernels.mhc import ref as mhc_ref

SWA_COMPRESSION_RATIO = 0
CSA_TOP_K = 512


@dataclass(frozen=True)
class Config:
    compression_ratios: tuple[int, ...]
    hidden: int
    heads: int
    head_dim: int
    hc_mult: int
    q_rank: int
    o_groups: int
    experts: int
    top_k: int
    window: int
    swiglu_limit: float
    norm_eps: float
    hc_eps: float
    sinkhorn_iters: int


def bf16(x):
    return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def _rms_norm(x, weight, eps):
    x = np.asarray(x, np.float32)
    scale = 1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return bf16(x * scale * np.asarray(weight, np.float32))


def _rope(x, positions, cos, sin, *, inverse=False):
    x = np.asarray(x, np.float32)
    rope_dim = 2 * csa_ref.ROPE_FREQUENCY_DIM
    prefix = x[..., :-rope_dim]
    pairs = x[..., -rope_dim:].reshape(*x.shape[:-1], csa_ref.ROPE_FREQUENCY_DIM, 2)
    table_shape = (
        (len(positions),) + (1,) * (pairs.ndim - 3) + (csa_ref.ROPE_FREQUENCY_DIM,)
    )
    c = np.asarray(cos, np.float32)[positions].reshape(table_shape)
    s = np.asarray(sin, np.float32)[positions].reshape(table_shape)
    if inverse:
        s = -s
    rotated = np.stack(
        (pairs[..., 0] * c - pairs[..., 1] * s, pairs[..., 0] * s + pairs[..., 1] * c),
        axis=-1,
    ).reshape(*x.shape[:-1], rope_dim)
    return bf16(np.concatenate((prefix, rotated), axis=-1))


def _attention_inputs(x, positions, weights, config):
    qr = _rms_norm(
        bf16(x @ weights["q_a"]),
        np.ones((config.q_rank,), np.float32),
        config.norm_eps,
    )
    q = bf16(qr @ weights["q_b"]).reshape(-1, config.heads, config.head_dim)
    q = _rms_norm(q, np.ones((config.head_dim,), np.float32), config.norm_eps)
    q = _rope(q, positions, weights["cos"], weights["sin"])
    kv = _rms_norm(
        bf16(x @ weights["kv"]),
        np.ones((config.head_dim,), np.float32),
        config.norm_eps,
    )
    kv = _rope(kv, positions, weights["cos"], weights["sin"])
    index_q = bf16(qr @ weights["index_q"]).reshape(-1, config.heads, csa_ref.INDEX_DIM)
    index_q = _rope(index_q, positions, weights["cos"], weights["sin"])
    index_weights = np.asarray(x, np.float32) @ weights["index_weight"]
    index_weights *= csa_ref.INDEX_DIM**-0.5 * config.heads**-0.5
    return q, kv, index_q, index_weights


def _attention_output(x, positions, weights, config):
    x = _rope(x, positions, weights["cos"], weights["sin"], inverse=True)
    grouped = x.reshape(
        x.shape[0],
        config.o_groups,
        config.heads * config.head_dim // config.o_groups,
    )
    low_rank = bf16(np.einsum("tgd,grd->tgr", grouped, weights["o_a"]))
    return bf16(low_rank.reshape(x.shape[0], -1) @ weights["o_b"])


def _dense_attention(q, values, sink, scale):
    scores = np.einsum("hd,kd->hk", q, values) * np.float32(scale)
    maximum = np.maximum(np.max(scores, axis=-1), sink)
    probability = np.exp(scores - maximum[:, None])
    denominator = np.sum(probability, axis=-1) + np.exp(sink - maximum)
    return np.einsum("hk,kd->hd", probability, values) / denominator[:, None]


def _swa_attention(q, kv, sink, window):
    output = []
    for position in range(q.shape[0]):
        values = kv[max(0, position - window + 1) : position + 1]
        output.append(_dense_attention(q[position], values, sink, q.shape[-1] ** -0.5))
    return bf16(np.stack(output))


def _hca_attention(hidden, q, kv, weights, config):
    stream = {
        "hidden": hidden[None],
        "q": q[None],
        "kv": kv[None],
    }
    hca_weights = {
        "wkv": weights["hca_kv"],
        "wgate": weights["hca_gate"],
        "ape": weights["hca_ape"],
        "norm": np.ones((config.head_dim,), np.float32),
        "cos": weights["cos"],
        "sin": weights["sin"],
        "sink": np.zeros((config.heads,), np.float32),
    }
    output = hca_ref.request_outputs(
        stream,
        0,
        np.arange(q.shape[0], dtype=np.int32),
        hca_weights,
        config.head_dim**-0.5,
    )
    return bf16(output)


def _initial_csa_state(width):
    state = np.zeros((1, csa_ref.STATE_SLOTS, 2, width), np.float32)
    state[:, :, 1] = -np.inf
    return state


def _csa_records(hidden, weights, config):
    positions = np.arange(hidden.shape[0], dtype=np.int32)
    main_width = 2 * config.head_dim
    main_positions, main_values, _ = csa_ref.compressor_ragged(
        hidden,
        _initial_csa_state(main_width),
        weights["csa_dual"][:, : 2 * main_width],
        weights["csa_main_ape"],
        np.ones((config.head_dim,), np.float32),
        weights["cos"],
        weights["sin"],
        positions,
        (hidden.shape[0],),
    )
    index_width = 2 * csa_ref.INDEX_DIM
    index_positions, index_values, _ = csa_ref.compressor_ragged(
        hidden,
        _initial_csa_state(index_width),
        weights["csa_dual"][:, 2 * main_width :],
        weights["csa_index_ape"],
        np.ones((csa_ref.INDEX_DIM,), np.float32),
        weights["cos"],
        weights["sin"],
        positions,
        (hidden.shape[0],),
    )
    np.testing.assert_array_equal(main_positions[0], index_positions[0])
    if not len(main_values[0]):
        return (
            np.zeros((0, config.head_dim), np.float32),
            np.zeros((0, csa_ref.INDEX_DIM), np.float32),
        )
    main_nope, main_rope = csa_ref.pack_main(main_values[0])
    main = csa_ref.decode_main(main_nope, main_rope)
    index = csa_ref.decode_index(csa_ref.pack_index(index_values[0]))
    return main, index


def _csa_attention(hidden, q, kv, index_q, index_weights, weights, config):
    main_records, index_records = _csa_records(hidden, weights, config)
    positions = np.arange(q.shape[0], dtype=np.int32)
    if main_records.shape[0] > CSA_TOP_K:
        topk = csa_ref.lightning_topk(
            index_q,
            index_records[None],
            index_weights,
            positions,
            np.zeros((q.shape[0],), np.int32),
            np.asarray([main_records.shape[0]], np.int32),
            selected=CSA_TOP_K,
        )
    else:
        topk = None
    sink = np.zeros((config.heads,), np.float32)
    output = []
    for position in positions:
        available = min(
            main_records.shape[0], (int(position) + 1) // csa_ref.COMPRESSION_RATIO
        )
        if topk is None:
            selected = main_records[:available]
        else:
            selected_indices = topk[position]
            selected_indices = selected_indices[selected_indices >= 0]
            selected = main_records[selected_indices]
        window = kv[max(0, int(position) - config.window + 1) : int(position) + 1]
        values = np.concatenate((window, selected), axis=0)
        output.append(
            _dense_attention(q[position], values, sink, config.head_dim**-0.5)
        )
    return bf16(np.stack(output))


def _moe(x, token_ids, weights, config, layer_id):
    scores = np.sqrt(np.logaddexp(0.0, np.asarray(x, np.float32) @ weights["route"]))
    if layer_id < 3:
        first = np.mod(token_ids, config.experts)
        second = np.mod(
            first + 1 + np.mod(token_ids, config.experts - 1), config.experts
        )
        selected = np.stack((first, second), axis=-1)
    else:
        selected = np.argsort(
            -(scores + weights["route_bias"]), axis=-1, kind="stable"
        )[:, : config.top_k]
    selected_scores = np.take_along_axis(scores, selected, axis=-1)
    selected_scores /= np.sum(selected_scores, axis=-1, keepdims=True)
    dispatch = np.zeros_like(scores)
    np.put_along_axis(dispatch, selected, selected_scores, axis=-1)

    gate = bf16(np.einsum("td,edi->tei", x, weights["expert_gate"]))
    up = bf16(np.einsum("td,edi->tei", x, weights["expert_up"]))
    gate = np.minimum(gate, config.swiglu_limit)
    up = np.clip(up, -config.swiglu_limit, config.swiglu_limit)
    expert_hidden = gate / (1.0 + np.exp(-gate)) * up
    expert_output = bf16(
        np.einsum("tei,eid->ted", bf16(expert_hidden), weights["expert_down"])
    )
    routed = np.sum(dispatch[..., None] * expert_output, axis=1)

    shared_gate = bf16(x @ weights["shared_gate"])
    shared_up = bf16(x @ weights["shared_up"])
    shared_gate = np.minimum(shared_gate, config.swiglu_limit)
    shared_up = np.clip(shared_up, -config.swiglu_limit, config.swiglu_limit)
    shared_hidden = shared_gate / (1.0 + np.exp(-shared_gate)) * shared_up
    shared = bf16(bf16(shared_hidden) @ weights["shared_down"])
    return bf16(routed + shared)


def _attention(x, positions, ratio, weights, config):
    q, kv, index_q, index_weights = _attention_inputs(x, positions, weights, config)
    if ratio == SWA_COMPRESSION_RATIO:
        output = _swa_attention(
            q, kv, np.zeros((config.heads,), np.float32), config.window
        )
    elif ratio == csa_ref.COMPRESSION_RATIO:
        output = _csa_attention(x, q, kv, index_q, index_weights, weights, config)
    elif ratio == hca_ref.RATIO:
        output = _hca_attention(x, q, kv, weights, config)
    else:
        raise ValueError(f"unsupported compression ratio {ratio}")
    return _attention_output(output, positions, weights, config)


def _inject_local_error(value, local_errors, label, *, bf16_output=True):
    if not local_errors or label not in local_errors:
        return value
    value = np.asarray(value, np.float32) + np.asarray(local_errors[label], np.float32)
    return bf16(value) if bf16_output else value


def _sublayer(
    streams,
    gate_weights,
    operation,
    weights,
    config,
    trace,
    label,
    local_errors,
):
    x, post, comb = mhc_ref.pre(
        streams,
        weights[f"hc_{gate_weights}_fn"],
        weights[f"hc_{gate_weights}_scale"],
        weights[f"hc_{gate_weights}_base"],
        hc_mult=config.hc_mult,
        sinkhorn_iters=config.sinkhorn_iters,
        norm_eps=config.norm_eps,
        hc_eps=config.hc_eps,
    )
    x = _inject_local_error(bf16(x), local_errors, f"{label}.pre")
    normalized = _rms_norm(x, np.ones((config.hidden,), np.float32), config.norm_eps)
    normalized = _inject_local_error(normalized, local_errors, f"{label}.norm")
    block_output = _inject_local_error(
        operation(normalized), local_errors, f"{label}.operator"
    )
    output = _inject_local_error(
        bf16(mhc_ref.post(block_output, streams, post, comb)),
        local_errors,
        f"{label}.post",
    )
    trace[f"{label}.pre"] = x
    trace[f"{label}.norm"] = normalized
    trace[f"{label}.operator"] = block_output
    trace[f"{label}.post"] = output
    return output


def _prepare_weights(weights):
    return {name: np.asarray(value, np.float32) for name, value in weights.items()}


def run(
    weights: Mapping[str, np.ndarray],
    token_ids,
    config: Config,
    *,
    local_errors: Mapping[str, np.ndarray] | None = None,
):
    """Evaluate one zero-prefix causal request and return logits plus layer traces."""
    weights = _prepare_weights(weights)
    token_ids = np.asarray(token_ids, np.int32)
    positions = np.arange(token_ids.shape[0], dtype=np.int32)
    streams = np.repeat(
        bf16(weights["embedding"])[token_ids, None], config.hc_mult, axis=1
    )
    trace = {}

    for layer_id, ratio in enumerate(config.compression_ratios):
        streams = _sublayer(
            streams,
            "attn",
            lambda x, ratio=ratio: _attention(x, positions, ratio, weights, config),
            weights,
            config,
            trace,
            f"layer{layer_id}.attention",
            local_errors,
        )
        streams = _sublayer(
            streams,
            "ffn",
            lambda x, layer_id=layer_id: _moe(x, token_ids, weights, config, layer_id),
            weights,
            config,
            trace,
            f"layer{layer_id}.ffn",
            local_errors,
        )

    hidden = bf16(
        mhc_ref.head_collapse(
            streams,
            weights["hc_head_fn"],
            weights["hc_head_scale"],
            weights["hc_head_base"],
            norm_eps=config.norm_eps,
            hc_eps=config.hc_eps,
        )
    )
    hidden = _inject_local_error(hidden, local_errors, "head.collapse")
    trace["head.collapse"] = hidden
    hidden = _rms_norm(hidden, np.ones((config.hidden,), np.float32), config.norm_eps)
    hidden = _inject_local_error(hidden, local_errors, "head.norm")
    trace["head.norm"] = hidden
    logits = np.asarray(hidden, np.float32) @ weights["lm_head"]
    logits = _inject_local_error(logits, local_errors, "logits", bf16_output=False)
    trace["logits"] = logits
    return logits, trace


def local_references(
    weights: Mapping[str, np.ndarray],
    token_ids,
    config: Config,
    actual: Mapping[str, np.ndarray],
):
    """Re-evaluate each boundary using the TPU trace as that operation's input."""
    weights = _prepare_weights(weights)
    token_ids = np.asarray(token_ids, np.int32)
    positions = np.arange(token_ids.shape[0], dtype=np.int32)
    streams = np.repeat(
        bf16(weights["embedding"])[token_ids, None], config.hc_mult, axis=1
    )
    reference = {}

    for layer_id, ratio in enumerate(config.compression_ratios):
        for kind in ("attention", "ffn"):
            label = f"layer{layer_id}.{kind}"
            gate_weights = "attn" if kind == "attention" else "ffn"
            pre, post, comb = mhc_ref.pre(
                streams,
                weights[f"hc_{gate_weights}_fn"],
                weights[f"hc_{gate_weights}_scale"],
                weights[f"hc_{gate_weights}_base"],
                hc_mult=config.hc_mult,
                sinkhorn_iters=config.sinkhorn_iters,
                norm_eps=config.norm_eps,
                hc_eps=config.hc_eps,
            )
            reference[f"{label}.pre"] = bf16(pre)
            reference[f"{label}.norm"] = _rms_norm(
                actual[f"{label}.pre"],
                np.ones((config.hidden,), np.float32),
                config.norm_eps,
            )
            if kind == "attention":
                operation = _attention(
                    actual[f"{label}.norm"], positions, ratio, weights, config
                )
            else:
                operation = _moe(
                    actual[f"{label}.norm"], token_ids, weights, config, layer_id
                )
            reference[f"{label}.operator"] = operation
            reference[f"{label}.post"] = bf16(
                mhc_ref.post(actual[f"{label}.operator"], streams, post, comb)
            )
            streams = np.asarray(actual[f"{label}.post"], np.float32)

    reference["head.collapse"] = bf16(
        mhc_ref.head_collapse(
            streams,
            weights["hc_head_fn"],
            weights["hc_head_scale"],
            weights["hc_head_base"],
            norm_eps=config.norm_eps,
            hc_eps=config.hc_eps,
        )
    )
    reference["head.norm"] = _rms_norm(
        actual["head.collapse"],
        np.ones((config.hidden,), np.float32),
        config.norm_eps,
    )
    reference["logits"] = (
        np.asarray(actual["head.norm"], np.float32) @ weights["lm_head"]
    )
    return reference


__all__ = ["Config", "local_references", "run"]
