"""Run the pinned official PyTorch layer code with independent CPU kernels.

Only GPU-specific kernel entry points and Hadamard CUDA are replaced. Official
Attention/Compressor/Indexer/Gate/Expert/mHC Python control flow is unmodified.
This module intentionally does not import any production JAX low-bit operators.
"""

import dataclasses
import importlib.util
import sys
import types

import numpy as np
import torch


def _numpy(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy()
    return tensor.numpy()


def _scales(tensor):
    raw = tensor.view(torch.uint8).numpy()
    return torch.from_numpy(np.exp2(raw.astype(np.float64) - 127).astype(np.float32))


def _act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, simulate=False):
    grouped = x.float().reshape(*x.shape[:-1], -1, block_size)
    amax = torch.clamp(grouped.abs().amax(-1, keepdim=True), min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax.double() / 448))).float()
    quantized = torch.clamp(grouped / scale, -448, 448).to(torch.float8_e4m3fn)
    if simulate:
        x.copy_((quantized.float() * scale).reshape(x.shape))
        return x
    scale_bytes = (torch.log2(scale).to(torch.int32) + 127).to(torch.uint8)
    return quantized.reshape(x.shape), scale_bytes.squeeze(-1).view(torch.float8_e8m0fnu)


def _fp4_act_quant(x, block_size=32, simulate=False):
    grouped = x.float().numpy().reshape(*x.shape[:-1], -1, block_size)
    maximum = np.maximum(np.max(np.abs(grouped), axis=-1, keepdims=True), np.float32(6 * 2.0**-126))
    scale = np.exp2(np.ceil(np.log2(maximum.astype(np.float64) / 6))).astype(np.float32)
    values = grouped / scale
    lut = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    priority = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    distances = np.abs(np.abs(values[..., None]) - lut)
    codes = priority[np.argmin(distances[..., priority], axis=-1)]
    decoded = lut[codes] * np.where(np.signbit(values), -1, 1) * scale
    if not simulate:
        raise NotImplementedError("CPU oracle only needs the official indexer QAT path")
    x.copy_(torch.from_numpy(decoded.reshape(x.shape).astype(np.float32)))
    return x


def _weight(weight, scales, fp4):
    raw = weight.view(torch.uint8).numpy()
    if fp4:
        lut = np.array(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
        )
        value = np.empty((raw.shape[0], 2 * raw.shape[1]), np.float32)
        value[:, 0::2], value[:, 1::2] = lut[raw & 15], lut[raw >> 4]
        return torch.from_numpy(value) * _scales(scales).repeat_interleave(32, dim=-1)
    return (
        weight.float()
        * _scales(scales)
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)[: weight.shape[0], : weight.shape[1]]
    )


def _gemm(x, xs, weight, ws, scale_dtype=None, *, fp4):
    value = x.float() * _scales(xs).repeat_interleave(128, dim=-1)
    return (value @ _weight(weight, ws, fp4).T).to(torch.bfloat16)


def _sparse_attn(q, kv, sink, indices, scale):
    batch, length, heads, dim = q.shape
    output = torch.empty_like(q)
    # Each query gets the same 64-key block boundaries as TileLang, including
    # -1 padding. Probabilities round to BF16 before the value matmul.
    for b in range(batch):
        maximum = torch.full((length, heads), -float("inf"), dtype=torch.float32)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((length, heads, dim), dtype=torch.float32)
        for begin in range(0, indices.shape[-1], 64):
            ids = indices[b, :, begin : begin + 64].long()
            keys = kv[b, ids.clamp(min=0)]
            score = torch.einsum("mhd,mkd->mhk", q[b].float(), keys.float()) * scale
            valid = ids[:, None, :] >= 0
            score = torch.where(valid, score, -1e30)
            new_max = torch.maximum(maximum, score.amax(-1))
            alpha = torch.exp(maximum - new_max)
            probability = torch.where(valid, torch.exp(score - new_max[..., None]), 0.0)
            numerator = numerator * alpha[..., None] + torch.einsum(
                "mhk,mkd->mhd", probability.to(torch.bfloat16).float(), keys.float()
            )
            denominator = denominator * alpha + probability.sum(-1)
            maximum = new_max
        final_max = torch.maximum(maximum, sink[None, :])
        alpha = torch.exp(maximum - final_max)
        denominator = denominator * alpha + torch.exp(sink[None, :] - final_max)
        output[b] = numerator * alpha[..., None] / denominator[..., None]
    return output


def _sinkhorn(mixes, scale, base, hc, iterations, eps):
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).reshape(*mixes.shape[:-1], hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb /= comb.sum(-2, keepdim=True) + eps
    for _ in range(iterations - 1):
        comb /= comb.sum(-1, keepdim=True) + eps
        comb /= comb.sum(-2, keepdim=True) + eps
    return pre, post, comb


def _hadamard(x):
    # Dense independently constructed Sylvester matrix, not JAX's butterfly.
    matrix = np.ones((1, 1), np.float32)
    while matrix.shape[0] < x.shape[-1]:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return (x.float() @ torch.from_numpy(matrix) * x.shape[-1] ** -0.5).to(torch.bfloat16)


def official_module(checkpoint, max_context=256):
    from functools import partial

    shim = types.ModuleType("kernel")
    shim.act_quant, shim.fp4_act_quant = _act_quant, _fp4_act_quant
    shim.fp4_gemm, shim.fp8_gemm = partial(_gemm, fp4=True), partial(_gemm, fp4=False)
    shim.sparse_attn, shim.hc_split_sinkhorn = _sparse_attn, _sinkhorn
    previous = sys.modules.get("kernel")
    sys.modules["kernel"] = shim
    try:
        spec = importlib.util.spec_from_file_location(
            "deepseek_v4_official_cpu", checkpoint.directory / "inference/model.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["kernel"]
        else:
            sys.modules["kernel"] = previous
    module.rotate_activation = _hadamard
    module.default_dtype = torch.float8_e4m3fn
    module.scale_fmt, module.scale_dtype = "ue8m0", torch.float8_e8m0fnu
    c = checkpoint.config
    mapping = {
        "dim": "hidden_size",
        "moe_inter_dim": "moe_intermediate_size",
        "n_layers": "num_hidden_layers",
        "n_hash_layers": "num_hash_layers",
        "n_mtp_layers": "num_nextn_predict_layers",
        "n_heads": "num_attention_heads",
        "n_activated_experts": "num_experts_per_tok",
        "route_scale": "routed_scaling_factor",
        "rope_head_dim": "qk_rope_head_dim",
        "norm_eps": "rms_norm_eps",
        "window_size": "sliding_window",
        "score_func": "scoring_func",
    }
    args = {
        field.name: c[mapping.get(field.name, field.name)]
        for field in dataclasses.fields(module.ModelArgs)
        if mapping.get(field.name, field.name) in c
    }
    yarn = c["rope_scaling"]
    args.update(
        max_batch_size=1,
        max_seq_len=max_context,
        dtype="fp8",
        scale_dtype="fp8",
        scale_fmt="ue8m0",
        original_seq_len=yarn["original_max_position_embeddings"],
        rope_factor=yarn["factor"],
        beta_fast=yarn["beta_fast"],
        beta_slow=yarn["beta_slow"],
    )
    return module, module.ModelArgs(**args)


def tensor_from_checkpoint(checkpoint, name):
    info = checkpoint.tensor_info(name)
    raw = checkpoint.read_tensor(name)
    if info["dtype"] == "BF16":
        return torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16)
    if info["dtype"] in ("F8_E4M3", "F8_E8M0", "I8"):
        dtype = {
            "F8_E4M3": torch.float8_e4m3fn,
            "F8_E8M0": torch.float8_e8m0fnu,
            "I8": torch.float4_e2m1fn_x2,
        }[info["dtype"]]
        return torch.from_numpy(raw.view(np.uint8)).view(dtype)
    return torch.from_numpy(raw)


def load_module_weights(module, checkpoint, prefix):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            tensor = tensor_from_checkpoint(checkpoint, prefix + name)
            if name.endswith("wo_a.weight") and parameter.dtype == torch.bfloat16:
                tensor = _weight(
                    tensor,
                    tensor_from_checkpoint(
                        checkpoint, prefix + name.removesuffix("weight") + "scale"
                    ),
                    False,
                )
            if parameter.dtype == torch.float4_e2m1fn_x2:
                parameter.view(torch.uint8).copy_(tensor.view(torch.uint8))
            else:
                parameter.copy_(tensor)
    return module
