"""Measure how local implementation errors propagate through the test-only V4 slice."""

from __future__ import annotations

import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as np
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

from test.srt.model_executor import deepseek_v4_oracle
from test.srt.model_executor import test_deepseek_v4_vertical_slice as vertical


def _xla_sinkhorn(mixes, scale, base, *, hc, iterations, eps):
    pre = jax.nn.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * jax.nn.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[:, 2 * hc :].reshape(-1, hc, hc)
    comb = comb * scale[2] + base[2 * hc :].reshape(1, hc, hc)
    comb = jax.nn.softmax(comb, axis=-1) + eps
    comb /= jnp.sum(comb, axis=-2, keepdims=True) + eps
    for _ in range(iterations - 1):
        comb /= jnp.sum(comb, axis=-1, keepdims=True) + eps
        comb /= jnp.sum(comb, axis=-2, keepdims=True) + eps
    return pre, post, comb


@functools.partial(
    jax.jit,
    static_argnames=(
        "hc_mult",
        "sinkhorn_iters",
        "norm_eps",
        "hc_eps",
        "block_tokens",
        "interpret",
        "dot_precision",
    ),
)
def _xla_pre(
    streams,
    weight,
    scale,
    base,
    *,
    hc_mult,
    sinkhorn_iters,
    norm_eps,
    hc_eps,
    block_tokens=None,
    interpret=None,
    dot_precision=jax.lax.Precision.DEFAULT,
):
    del block_tokens, interpret
    dtype = streams.dtype
    flat = streams.reshape(streams.shape[0], -1).astype(jnp.float32)
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + norm_eps)
    mixes = jax.lax.dot_general(
        flat,
        weight.astype(jnp.float32),
        (((1,), (1,)), ((), ())),
        precision=dot_precision,
        preferred_element_type=jnp.float32,
    )
    mixes *= rms
    pre, post, comb = _xla_sinkhorn(
        mixes,
        scale.astype(jnp.float32),
        base.astype(jnp.float32),
        hc=hc_mult,
        iterations=sinkhorn_iters,
        eps=hc_eps,
    )
    collapsed = jnp.sum(pre[..., None] * streams.astype(jnp.float32), axis=-2).astype(
        dtype
    )
    return collapsed, post, comb


@functools.partial(
    jax.jit,
    static_argnames=("block_tokens", "backend", "interpret", "precision"),
)
def _xla_post(
    x,
    residual,
    post,
    comb,
    *,
    block_tokens=None,
    backend="auto",
    interpret=None,
    precision=jax.lax.Precision.DEFAULT,
):
    del block_tokens, backend, interpret
    mixed = jnp.einsum(
        "tij,tid->tjd",
        comb.astype(jnp.float32),
        residual.astype(jnp.float32),
        precision=precision,
    )
    return (
        post.astype(jnp.float32)[..., None] * x.astype(jnp.float32)[:, None] + mixed
    ).astype(x.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "hc_mult",
        "norm_eps",
        "hc_eps",
        "block_tokens",
        "interpret",
        "dot_precision",
    ),
)
def _xla_head(
    streams,
    weight,
    scale,
    base,
    *,
    hc_mult,
    norm_eps,
    hc_eps,
    block_tokens=None,
    interpret=None,
    dot_precision=jax.lax.Precision.DEFAULT,
):
    del hc_mult, block_tokens, interpret
    dtype = streams.dtype
    flat = streams.reshape(streams.shape[0], -1).astype(jnp.float32)
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(flat), axis=-1, keepdims=True) + norm_eps)
    normalized = (flat * rms).astype(jnp.bfloat16)
    mixes = jax.lax.dot_general(
        normalized,
        weight.astype(jnp.float32),
        (((1,), (1,)), ((), ())),
        precision=dot_precision,
        preferred_element_type=jnp.float32,
    )
    pre = jax.nn.sigmoid(mixes * scale.astype(jnp.float32)[0] + base) + hc_eps
    return jnp.sum(pre[..., None] * streams.astype(jnp.float32), axis=-2).astype(dtype)


def _nrmse(actual, expected):
    difference = np.asarray(actual, np.float32) - np.asarray(expected, np.float32)
    return float(
        np.sqrt(np.mean(np.square(difference)))
        / np.sqrt(np.mean(np.square(expected, dtype=np.float32)))
    )


def _group_nrmse(errors, reference, names):
    squared_error = sum(
        float(np.sum(np.square(errors[name], dtype=np.float64))) for name in names
    )
    squared_reference = sum(
        float(np.sum(np.square(reference[name], dtype=np.float64))) for name in names
    )
    return float(np.sqrt(squared_error / squared_reference))


def _topk_overlap(actual, expected, k=5):
    actual_topk = np.argpartition(actual, -k, axis=-1)[:, -k:]
    expected_topk = np.argpartition(expected, -k, axis=-1)[:, -k:]
    overlap = [
        len(set(actual_row) & set(expected_row)) / k
        for actual_row, expected_row in zip(actual_topk, expected_topk, strict=True)
    ]
    return float(np.mean(overlap))


def _kl(actual, expected):
    def probability(logits):
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exponential = np.exp(shifted)
        return exponential / np.sum(exponential, axis=-1, keepdims=True)

    actual_probability = probability(np.asarray(actual, np.float64))
    expected_probability = probability(np.asarray(expected, np.float64))
    return float(
        np.mean(
            np.sum(
                expected_probability
                * np.log(expected_probability / actual_probability),
                axis=-1,
            )
        )
    )


def _metrics(actual, expected):
    return {
        "logits_nrmse": _nrmse(actual, expected),
        "top1_flips": int(
            np.sum(np.argmax(actual, axis=-1) != np.argmax(expected, axis=-1))
        ),
        "top5_overlap": _topk_overlap(actual, expected),
        "kl": _kl(actual, expected),
    }


def _execute_model(config, weights, max_context, mesh, token_ids, *, xla_mhc):
    original = (
        vertical.mhc_pre_fused,
        vertical.mhc_post_fused,
        vertical.mhc_head_collapse_fused,
    )
    if xla_mhc:
        (
            vertical.mhc_pre_fused,
            vertical.mhc_post_fused,
            vertical.mhc_head_collapse_fused,
        ) = (_xla_pre, _xla_post, _xla_head)
    try:
        with jax.set_mesh(mesh):
            model = vertical._model(config, weights, 1, max_context, mesh)
            logits, trace = model.step(
                token_ids,
                (len(token_ids),),
                (0,),
                ForwardMode.EXTEND,
                return_trace=True,
            )
            jax.block_until_ready((logits, trace))
    finally:
        (
            vertical.mhc_pre_fused,
            vertical.mhc_post_fused,
            vertical.mhc_head_collapse_fused,
        ) = original
    return np.asarray(logits, np.float32), {
        name: np.asarray(value, np.float32) for name, value in trace.items()
    }


def _run_seed(seed):
    config = vertical._Config()
    max_context = 256
    mesh = vertical._mesh()
    weights = vertical._make_weights(config, max_context, seed=seed)
    token_rng = np.random.default_rng(seed ^ 0x5A17)
    token_ids = token_rng.integers(0, config.vocab, 133, dtype=np.int32)

    actual_logits, actual_trace = _execute_model(
        config, weights, max_context, mesh, token_ids, xla_mhc=False
    )
    xla_logits, xla_trace = _execute_model(
        config, weights, max_context, mesh, token_ids, xla_mhc=True
    )
    reference_weights, reference_config = vertical._reference_inputs(config, weights)
    reference_logits, _ = deepseek_v4_oracle.run(
        reference_weights, token_ids, reference_config
    )
    local_reference = deepseek_v4_oracle.local_references(
        reference_weights, token_ids, reference_config, actual_trace
    )
    local_errors = {
        name: actual_trace[name] - local_reference[name] for name in actual_trace
    }
    xla_local_reference = deepseek_v4_oracle.local_references(
        reference_weights, token_ids, reference_config, xla_trace
    )
    xla_local_errors = {
        name: xla_trace[name] - xla_local_reference[name] for name in xla_trace
    }

    mhc_pre = {name for name in local_errors if name.endswith(".pre")}
    mhc_post = {name for name in local_errors if name.endswith(".post")}
    mhc_head = {"head.collapse"}
    csa = {
        f"layer{layer}.attention.operator"
        for layer in range(len(config.compression_ratios))
        if config.compression_ratios[layer] == vertical.CSA_COMPRESSION_RATIO
    }
    hca = {
        f"layer{layer}.attention.operator"
        for layer in range(len(config.compression_ratios))
        if config.compression_ratios[layer] == vertical._HCA_COMPRESSION_RATIO
    }
    target = mhc_pre | mhc_post | mhc_head | csa | hca
    mhc = mhc_pre | mhc_post | mhc_head
    swa = {
        f"layer{layer}.attention.operator"
        for layer in range(len(config.compression_ratios))
        if config.compression_ratios[layer] == vertical._SWA_COMPRESSION_RATIO
    }
    moe = {name for name in local_errors if name.endswith(".ffn.operator")}
    normalization = {
        name for name in local_errors if name.endswith(".norm") or name == "head.norm"
    }
    all_stages = set(local_errors) - {"logits"}
    groups = {
        "mHC pre": mhc_pre,
        "mHC post": mhc_post,
        "mHC head": mhc_head,
        "mHC total": mhc,
        "CSA attention": csa,
        "HCA attention": hca,
        "target total": target,
        "SWA attention": swa,
        "MoE": moe,
        "normalization": normalization,
        "non-target total": all_stages - target,
        "all local errors": all_stages,
    }

    results = {}
    for label, names in groups.items():
        injected_logits, _ = deepseek_v4_oracle.run(
            reference_weights,
            token_ids,
            reference_config,
            local_errors={name: local_errors[name] for name in names},
        )
        results[label] = {
            "source_nrmse": _group_nrmse(local_errors, local_reference, names),
            **_metrics(injected_logits, reference_logits),
        }
        source_nrmse = results[label]["source_nrmse"]
        results[label]["gain"] = (
            results[label]["logits_nrmse"] / source_nrmse if source_nrmse else 0.0
        )

    reconstructed_logits, reconstructed_trace = deepseek_v4_oracle.run(
        reference_weights,
        token_ids,
        reference_config,
        local_errors=local_errors,
    )
    reconstruction = {
        "logits_nrmse": _nrmse(reconstructed_logits, actual_logits),
        "max_trace_nrmse": max(
            _nrmse(reconstructed_trace[name], actual_trace[name])
            for name in actual_trace
        ),
    }
    individual = {}
    for name in sorted(target):
        injected_logits, _ = deepseek_v4_oracle.run(
            reference_weights,
            token_ids,
            reference_config,
            local_errors={name: local_errors[name]},
        )
        source_nrmse = _nrmse(actual_trace[name], local_reference[name])
        logits_nrmse = _nrmse(injected_logits, reference_logits)
        individual[name] = {
            "source_nrmse": source_nrmse,
            "logits_nrmse": logits_nrmse,
            "gain": logits_nrmse / source_nrmse if source_nrmse else 0.0,
            "top1_flips": _metrics(injected_logits, reference_logits)["top1_flips"],
        }
    xla_mhc_logits, _ = deepseek_v4_oracle.run(
        reference_weights,
        token_ids,
        reference_config,
        local_errors={name: xla_local_errors[name] for name in mhc},
    )
    return {
        "seed": seed,
        "baseline": _metrics(actual_logits, reference_logits),
        "xla_mhc_control": _metrics(xla_logits, reference_logits),
        "production_vs_xla_mhc": _metrics(actual_logits, xla_logits),
        "xla_mhc_local_effect": {
            "source_nrmse": _group_nrmse(xla_local_errors, xla_local_reference, mhc),
            **_metrics(xla_mhc_logits, reference_logits),
        },
        "reconstruction": reconstruction,
        "groups": results,
        "individual": individual,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    results = []
    for offset in range(args.seeds):
        result = _run_seed(20260904 + args.seed_offset + offset)
        results.append(result)
        if args.verbose:
            print(result, flush=True)
        else:
            compact = {
                "seed": result["seed"],
                "production": result["baseline"],
                "xla_mhc": result["xla_mhc_control"],
                "production_vs_xla_mhc": result["production_vs_xla_mhc"],
                "production_mhc_effect": result["groups"]["mHC total"],
                "xla_mhc_effect": result["xla_mhc_local_effect"],
                "csa_effect": result["groups"]["CSA attention"],
                "hca_effect": result["groups"]["HCA attention"],
                "reconstruction": result["reconstruction"],
            }
            print(compact, flush=True)

    if len(results) > 1:
        metrics = {
            "production_logits_nrmse": [
                result["baseline"]["logits_nrmse"] for result in results
            ],
            "xla_mhc_logits_nrmse": [
                result["xla_mhc_control"]["logits_nrmse"] for result in results
            ],
            "production_vs_xla_mhc_nrmse": [
                result["production_vs_xla_mhc"]["logits_nrmse"] for result in results
            ],
            "production_mhc_effect_nrmse": [
                result["groups"]["mHC total"]["logits_nrmse"] for result in results
            ],
            "xla_mhc_effect_nrmse": [
                result["xla_mhc_local_effect"]["logits_nrmse"] for result in results
            ],
            "csa_effect_nrmse": [
                result["groups"]["CSA attention"]["logits_nrmse"] for result in results
            ],
            "hca_effect_nrmse": [
                result["groups"]["HCA attention"]["logits_nrmse"] for result in results
            ],
        }
        print(
            {
                name: {
                    "mean": float(np.mean(values)),
                    "max": float(np.max(values)),
                }
                for name, values in metrics.items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
