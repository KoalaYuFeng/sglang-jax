"""Diagnostic: preserve the legacy scan-pooling repro beside the current path."""

import argparse
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.low_bit.formats import activation_fp8_roundtrip
from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    compress,
    config_for_layer,
    empty_cache,
    load_layer,
    rms_norm,
    rope,
)
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.test.deepseek_v4_cpu_oracle import load_module_weights, official_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    c = DeepSeekV4Checkpoint(args.checkpoint)
    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    config = config_for_layer(c.config, 2, 256)
    w = load_layer(c, 2, mesh, include_experts=False)
    x = np.random.default_rng(20260906).normal(size=(132, 4096)).astype(ml_dtypes.bfloat16)

    def calculate(x, w):
        values = jnp.matmul(
            x, w["attn.compressor.wkv.weight"].T, preferred_element_type=jnp.float32
        ).reshape(33, 4, 1024)
        scores = (
            jnp.matmul(
                x, w["attn.compressor.wgate.weight"].T, preferred_element_type=jnp.float32
            ).reshape(33, 4, 1024)
            + w["attn.compressor.ape"][None]
        )
        pv = jnp.concatenate(
            (jnp.concatenate((jnp.zeros((1, 4, 512)), values[:-1, :, :512])), values[:, :, 512:]),
            axis=1,
        )
        ps = jnp.concatenate(
            (
                jnp.concatenate((jnp.full((1, 4, 512), -jnp.inf), scores[:-1, :, :512])),
                scores[:, :, 512:],
            ),
            axis=1,
        )
        pooled = jnp.sum(pv * jax.nn.softmax(ps, axis=1), axis=1).astype(jnp.bfloat16)
        normalized = rms_norm(pooled, w["attn.compressor.norm.weight"], config.eps)
        rotated = rope(normalized, jnp.arange(33, dtype=jnp.int32) * 4, config)
        final = jnp.concatenate(
            (activation_fp8_roundtrip(rotated[:, :-64], 64), rotated[:, -64:]), axis=-1
        )
        current = compress(x, jnp.arange(132, dtype=jnp.int32), w, empty_cache(config), config)

        def single(item):
            pooled, pos = item
            normalized = rms_norm(pooled[None], w["attn.compressor.norm.weight"], config.eps)
            rotated = rope(normalized, pos[None], config)
            final = jnp.concatenate(
                (activation_fp8_roundtrip(rotated[:, :-64], 64), rotated[:, -64:]), axis=-1
            )
            return normalized[0], rotated[0], final[0]

        per_row = jax.lax.map(single, (pooled, jnp.arange(33, dtype=jnp.int32) * 4))

        def inspect_step(state, item):
            v, s = state
            value, score, pos = item
            v = v.at[4 + pos % 4].set(value)
            s = s.at[4 + pos % 4].set(score)
            pv = jnp.concatenate((v[:4, :512], v[4:, 512:]))
            ps = jnp.concatenate((s[:4, :512], s[4:, 512:]))
            pooled = jnp.sum(pv * jax.nn.softmax(ps, axis=0), axis=0).astype(jnp.bfloat16)
            v, s = jax.lax.cond(
                pos % 4 == 3,
                lambda vs: (vs[0].at[:4].set(vs[0][4:]), vs[1].at[:4].set(vs[1][4:])),
                lambda vs: vs,
                (v, s),
            )
            return (v, s), pooled

        _, scan_pool = jax.lax.scan(
            inspect_step,
            (jnp.zeros((8, 1024), jnp.float32), jnp.full((8, 1024), -jnp.inf, jnp.float32)),
            (values.reshape(132, 1024), scores.reshape(132, 1024), jnp.arange(132)),
        )
        return (
            pooled,
            normalized,
            rotated,
            final,
            current["main.compressed"][:33],
            per_row,
            scan_pool[3::4],
        )

    fn = jax.jit(
        jax.shard_map(calculate, mesh=mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False)
    )
    output = jax.device_get(fn(jax.device_put(x, NamedSharding(mesh, P())), w))
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(16)
    official, cfg = official_module(c)
    attn = load_module_weights(official.Attention(2, cfg), c, "layers.2.attn.")
    capture = {}
    attn.compressor.norm.register_forward_pre_hook(
        lambda m, a: capture.update(pooled=a[0][0].float().clone().numpy())
    )
    attn.compressor.norm.register_forward_hook(
        lambda m, a, o: capture.update(normalized=o[0].float().clone().numpy())
    )
    with torch.inference_mode():
        attn(torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16)[None], 0)
    expected = attn.kv_cache[0, 128:161].float().numpy()

    def show(label, a, b):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        print(
            json.dumps(
                {
                    "label": label,
                    "nrmse": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                    "first_nrmse": float(np.linalg.norm(a[0] - b[0]) / np.linalg.norm(b[0])),
                    "a_first": a[0, :8].tolist(),
                    "b_first": b[0, :8].tolist(),
                }
            ),
            flush=True,
        )

    show("vector_pool_vs_cpu", output[0], capture["pooled"])
    show("vector_norm_vs_cpu", output[1], capture["normalized"])
    show("vector_final_vs_cpu", output[3], expected)
    show("current_vs_vector", output[4], output[3])
    show("current_vs_cpu", output[4], expected)
    show("per_row_norm_vs_vector", output[5][0], output[1])
    show("per_row_rope_vs_vector", output[5][1], output[2])
    show("per_row_vs_cpu", output[5][2], expected)
    show("legacy_scan_pool_vs_vector", output[6], output[0])


if __name__ == "__main__":
    main()
