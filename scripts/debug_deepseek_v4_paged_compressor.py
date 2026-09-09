"""Compare ragged compressor intermediates to host arithmetic on small fixtures."""

import json
import jax
import jax.numpy as jnp
import numpy as np
from sgl_jax.srt.kernels.deepseek_v4.compressor import compress
from sgl_jax.srt.kernels.deepseek_v4.numerics import V4LayerConfig
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.test.test_deepseek_v4_paged import make_batch, make_cache, _compressor_weights


def main():
    rng = np.random.default_rng(20260907)
    config = V4LayerConfig(hidden=128, head_dim=128, index_dim=128, max_context=384, ratio=4)
    weights = _compressor_weights(rng, config)
    x = jnp.asarray(rng.normal(size=(132, 128)), jnp.bfloat16)
    batch = make_batch([0], [132])
    meta = V4PagedBackend(max_context=384).get_forward_metadata(batch)

    @jax.jit
    def run(x, weights, cache, meta):
        trace = {}
        cache = compress(x, weights, cache, config, meta, trace=trace)
        return cache, trace

    _, trace = jax.device_get(run(x, weights, make_cache(config), meta))
    host_x = np.asarray(x, np.float32)
    host_w = {k: np.asarray(v, np.float32) for k, v in weights.items()}
    kv = host_x @ host_w["attn.compressor.wkv.weight"].T
    score = host_x @ host_w["attn.compressor.wgate.weight"].T
    values = kv.reshape(33, 4, 256)
    scores = score.reshape(33, 4, 256) + host_w["attn.compressor.ape"][None]
    gv = np.concatenate(
        (np.concatenate((np.zeros((1, 4, 128)), values[:-1, :, :128])), values[:, :, 128:]), axis=1
    )
    gs = np.concatenate(
        (np.concatenate((np.full((1, 4, 128), -np.inf), scores[:-1, :, :128])), scores[:, :, 128:]),
        axis=1,
    )
    probability = np.exp(
        gs.astype(np.float32) - np.max(gs, axis=1, keepdims=True).astype(np.float32)
    )
    pooled = (
        np.sum(
            gv.astype(np.float32) * (probability / probability.sum(axis=1, keepdims=True)), axis=1
        )
        .astype(jnp.bfloat16)
        .astype(np.float32)
    )
    normalized = (
        pooled
        / np.sqrt(np.mean(pooled**2, axis=-1, keepdims=True) + 1e-6)
        * host_w["attn.compressor.norm.weight"]
    )
    expected = dict(
        kv=kv,
        scores=score,
        group_values=gv,
        group_scores=gs,
        pooled=pooled,
        normalized=normalized.astype(jnp.bfloat16).astype(np.float32),
    )
    print("positions", trace["group_positions"][:2].tolist(), flush=True)
    for row in range(8):
        print(
            "group row",
            row,
            "raw",
            trace["raw_group_values"][0, row, :4].tolist(),
            "tail",
            trace["raw_group_values"][0, row, 128:132].tolist(),
            "actual",
            trace["group_values"][0, row, :4].tolist(),
            "expected",
            gv[0, row, :4].tolist(),
            flush=True,
        )
    for key, value in expected.items():
        actual = np.asarray(trace[key][: len(value)], np.float32)
        finite = np.isfinite(value) & np.isfinite(actual)
        diff = actual[finite] - value[finite]
        print(
            json.dumps(
                {
                    "stage": key,
                    "max_abs": float(np.max(np.abs(diff), initial=0)),
                    "nrmse": float(
                        np.linalg.norm(diff) / max(np.linalg.norm(value[finite]), 1e-12)
                    ),
                    "actual_head": actual.ravel()[:8].tolist(),
                    "expected_head": value.ravel()[:8].tolist(),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
