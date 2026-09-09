"""Isolate output-head layout/fusion from the full-model framework graph."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from run_deepseek_v4_framework import compare_arrays

from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    compiled_head,
    config_for_layer,
    official_head_collapse,
    rms_norm,
)
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    config = config_for_layer(checkpoint.config, 0, 256)
    mesh = create_device_mesh([1, 4], [1, 1])
    shared = {
        name: jax.device_put(checkpoint.read_tensor(name), NamedSharding(mesh, P()))
        for name in (
            "hc_head_fn",
            "hc_head_scale",
            "hc_head_base",
            "norm.weight",
            "head.weight",
        )
    }

    def head(value, shared):
        collapsed = official_head_collapse(
            value,
            shared["hc_head_fn"],
            shared["hc_head_scale"],
            shared["hc_head_base"],
            eps=config.eps,
            hc_eps=config.hc_eps,
        )
        normalized = rms_norm(collapsed, shared["norm.weight"], config.eps)
        return jnp.matmul(
            normalized, shared["head.weight"].T, preferred_element_type=jnp.float32
        )

    def build(barrier):
        @jax.jit
        def run(value, weights):
            logits = jax.shard_map(
                head, mesh=mesh, in_specs=(P(), P()), out_specs=P(), check_vma=False
            )(value, weights)
            if barrier:
                logits = jax.lax.optimization_barrier(logits)
            return jax.reshard(logits, NamedSharding(mesh, P("data", "tensor")))

        return run

    variants = {"resharded": build(False), "barrier_resharded": build(True)}
    traces = np.load(args.reference_trace)
    report = {"scope": "head-only, real saved last-layer activations", "checks": []}
    with jax.set_mesh(mesh):
        for key in (k for k in traces.files if k.endswith("/42/streams")):
            value = jax.device_put(
                jnp.asarray(traces[key][None], jnp.bfloat16), NamedSharding(mesh, P())
            )
            expected = compiled_head(config, mesh)(value, shared)
            for name, fn in variants.items():
                actual = fn(value, shared)
                check = {
                    "activation": key,
                    "variant": name,
                    **compare_arrays(expected, actual),
                }
                report["checks"].append(check)
                print(json.dumps(check), flush=True)
                path = args.output / f"{name}.hlo.txt"
                if not path.exists():
                    path.write_text(fn.lower(value, shared).compile().as_text())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
