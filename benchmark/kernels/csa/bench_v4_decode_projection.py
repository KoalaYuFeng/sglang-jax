"""Paired decode projection boundary benchmark, no model/Engine allocation.

The original real 7979 activation/weights are frozen inputs. Larger batches
repeat/sign-flip that activation; they are shape fixtures, not independently
captured concurrent requests. All variants keep FP32 state projections.
"""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.kernels.csa.compressor import (
    csa_project_decode_pallas,
    csa_project_pallas,
)
from sgl_jax.srt.kernels.deepseek_v4 import csa
from sgl_jax.srt.kernels.deepseek_v4.numerics import V4LayerConfig
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import V4PagedBackend
from sgl_jax.test.kernels.csa_compressor_cases import read_arrays
from sgl_jax.test.test_deepseek_v4_paged import make_batch

from benchmark.kernels.csa.bench_v4_compressor import (
    memory_bytes,
    paired_timings,
    profile_call,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 9])
    parser.add_argument("--chips", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        raise RuntimeError("requires four real TPU chips")
    if any(c not in (1, 4) for c in args.chips) or any(b < 1 for b in args.batches):
        raise ValueError("chips must be 1/4 and batch sizes positive")
    args.output.mkdir(parents=True, exist_ok=False)
    names = ["activation"] + [
        f"{kind}.{field}"
        for kind in ("main", "index")
        for field in ("wkv.weight", "wgate.weight")
    ]
    fixture = read_arrays(args.fixture, names)
    from analyze_deepseek_v4_native_profile import fingerprint

    report = {
        "complete": False,
        "scope": __doc__,
        "framework_source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture": str(args.fixture),
        "fixture_sha256": hashlib.sha256(
            args.fixture.with_suffix(".npz").read_bytes()
        ).hexdigest(),
        "checks": [],
        "cases": [],
    }

    def flush():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    flush()
    try:
        config = V4LayerConfig(ratio=4, max_context=8192)
        for chips in args.chips:
            mesh = Mesh(np.asarray(jax.devices()[:chips]), ("tensor",))
            with jax.set_mesh(mesh):
                sharding = NamedSharding(mesh, P())
                for batch in args.batches:
                    pages = [list(range(1 + i * 64, 65 + i * 64)) for i in range(batch)]
                    metadata = V4PagedBackend(
                        max_context=8192, max_requests=batch
                    ).get_forward_metadata(
                        make_batch(
                            [7979] * batch, [1] * batch, pages=pages, decode=True
                        )
                    )
                    for kind in ("main", "index"):
                        label = f"chips{chips}/B{batch}/{kind}"
                        print(
                            json.dumps({"event": "compile", "case": label}), flush=True
                        )

                        def adapter(x, a, b, m):
                            return csa.project(x, a, b, m, config)

                        def sequential(x, a, b, m):
                            fused = jnp.concatenate((a.T, b.T), axis=1)
                            result = jax.lax.map(
                                lambda token: csa_project_pallas(
                                    token[None], fused, projection_mode="v4_gemv"
                                )[0],
                                x,
                            )
                            return result[:, : a.shape[0]], result[:, a.shape[0] :]

                        def batched(x, a, b, m):
                            result = csa_project_decode_pallas(
                                x, jnp.concatenate((a.T, b.T), axis=1)
                            )
                            return result[:, : a.shape[0]], result[:, a.shape[0] :]

                        ring = []
                        for phase in range(4):
                            signs = np.where((np.arange(batch) + phase) % 2, -1, 1)
                            x = (
                                np.repeat(fixture["activation"], batch, axis=0).astype(
                                    np.float32
                                )
                                * signs[:, None]
                            ).astype(fixture["activation"].dtype)
                            values = (
                                x,
                                fixture[f"{kind}.wkv.weight"],
                                fixture[f"{kind}.wgate.weight"],
                                metadata,
                            )
                            ring.append(
                                jax.tree.map(
                                    lambda a, sharding=sharding: jax.device_put(
                                        a, sharding
                                    ),
                                    values,
                                )
                            )
                        calls, variants = {}, {}
                        for name, fn in (
                            ("adapter", adapter),
                            ("sequential", sequential),
                            ("batched", batched),
                        ):
                            run = jax.jit(
                                jax.shard_map(
                                    fn,
                                    mesh=mesh,
                                    in_specs=(P(),) * 4,
                                    out_specs=(P(), P()),
                                    check_vma=False,
                                )
                            )
                            compiled = run.lower(*ring[0]).compile()
                            hlo = compiled.as_text()
                            (
                                args.output
                                / f"chips{chips}-B{batch}-{kind}-{name}.hlo.txt"
                            ).write_text(hlo)
                            calls[name] = compiled
                            variants[name] = {
                                "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
                                "memory_analysis": memory_bytes(compiled),
                            }
                        for phase, operands in enumerate(ring):
                            expected = jax.tree.map(
                                np.asarray, calls["adapter"](*operands)
                            )
                            for name, fn in calls.items():
                                result = fn(*operands)
                                for component, (a, value) in enumerate(
                                    zip(expected, result, strict=True)
                                ):
                                    for shard in value.addressable_shards:
                                        passed = bool(
                                            np.array_equal(
                                                a.view(np.uint32),
                                                np.asarray(shard.data).view(np.uint32),
                                            )
                                        )
                                        row = {
                                            "label": f"{label}/{name}/phase{phase}/component{component}/chip{shard.device.id}",
                                            "bitwise_equal": passed,
                                            "passed": passed,
                                        }
                                        report["checks"].append(row)
                                        if not passed:
                                            raise AssertionError(row)
                        timings = paired_timings(calls, ring, args.iterations)
                        for name, fn in calls.items():
                            variants[name]["host_ms"] = timings[name]
                            if args.profile and batch in (1, 4):
                                variants[name]["profile"] = profile_call(
                                    fn,
                                    ring,
                                    args.output
                                    / "traces"
                                    / f"chips{chips}-B{batch}-{kind}-{name}",
                                    10,
                                )
                        report["cases"].append({"label": label, "variants": variants})
                        print(
                            json.dumps(
                                {
                                    "event": "case_passed",
                                    "case": label,
                                    "host_ms": {
                                        n: v["median"] for n, v in timings.items()
                                    },
                                }
                            ),
                            flush=True,
                        )
                        flush()
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        flush()


if __name__ == "__main__":
    main()
