"""Independent stateful head-TP gate; never constructs a ModelRunner/Engine.

Cold inputs are seeded synthetic BF16 activations with official layer weights.
Near-8K inputs start with frozen real 8023 B2 cache/activations cloned into four
private requests; subsequent activations repeat/sign-flip those captured rows.
They are operator stress fixtures, not freshly computed whole-model hidden
states. Each variant advances its own cache; no reference state is injected
into the TP candidate between steps. This script does not measure performance.
"""

import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from analyze_deepseek_v4_native_profile import fingerprint
from benchmark_deepseek_v4_attention_tp import (
    array_digest,
    build,
    clone_b4,
    metrics,
    place,
    read_all,
)
from jax.sharding import Mesh
from sgl_jax.srt.kernels.deepseek_v4.numerics import config_for_layer
from sgl_jax.srt.layers.attention.deepseek_v4_paged_backend import (
    V4PagedBackend,
    V4PagedMetadata,
)
from sgl_jax.srt.mem_cache.deepseek_v4_paged_pool import layer_buffer_specs
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer
from sgl_jax.test.kernels.csa_compressor_cases import read_arrays
from sgl_jax.test.test_deepseek_v4_paged import make_batch


def cold_plan():
    # Packed query counts never exceed 128. Inert slots, odd chunks and request
    # reordering exercise the same dynamic metadata/compilation bucket.
    return [
        [3, 0, 0, 0],
        [7, 5, 0, 0],
        [31, 33, 29, 35],
        [63, 61, 67, 65],
        [95, 97, 93, 99],
        [127, 128, 125, 130],
        [129, 131, 126, 131],
        [161, 163, 158, 163],
        [193, 195, 190, 195],
        [225, 227, 222, 227],
        [255, 256, 253, 258],
        [259, 260, 257, 262],
        [260, 261, 258, 263],
    ]


def near8k_plan(previous):
    cursor = np.asarray(previous, np.int32) + 1
    if np.any(cursor + 11 >= 8191):
        raise ValueError("near-8K fixture needs room for boundary continuation")
    plan = [cursor.tolist()]
    cursor = cursor + np.asarray([7, 9, 5, 11])
    plan.append(cursor.tolist())
    while np.any(cursor < 8191):
        cursor = np.minimum(cursor + 31, 8191)
        plan.append(cursor.tolist())
    plan.append([8192] * 4)
    return plan


def make_frame(previous, ends, order, pages, inputs, *, decode=False):
    previous, ends = np.asarray(previous), np.asarray(ends)
    counts = ends - previous
    if np.any(counts < 0) or (decode and np.any((counts != 0) & (counts != 1))):
        raise ValueError("invalid advancing query lengths")
    if not np.any(counts > 0):
        raise ValueError("fixture must contain a live query")
    bucket = 4 if decode else 128
    if counts.sum() > bucket:
        raise ValueError("query exceeds the production packed-token bucket")
    selected_pages = [pages[i] for i in order]
    batch = make_batch(
        np.where(counts > 0, previous, 0)[order],
        counts[order],
        slots=order,
        pages=selected_pages,
        padding=bucket - int(counts.sum()),
        decode=decode,
    )
    hidden = np.zeros((bucket, inputs[0].shape[-1]), inputs[0].dtype)
    if decode:
        # Decode token rows are slot-aligned, unlike the packed EXTEND helper.
        batch.positions = np.zeros(bucket, np.int32)
        batch.out_cache_loc = np.full(bucket, -1, np.int32)
        for row, request in enumerate(order):
            if counts[request]:
                position = int(previous[request])
                batch.positions[row] = position
                batch.out_cache_loc[row] = (
                    pages[request][position // 128] * 128 + position % 128
                )
                hidden[row] = inputs[request][position]
    else:
        values = np.concatenate([inputs[i][previous[i] : ends[i]] for i in order])
        hidden[: len(values)] = values
        live = counts[counts > 0]
        if np.any(live == 1) and np.any(live > 1):
            batch.forward_mode = ForwardMode.MIXED
    metadata = V4PagedBackend(max_context=8192).get_forward_metadata(batch)
    return hidden, batch, metadata


def check_heads(observed, tokens, heads):
    shapes = [list(shard.data.shape) for shard in observed[2]["q"].addressable_shards]
    if len(shapes) != 4 or any(shape != [tokens, heads, 512] for shape in shapes):
        raise AssertionError(
            {"unexpected_query_shards": shapes, "expected_heads": heads}
        )
    return shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 2, 3])
    parser.add_argument(
        "--suites", nargs="+", choices=["cold", "near8k"], default=["cold", "near8k"]
    )
    args = parser.parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 4:
        raise RuntimeError("requires exactly four physical TPU chips")
    args.output.mkdir(parents=True, exist_ok=False)
    capture = json.loads((args.capture / "report.json").read_text())
    replay = json.loads((args.replay / "report.json").read_text())
    if (
        not replay["faithful"]
        or replay["framework_source_fingerprint"]
        != capture["framework_source_fingerprint"]
    ):
        raise ValueError("requires the matching immutable historical capture/replay")
    if capture["checkpoint"] != args.checkpoint:
        raise ValueError("checkpoint must match the real operator fixtures")
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    report = {
        "complete": False,
        "scope": __doc__,
        "framework_source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("benchmark_deepseek_v4_attention_tp.py",)
        },
        "checkpoint": args.checkpoint,
        "source_receipts": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.capture / "report.json", args.replay / "report.json")
        },
        "cases": [],
        "checks": [],
    }

    def flush():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def check(label, expected, actual, *, tolerance=0, exact=True):
        row = {
            "label": label,
            **metrics(expected, actual, tolerance=tolerance, exact=exact),
        }
        report["checks"].append(row)
        if not row["passed"]:
            flush()
            raise AssertionError(row)

    def compare(label, expected, actual):
        check(label + "/output", expected[0], actual[0], tolerance=0.005, exact=False)
        for key, value in expected[1].items():
            check(label + "/cache/" + key, value, actual[1][key])
        if len(expected) == len(actual) == 3:
            for key, value in expected[2].items():
                tolerance = 2e-4 if key == "attention_value" else 0
                check(
                    label + "/trace/" + key,
                    value,
                    actual[2][key],
                    tolerance=tolerance,
                    exact=tolerance == 0,
                )

    mesh = Mesh(np.asarray(jax.devices()), ("tensor",))
    flush()
    try:
        with jax.set_mesh(mesh):
            for layer in args.layers:
                config = config_for_layer(checkpoint.config, layer, 8192)
                staged = load_layer(checkpoint, layer, mesh, include_experts=False)
                weights = {
                    key: np.asarray(value)
                    for key, value in staged.items()
                    if key.startswith("attn.")
                }
                del staged
                for suite in args.suites:
                    label = f"layer{layer}/{suite}"
                    if suite == "cold":
                        rng = np.random.default_rng(8023 + layer)
                        inputs = [
                            rng.normal(size=(384, 4096)).astype(jnp.bfloat16)
                            for _ in range(4)
                        ]
                        pages = [[i * 3 + 1, i * 3 + 3, i * 3 + 2] for i in range(4)]
                        cache = {
                            key: np.full(shape, fill, dtype)
                            for key, (shape, dtype, fill) in layer_buffer_specs(
                                config, 1792, 4
                            ).items()
                        }
                        previous = np.zeros(4, np.int32)
                        plan = cold_plan()
                    else:
                        folder = args.capture / "B2"
                        captured = read_all(folder / "inputs")
                        cache = read_all(folder / "before" / f"layer-{layer:02d}")
                        md = V4PagedMetadata(**read_all(folder / "metadata"))
                        trace = args.replay / f"B2-layer-{layer:02d}-trace"
                        hidden = read_arrays(trace, ["attn.norm"])["attn.norm"]
                        hidden, captured, cache, md = clone_b4(
                            hidden, captured, cache, md, config
                        )
                        previous = np.asarray(captured["positions"], np.int32)
                        pages = [
                            list(range(1 + i * 64, 1 + (i + 1) * 64)) for i in range(4)
                        ]
                        inputs = []
                        for i in range(4):
                            values = np.tile(hidden[i], (8192, 1))
                            values[previous[i] + 1 :: 2] *= -1
                            inputs.append(values)
                        plan = near8k_plan(previous)
                    case = {
                        "label": label,
                        "ratio": config.ratio,
                        "input_digests": [array_digest(value) for value in inputs],
                        "weight_digests": {
                            key: array_digest(value) for key, value in weights.items()
                        },
                        "initial_cache_digests": {
                            key: array_digest(value) for key, value in cache.items()
                        },
                        "steps": [],
                    }
                    report["cases"].append(case)
                    states = {
                        name: jax.tree.map(
                            lambda value: np.array(value, copy=True), cache
                        )
                        for name in ("replicated4", "head_tp4")
                    }
                    runners = {
                        name: build(mesh, config, weights, tp=tp, trace=True)
                        for name, tp in (("replicated4", False), ("head_tp4", True))
                    }
                    compiled = {}

                    def step(
                        tag,
                        previous,
                        ends,
                        order,
                        pages,
                        inputs,
                        *,
                        decode=False,
                        validate_untraced=False,
                        case=case,
                        label=label,
                        states=states,
                        runners=runners,
                        weights=weights,
                        compiled=compiled,
                        config=config,
                        layer=layer,
                        suite=suite,
                    ):
                        hidden, batch, metadata = make_frame(
                            previous, ends, order, pages, inputs, decode=decode
                        )
                        entry = {
                            "tag": tag,
                            "prefixes": list(map(int, previous)),
                            "ends": list(map(int, ends)),
                            "order": order,
                            "mode": batch.forward_mode.name,
                            "input_digest": array_digest(hidden),
                            "variants": {},
                        }
                        case["steps"].append(entry)
                        print(
                            json.dumps(
                                {
                                    "event": "step",
                                    "case": label,
                                    **{
                                        key: entry[key]
                                        for key in ("tag", "prefixes", "ends", "mode")
                                    },
                                }
                            ),
                            flush=True,
                        )
                        results = {}
                        for name in states:
                            runner, specs = runners[name]
                            operands = (
                                hidden,
                                batch.positions,
                                weights,
                                states[name],
                                metadata,
                                batch.out_cache_loc,
                            )
                            placed = place(mesh, specs, operands)
                            signature = (
                                name,
                                tuple(
                                    (value.shape, str(value.dtype))
                                    for value in jax.tree.leaves(placed)
                                ),
                            )
                            if signature not in compiled:
                                executable = runner.lower(*placed).compile()
                                hlo = executable.as_text()
                                hlo_path = (
                                    args.output
                                    / f"layer{layer}-{suite}-{name}-bucket{len(compiled)}.hlo.txt"
                                )
                                hlo_path.write_text(hlo)
                                if name == "head_tp4" and (
                                    "all-gather(" not in hlo or "all-reduce(" in hlo
                                ):
                                    raise AssertionError(
                                        "head TP requires gather without floating sum collective"
                                    )
                                if config.ratio:
                                    family = (
                                        "csa-joint-attention"
                                        if config.ratio == 4
                                        else "hca-paged-stream"
                                    )
                                    calls = [
                                        line
                                        for line in hlo.splitlines()
                                        if "custom-call(" in line and family in line
                                    ]
                                    if (
                                        len(calls) != 1
                                        or f"-h{16 if name == 'head_tp4' else 64}-d512-v4"
                                        not in calls[0]
                                    ):
                                        raise AssertionError(
                                            "missing original local-head Pallas call"
                                        )
                                compiled[signature] = (
                                    executable,
                                    str(hlo_path),
                                    hashlib.sha256(hlo.encode()).hexdigest(),
                                )
                            executable, hlo_path, digest = compiled[signature]
                            observed = executable(*placed)
                            results[name] = jax.tree.map(np.asarray, observed)
                            states[name] = observed[1]
                            entry["variants"][name] = {
                                "hlo": hlo_path,
                                "hlo_sha256": digest,
                                "query_shards": check_heads(
                                    observed,
                                    len(hidden),
                                    16 if name == "head_tp4" else 64,
                                ),
                            }
                            if validate_untraced:
                                run, plain_specs = build(
                                    mesh,
                                    config,
                                    weights,
                                    tp=name == "head_tp4",
                                    trace=False,
                                )
                                fresh = jax.tree.map(
                                    lambda value: np.array(value, copy=True), placed[3]
                                )
                                plain_args = place(
                                    mesh,
                                    plain_specs,
                                    operands[:3] + (fresh,) + operands[4:],
                                )
                                plain = run.lower(*plain_args).compile()
                                plain_hlo = plain.as_text()
                                plain_path = (
                                    args.output
                                    / f"layer{layer}-{suite}-{tag}-{name}-untraced.hlo.txt"
                                )
                                plain_path.write_text(plain_hlo)
                                entry["variants"][name]["untraced_hlo"] = str(
                                    plain_path
                                )
                                entry["variants"][name]["untraced_hlo_sha256"] = (
                                    hashlib.sha256(plain_hlo.encode()).hexdigest()
                                )
                                compare(
                                    label + "/" + tag + "/untraced/" + name,
                                    results[name],
                                    jax.tree.map(np.asarray, plain(*plain_args)),
                                )
                        compare(
                            label + "/" + tag,
                            results["replicated4"],
                            results["head_tp4"],
                        )
                        flush()
                        return results

                    for index, ends in enumerate(plan):
                        ends = np.asarray(ends, np.int32)
                        decode = bool(np.all(ends - previous <= 1))
                        order = [0, 1, 2, 3] if index % 2 == 0 else [3, 1, 0, 2]
                        step(
                            f"advance{index}",
                            previous,
                            ends,
                            order,
                            pages,
                            inputs,
                            decode=decode,
                            validate_untraced=index == len(plan) - 1,
                        )
                        previous = ends
                    if suite == "cold":
                        # Reclaim slot 3 and fork request 0's immutable full pages.
                        # Its previous scratch is dirty and must be restored from
                        # page 2's snapshot, with all other request slots unchanged.
                        saved = {
                            name: jax.tree.map(np.asarray, state)
                            for name, state in states.items()
                        }
                        fork_pages, fork_inputs = list(pages), list(inputs)
                        fork_pages[3] = pages[0][:2] + [13]
                        fork_inputs[3] = inputs[0]
                        results = step(
                            "prefix_fork",
                            [0, 0, 0, 256],
                            [0, 0, 0, 263],
                            [3, 1, 0, 2],
                            fork_pages,
                            fork_inputs,
                            validate_untraced=True,
                        )
                        for name, result in results.items():
                            for key in result[1]:
                                if key.endswith((".kv", ".score")):
                                    check(
                                        label
                                        + "/fork_private_untouched/"
                                        + name
                                        + "/"
                                        + key,
                                        saved[name][key][1:4],
                                        result[1][key][1:4],
                                    )
                                elif ".snapshot_" in key:
                                    check(
                                        label
                                        + "/fork_shared_untouched/"
                                        + name
                                        + "/"
                                        + key,
                                        saved[name][key][pages[0][:2]],
                                        result[1][key][pages[0][:2]],
                                    )
                                else:
                                    rows_per_page = (
                                        128 if key == "window" else 128 // config.ratio
                                    )
                                    for page in pages[0][:2]:
                                        rows = slice(
                                            page * rows_per_page,
                                            (page + 1) * rows_per_page,
                                        )
                                        check(
                                            label
                                            + f"/fork_shared_untouched/{name}/{key}/page{page}",
                                            saved[name][key][rows],
                                            result[1][key][rows],
                                        )
                    for name, state in states.items():
                        for key, value in state.items():
                            for rank, shard in enumerate(value.addressable_shards):
                                check(
                                    f"{label}/final_replica/{name}/{rank}/{key}",
                                    np.asarray(value),
                                    np.asarray(shard.data),
                                )
                    case["complete"] = True
                    flush()
                    del step, compiled, runners, states, cache, inputs
                    jax.clear_caches()
                    gc.collect()
                del weights
                gc.collect()
        report["complete"] = True
        report["summary"] = {
            "checks": len(report["checks"]),
            "bitwise": sum(row["bitwise_equal"] for row in report["checks"]),
            "cases": len(report["cases"]),
        }
        flush()
        print(json.dumps(report["summary"]), flush=True)
    except BaseException:
        report["error"] = traceback.format_exc()
        flush()
        raise


if __name__ == "__main__":
    main()
