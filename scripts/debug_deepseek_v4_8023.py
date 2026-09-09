"""Capture exact production B1/B2 inputs/states at the first 8K numerical failure.

No model/kernel monkeypatch or added JIT outputs: the ordinary ModelWorker
executes unchanged. Only the test's call wrapper saves buffers before/after
the selected step, outside its compiled program. BF16 bits are preserved.
"""

import argparse
import dataclasses
import hashlib
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from run_deepseek_v4_8k_native import CAPACITY, CONTEXT, PROMPT, restore_prefix
from run_deepseek_v4_framework import compare_arrays, framework_fingerprint
from run_deepseek_v4_paged import ModelWorker, PagedWorkerSession, server_args
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

POSITION = 8023
INDEX = POSITION - PROMPT


def save_arrays(path, arrays, *, compressed=True):
    values, schema = {}, {}
    for name, array in arrays.items():
        value = np.asarray(jax.device_get(array))
        schema[name] = {"dtype": str(value.dtype), "shape": list(value.shape)}
        values[name] = value.view(np.uint16) if value.dtype == jnp.bfloat16 else value
    writer = np.savez_compressed if compressed else np.savez
    writer(path.with_suffix(".npz"), **values)
    path.with_suffix(".json").write_text(json.dumps(schema, indent=2) + "\n")


def load_arrays(path):
    schema = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
        result = {}
        for name, info in schema.items():
            value = saved[name]
            if info["dtype"] == "bfloat16":
                value = value.view(jnp.bfloat16)
            if list(value.shape) != info["shape"] or str(value.dtype) != info["dtype"]:
                raise ValueError(f"invalid diagnostic array {path}/{name}")
            result[name] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.golden_report.read_text())
    fingerprint = framework_fingerprint()
    if previous["framework_source_fingerprint"] != fingerprint:
        raise ValueError("reproduction requires the original failing source")
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.golden_report.parent / "golden_logits.npz") as data:
        goldens = [data[f"case{i}"] for i in range(2)]
    prompts = previous["prompts"]
    report = {
        "finished": False,
        "framework_source_fingerprint": fingerprint,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": previous["checkpoint"],
        "golden_report": str(args.golden_report),
        "mhc_backend": previous.get("mhc_backend", "reference"),
        "hca_backend": previous.get("hca_backend", "reference"),
        "csa_backend": previous.get("csa_backend", "reference"),
        "position": POSITION,
        "events": [],
        "checks": [],
        "frames": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(event), flush=True)

    def check(label, expected, actual):
        metrics = compare_arrays(expected, actual)
        if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
            raise AssertionError("nonfinite reproduction logits")
        metrics["top1_equal"] = bool(np.array_equal(np.argmax(expected, -1), np.argmax(actual, -1)))
        record = {"label": label, **metrics}
        report["checks"].append(record)
        if metrics["nrmse"] > 0.005 or label.endswith(f"/{INDEX}"):
            emit({"event": "comparison", **record})
        return metrics

    try:
        sa = server_args(previous["checkpoint"], CONTEXT)
        sa.json_model_override_args = json.dumps(
            {
                "v4_mhc_backend": report["mhc_backend"],
                "v4_hca_backend": report["hca_backend"],
                "v4_csa_backend": report["csa_backend"],
            }
        )
        sa.max_total_tokens = CAPACITY
        emit({"event": "loading"})
        worker = ModelWorker(sa, create_device_mesh([1, 4], [1, 1]))
        session, runner = PagedWorkerSession(worker), worker.model_runner
        original_forward = worker.forward_batch_generation
        selected = None

        def save_cache(folder, when):
            target = folder / when
            target.mkdir()
            for layer, cache in enumerate(runner.token_to_kv_pool.layers):
                save_arrays(target / f"layer-{layer:02d}", cache)
            emit({"event": "cache_saved", "frame": folder.name, "when": when})

        def forward(batch):
            if selected is None:
                return original_forward(batch)
            folder = args.output / selected["label"]
            folder.mkdir()
            metadata = runner.attn_backend.get_forward_metadata(batch)
            save_arrays(folder / "metadata", dataclasses.asdict(metadata))
            save_arrays(
                folder / "inputs",
                {
                    "ids": batch.input_ids,
                    "positions": batch.positions,
                    "locations": batch.out_cache_loc,
                    "request_order": np.asarray(selected["order"], np.int32),
                },
            )
            save_cache(folder, "before")
            output, sampled, misses = original_forward(batch)
            jax.block_until_ready((output, sampled))
            save_arrays(folder / "production", {"logits": output.next_token_logits})
            save_cache(folder, "after")
            report["frames"].append(selected["label"])
            emit({"event": "frame_saved", **selected})
            return output, sampled, misses

        worker.forward_batch_generation = forward
        emit({"event": "loaded"})
        saved = []
        for case, prompt in enumerate(prompts):
            session.new(case)
            for begin in range(0, PROMPT, 128):
                session.step([(case, prompt[begin : begin + 128])], bucket=1)
                if begin % 1024 == 0:
                    emit({"event": "B1_prefill", "case": case, "position": begin})
            saved.append(runner.token_to_kv_pool.get_cpu_copy(session.requests[case].locations))
            for index in range(INDEX + 1):
                selected = {"label": f"B1-case{case}", "order": [case]} if index == INDEX else None
                logits, _ = session.step(
                    [(case, [int(np.argmax(goldens[case][index]))])], decode=True, bucket=1
                )
                metrics = check(f"B1/{case}/{index}", goldens[case][index + 1], logits[0])
                if not metrics["top1_equal"] or metrics["nrmse"] > 0.005:
                    raise AssertionError("B1 baseline no longer reproduces")
            selected = None
            session.free(case)
            emit({"event": "B1_complete", "case": case})

        with jax.set_mesh(runner.mesh):
            for case in range(2):
                restore_prefix(session, case, saved[case])
        first_failure = None
        for index in range(INDEX + 1):
            order = [0, 1] if index % 2 == 0 else [1, 0]
            selected = {"label": "B2", "order": order} if index == INDEX else None
            logits, _ = session.step(
                [(case, [int(np.argmax(goldens[case][index]))]) for case in order],
                decode=True,
                bucket=2,
            )
            for row, case in enumerate(order):
                metrics = check(f"B2/{case}/{index}", goldens[case][index + 1], logits[row])
                if metrics["nrmse"] > 0.005 or not metrics["top1_equal"]:
                    first_failure = first_failure or {"index": index, "case": case, **metrics}
            if index % 16 == 0:
                emit({"event": "B2_decode", "index": index})
        report["first_failure"] = first_failure
        if first_failure is None or first_failure["index"] != INDEX or first_failure["case"] != 1:
            raise AssertionError(f"expected original first failure, got {first_failure}")
        if not np.isclose(first_failure["nrmse"], 0.10763054341077805, rtol=1e-5):
            raise AssertionError("reproduction error differs from original failure")
        report["finished"] = True
        emit({"event": "original_failure_reproduced", **first_failure})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
