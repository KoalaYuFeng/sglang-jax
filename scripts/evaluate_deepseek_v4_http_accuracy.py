"""Owned full-model HTTP generation/logprob controls and frozen MMLU-Pro replay."""

import argparse
import hashlib
import importlib.util
import json
import time
import traceback
from pathlib import Path

import httpx
import numpy as np
from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_execution_options import assert_server_options
from run_deepseek_v4_http_stress import validate_idle


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    old = out.parent / "v4-mmlu-accuracy-20260911-01"
    fixture = json.loads((out / "fixture.json").read_text())
    receipt = json.loads((out / "server_start.json").read_text())
    assert (
        fingerprint()
        == fixture["source_fingerprint"]
        == receipt["framework_source_fingerprint"]
    )
    actual = (
        Path(f"/proc/{receipt['pid']}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    )
    assert [v.decode() for v in actual] == receipt["command"]
    protocol = json.loads((old / "protocol.json").read_text())
    assert sha(old / "selected.json") == protocol["selected_sha256"]
    assert sha(old / "evaluate.py") == protocol["script_sha256"]
    spec = importlib.util.spec_from_file_location(
        "frozen_mmlu_evaluation", old / "evaluate.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    cases = json.loads((old / "selected.json").read_text())
    assert len(cases) == 56
    previous = {
        r["question_id"]: r
        for r in json.loads((old / "report.json").read_text())["cases"]
    }
    target = out / "http-evaluation"
    target.mkdir(exist_ok=False)
    (target / "selected.json").write_bytes((old / "selected.json").read_bytes())
    frozen = {
        "previous_protocol": protocol,
        "previous_protocol_sha256": sha(old / "protocol.json"),
        "source_fingerprint": fingerprint(),
        "script_sha256": sha(Path(__file__)),
        "scope": "Full 43-layer production HTTP; generation/logprob controls and exact frozen 56-question MMLU-Pro replay",
        "sampling": protocol["sampling"],
        "denominator": 56,
        "no_score_dependent_retry": True,
        "logits_protocol": "Full-vocabulary HTTP log probabilities compared with normalized CPU logits; raw HTTP logits are not exposed",
    }
    (target / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    report = {
        "complete": False,
        "generation": [],
        "teacher_forced_rows": 0,
        "mmlu_cases": [],
        "denominator": 56,
    }

    def emit(event, **values):
        (target / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        print(
            json.dumps(
                {"event": event, "time": time.time(), **values}, ensure_ascii=False
            ),
            flush=True,
        )

    try:
        with httpx.Client(base_url="http://127.0.0.1:30126", timeout=1200) as client:
            deadline = time.monotonic() + 1800
            last_emit = 0
            while True:
                try:
                    info = (
                        client.get("/get_server_info", timeout=5)
                        .raise_for_status()
                        .json()
                    )
                    if info.get("status") == "ready":
                        break
                except (httpx.HTTPError, ValueError):
                    pass
                assert time.monotonic() < deadline, "Server startup timeout"
                if time.monotonic() - last_emit > 30:
                    emit("waiting_server")
                    last_emit = time.monotonic()
                time.sleep(2)
            assert validate_idle(info["internal_states"], 33280)
            assert info["context_length"] == 8192 and info["max_running_requests"] == 4
            assert_server_options(info, receipt["execution"])
            (target / "server-before.json").write_text(
                json.dumps(info, indent=2) + "\n"
            )
            for repeat in range(2):
                emit("generation_start", repeat=repeat)
                response = (
                    client.post(
                        "/generate",
                        json={
                            "input_ids": fixture["prompt_ids"],
                            "sampling_params": {
                                "temperature": 0,
                                "top_p": 1,
                                "max_new_tokens": 128,
                                "ignore_eos": False,
                            },
                            "stream": False,
                            "rid": f"v4-e2e-generation-{repeat}",
                        },
                    )
                    .raise_for_status()
                    .json()
                )
                (target / f"generation-{repeat}.json").write_text(
                    json.dumps(response, indent=2, ensure_ascii=False) + "\n"
                )
                report["generation"].append(
                    {
                        "repeat": repeat,
                        "text": response["text"],
                        "meta_info": response["meta_info"],
                    }
                )
                emit("generation_complete", repeat=repeat, text=response["text"])
            report["generation_text_repeat_equal"] = (
                report["generation"][0]["text"] == report["generation"][1]["text"]
            )
            vocab = fixture["vocab_size"]
            rows = []
            for step in range(len(fixture["continuation_ids"]) + 1):
                emit("teacher_start", step=step)
                ids = fixture["prompt_ids"] + fixture["continuation_ids"][:step]
                response = (
                    client.post(
                        "/generate",
                        json={
                            "input_ids": ids,
                            "sampling_params": {
                                "temperature": 0,
                                "top_p": 1,
                                "max_new_tokens": 1,
                                "ignore_eos": False,
                            },
                            "stream": False,
                            "return_logprob": True,
                            "logprob_start_len": -1,
                            "token_ids_logprob": list(range(vocab)),
                            "rid": f"v4-e2e-teacher-{step}",
                        },
                    )
                    .raise_for_status()
                    .json()
                )
                (target / f"teacher-{step:02d}.json").write_text(
                    json.dumps(response, ensure_ascii=False) + "\n"
                )
                records = response["meta_info"]["output_token_ids_logprobs"]
                assert len(records) == 1 and len(records[0]) == vocab
                values = np.asarray([v[0] for v in records[0]], np.float32)
                indices = np.asarray([v[1] for v in records[0]], np.int64)
                assert (
                    np.array_equal(indices, np.arange(vocab))
                    and np.isfinite(values).all()
                )
                assert abs(np.exp(values.astype(np.float64)).sum() - 1) < 2e-4
                rows.append(values)
                np.savez_compressed(
                    target / "teacher-logprobs.npz", logprobs=np.stack(rows)
                )
                report["teacher_forced_rows"] = len(rows)
                emit("teacher_complete", step=step, top1=int(values.argmax()))
            errors = 0
            for case in cases:
                emit(
                    "mmlu_start",
                    question_id=case["question_id"],
                    completed=len(report["mmlu_cases"]),
                )
                result = {
                    "question_id": case["question_id"],
                    "category": case["category"],
                    "gold": case["gold"],
                    "prediction": None,
                    "correct": False,
                    "invalid": True,
                    "truncated": False,
                    "previous_correct": previous[case["question_id"]]["correct"],
                }
                start = time.monotonic()
                try:
                    assert case["fits_context"]
                    generated = (
                        client.post(
                            "/generate",
                            json={
                                "input_ids": case["input_ids"],
                                "sampling_params": protocol["sampling"],
                                "stream": False,
                                "rid": f"v4-e2e-mmlu-{case['question_id']}",
                            },
                        )
                        .raise_for_status()
                        .json()
                    )
                    pred, tier = evaluator.extract(generated["text"])
                    valid = pred is not None and evaluator.LETTERS.index(pred) < len(
                        [o for o in case["raw_test"]["options"] if o != "N/A"]
                    )
                    truncated = (
                        generated["meta_info"]["finish_reason"].get("type") == "length"
                    )
                    assert (
                        generated["meta_info"]["prompt_tokens"] == case["prompt_tokens"]
                    )
                    result.update(
                        response=generated,
                        prediction=pred,
                        extraction_tier=tier,
                        invalid=not valid,
                        truncated=truncated,
                        correct=valid and not truncated and pred == case["gold"],
                    )
                    errors = 0
                except (
                    httpx.HTTPError,
                    ValueError,
                    AssertionError,
                    KeyError,
                    IndexError,
                    TypeError,
                ):
                    result["error"] = traceback.format_exc()
                    errors += 1
                result["seconds_including_possible_compilation"] = (
                    time.monotonic() - start
                )
                with (target / "mmlu-responses.jsonl").open("a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                report["mmlu_cases"].append(
                    {k: v for k, v in result.items() if k != "response"}
                )
                report["correct"] = sum(r["correct"] for r in report["mmlu_cases"])
                report["accuracy"] = report["correct"] / 56
                emit(
                    "mmlu_answer",
                    completed=len(report["mmlu_cases"]),
                    correct=report["correct"],
                    prediction=result["prediction"],
                    gold=case["gold"],
                )
                if errors >= 3:
                    raise RuntimeError(
                        "Three consecutive HTTP errors; retained incomplete report"
                    )
            info = client.get("/get_server_info").raise_for_status().json()
            assert validate_idle(info["internal_states"], 33280)
            (target / "server-after.json").write_text(json.dumps(info, indent=2) + "\n")
        assert fingerprint() == fixture["source_fingerprint"]
        report.update(
            complete=True,
            invalid=sum(r["invalid"] for r in report["mmlu_cases"]),
            truncated=sum(r["truncated"] for r in report["mmlu_cases"]),
            failed=sum("error" in r for r in report["mmlu_cases"]),
        )
        emit("complete", correct=report["correct"], total=56)
    except BaseException:
        report["error"] = traceback.format_exc()
        emit("failed", error=report["error"])
        raise


if __name__ == "__main__":
    main()
