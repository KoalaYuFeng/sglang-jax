"""Resumable, fixed-protocol full MMLU-Pro evaluation of an owned V4 server.

No serving changes, truncation, score-dependent retry, or answer-based selection.
Atomic per-question records survive interruption; unknown in-flight attempts are
retained as failed rather than silently retried. Standard library imports permit
CPU-only unit testing of scoring and checkpoint behavior.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import time
import traceback
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_jsonl(path):
    # str.splitlines() also splits U+0085/U+2028 inside valid JSON strings.
    # Text-file iteration recognizes only the physical record delimiters.
    with path.open() as f:
        return [json.loads(line) for line in f]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def old_evaluator(base):
    old = base / "v4-mmlu-accuracy-20260911-01"
    protocol = json.loads((old / "protocol.json").read_text())
    assert sha(old / "evaluate.py") == protocol["script_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_mmlu", old / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, protocol


def result_from_response(case, response, extract):
    prediction, tier = extract(response["text"])
    valid = prediction in "ABCDEFGHIJ" if isinstance(prediction, str) else False
    valid = (
        valid
        and len(prediction) == 1
        and ord(prediction) - ord("A")
        < len([o for o in case["raw_test"]["options"] if o != "N/A"])
    )
    truncated = response["meta_info"]["finish_reason"].get("type") == "length"
    assert response["meta_info"]["prompt_tokens"] == case["prompt_tokens"]
    return {
        "response": response,
        "prediction": prediction,
        "extraction_tier": tier,
        "invalid": not valid,
        "truncated": truncated,
        "correct": bool(valid and not truncated and prediction == case["gold"]),
        "failed": False,
        "context_limit": False,
    }


def failure(case, reason, *, context_limit=False):
    return {
        "question_id": case["question_id"],
        "category": case["category"],
        "gold": case["gold"],
        "prediction": None,
        "correct": False,
        "invalid": True,
        "truncated": False,
        "failed": True,
        "context_limit": context_limit,
        "error": reason,
    }


def summarize(results, total):
    assert len(results) <= total
    assert len({r["question_id"] for r in results}) == len(results)
    correct = sum(r["correct"] for r in results)
    return {
        "completed": len(results),
        "total": total,
        "correct": correct,
        "accuracy_completed": correct / len(results) if results else None,
        "full_denominator_lower_bound": correct / total,
        "complete": len(results) == total,
        "failed": sum(r["failed"] for r in results),
        "truncated": sum(r["truncated"] for r in results),
        "invalid": sum(r["invalid"] for r in results),
        "context_limit": sum(r["context_limit"] for r in results),
        "by_category": {
            category: {
                "completed": sum(r["category"] == category for r in results),
                "correct": sum(
                    r["correct"] for r in results if r["category"] == category
                ),
            }
            for category in sorted({r["category"] for r in results})
        },
    }


def pending_action(case, completed, attempted):
    if case["question_id"] in completed:
        return "skip"
    if attempted:
        return "interrupted"
    return "request" if case["fits_context"] else "context_limit"


def prepare(out):
    from collections import Counter, defaultdict

    from analyze_deepseek_v4_native_profile import fingerprint
    from datasets import Dataset
    from transformers import AutoTokenizer

    out.mkdir(exist_ok=False)
    evaluator, previous = old_evaluator(out.parent)
    old = out.parent / "v4-mmlu-accuracy-20260911-01"
    cache = json.loads((old / "dataset-cache-audit.json").read_text())
    data = {}
    for split, source in cache["files"].items():
        assert sha(source["path"]) == source["sha256"]
        data[split] = Dataset.from_file(source["path"])
    assert len(data["test"]) == 12032 and len(data["validation"]) == 70
    examples = defaultdict(list)
    for row in data["validation"]:
        examples[row["category"]].append(dict(row))
    assert len(examples) == 14 and all(len(rows) == 5 for rows in examples.values())
    assert dict(examples) == json.loads((old / "validation-examples.json").read_text())
    token_dir = out.parent / "v4-standard-bench-20260910-01/client-tokenizer"
    for name, digest in previous["tokenizer_sha256"].items():
        assert sha(token_dir / name) == digest
    tokenizer = AutoTokenizer.from_pretrained(token_dir, local_files_only=True)
    encoder_path = Path(previous["checkpoint"]) / "encoding/encoding_dsv4.py"
    assert sha(encoder_path) == previous["encoder_sha256"]
    spec = importlib.util.spec_from_file_location("v4_encoder", encoder_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    rows = sorted(
        map(dict, data["test"]),
        key=lambda r: hashlib.sha256(
            f"20260911:full:{r['category']}:{r['question_id']}".encode()
        ).hexdigest(),
    )
    assert len({r["question_id"] for r in rows}) == 12032
    validation_questions = {r["question"] for r in data["validation"]}
    lengths, overflows = [], []
    old_cases = {
        r["question_id"]: r for r in json.loads((old / "selected.json").read_text())
    }
    checked = 0
    with (out / "cases.jsonl").open("x") as f:
        for row in rows:
            assert row["question"] not in validation_questions
            category = row["category"]
            instruction = (
                f"The following are multiple choice questions (with answers) about {category}. "
                "Think step by step and then output the answer in the format of "
                '"The answer is (X)" at the end.\n\n'
            )
            content = instruction + "".join(
                evaluator.format_question(e, True) for e in examples[category]
            )
            content += evaluator.format_question(row)
            prompt = encoder.encode_messages(
                [{"role": "user", "content": content}], thinking_mode="chat"
            )
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            case = {
                "question_id": row["question_id"],
                "category": category,
                "gold": row["answer"],
                "raw_test": row,
                "input_ids": ids,
                "prompt": prompt,
                "prompt_tokens": len(ids),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "fits_context": len(ids) + previous["sampling"]["max_new_tokens"] + 2
                <= 8192,
            }
            if row["question_id"] in old_cases:
                before = old_cases[row["question_id"]]
                assert ids == before["input_ids"] and prompt == before["prompt"]
                checked += 1
            lengths.append(len(ids))
            if not case["fits_context"]:
                overflows.append(
                    {k: case[k] for k in ("question_id", "category", "prompt_tokens")}
                )
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    assert checked == 56
    save(out / "validation-examples.json", dict(examples))
    save(out / "dataset-cache-audit.json", cache)
    protocol = {
        "dataset": previous["dataset"],
        "dataset_revision": previous["dataset_revision"],
        "total": 12032,
        "category_totals": dict(Counter(r["category"] for r in rows)),
        "order": "SHA256(20260911:full:category:question_id), all rows, no filtering",
        "source_fingerprint": fingerprint(),
        "checkpoint": previous["checkpoint"],
        "sampling": previous["sampling"],
        "shots": 5,
        "thinking_mode": "chat (Non-Think)",
        "concurrency": 1,
        "max_run_hours": 72,
        "context_length": 8192,
        "prompt_tokens_min": min(lengths),
        "prompt_tokens_max": max(lengths),
        "context_overflows": overflows,
        "cases_sha256": sha(out / "cases.jsonl"),
        "script_sha256": sha(__file__),
        "evaluator_sha256": previous["script_sha256"],
        "previous_protocol_sha256": sha(old / "protocol.json"),
        "tokenizer_sha256": previous["tokenizer_sha256"],
        "encoder_sha256": previous["encoder_sha256"],
        "previous_56_prompts_identical": checked,
        "retry_policy": "No score-dependent or automatic transport retry. Unknown interrupted attempts count failed. Resume unattempted IDs only.",
        "overflow_policy": "No prompt shortening, sample removal or output-budget reduction; context-limit cases count wrong in denominator 12032.",
        "scoring": "Frozen generated-answer extractor; failed/invalid/truncated/context-limit answers wrong. Full-test micro accuracy and category accuracy, not official protocol parity.",
    }
    save(out / "protocol.json", protocol)
    (out / "results").mkdir()
    (out / "attempts").mkdir()
    print(json.dumps(protocol, ensure_ascii=False), flush=True)


def run(out, receipt_path):
    import httpx
    from analyze_deepseek_v4_native_profile import fingerprint
    from deepseek_v4_execution_options import assert_server_options
    from run_deepseek_v4_http_stress import validate_idle

    lock = (out / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((out / "protocol.json").read_text())
    assert fingerprint() == protocol["source_fingerprint"]
    assert sha(__file__) == protocol["script_sha256"]
    assert sha(out / "cases.jsonl") == protocol["cases_sha256"]
    evaluator, previous = old_evaluator(out.parent)
    assert previous["script_sha256"] == protocol["evaluator_sha256"]
    receipt = json.loads(receipt_path.read_text())
    assert receipt["framework_source_fingerprint"] == fingerprint()
    actual = (
        Path(f"/proc/{receipt['pid']}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    )
    assert [v.decode() for v in actual] == receipt["command"]
    cases = read_jsonl(out / "cases.jsonl")
    assert len(cases) == protocol["total"]
    indexed = {c["question_id"]: c for c in cases}
    results = []
    for path in sorted((out / "results").glob("*.json")):
        record = json.loads(path.read_text())
        case = indexed[record["question_id"]]
        assert record["protocol_sha256"] == sha(out / "protocol.json")
        assert record["gold"] == case["gold"]
        if "response" in record:
            audited = result_from_response(case, record["response"], evaluator.extract)
            assert all(record[k] == audited[k] for k in audited)
        else:
            assert record["failed"] and not record["correct"]
        results.append(record)
    summarize(results, protocol["total"])
    completed = {r["question_id"] for r in results}
    run_id = str(time.time_ns())
    elapsed_start = time.monotonic()
    start_count = len(results)
    source_checks = 0

    def status(state, **extra):
        summary = summarize(results, protocol["total"])
        elapsed = time.monotonic() - elapsed_start
        done = len(results) - start_count
        summary.update(
            state=state,
            updated_unix=time.time(),
            pid=os.getpid(),
            run_id=run_id,
            protocol_sha256=sha(out / "protocol.json"),
            source_fingerprint=protocol["source_fingerprint"],
            run_seconds=elapsed,
            run_completed=done,
            estimated_remaining_hours=(protocol["total"] - len(results))
            * elapsed
            / done
            / 3600
            if done
            else None,
            **extra,
        )
        save(out / "progress.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    def persist(case, record):
        record.update(
            protocol_sha256=sha(out / "protocol.json"),
            question_id=case["question_id"],
            category=case["category"],
            gold=case["gold"],
            run_id=run_id,
            finished_unix=time.time(),
        )
        path = out / "results" / f"{case['question_id']:05d}.json"
        assert not path.exists()
        save(path, record)
        results.append(record)

    try:
        with httpx.Client(base_url="http://127.0.0.1:30126", timeout=1200) as client:

            def check_server():
                info = (
                    client.get("/get_server_info", timeout=15).raise_for_status().json()
                )
                assert info["status"] == "ready"
                assert info["context_length"] == 8192 and info["tp_size"] == 4
                assert info["ep_size"] == 4 and info["dp_size"] == 1
                assert info["model_path"] == protocol["checkpoint"]
                assert info["chunked_prefill_size"] == 128
                assert info["max_running_requests"] == 4
                assert validate_idle(info["internal_states"], 33280)
                assert_server_options(info, receipt["execution"])
                return info

            save(out / f"server-before-{run_id}.json", check_server())
            save(out / f"server-receipt-{run_id}.json", receipt)
            status("running")
            errors = 0
            for case in cases:
                qid = case["question_id"]
                attempt = out / "attempts" / f"{qid:05d}.json"
                action = pending_action(case, completed, attempt.exists())
                if action == "skip":
                    continue
                assert (
                    time.monotonic() - elapsed_start < protocol["max_run_hours"] * 3600
                ), "Run lease expired; resume remaining IDs explicitly"
                if action == "interrupted":
                    prior_attempt = json.loads(attempt.read_text())
                    assert prior_attempt["question_id"] == qid
                    assert prior_attempt["protocol_sha256"] == sha(
                        out / "protocol.json"
                    )
                    persist(
                        case,
                        failure(
                            case,
                            "INTERRUPTED: prior attempt has no durable result; not retried",
                        ),
                    )
                    status("running")
                    continue
                if action == "context_limit":
                    persist(
                        case,
                        failure(
                            case,
                            "CONTEXT_LIMIT: full prompt + declared output budget exceeds 8192",
                            context_limit=True,
                        ),
                    )
                    status("running")
                    continue
                if source_checks % 25 == 0:
                    assert fingerprint() == protocol["source_fingerprint"]
                    check_server()
                source_checks += 1
                save(
                    attempt,
                    {
                        "question_id": qid,
                        "run_id": run_id,
                        "started_unix": time.time(),
                        "protocol_sha256": sha(out / "protocol.json"),
                    },
                )
                started = time.monotonic()
                status("running", inflight_question_id=qid)
                try:
                    response = (
                        client.post(
                            "/generate",
                            json={
                                "input_ids": case["input_ids"],
                                "sampling_params": protocol["sampling"],
                                "stream": False,
                                "rid": f"v4-mmlu-full-{qid}-{run_id}",
                            },
                        )
                        .raise_for_status()
                        .json()
                    )
                    record = result_from_response(case, response, evaluator.extract)
                    errors = 0
                except (
                    httpx.HTTPError,
                    ValueError,
                    AssertionError,
                    KeyError,
                    IndexError,
                    TypeError,
                ):
                    record = failure(case, traceback.format_exc())
                    # Preserve received data even if metadata/scoring validation fails.
                    if "response" in locals():
                        record["unvalidated_response"] = response
                    errors += 1
                finally:
                    if "response" in locals():
                        del response
                record["seconds"] = time.monotonic() - started
                persist(case, record)
                status("running")
                if errors >= 3:
                    raise RuntimeError(
                        "Three consecutive request errors; stop without retry"
                    )
            assert fingerprint() == protocol["source_fingerprint"]
            save(out / "server-after.json", check_server())
        status("complete")
    except BaseException:
        status("paused_error", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-receipt", type=Path)
    options = parser.parse_args()
    if options.mode == "prepare":
        prepare(options.output)
    else:
        assert options.server_receipt is not None
        run(options.output, options.server_receipt)
