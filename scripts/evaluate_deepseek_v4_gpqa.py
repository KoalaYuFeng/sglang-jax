"""Private-data GPQA Diamond Non-Think evaluation; no serving changes.

All 198 questions, one generation each, fixed option shuffle, temperature 1.0,
8K total context. This is not a claim of exact official-harness reproduction.
Do not publish question text, options, explanations or generated responses.
"""

import argparse
import csv
import fcntl
import importlib.util
import json
import os
import random
import re
import time
import traceback
from pathlib import Path

from evaluate_deepseek_v4_mmlu_full import read_jsonl, save, sha

LETTERS = "ABCD"
INSTRUCTION = (
    "Solve the following multiple-choice science question. Explain your reasoning, "
    "then finish with a separate line in the format Answer: X, "
    "where X is exactly one of A, B, C, D.\n\n"
)


def question_content(row, index):
    values = [row["Correct Answer"]] + [row[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
    assert all(isinstance(v, str) and v.strip() for v in values)
    order = list(range(4))
    random.Random(20260911 + index).shuffle(order)
    content = INSTRUCTION + row["Question"] + "\n\n"
    content += "\n".join(f"{LETTERS[i]}. {values[j]}" for i, j in enumerate(order))
    return content, LETTERS[order.index(0)], order


def extract(text):
    text = text.replace("**", "")
    matches = re.findall(
        r"(?im)^\s*(?:the\s+)?(?:correct\s+|final\s+)?answer\s*(?::|is)\s*\(?([A-D])\)?(?:[.\s]|$)",
        text,
    )
    if matches:
        matches = [m.upper() for m in matches]
        return matches[-1], "explicit_final", len(set(matches)) > 1
    matches = re.findall(r"\\boxed\{\s*\(?([A-D])\)?\s*\}", text)
    if matches:
        return matches[-1], "boxed", len(set(matches)) > 1
    match = re.fullmatch(r"\s*\(?([A-D])\)?\.?\s*", text)
    return (match[1], "letter_only", False) if match else (None, "invalid", False)


def score(case, response):
    prediction, tier, conflict = extract(response["text"])
    meta = response["meta_info"]
    assert meta["prompt_tokens"] == case["prompt_tokens"]
    truncated = meta["finish_reason"].get("type") == "length"
    return {
        "prediction": prediction,
        "extraction_tier": tier,
        "conflicting_explicit_answers": conflict,
        "invalid": prediction is None,
        "truncated": truncated,
        "failed": False,
        "correct": prediction == case["gold"] and not truncated,
    }


def prepare(out):
    from analyze_deepseek_v4_native_profile import fingerprint
    from transformers import AutoTokenizer

    assert not (out / "protocol.json").exists()
    source = json.loads((out / "dataset-source.json").read_text())
    assert sha(out / "gpqa_diamond.csv") == source["csv_sha256"]
    with (out / "gpqa_diamond.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 198 and len({r["Record ID"] for r in rows}) == 198
    old = json.loads(
        (out.parent / "v4-mmlu-accuracy-20260911-01/protocol.json").read_text()
    )
    token_dir = out.parent / "v4-standard-bench-20260910-01/client-tokenizer"
    for name, digest in old["tokenizer_sha256"].items():
        assert sha(token_dir / name) == digest
    tokenizer = AutoTokenizer.from_pretrained(token_dir, local_files_only=True)
    encoder_path = Path(old["checkpoint"]) / "encoding/encoding_dsv4.py"
    assert sha(encoder_path) == old["encoder_sha256"]
    spec = importlib.util.spec_from_file_location("gpqa_v4_encoder", encoder_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    cases = []
    for i, row in enumerate(rows):
        content, gold, order = question_content(row, i)
        prompt = encoder.encode_messages(
            [{"role": "user", "content": content}], thinking_mode="chat"
        )
        assert prompt.endswith("</think>")
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        budget = 8192 - len(ids) - 2
        assert budget > 0 and len(ids) < 8186
        cases.append(
            {
                "index": i,
                "record_id": row["Record ID"],
                "domain": row["High-level domain"],
                "gold": gold,
                "choice_order": order,
                "prompt": prompt,
                "input_ids": ids,
                "prompt_tokens": len(ids),
                "max_new_tokens": budget,
            }
        )
    with (out / "cases.jsonl").open("x") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    protocol = {
        "dataset": "GPQA Diamond",
        "total": 198,
        "shots": 0,
        "samples_per_question": 1,
        "dataset_source": source,
        "thinking_mode": "chat (Non-Think)",
        "sampling": {"temperature": 1.0, "top_p": 1.0, "ignore_eos": False},
        "output_budget": "8192 - prompt_tokens - 2; EOS allowed; no shortened question or output-only cap",
        "context_length": 8192,
        "concurrency": 1,
        "option_shuffle_seed_base": 20260911,
        "prompt_tokens_min": min(c["prompt_tokens"] for c in cases),
        "prompt_tokens_max": max(c["prompt_tokens"] for c in cases),
        "cases_sha256": sha(out / "cases.jsonl"),
        "script_sha256": sha(__file__),
        "helper_sha256": sha(
            Path(__file__).with_name("evaluate_deepseek_v4_mmlu_full.py")
        ),
        "checkpoint": old["checkpoint"],
        "source_fingerprint": fingerprint(),
        "tokenizer_sha256": old["tokenizer_sha256"],
        "encoder_sha256": old["encoder_sha256"],
        "prompt_template": INSTRUCTION,
        "scoring": "Last explicit answer line, then boxed, then bare letter; no guessing. Failed/invalid/truncated count wrong in denominator 198.",
        "resume": "Completed records skipped; unknown in-flight attempts recorded as interrupted, never retried automatically",
        "max_run_hours": 4,
        "official_reference_percent": 71.2,
        "limitations": [
            "Official temperature and Non-Think context matched, but exact official GPQA prompt, option shuffle and repeat count not established",
            "Single stochastic sample per question, not an official-runtime same-input A/B",
            "Server random seed recorded; deterministic per-request sampling is not enabled",
            "Keep dataset examples and responses private",
        ],
    }
    save(out / "protocol.json", protocol)
    (out / "results").mkdir()
    (out / "attempts").mkdir()
    print(
        json.dumps(
            {
                k: protocol[k]
                for k in (
                    "total",
                    "prompt_tokens_min",
                    "prompt_tokens_max",
                    "source_fingerprint",
                    "script_sha256",
                )
            }
        ),
        flush=True,
    )


def run(out):
    import httpx
    from analyze_deepseek_v4_native_profile import fingerprint
    from deepseek_v4_execution_options import assert_server_options
    from run_deepseek_v4_http_stress import validate_idle

    lock = (out / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = json.loads((out / "protocol.json").read_text())
    assert p["script_sha256"] == sha(__file__)
    assert p["helper_sha256"] == sha(
        Path(__file__).with_name("evaluate_deepseek_v4_mmlu_full.py")
    )
    assert p["cases_sha256"] == sha(out / "cases.jsonl")
    assert fingerprint() == p["source_fingerprint"]
    receipt = json.loads(
        (out.parent / "v4-endtoend-accuracy-20260911-01/server_start.json").read_text()
    )
    assert receipt["framework_source_fingerprint"] == fingerprint()
    actual = (
        Path(f"/proc/{receipt['pid']}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    )
    assert [v.decode() for v in actual] == receipt["command"]
    cases = read_jsonl(out / "cases.jsonl")
    records = {}
    for path in (out / "results").glob("*.json"):
        r = json.loads(path.read_text())
        assert r["protocol_sha256"] == sha(out / "protocol.json")
        if "response" in r:
            assert all(
                r[k] == v for k, v in score(cases[r["index"]], r["response"]).items()
            )
        records[r["index"]] = r
    started = time.monotonic()
    run_id = str(time.time_ns())

    def status(state, **extra):
        result = {
            "state": state,
            "complete": state == "complete",
            "total": 198,
            "completed": len(records),
            "correct": sum(r["correct"] for r in records.values()),
            "failed": sum(r["failed"] for r in records.values()),
            "invalid": sum(r["invalid"] for r in records.values()),
            "truncated": sum(r["truncated"] for r in records.values()),
            "pid": os.getpid(),
            "run_id": run_id,
            "updated_unix": time.time(),
            "run_seconds": time.monotonic() - started,
            "protocol_sha256": sha(out / "protocol.json"),
            **extra,
        }
        save(out / "progress.json", result)
        print(json.dumps(result), flush=True)

    try:
        with httpx.Client(base_url="http://127.0.0.1:30126", timeout=1200) as client:

            def check_server():
                info = (
                    client.get("/get_server_info", timeout=15).raise_for_status().json()
                )
                assert info["status"] == "ready" and validate_idle(
                    info["internal_states"], 33280
                )
                assert (
                    info["context_length"] == 8192
                    and info["model_path"] == p["checkpoint"]
                )
                assert (info["tp_size"], info["ep_size"], info["dp_size"]) == (4, 4, 1)
                assert info["chunked_prefill_size"] == 128
                assert_server_options(info, receipt["execution"])
                return info

            save(out / f"server-before-{run_id}.json", check_server())
            status("running")
            consecutive_errors = 0
            for case in cases:
                i = case["index"]
                if i in records:
                    continue
                assert time.monotonic() - started < p["max_run_hours"] * 3600
                assert fingerprint() == p["source_fingerprint"]
                attempt = out / "attempts" / f"{i:03d}.json"
                r = {
                    "index": i,
                    "gold": case["gold"],
                    "domain": case["domain"],
                    "protocol_sha256": sha(out / "protocol.json"),
                }
                if attempt.exists():
                    assert json.loads(attempt.read_text())["protocol_sha256"] == sha(
                        out / "protocol.json"
                    )
                    r.update(
                        failed=True,
                        invalid=True,
                        truncated=False,
                        correct=False,
                        error="Interrupted prior attempt; no automatic retry",
                    )
                else:
                    if i % 25 == 0:
                        check_server()
                    rid = f"v4-gpqa-nonthink-{i}-{run_id}"
                    save(
                        attempt,
                        {
                            "index": i,
                            "rid": rid,
                            "started_unix": time.time(),
                            "protocol_sha256": sha(out / "protocol.json"),
                        },
                    )
                    status("running", inflight_index=i)
                    begin = time.monotonic()
                    try:
                        response = (
                            client.post(
                                "/generate",
                                json={
                                    "input_ids": case["input_ids"],
                                    "sampling_params": {
                                        **p["sampling"],
                                        "max_new_tokens": case["max_new_tokens"],
                                    },
                                    "stream": False,
                                    "rid": rid,
                                },
                            )
                            .raise_for_status()
                            .json()
                        )
                        r["response"] = response
                        r.update(score(case, response))
                        consecutive_errors = 0
                    except (
                        httpx.HTTPError,
                        AssertionError,
                        ValueError,
                        KeyError,
                        TypeError,
                    ):
                        if "response" in r:
                            r["unvalidated_response"] = r.pop("response")
                        r.update(
                            failed=True,
                            invalid=True,
                            truncated=False,
                            correct=False,
                            error=traceback.format_exc(),
                        )
                        consecutive_errors += 1
                    r["seconds"] = time.monotonic() - begin
                save(out / "results" / f"{i:03d}.json", r)
                records[i] = r
                status("running")
                if consecutive_errors >= 3:
                    raise RuntimeError(
                        "Three consecutive errors; stopped without retry"
                    )
            assert fingerprint() == p["source_fingerprint"]
            save(out / "server-after.json", check_server())
        status("complete")
        audit(out)
    except BaseException:
        status("paused_error", error=traceback.format_exc())
        raise


def audit(out):
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["script_sha256"] == sha(__file__)
    assert protocol["cases_sha256"] == sha(out / "cases.jsonl")
    cases = read_jsonl(out / "cases.jsonl")
    records, manifest = [], {}
    for i, case in enumerate(cases):
        path = out / "results" / f"{i:03d}.json"
        r = json.loads(path.read_text())
        assert r["index"] == i and r["gold"] == case["gold"]
        assert r["protocol_sha256"] == sha(out / "protocol.json")
        if "response" in r:
            assert all(r[k] == v for k, v in score(case, r["response"]).items())
        else:
            assert r["failed"] and not r["correct"]
        records.append(r)
        manifest[str(path.relative_to(out))] = sha(path)
    assert len(records) == 198
    report = {
        "complete": True,
        "correct": sum(r["correct"] for r in records),
        "total": 198,
        "failed": sum(r["failed"] for r in records),
        "invalid": sum(r["invalid"] for r in records),
        "truncated": sum(r["truncated"] for r in records),
        "conflicting_explicit_answers": sum(
            r.get("conflicting_explicit_answers", False) for r in records
        ),
        "by_domain": {
            d: {
                "correct": sum(r["correct"] for r in records if r["domain"] == d),
                "total": sum(r["domain"] == d for r in records),
            }
            for d in sorted({r["domain"] for r in records})
        },
        "protocol_sha256": sha(out / "protocol.json"),
        "official_reference_percent": 71.2,
        "official_protocol_equivalence_established": False,
    }
    report["accuracy_percent"] = 100 * report["correct"] / 198
    save(out / "audit-manifest.json", manifest)
    save(out / "audit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "audit": audit}[args.mode](args.output)
