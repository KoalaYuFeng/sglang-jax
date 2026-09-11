"""Frozen GSM8K Non-Think evaluation: full test or legacy 128-row smoke test.

Keep dataset, prompts and responses in a private output directory. No tools,
calculator, score-dependent retry, runtime changes or server restart are used.
"""

import argparse
import fcntl
import importlib.util
import json
import os
import random
import re
import time
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

from evaluate_deepseek_v4_mmlu_full import read_jsonl, save, sha

TOTAL = 128
SEED = 20260911
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
INSTRUCTION = (
    "Solve this math problem. Show your calculation briefly, then end with a "
    "separate line '#### <number>' containing only the final numeric answer. "
    "Do not use tools.\n\n"
)


def normalize(value):
    if not re.fullmatch(NUMBER, value):
        raise ValueError("Not a plain decimal number")
    number = Decimal(value.replace(",", "")).normalize()
    return "0" if number == 0 else str(number)


def extract(text):
    matches = re.findall(rf"(?m)^\s*####\s*\$?({NUMBER})\s*$", text.replace("**", ""))
    values = [normalize(value) for value in matches]
    return (values[-1] if values else None), len(set(values)) > 1


def score(case, response):
    prediction, conflict = extract(response["text"])
    meta = response["meta_info"]
    assert meta["prompt_tokens"] == case["prompt_tokens"]
    assert 0 < meta["completion_tokens"] <= case["max_new_tokens"]
    assert meta["finish_reason"]["type"] in ("stop", "length")
    truncated = meta["finish_reason"]["type"] == "length"
    return {
        "prediction": prediction,
        "correct": prediction == case["gold"] and not truncated,
        "invalid": prediction is None,
        "truncated": truncated,
        "failed": False,
        "conflicting_final_numbers": conflict,
    }


def select_indices(size, full_test=False):
    if size != 1319:
        raise ValueError("Expected the complete original 1319-row test split")
    return list(range(size)) if full_test else sorted(random.Random(SEED).sample(range(size), TOTAL))


def prepare(args):
    from analyze_deepseek_v4_native_profile import fingerprint
    from transformers import AutoTokenizer

    out = args.output
    assert not (out / "protocol.json").exists()
    reference = json.loads(args.reference_protocol.read_text())
    source = json.loads((out / "dataset-source.json").read_text())
    assert sha(out / "test.jsonl") == source["sha256"]
    rows = read_jsonl(out / "test.jsonl")
    indices = select_indices(len(rows), args.full_test)
    for name, digest in reference["tokenizer_sha256"].items():
        assert sha(args.tokenizer / name) == digest
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoder_path = Path(reference["checkpoint"]) / "encoding/encoding_dsv4.py"
    assert sha(encoder_path) == reference["encoder_sha256"]
    spec = importlib.util.spec_from_file_location(
        "gsm8k_official_encoder", encoder_path
    )
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    cases = []
    for index, source_index in enumerate(indices):
        row = rows[source_index]
        gold, conflict = extract(row["answer"])
        assert gold is not None and not conflict
        # The dataset solution is never included in the user prompt.
        content = INSTRUCTION + row["question"]
        prompt = encoder.encode_messages(
            [{"role": "user", "content": content}], thinking_mode="chat"
        )
        assert prompt.endswith("</think>")
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(ids) + 2048 + 2 <= 8192
        cases.append(
            {
                "index": index,
                "source_index": source_index,
                "gold": gold,
                "prompt": prompt,
                "input_ids": ids,
                "prompt_tokens": len(ids),
                "max_new_tokens": 2048,
            }
        )
    with (out / "cases.jsonl").open("x") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    protocol = {
        "dataset": "GSM8K original test",
        "dataset_source": source,
        "total": len(indices),
        "full_test_size": 1319,
        "subset": not args.full_test,
        "subset_seed": None if args.full_test else SEED,
        "source_indices": indices,
        "shots": 0,
        "samples_per_question": 1,
        "thinking_mode": "chat (Non-Think)",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "ignore_eos": False},
        "context_length": 8192,
        "max_new_tokens": 2048,
        "concurrency": 1,
        "prompt_template": INSTRUCTION,
        "tools": False,
        "scoring": "Last explicit #### decimal; exact Decimal equality. Invalid/failed/truncated count wrong. No retry.",
        "checkpoint": reference["checkpoint"],
        "tokenizer_sha256": reference["tokenizer_sha256"],
        "encoder_sha256": reference["encoder_sha256"],
        "source_fingerprint": fingerprint(),
        "script_sha256": sha(__file__),
        "helper_sha256": sha(
            Path(__file__).with_name("evaluate_deepseek_v4_mmlu_full.py")
        ),
        "cases_sha256": sha(out / "cases.jsonl"),
        "official_protocol_equivalence_established": False,
        "limitation": "Custom zero-shot greedy protocol, not a matched official Instruct or Base protocol. Coverage is recorded separately by subset/total.",
    }
    save(out / "protocol.json", protocol)
    (out / "results").mkdir()
    (out / "attempts").mkdir()
    print(
        json.dumps(
            {
                "prepared": len(indices),
                "prompt_min": min(c["prompt_tokens"] for c in cases),
                "prompt_max": max(c["prompt_tokens"] for c in cases),
                "protocol_sha256": sha(out / "protocol.json"),
            }
        ),
        flush=True,
    )


def load_protocol(out):
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["script_sha256"] == sha(__file__)
    assert protocol["helper_sha256"] == sha(
        Path(__file__).with_name("evaluate_deepseek_v4_mmlu_full.py")
    )
    assert protocol["cases_sha256"] == sha(out / "cases.jsonl")
    cases = read_jsonl(out / "cases.jsonl")
    full_test = not protocol["subset"]
    assert len(cases) == protocol["total"] == (1319 if full_test else TOTAL)
    assert (
        [c["source_index"] for c in cases]
        == select_indices(1319, full_test)
        == protocol["source_indices"]
    )
    return protocol, cases


def audit(out):
    protocol, cases = load_protocol(out)
    records, manifest = [], {}
    for i, case in enumerate(cases):
        path = out / "results" / f"{i:03d}.json"
        record = json.loads(path.read_text())
        assert record["index"] == case["index"] == i
        assert record["source_index"] == case["source_index"]
        assert record["gold"] == case["gold"]
        assert record["protocol_sha256"] == sha(out / "protocol.json")
        if "response" in record:
            assert all(
                record[k] == v for k, v in score(case, record["response"]).items()
            )
        else:
            assert record["failed"] and not record["correct"]
        records.append(record)
        manifest[str(path.relative_to(out))] = sha(path)
    report = {
        "complete": True,
        "total": protocol["total"],
        "subset": protocol["subset"],
        "full_test_size": 1319,
        "protocol_sha256": sha(out / "protocol.json"),
        "official_protocol_equivalence_established": False,
        "source_fingerprint": protocol["source_fingerprint"],
    }
    for key in (
        "correct",
        "failed",
        "invalid",
        "truncated",
        "conflicting_final_numbers",
    ):
        report[key] = sum(r.get(key, False) for r in records)
    report["accuracy_percent"] = report["correct"] / protocol["total"] * 100
    save(out / "audit-manifest.json", manifest)
    save(out / "audit.json", report)
    print(json.dumps(report), flush=True)


def run(args):
    import httpx
    from analyze_deepseek_v4_native_profile import fingerprint
    from deepseek_v4_execution_options import assert_server_options
    from run_deepseek_v4_http_stress import validate_idle

    assert urlparse(args.base_url).hostname in ("127.0.0.1", "localhost")
    out = args.output
    lock = (out / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol, cases = load_protocol(out)
    receipt = json.loads(args.server_receipt.read_text())
    actual = (
        Path(f"/proc/{receipt['pid']}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    )
    assert [v.decode() for v in actual] == receipt["command"]
    assert (
        fingerprint()
        == protocol["source_fingerprint"]
        == receipt["framework_source_fingerprint"]
    )
    save(out / "server-start.json", receipt)
    # Fresh run only. A failed/interrupted experiment must not silently resample.
    assert not list((out / "attempts").glob("*.json"))
    records = []
    run_id, start = str(time.time_ns()), time.monotonic()

    def status(state, **extra):
        value = dict(
            state=state,
            complete=state == "complete",
            total=protocol["total"],
            completed=len(records),
            pid=os.getpid(),
            run_id=run_id,
            updated_unix=time.time(),
            run_seconds=time.monotonic() - start,
            protocol_sha256=sha(out / "protocol.json"),
            **extra,
        )
        for key in ("correct", "failed", "invalid", "truncated"):
            value[key] = sum(r[key] for r in records)
        save(out / "progress.json", value)
        print(json.dumps(value), flush=True)

    try:
        with httpx.Client(base_url=args.base_url, timeout=600) as client:

            def check_server():
                info = (
                    client.get("/get_server_info", timeout=20).raise_for_status().json()
                )
                assert info["status"] == "ready" and validate_idle(
                    info["internal_states"], 33280
                )
                assert (
                    info["model_path"] == protocol["checkpoint"]
                    and info["context_length"] == 8192
                )
                assert (info["tp_size"], info["ep_size"], info["dp_size"]) == (4, 4, 1)
                assert info["chunked_prefill_size"] == 128
                assert_server_options(info, receipt["execution"])
                return info

            save(out / "server-before.json", check_server())
            errors = 0
            status("running")
            for case in cases:
                assert time.monotonic() - start < 14400, "Four-hour client lease expired"
                assert fingerprint() == protocol["source_fingerprint"]
                i = case["index"]
                if i % 32 == 0:
                    check_server()
                rid = f"v4-gsm8k-{protocol['total']}-{run_id}-{i}"
                save(
                    out / "attempts" / f"{i:03d}.json",
                    {
                        "index": i,
                        "rid": rid,
                        "started_unix": time.time(),
                        "protocol_sha256": sha(out / "protocol.json"),
                    },
                )
                status("running", inflight_index=i)
                record = {k: case[k] for k in ("index", "source_index", "gold")}
                record["protocol_sha256"] = sha(out / "protocol.json")
                begin = time.monotonic()
                try:
                    response = (
                        client.post(
                            "/generate",
                            json={
                                "input_ids": case["input_ids"],
                                "stream": False,
                                "rid": rid,
                                "sampling_params": {
                                    **protocol["sampling"],
                                    "max_new_tokens": case["max_new_tokens"],
                                },
                            },
                        )
                        .raise_for_status()
                        .json()
                    )
                    record["response"] = response
                    checked = score(case, response)
                    record.update(response=response, **checked)
                    errors = 0
                except (
                    httpx.HTTPError,
                    AssertionError,
                    KeyError,
                    ValueError,
                    TypeError,
                ) as error:
                    if "response" in record:
                        record["unvalidated_response"] = record.pop("response")
                    record.update(
                        failed=True,
                        correct=False,
                        invalid=True,
                        truncated=False,
                        error_type=type(error).__name__,
                    )
                    errors += 1
                record["seconds"] = time.monotonic() - begin
                save(out / "results" / f"{i:03d}.json", record)
                records.append(record)
                status("running")
                assert errors < 3, "Three consecutive failures; stopped without retry"
            assert fingerprint() == protocol["source_fingerprint"]
            save(out / "server-after.json", check_server())
        audit(out)
        status("complete")
    except BaseException as error:
        status("paused_error", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-protocol", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--full-test", action="store_true", help="Prepare all 1319 original test questions")
    parser.add_argument("--server-receipt", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30126")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        run(args)
    else:
        audit(args.output)
