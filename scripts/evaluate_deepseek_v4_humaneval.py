"""Full official HumanEval (164), Non-Think greedy pass@1 on an owned V4 server.

Original tests execute only inside a restricted, pinned Python container.
Keep benchmark content and generated code in the private experiment directory.
"""

import argparse
import ast
import fcntl
import gzip
import importlib.util
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from evaluate_deepseek_v4_mmlu_full import read_jsonl, save, sha

TOTAL = 164
REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
DATA_SHA = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
EXEC_SHA = "79901c6f5b59701c465b164aed290fa53abc35abc80a5378fc10f4dac1f9a84c"
IMAGE = "sha256:8e79d4490b856f2e823cb4d20b5a8061c867f1b79951e9a3fb7571185187ea06"
INSTRUCTION = (
    "Read the following Python function signature and docstring, and fully implement "
    "the function described. Return only Python code, including the complete function "
    "definition and any required imports or helper functions. Do not use tools.\n\n"
)


def extract_code(text, entry_point):
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if "```" in text:
        if len(fences) != 1 or text.count("```") != 2:
            raise ValueError("Expected exactly one Python code block")
        code = fences[0].strip() + "\n"
    else:
        code = text.strip() + "\n"
    tree = ast.parse(code)
    if not any(isinstance(n, ast.FunctionDef) and n.name == entry_point for n in tree.body):
        raise ValueError("Missing entry-point function")
    # Original prompt is a valid module ending in a docstring-only function.
    # A full generated definition replaces it; preserve imports/helpers verbatim.
    return "\n\n" + code


def docker_command(out, name, image=IMAGE):
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", image)
    return [
        "sudo",
        "-n",
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "512m",
        "--memory-swap",
        "512m",
        "--cpus",
        "1",
        "--shm-size",
        "16m",
        "--log-driver",
        "none",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
        "--mount",
        f"type=bind,src={out.resolve() / 'sandbox'},dst=/eval,readonly",
        "--workdir",
        "/tmp",
        "--env",
        "HOME=/tmp",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--env",
        "MKL_NUM_THREADS=1",
        image,
        "python",
        "-B",
        "/eval/runner.py",
    ]


def sandbox(out, payload, timeout=20):
    name = "v4-humaneval-" + uuid.uuid4().hex
    try:
        value = subprocess.run(
            docker_command(out, name, json.loads((out / "protocol.json").read_text())["image"]),
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["sudo", "-n", "docker", "rm", "-f", name],
            capture_output=True,
            timeout=20,
            check=False,
        )
        raise
    return json.loads(value.stdout)


def problems(out):
    assert sha(out / "HumanEval.jsonl.gz") == DATA_SHA
    rows = [
        json.loads(x)
        for x in gzip.decompress((out / "HumanEval.jsonl.gz").read_bytes()).split(b"\n")
        if x
    ]
    rows.sort(key=lambda r: int(r["task_id"].split("/")[1]))
    assert [r["task_id"] for r in rows] == [f"HumanEval/{i}" for i in range(TOTAL)]
    return rows


def load(out):
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["script_sha256"] == sha(__file__)
    assert protocol["helper_sha256"] == sha(
        Path(__file__).with_name("evaluate_deepseek_v4_mmlu_full.py")
    )
    assert sha(out / "sandbox/execution.py") == EXEC_SHA
    assert sha(out / "sandbox/runner.py") == protocol["sandbox_runner_sha256"]
    assert sha(out / "cases.jsonl") == protocol["cases_sha256"]
    cases = read_jsonl(out / "cases.jsonl")
    rows = problems(out)
    assert len(cases) == protocol["total"] == TOTAL
    assert [c["task_id"] for c in cases] == [r["task_id"] for r in rows]
    return protocol, cases, rows


def prepare(args):
    from analyze_deepseek_v4_native_profile import fingerprint
    from transformers import AutoTokenizer

    out = args.output
    assert not (out / "protocol.json").exists()
    reference = json.loads(args.reference_protocol.read_text())
    for name, digest in reference["tokenizer_sha256"].items():
        assert sha(args.tokenizer / name) == digest
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoder_path = Path(reference["checkpoint"]) / "encoding/encoding_dsv4.py"
    assert sha(encoder_path) == reference["encoder_sha256"]
    spec = importlib.util.spec_from_file_location("humaneval_official_encoder", encoder_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    cases = []
    for i, row in enumerate(problems(out)):
        ast.parse(row["prompt"])
        prompt = encoder.encode_messages(
            [{"role": "user", "content": INSTRUCTION + row["prompt"]}],
            thinking_mode="chat",
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert prompt.endswith("</think>") and len(ids) + 2048 + 2 <= 8192
        cases.append(
            {
                "index": i,
                "task_id": row["task_id"],
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
        "benchmark": "HumanEval",
        "total": TOTAL,
        "subset": False,
        "source_revision": REVISION,
        "dataset_sha256": DATA_SHA,
        "official_execution_sha256": EXEC_SHA,
        "image": args.image,
        "samples_per_task": 1,
        "metric": "greedy pass@1",
        "shots": 0,
        "thinking_mode": "chat (Non-Think)",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "ignore_eos": False},
        "context_length": 8192,
        "max_new_tokens": 2048,
        "concurrency": 1,
        "tools": False,
        "prompt_template": INSTRUCTION,
        "checkpoint": reference["checkpoint"],
        "tokenizer_sha256": reference["tokenizer_sha256"],
        "encoder_sha256": reference["encoder_sha256"],
        "source_fingerprint": fingerprint(),
        "script_sha256": sha(__file__),
        "helper_sha256": sha(Path(__file__).with_name("evaluate_deepseek_v4_mmlu_full.py")),
        "sandbox_runner_sha256": sha(out / "sandbox/runner.py"),
        "cases_sha256": sha(out / "cases.jsonl"),
        "execution_timeout_seconds": 3,
        "official_protocol_equivalence_established": False,
        "scoring": "Original tests, complete generated definition after original prompt; single fenced or plain Python. Invalid/failed/truncated wrong. No retry.",
    }
    save(out / "protocol.json", protocol)
    (out / "attempts").mkdir()
    (out / "results").mkdir()
    print(
        json.dumps(
            {
                "prepared": TOTAL,
                "prompt_min": min(c["prompt_tokens"] for c in cases),
                "prompt_max": max(c["prompt_tokens"] for c in cases),
            }
        ),
        flush=True,
    )


def preflight(out):
    protocol, _, rows = load(out)
    probe = sandbox(out, {"probe": True})
    canonical = sandbox(
        out,
        {"tasks": [{"problem": r, "completion": r["canonical_solution"]} for r in rows]},
        timeout=900,
    )
    assert len(canonical) == TOTAL and all(r["passed"] for r in canonical)
    example = {
        "task_id": "synthetic",
        "prompt": "def f():\n    pass\n",
        "entry_point": "f",
        "test": "def check(candidate):\n    assert candidate() == 42\n",
    }
    negatives = sandbox(
        out,
        {
            "tasks": [
                {"problem": example, "completion": "\ndef f():\n    return 41\n"},
                {
                    "problem": example,
                    "completion": "\ndef f():\n    while True: pass\n",
                },
            ]
        },
    )
    assert not any(r["passed"] for r in negatives)
    assert negatives[1]["result"] == "timed out"
    save(
        out / "preflight.json",
        {
            "complete": True,
            "probe": probe,
            "canonical": canonical,
            "negatives": negatives,
            "protocol_sha256": sha(out / "protocol.json"),
            "image": protocol["image"],
        },
    )
    print(
        json.dumps({"preflight": True, "canonical_passed": TOTAL, "negative_controls": 2}),
        flush=True,
    )


def audit(out):
    protocol, cases, rows = load(out)
    records, manifest = [], {}
    for case, row in zip(cases, rows):
        path = out / "results" / f"{case['index']:03d}.json"
        r = json.loads(path.read_text())
        assert r["task_id"] == case["task_id"]
        assert r["protocol_sha256"] == sha(out / "protocol.json")
        if not r["failed"]:
            meta = r["response"]["meta_info"]
            assert meta["prompt_tokens"] == case["prompt_tokens"]
            assert 0 < meta["completion_tokens"] <= 2048
            assert r["truncated"] == (meta["finish_reason"]["type"] == "length")
            try:
                code = extract_code(r["response"]["text"], row["entry_point"])
            except (SyntaxError, ValueError):
                assert r["invalid"] and not r["correct"]
            else:
                assert not r["invalid"] and r["completion"] == code
                if not r["truncated"]:
                    assert r["execution"]["task_id"] == row["task_id"]
                    assert r["correct"] == r["execution"]["passed"]
        assert not r["correct"] or not any(r[k] for k in ("failed", "invalid", "truncated"))
        records.append(r)
        manifest[str(path.relative_to(out))] = sha(path)
    report = {
        "complete": True,
        "total": TOTAL,
        "subset": False,
        "metric": "greedy pass@1",
        "protocol_sha256": sha(out / "protocol.json"),
        "source_fingerprint": protocol["source_fingerprint"],
    }
    for key in ("correct", "failed", "invalid", "truncated"):
        report[key] = sum(r[key] for r in records)
    report["pass_at_1_percent"] = report["correct"] / TOTAL * 100
    save(out / "audit-manifest.json", manifest)
    save(out / "audit.json", report)
    print(json.dumps(report), flush=True)


def run(args):
    import httpx
    from analyze_deepseek_v4_native_profile import fingerprint
    from deepseek_v4_execution_options import assert_server_options
    from run_deepseek_v4_http_stress import validate_idle

    out = args.output
    lock = (out / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol, cases, rows = load(out)
    pre = json.loads((out / "preflight.json").read_text())
    assert pre["complete"] and pre["protocol_sha256"] == sha(out / "protocol.json")
    receipt = json.loads(args.server_receipt.read_text())
    command = Path(f"/proc/{receipt['pid']}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    assert [x.decode() for x in command] == receipt["command"]
    assert (
        fingerprint() == protocol["source_fingerprint"] == receipt["framework_source_fingerprint"]
    )
    assert not list((out / "attempts").glob("*.json"))
    save(out / "server-start.json", receipt)
    records, start, run_id = [], time.monotonic(), str(time.time_ns())

    def status(state, **extra):
        value = dict(
            state=state,
            complete=state == "complete",
            total=TOTAL,
            completed=len(records),
            pid=os.getpid(),
            updated_unix=time.time(),
            run_seconds=time.monotonic() - start,
            **extra,
        )
        for key in ("correct", "failed", "invalid", "truncated"):
            value[key] = sum(r[key] for r in records)
        save(out / "progress.json", value)
        print(json.dumps(value), flush=True)

    try:
        with httpx.Client(base_url="http://127.0.0.1:30126", timeout=600) as client:

            def check_server():
                info = client.get("/get_server_info", timeout=20).raise_for_status().json()
                assert info["status"] == "ready" and validate_idle(info["internal_states"], 33280)
                assert (
                    info["model_path"] == protocol["checkpoint"] and info["context_length"] == 8192
                )
                assert (info["tp_size"], info["ep_size"], info["dp_size"]) == (4, 4, 1)
                assert info["chunked_prefill_size"] == 128
                assert_server_options(info, receipt["execution"])
                return info

            save(out / "server-before.json", check_server())
            errors = 0
            status("running")
            for case, row in zip(cases, rows):
                assert time.monotonic() - start < 10800, "Three-hour lease expired"
                assert fingerprint() == protocol["source_fingerprint"]
                if case["index"] % 32 == 0:
                    check_server()
                rid = f"v4-humaneval-{run_id}-{case['index']}"
                save(
                    out / "attempts" / f"{case['index']:03d}.json",
                    {
                        "index": case["index"],
                        "rid": rid,
                        "protocol_sha256": sha(out / "protocol.json"),
                    },
                )
                r = {
                    "index": case["index"],
                    "task_id": case["task_id"],
                    "correct": False,
                    "failed": False,
                    "invalid": False,
                    "truncated": False,
                    "protocol_sha256": sha(out / "protocol.json"),
                }
                begin = time.monotonic()
                status("running", inflight_index=case["index"])
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
                                    "max_new_tokens": 2048,
                                },
                            },
                        )
                        .raise_for_status()
                        .json()
                    )
                    r["response"] = response
                    meta = response["meta_info"]
                    assert meta["prompt_tokens"] == case["prompt_tokens"]
                    assert 0 < meta["completion_tokens"] <= 2048
                    assert meta["finish_reason"]["type"] in ("stop", "length")
                    r["truncated"] = meta["finish_reason"]["type"] == "length"
                    try:
                        r["completion"] = extract_code(response["text"], row["entry_point"])
                    except (SyntaxError, ValueError):
                        r["invalid"] = True
                    if not r["invalid"] and not r["truncated"]:
                        result = sandbox(
                            out,
                            {"tasks": [{"problem": row, "completion": r["completion"]}]},
                        )
                        assert len(result) == 1 and result[0]["task_id"] == row["task_id"]
                        r["execution"] = result[0]
                        r["correct"] = result[0]["passed"]
                    errors = 0
                except (
                    httpx.HTTPError,
                    AssertionError,
                    KeyError,
                    ValueError,
                    TypeError,
                    OSError,
                    subprocess.SubprocessError,
                ) as error:
                    r.update(failed=True, correct=False, error_type=type(error).__name__)
                    errors += 1
                r["seconds"] = time.monotonic() - begin
                save(out / "results" / f"{case['index']:03d}.json", r)
                records.append(r)
                status("running")
                assert (
                    errors < 3
                ), "Three consecutive infrastructure failures; stopped without retry"
            save(out / "server-after.json", check_server())
        audit(out)
        status("complete")
    except BaseException as error:
        status("paused_error", error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "run", "audit"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-protocol", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--server-receipt", type=Path)
    parser.add_argument(
        "--image",
        default=IMAGE,
        help="Immutable local Docker image ID from the supplied Dockerfile",
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "preflight":
        preflight(args.output)
    elif args.mode == "run":
        run(args)
    else:
        audit(args.output)
