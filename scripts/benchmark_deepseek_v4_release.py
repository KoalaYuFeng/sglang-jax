"""Short same-source HTTP performance check using unmodified bench_serving.

Three cases, two warmup waves (up to four if needed), three measured waves.
Not a full stress test or a replacement for the historical ModelWorker matrix.
All raw prompts/output/logs stay in the private output directory.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_execution_options import assert_server_options
from evaluate_deepseek_v4_mmlu_full import read_jsonl, save, sha

ROOT = Path(__file__).resolve().parents[1]
CASES = ((128, 1), (1024, 1), (1024, 4))


def compilation_evidence(segment):
    """The pinned scheduler logs every nonzero miss, not explicit zeroes.

    Require real prefill observations so an empty/wrong log cannot pass.
    Both prefill and decode misses force an INFO log in the retained runtime.
    """
    assert "Prefill batch. #new-seq:" in segment, "Missing prefill log coverage"
    return [int(v) for v in re.findall(r"#cache_miss:\s*(\d+)", segment)]


def main(args):
    import httpx
    from run_deepseek_v4_http_stress import validate_idle

    assert urlparse(args.base_url).hostname in ("127.0.0.1", "localhost")
    out = args.output
    out.mkdir(mode=0o700, exist_ok=False)
    receipt = json.loads(args.server_receipt.read_text())
    actual = (
        Path(f"/proc/{receipt['pid']}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    )
    assert [v.decode() for v in actual] == receipt["command"]
    assert json.loads(args.accuracy_progress.read_text())["state"] == "complete"
    reference = json.loads(args.reference_protocol.read_text())
    data = reference["performance_dataset"]
    assert sha(data["path"]) == data["sha256"]
    bench_path = ROOT / "python/sgl_jax/bench_serving.py"
    source = fingerprint()
    assert source == receipt["framework_source_fingerprint"]
    protocol = {
        "source_fingerprint": source,
        "script_sha256": sha(__file__),
        "bench_serving_sha256": sha(bench_path),
        "compilation_evidence": "Pinned INFO logging emits every nonzero prefill/decode miss; require actual prefill log coverage before accepting an empty miss list",
        "compile_logging_source_sha256": {
            name: sha(ROOT / "python/sgl_jax/srt/managers" / name)
            for name in (
                "scheduler_output_processor_mixin.py",
                "scheduler_metrics_mixin.py",
            )
        },
        "dataset": data,
        "tokenizer": reference["tokenizer_directory"],
        "cases": [
            {"input_tokens": n, "concurrency": c, "output_tokens": 32} for n, c in CASES
        ],
        "warmups": "At least two, at most four; last warmup must be compilation-miss-free",
        "measured_rounds": 3,
        "request_count_per_round": "equal to concurrency; finite request batch, no sustained load",
        "seed": 20260911,
        "cache_policy": "flush idle prefix cache before each wave; require zero cached tokens",
        "metric_notes": "HTTP TTFT/TPOT include scheduling and transport. Input/output throughput use complete batch wall time, not prefill-only or decode-only rates.",
    }
    save(out / "protocol.json", protocol)
    save(out / "server-start.json", receipt)
    report = {
        "complete": False,
        "source_fingerprint": source,
        "waves": [],
        "summary": [],
    }
    start = time.monotonic()
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    try:
        with httpx.Client(base_url=args.base_url, timeout=30) as client:

            def check():
                assert (
                    fingerprint() == source
                    and sha(bench_path) == protocol["bench_serving_sha256"]
                )
                info = client.get("/get_server_info").raise_for_status().json()
                assert info["status"] == "ready" and validate_idle(
                    info["internal_states"], 33280
                )
                assert (
                    info["model_path"]
                    == receipt["command"][receipt["command"].index("--model-path") + 1]
                )
                assert (
                    info["context_length"] == 8192
                    and info["chunked_prefill_size"] == 128
                )
                assert (info["tp_size"], info["ep_size"], info["dp_size"]) == (4, 4, 1)
                assert info["log_level"] in ("info", "debug")
                assert_server_options(info, receipt["execution"])
                return info

            def wave(n, c, phase, repeat):
                assert time.monotonic() - start < 1800, (
                    "30-minute controller lease expired"
                )
                info = check()
                client.post("/flush_cache").raise_for_status()
                info = check()
                assert all(s["tree_cache_size"] == 0 for s in info["internal_states"])
                label = f"i{n}-o32-c{c}-{phase}-{repeat}"
                offset = args.server_log.stat().st_size
                command = [
                    sys.executable,
                    "-m",
                    "sgl_jax.bench_serving",
                    "--backend",
                    "sgl-jax",
                    "--base-url",
                    args.base_url,
                    "--model",
                    info["model_path"],
                    "--tokenizer",
                    reference["tokenizer_directory"],
                    "--served-model-name",
                    info["served_model_name"],
                    "--dataset-name",
                    "random",
                    "--dataset-path",
                    data["path"],
                    "--num-prompts",
                    str(c),
                    "--random-input-len",
                    str(n),
                    "--random-output-len",
                    "32",
                    "--random-range-ratio",
                    "1",
                    "--max-concurrency",
                    str(c),
                    "--request-rate",
                    "inf",
                    "--seed",
                    "20260911",
                    "--warmup-requests",
                    "0",
                    "--flush-cache",
                    "--tokenize-prompt",
                    "--disable-tqdm",
                    "--output-details",
                    "--output-file",
                    str(out / f"{label}.jsonl"),
                    "--tag",
                    label,
                ]
                save(out / f"{label}.command.json", command)
                print(json.dumps({"event": "wave_start", "label": label}), flush=True)
                with (out / f"{label}.log").open("x") as log:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=300,
                    )
                check()
                with args.server_log.open("rb") as log:
                    log.seek(offset)
                    segment = log.read().decode(errors="replace")
                (out / f"{label}.server.log").write_text(segment)
                misses = compilation_evidence(segment)
                result = read_jsonl(out / f"{label}.jsonl")
                assert len(result) == 1
                result = result[0]
                assert result["completed"] == c and not any(result["errors"])
                assert (
                    result["input_lens"] == [n] * c
                    and result["output_lens"] == [32] * c
                )
                assert result["total_cached_tokens"] == 0
                record = {
                    "label": label,
                    "phase": phase,
                    "repeat": repeat,
                    "input_tokens": n,
                    "concurrency": c,
                    "compile_misses": misses,
                    "result_sha256": sha(out / f"{label}.jsonl"),
                }
                for key in (
                    "duration",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                    "input_throughput",
                    "output_throughput",
                ):
                    record[key] = result[key]
                report["waves"].append(record)
                save(out / "report.json", report)
                print(json.dumps(record), flush=True)
                return record

            save(out / "server-before.json", check())
            for n, c in CASES:
                for i in range(4):
                    warmup = wave(n, c, "warmup", i)
                    if i >= 1 and not any(warmup["compile_misses"]):
                        break
                assert not any(warmup["compile_misses"]), "Warmup failed to converge"
                measured = []
                for i in range(3):
                    sample = wave(n, c, "measured", i)
                    assert not any(sample["compile_misses"]), (
                        "Compilation in measured wave"
                    )
                    measured.append(sample)
                summary = {"input_tokens": n, "output_tokens": 32, "concurrency": c}
                for key in (
                    "duration",
                    "mean_ttft_ms",
                    "mean_tpot_ms",
                    "input_throughput",
                    "output_throughput",
                ):
                    summary[key] = statistics.median(v[key] for v in measured)
                report["summary"].append(summary)
            save(out / "server-after.json", check())
        report["complete"] = True
        report["run_seconds"] = time.monotonic() - start
    except BaseException as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        save(out / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "waves"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-receipt", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--accuracy-progress", type=Path, required=True)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30126")
    main(parser.parse_args())
