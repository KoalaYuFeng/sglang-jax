"""Offline recheck of saved short HTTP benchmark metrics and evidence."""

import argparse
import json
import math
import statistics
from pathlib import Path

from benchmark_deepseek_v4_release import compilation_evidence
from evaluate_deepseek_v4_mmlu_full import read_jsonl, save, sha

METRICS = (
    "duration",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "input_throughput",
    "output_throughput",
)


def validate_metrics(result, n, c):
    assert result["completed"] == c and not any(result["errors"])
    assert result["input_lens"] == [n] * c and result["output_lens"] == [32] * c
    assert result["total_cached_tokens"] == 0 and result["cached_tokens"] == [0] * c
    assert (
        result["total_input_tokens"] == n * c
        and result["total_output_tokens"] == 32 * c
    )
    assert len(result["ttfts"]) == c
    expected = {
        "mean_ttft_ms": statistics.mean(result["ttfts"]) * 1000,
        "input_throughput": n * c / result["duration"],
        "output_throughput": 32 * c / result["duration"],
        # All completions have length 32, so averaging commutes with /31.
        "mean_tpot_ms": (result["mean_e2e_latency_ms"] - result["mean_ttft_ms"]) / 31,
    }
    for key in METRICS:
        assert math.isfinite(result[key]) and result[key] > 0
    for key, value in expected.items():
        assert math.isclose(result[key], value, rel_tol=1e-10, abs_tol=1e-6), key


def audit(out):
    protocol = json.loads((out / "protocol.json").read_text())
    report = json.loads((out / "report.json").read_text())
    assert report["complete"]
    assert report["source_fingerprint"] == protocol["source_fingerprint"]
    manifest = {}
    for wave in report["waves"]:
        label = wave["label"]
        assert Path(label).name == label
        path = out / f"{label}.jsonl"
        assert sha(path) == wave["result_sha256"]
        rows = read_jsonl(path)
        assert len(rows) == 1
        result = rows[0]
        validate_metrics(result, wave["input_tokens"], wave["concurrency"])
        for key in METRICS:
            assert result[key] == wave[key]
        log = out / f"{label}.server.log"
        assert compilation_evidence(log.read_text()) == wave["compile_misses"]
        manifest[path.name] = sha(path)
        manifest[log.name] = sha(log)
        if wave["phase"] == "measured":
            assert not any(wave["compile_misses"])
    assert len(report["summary"]) == len(protocol["cases"]) == 3
    for summary, case in zip(report["summary"], protocol["cases"], strict=True):
        assert all(summary[k] == v for k, v in case.items())
        waves = [
            w
            for w in report["waves"]
            if w["input_tokens"] == case["input_tokens"]
            and w["concurrency"] == case["concurrency"]
        ]
        measured = [w for w in waves if w["phase"] == "measured"]
        warmups = [w for w in waves if w["phase"] == "warmup"]
        assert len(measured) == 3 and 2 <= len(warmups) <= 4
        assert not any(warmups[-1]["compile_misses"])
        for key in METRICS:
            assert summary[key] == statistics.median(w[key] for w in measured)
    result = {
        "complete": True,
        "measured_waves": 9,
        "summary": report["summary"],
        "source_fingerprint": report["source_fingerprint"],
        "protocol_sha256": sha(out / "protocol.json"),
        "report_sha256": sha(out / "report.json"),
        "metric_note": "TTFT recomputed from per-request values; TPOT checked against saved mean E2E for uniform 32-token outputs; throughput and medians recomputed.",
    }
    save(out / "audit-manifest.json", manifest)
    save(out / "audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
