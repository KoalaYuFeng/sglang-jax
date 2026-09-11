"""Offline audit of full-test records, including partial local backups."""

import argparse
import importlib.util
import json
from pathlib import Path

from evaluate_deepseek_v4_mmlu_full import result_from_response, save, sha, summarize


def audit(out, allow_partial=False):
    protocol = json.loads((out / "protocol.json").read_text())
    assert sha(out / "cases.jsonl") == protocol["cases_sha256"]
    source = out / "source/frozen-evaluate.py"
    assert sha(source) == protocol["evaluator_sha256"]
    assert (
        sha(out / "source/evaluate_deepseek_v4_mmlu_full.py")
        == protocol["script_sha256"]
    )
    spec = importlib.util.spec_from_file_location("frozen_mmlu_audit", source)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    indexed = {}
    with (out / "cases.jsonl").open() as f:
        for line in f:
            raw = json.loads(line)
            assert raw["question_id"] not in indexed
            indexed[raw["question_id"]] = {
                k: raw[k]
                for k in (
                    "question_id",
                    "gold",
                    "category",
                    "raw_test",
                    "prompt_tokens",
                )
            }
    assert len(indexed) == protocol["total"] == 12032
    records = []
    manifest = {}
    for path in sorted((out / "results").glob("*.json")):
        result = json.loads(path.read_text())
        assert result["protocol_sha256"] == sha(out / "protocol.json")
        assert path.stem == f"{result['question_id']:05d}"
        case = indexed[result["question_id"]]
        assert result["gold"] == case["gold"] and result["category"] == case["category"]
        if "response" in result:
            recalculated = result_from_response(
                case, result["response"], evaluator.extract
            )
            assert all(result[k] == recalculated[k] for k in recalculated)
        else:
            assert result["failed"] and not result["correct"] and result["error"]
        manifest[str(path.relative_to(out))] = sha(path)
        records.append({k: v for k, v in result.items() if k != "response"})
    report = summarize(records, protocol["total"])
    if not allow_partial:
        assert report["complete"]
        assert json.loads((out / "progress.json").read_text())["state"] == "complete"
    report["protocol_sha256"] = sha(out / "protocol.json")
    report["audit_script_sha256"] = sha(__file__)
    report["responses_audited"] = len(records)
    report["macro_category_accuracy"] = (
        sum(
            report["by_category"][c]["correct"] / total
            for c, total in protocol["category_totals"].items()
        )
        / 14
        if report["complete"]
        else None
    )
    report["interpretation"] = (
        "Full fixed-protocol deployment score only when complete; not official score equivalence or cause attribution."
    )
    for name in (
        "cases.jsonl",
        "protocol.json",
        "source/frozen-evaluate.py",
        "source/evaluate_deepseek_v4_mmlu_full.py",
    ):
        manifest[name] = sha(out / name)
    stem = "audit" if report["complete"] else "audit-partial"
    save(out / f"{stem}-manifest.json", manifest)
    save(out / f"{stem}.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    options = parser.parse_args()
    audit(options.output, options.allow_partial)
