"""Audit full-model CPU/HTTP probabilities and frozen generation accuracy."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from analyze_deepseek_v4_native_profile import fingerprint
from deepseek_v4_endtoend_metrics import compare_distributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    fixture = json.loads((out / "fixture.json").read_text())
    cpu = json.loads((out / "cpu-reference/report.json").read_text())
    http = json.loads((out / "http-evaluation/report.json").read_text())
    assert (
        cpu["complete"]
        and cpu["full_model"]
        and len(cpu["layers"]) == 43
        and http["complete"]
    )
    assert fingerprint() == fixture["source_fingerprint"] == cpu["source_fingerprint"]
    logits = np.load(out / "cpu-reference/logits.npz")["logits"]
    lp = np.load(out / "http-evaluation/teacher-logprobs.npz")["logprobs"]
    assert (
        logits.shape
        == lp.shape
        == (len(fixture["continuation_ids"]) + 1, fixture["vocab_size"])
    )
    distributions = compare_distributions(logits, lp, fixture["continuation_ids"])
    old = out.parent / "v4-mmlu-accuracy-20260911-01"
    protocol = json.loads((old / "protocol.json").read_text())
    assert (
        hashlib.sha256((old / "evaluate.py").read_bytes()).hexdigest()
        == protocol["script_sha256"]
    )
    spec = importlib.util.spec_from_file_location("old_mmlu_eval", old / "evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    selected = json.loads((out / "http-evaluation/selected.json").read_text())
    assert (
        hashlib.sha256((out / "http-evaluation/selected.json").read_bytes()).hexdigest()
        == protocol["selected_sha256"]
    )
    previous = {
        r["question_id"]: r
        for r in map(json.loads, (old / "responses.jsonl").read_text().splitlines())
    }
    results = list(
        map(
            json.loads,
            (out / "http-evaluation/mmlu-responses.jsonl").read_text().splitlines(),
        )
    )
    assert len(results) == len(selected) == 56 and [
        r["question_id"] for r in results
    ] == [r["question_id"] for r in selected]
    changes = []
    exact_text = 0
    for case, result in zip(selected, results):
        assert result["gold"] == case["gold"]
        if "error" not in result:
            pred, tier = evaluator.extract(result["response"]["text"])
            valid = pred is not None and evaluator.LETTERS.index(pred) < len(
                [o for o in case["raw_test"]["options"] if o != "N/A"]
            )
            truncated = (
                result["response"]["meta_info"]["finish_reason"].get("type") == "length"
            )
            assert result["correct"] == (
                valid and not truncated and pred == case["gold"]
            )
            assert result["prediction"] == pred and result["extraction_tier"] == tier
            exact_text += (
                result["response"]["text"]
                == previous[case["question_id"]]["response"]["text"]
            )
        else:
            assert not result["correct"]
        before = previous[case["question_id"]]
        if (
            before["prediction"] != result["prediction"]
            or before["correct"] != result["correct"]
        ):
            changes.append(
                {
                    "question_id": case["question_id"],
                    "category": case["category"],
                    "gold": case["gold"],
                    "previous_prediction": before["prediction"],
                    "prediction": result["prediction"],
                    "previous_correct": before["correct"],
                    "correct": result["correct"],
                }
            )
    correct = sum(r["correct"] for r in results)
    assert correct == http["correct"]
    report = {
        "complete": True,
        "source_fingerprint": fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "full_model_distribution_comparison": distributions,
        "generation_text_repeat_equal": http["generation_text_repeat_equal"],
        "generated_text": http["generation"][0]["text"],
        "mmlu": {
            "correct": correct,
            "total": 56,
            "accuracy": correct / 56,
            "previous_correct": sum(r["correct"] for r in previous.values()),
            "previous_source_fingerprint": protocol["source_fingerprint"],
            "same_generated_text_count": int(exact_text),
            "answer_changes": changes,
            "correct_to_incorrect": sum(
                r["previous_correct"] and not r["correct"] for r in changes
            ),
            "incorrect_to_correct": sum(
                not r["previous_correct"] and r["correct"] for r in changes
            ),
            "invalid": http["invalid"],
            "truncated": http["truncated"],
            "failed": http["failed"],
            "by_category": {
                c: {
                    "correct": sum(r["correct"] for r in results if r["category"] == c),
                    "total": 4,
                }
                for c in sorted({r["category"] for r in results})
            },
        },
        "limitations": [
            "One fixed natural prompt and eight fixed continuation tokens, not a broad logits corpus",
            "CPU reference is official Python with CPU shims, not official GPU or API",
            "MMLU-Pro is the previously seen 56-question regression sample, not a new held-out or full-suite estimate",
            "No FP64 equality gate; no numerical tolerance relaxed; no official accuracy equivalence claim",
            "Historical 15/18 refers to isolated compressor matrix, not full-model accuracy",
        ],
    }
    destination = out / "analysis.json"
    assert not destination.exists()
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
