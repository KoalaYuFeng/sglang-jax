"""CPU-only scoring, checkpoint and no-score-dependent-retry coverage."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "v4_full_eval",
    Path(__file__).resolve().parents[3] / "scripts/evaluate_deepseek_v4_mmlu_full.py",
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def case():
    return dict(
        question_id=12,
        category="math",
        gold="B",
        prompt_tokens=30,
        raw_test={"options": ["foo", "bar", "N/A"]},
        fits_context=True,
    )


def response(text, reason="stop"):
    return dict(text=text, meta_info=dict(prompt_tokens=30, finish_reason={"type": reason}))


@pytest.mark.parametrize(
    "answer,correct,invalid",
    [
        ("B", True, False),
        ("A", False, False),
        ("J", False, True),
        (None, False, True),
        ("", False, True),
    ],
)
def test_generated_answer_scoring(answer, correct, invalid):
    result = evaluation.result_from_response(case(), response("generated"), lambda _: (answer, 1))
    assert result["correct"] == correct and result["invalid"] == invalid


def test_truncated_correct_letter_counts_wrong():
    result = evaluation.result_from_response(case(), response("B", "length"), lambda _: ("B", 1))
    assert not result["correct"] and result["truncated"] and not result["invalid"]


def test_prompt_length_mismatch_rejected():
    value = response("B")
    value["meta_info"]["prompt_tokens"] = 29
    with pytest.raises(AssertionError):
        evaluation.result_from_response(case(), value, lambda _: ("B", 1))


@pytest.mark.parametrize(
    "completed,attempted,fits,action",
    [
        ({12}, True, True, "skip"),
        ({12}, False, True, "skip"),
        (set(), True, True, "interrupted"),
        (set(), False, False, "context_limit"),
        (set(), False, True, "request"),
    ],
)
def test_resume_never_retries_completed_or_unknown_attempt(completed, attempted, fits, action):
    value = case()
    value["fits_context"] = fits
    assert evaluation.pending_action(value, completed, attempted) == action


def test_failure_stays_in_denominator():
    result = evaluation.failure(case(), "out of context", context_limit=True)
    summary = evaluation.summarize([result], 12032)
    assert summary["completed"] == 1 and summary["total"] == 12032
    assert summary["failed"] == 1 and summary["context_limit"] == 1
    assert summary["accuracy_completed"] == 0 and not summary["complete"]


def test_duplicate_result_rejected():
    result = evaluation.failure(case(), "error")
    with pytest.raises(AssertionError):
        evaluation.summarize([result, result], 12032)


def test_atomic_json_checkpoint(tmp_path):
    path = tmp_path / "record.json"
    evaluation.save(path, {"state": "first"})
    evaluation.save(path, {"state": "second"})
    assert json.loads(path.read_text()) == {"state": "second"}
    assert not path.with_suffix(".json.tmp").exists()


def test_unicode_separators_are_not_jsonl_boundaries(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [{"text": "hello\u0085world\u2028next\u2029paragraph"}, {"text": "line\nend"}]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert evaluation.read_jsonl(path) == rows
