"""Synthetic examples only: no GPQA benchmark contents in public source."""

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("gpqa_eval", SCRIPTS / "evaluate_deepseek_v4_gpqa.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


@pytest.mark.parametrize(
    "text,answer,tier",
    [
        ("Explanation\nAnswer: B", "B", "explicit_final"),
        ("**Answer: (C)**", "C", "explicit_final"),
        ("Answer: a", "A", "explicit_final"),
        ("Result: \\boxed{D}", "D", "boxed"),
        ("(A)", "A", "letter_only"),
        ("I discussed option A but no final answer", None, "invalid"),
        ("Answer: E", None, "invalid"),
        ("Answer: ABC", None, "invalid"),
    ],
)
def test_frozen_extractor(text, answer, tier):
    pred, actual_tier, _ = evaluation.extract(text)
    assert pred == answer and actual_tier == tier


def test_final_correction_and_conflict_recorded():
    assert evaluation.extract("Answer: A\nReconsidering.\nAnswer: C") == (
        "C",
        "explicit_final",
        True,
    )


def test_no_solution_or_validation_metadata_in_prompt():
    row = {
        "Question": "Which synthetic value?",
        "Correct Answer": "correct-option",
        "Incorrect Answer 1": "wrong-one",
        "Incorrect Answer 2": "wrong-two",
        "Incorrect Answer 3": "wrong-three",
        "Explanation": "SECRET_EXPLANATION",
        "Extra Revised Explanation": "SECRET_REVISION",
    }
    a = evaluation.question_content(row, 7)
    assert a == evaluation.question_content(row, 7)
    content, gold, order = a
    assert order[evaluation.LETTERS.index(gold)] == 0
    assert "SECRET" not in content
    assert all(
        content.count(value) == 1
        for value in ("correct-option", "wrong-one", "wrong-two", "wrong-three")
    )


def test_truncated_correct_answer_counts_wrong():
    response = {
        "text": "Answer: B",
        "meta_info": {"prompt_tokens": 10, "finish_reason": {"type": "length"}},
    }
    result = evaluation.score({"gold": "B", "prompt_tokens": 10}, response)
    assert result["truncated"] and not result["correct"]


def test_prompt_token_count_checked():
    with pytest.raises(AssertionError):
        evaluation.score(
            {"gold": "A", "prompt_tokens": 11}, {"text": "A", "meta_info": {"prompt_tokens": 10}}
        )
