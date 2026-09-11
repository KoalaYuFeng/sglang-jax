"""Synthetic protocol tests; no benchmark examples or serving imports."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from evaluate_deepseek_v4_gsm8k import extract, normalize, score, select_indices
import evaluate_deepseek_v4_gsm8k as evaluator


@pytest.mark.parametrize(
    "text,value",
    [
        ("A calculation.\n#### 42", "42"),
        ("#### -42", "-42"),
        ("#### 1,234.50", "1234.5"),
        ("#### $42", "42"),
        ("**#### 42**", "42"),
        ("#### 42\n\n", "42"),
        ("Result is 42", None),
        ("#### 4/2", None),
        ("#### NaN", None),
        ("#### 42 apples", None),
        ("#### 1,23", None),
    ],
)
def test_extract(text, value):
    assert extract(text)[0] == value


def test_last_explicit_and_conflict():
    assert extract("#### 41\n#### 42") == ("42", True)
    assert extract("#### 42\n#### 42.0") == ("42", False)


def test_numeric_equality():
    assert normalize("1,000.00") == normalize("1000")
    with pytest.raises(ValueError):
        normalize("2**10")


def test_fixed_subset():
    a = select_indices(1319)
    assert len(a) == len(set(a)) == 128 and a == sorted(a)
    assert a == select_indices(1319)
    assert min(a) >= 0 and max(a) < 1319
    with pytest.raises(ValueError):
        select_indices(128)


def test_complete_test_in_original_order():
    assert select_indices(1319, full_test=True) == list(range(1319))
    with pytest.raises(ValueError):
        select_indices(128, full_test=True)


def test_truncation_and_invalid_count_wrong():
    case = {"prompt_tokens": 20, "max_new_tokens": 2048, "gold": "42"}
    response = {
        "text": "#### 42",
        "meta_info": {
            "prompt_tokens": 20,
            "completion_tokens": 2048,
            "finish_reason": {"type": "length"},
        },
    }
    assert score(case, response)["truncated"] and not score(case, response)["correct"]
    response["meta_info"].update(completion_tokens=8, finish_reason={"type": "stop"})
    assert score(case, response)["correct"]
    response["text"] = "No final number"
    assert score(case, response)["invalid"] and not score(case, response)["correct"]


def test_reject_wrong_token_accounting():
    case = {"prompt_tokens": 20, "max_new_tokens": 2048, "gold": "42"}
    response = {
        "text": "#### 42",
        "meta_info": {
            "prompt_tokens": 21,
            "completion_tokens": 8,
            "finish_reason": {"type": "stop"},
        },
    }
    with pytest.raises(AssertionError):
        score(case, response)


def test_full_audit_denominator_and_four_digit_record_names(tmp_path, monkeypatch):
    protocol = {"total": 1319, "subset": False, "source_fingerprint": "synthetic"}
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "results").mkdir()
    cases = [{"index": i, "source_index": i, "gold": "42", "prompt_tokens": 1,
              "max_new_tokens": 2048} for i in range(1319)]
    monkeypatch.setattr(evaluator, "load_protocol", lambda out: (protocol, cases))
    for case in cases:
        record = {k: case[k] for k in ("index", "source_index", "gold")}
        record.update(protocol_sha256=evaluator.sha(tmp_path / "protocol.json"),
                      failed=True, correct=False, invalid=True, truncated=False)
        (tmp_path / "results" / f"{case['index']:03d}.json").write_text(json.dumps(record))
    evaluator.audit(tmp_path)
    result = json.loads((tmp_path / "audit.json").read_text())
    assert result["total"] == result["failed"] == 1319
    assert result["accuracy_percent"] == 0 and not result["subset"]
    assert len(json.loads((tmp_path / "audit-manifest.json").read_text())) == 1319
