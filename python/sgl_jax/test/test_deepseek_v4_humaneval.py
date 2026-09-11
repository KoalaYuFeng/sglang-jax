"""Synthetic HumanEval extraction/isolation checks; no benchmark content."""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from evaluate_deepseek_v4_humaneval import IMAGE, docker_command, extract_code


@pytest.mark.parametrize(
    "text",
    [
        "def f():\n    return 42",
        "```python\ndef f():\n    return 42\n```",
        "```\ndef f():\n    return 42\n```",
        "Here is the code:\n```python\ndef f():\n    return 42\n```",
    ],
)
def test_extract_complete_definition(text):
    code = extract_code(text, "f")
    assert code.startswith("\n\ndef f")
    assert isinstance(ast.parse("def f():\n    'doc'\n" + code).body[-1], ast.FunctionDef)


@pytest.mark.parametrize(
    "text",
    [
        "return 42",
        "def other():\n    return 42",
        "not valid python!",
        "```python\ndef f(): pass\n```\n```python\nx=1\n```",
        "```javascript\nfunction f() {}\n```",
        "```python\ndef f(): pass",
    ],
)
def test_reject_ambiguous_or_invalid_output(text):
    with pytest.raises((ValueError, SyntaxError)):
        extract_code(text, "f")


def test_preserve_helpers_and_imports():
    code = "import math\ndef helper(): return math.sqrt(4)\ndef f(): return helper()\n"
    assert extract_code(code, "f") == "\n\n" + code


def test_container_isolation_flags(tmp_path):
    command = docker_command(tmp_path, "v4-humaneval-synthetic")
    assert command[:4] == ["sudo", "-n", "docker", "run"]
    for flag, value in (
        ("--network", "none"),
        ("--user", "65534:65534"),
        ("--cap-drop", "ALL"),
        ("--security-opt", "no-new-privileges"),
        ("--memory", "512m"),
        ("--pids-limit", "64"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--read-only" in command and "--privileged" not in command
    assert command.count("--mount") == 1
    mount = command[command.index("--mount") + 1]
    assert mount.endswith("/sandbox,dst=/eval,readonly")
    assert IMAGE in command and IMAGE.startswith("sha256:")


def test_reject_mutable_image_tag(tmp_path):
    with pytest.raises(AssertionError):
        docker_command(tmp_path, "v4-humaneval-synthetic", image="python:latest")
