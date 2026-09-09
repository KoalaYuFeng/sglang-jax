"""Small, reproducible MMLU-Pro accuracy gate for the native V4 TPU path.

This is deliberately an implementation-regression benchmark, not a claimed
reproduction of DeepSeek's published full-suite score.  It selects real test
questions whose officially encoded prompts share one static length, then runs
the unchanged checkpoint reference and the production ModelWorker path on
every question.  Full-vocabulary logits and predictions must agree bitwise.

The single prompt length avoids compiling a separate B1 executable for every
question while the bounded V4 integration is still specialized to 256 tokens.
"""

import argparse
import hashlib
import importlib.util
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from run_deepseek_v4_framework import (
    WorkerSession,
    compare_arrays,
    framework_fingerprint,
    memory_snapshot,
    server_args,
)
from sgl_jax.srt.model_executor.deepseek_v4_reference import (
    DeepSeekV4Reference,
    source_fingerprint,
)
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


LETTERS = "ABCDEFGHIJ"
DEFAULT_DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"


def load_official_encoding(checkpoint: Path):
    path = checkpoint / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("pinned_deepseek_v4_encoding", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official encoding from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def question_text(row: dict) -> str:
    options = [option for option in row["options"] if option != "N/A"]
    if not 2 <= len(options) <= len(LETTERS):
        raise ValueError(f"question {row['question_id']} has {len(options)} options")
    rendered_options = "\n".join(
        f"{LETTERS[index]}. {option}" for index, option in enumerate(options)
    )
    return (
        "Answer this multiple-choice question. Reply with only the single letter "
        "of the correct option.\n\n"
        f"Question: {row['question']}\n"
        f"Options:\n{rendered_options}\n"
        "Answer:"
    )


def encoded_case(row: dict, tokenizer, encoding) -> dict:
    content = question_text(row)
    prompt = encoding.encode_messages(
        [{"role": "user", "content": content}], thinking_mode="chat"
    )
    input_ids = [int(token) for token in tokenizer.encode(prompt)]
    return {
        "question_id": int(row["question_id"]),
        "category": row["category"],
        "answer": row["answer"],
        "answer_index": int(row["answer_index"]),
        "input_ids": input_ids,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def stable_order(case: dict, seed: int) -> str:
    value = f"{seed}:{case['question_id']}:{case['prompt_sha256']}"
    return hashlib.sha256(value.encode()).hexdigest()


def select_cases(dataset, tokenizer, encoding, *, count: int, tokens: int, seed: int):
    by_category = defaultdict(list)
    prompt_length_histogram = defaultdict(int)
    eligible = 0
    for row in dataset:
        case = encoded_case(row, tokenizer, encoding)
        length = len(case["input_ids"])
        if length <= 252:
            eligible += 1
            prompt_length_histogram[length] += 1
        if length == tokens:
            by_category[case["category"]].append(case)

    for category in by_category:
        by_category[category].sort(key=lambda case: stable_order(case, seed))

    categories = sorted(by_category)
    selected = []
    round_index = 0
    while len(selected) < count:
        progressed = False
        for category in categories:
            cases = by_category[category]
            if round_index < len(cases):
                selected.append(cases[round_index])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        round_index += 1
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} cases have the requested {tokens}-token shape; "
            f"need {count}"
        )
    return selected, {
        "eligible_within_252_tokens": eligible,
        "requested_prompt_tokens": tokens,
        "available_at_requested_length": sum(map(len, by_category.values())),
        "categories_at_requested_length": categories,
        "prompt_length_histogram": dict(sorted(prompt_length_histogram.items())),
        "selection": "round-robin categories; SHA256(seed, question_id, prompt) order",
    }


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054):
    if total == 0:
        return [0.0, 1.0]
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    half /= denominator
    return [float(center - half), float(center + half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=Path("/mnt/disks/deepseek-models/datasets"),
    )
    parser.add_argument("--dataset", default="TIGER-Lab/MMLU-Pro")
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--prompt-tokens", type=int, default=129)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if args.num_examples < 1 or not 1 <= args.prompt_tokens <= 252:
        raise ValueError("num-examples must be positive and prompt-tokens must be in [1, 252]")

    gate = json.loads(args.framework_report.read_text())
    checkpoint = Path(gate["checkpoint"])
    if (
        not gate.get("complete")
        or gate.get("source_fingerprint") != source_fingerprint()
        or gate.get("framework_source_fingerprint") != framework_fingerprint()
        or not checkpoint.is_dir()
    ):
        raise ValueError("requires the same-source complete native ModelWorker gate")

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "complete": False,
        "benchmark": "MMLU-Pro implementation-regression smoke",
        "scope": (
            "20 real MMLU-Pro test questions by default; official V4 chat/non-think "
            "encoding; zero-shot direct-answer; B1 TP4/EP4; one static prefill shape"
        ),
        "not_comparable_to_published_score_because": [
            "small length-conditioned subset rather than all 12,032 test questions",
            "zero-shot direct answer rather than DeepSeek's undisclosed exact MMLU-Pro prompt",
            "greedy implementation gate rather than temperature=1 sampling",
            "256-token integration limit rather than the published non-think 8K context",
        ],
        "published_reference": {
            "model": "DeepSeek-V4-Flash",
            "mode": "Non-think",
            "benchmark": "MMLU-Pro",
            "metric": "EM",
            "score_percent": 83.0,
            "temperature": 1.0,
            "context_tokens": 8192,
        },
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_snapshot": checkpoint.name,
        "framework_report": str(args.framework_report.resolve()),
        "source_fingerprint": source_fingerprint(),
        "framework_source_fingerprint": framework_fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "seed": args.seed,
        "events": [],
        "cases": [],
    }

    def emit(event):
        report["events"].append({"time": time.time(), **event})
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
        print(json.dumps(event, default=str), flush=True)

    try:
        encoding, encoding_path = load_official_encoding(checkpoint)
        report["official_encoding_sha256"] = hashlib.sha256(
            encoding_path.read_bytes()
        ).hexdigest()
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        choice_token_ids = {}
        for letter in LETTERS:
            ids = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1 or tokenizer.decode(ids) != letter:
                raise AssertionError(f"choice {letter!r} is not one exact token: {ids}")
            choice_token_ids[letter] = int(ids[0])
        report["choice_token_ids"] = choice_token_ids

        emit({"event": "dataset_load_start"})
        dataset = load_dataset(
            args.dataset,
            revision=args.dataset_revision,
            cache_dir=str(args.dataset_cache),
            split="test",
        )
        selected, selection = select_cases(
            dataset,
            tokenizer,
            encoding,
            count=args.num_examples,
            tokens=args.prompt_tokens,
            seed=args.seed,
        )
        report["dataset_fingerprint"] = dataset._fingerprint
        report["dataset_rows"] = len(dataset)
        report["selection_metadata"] = selection
        report["selected_question_ids"] = [case["question_id"] for case in selected]
        report["selected_categories"] = [case["category"] for case in selected]
        emit(
            {
                "event": "cases_selected",
                "count": len(selected),
                "prompt_tokens": args.prompt_tokens,
                "categories": sorted(set(report["selected_categories"])),
                "question_ids": report["selected_question_ids"],
            }
        )

        if jax.default_backend() != "tpu" or len(jax.devices()) != 4:
            raise RuntimeError("requires exactly four TPU devices")
        mesh = create_device_mesh([1, 4], [1, 1])
        sa = server_args(checkpoint)
        report["server_scope"] = {
            "context_length": sa.context_length,
            "tp_size": sa.tp_size,
            "ep_size": sa.ep_size,
            "dtype": sa.dtype,
            "moe_backend": sa.moe_backend,
            "attention_backend": sa.attention_backend,
        }
        emit({"event": "native_model_load_start"})
        start = time.perf_counter()
        worker = ModelWorker(sa, mesh)
        runner = worker.model_runner
        report["model_load_seconds"] = time.perf_counter() - start
        report["hbm_after_model_load"] = memory_snapshot()
        emit({"event": "native_model_loaded", "seconds": report["model_load_seconds"]})

        reference = DeepSeekV4Reference(checkpoint, max_context=256, progress=lambda _: None)
        reference.mesh = mesh
        reference.shared = runner.model.shared.get_value()
        reference.layers = list(runner.model.layers.get_value())
        session = WorkerSession(worker)
        token_vector = np.asarray([choice_token_ids[c] for c in LETTERS], np.int32)

        for index, case in enumerate(selected):
            emit(
                {
                    "event": "case_start",
                    "index": index,
                    "question_id": case["question_id"],
                    "category": case["category"],
                }
            )
            reference.reset()
            ref_start = time.perf_counter()
            with jax.set_mesh(mesh):
                reference_logits, _ = reference.step(case["input_ids"])
            reference_seconds = time.perf_counter() - ref_start
            reference_host = np.asarray(reference_logits, np.float32)

            session.reset()
            framework_logits, framework_greedy, framework_seconds, misses = session.step(
                case["input_ids"], device_sample=True
            )
            comparison = compare_arrays(reference_host, framework_logits)
            reference_greedy = int(np.argmax(reference_host[0]))
            if not comparison["bitwise_equal"] or framework_greedy != reference_greedy:
                raise AssertionError(
                    f"deployment/reference divergence on question {case['question_id']}"
                )

            reference_scores = reference_host[0, token_vector]
            framework_scores = framework_logits[0, token_vector]
            reference_choice = LETTERS[int(np.argmax(reference_scores))]
            framework_choice = LETTERS[int(np.argmax(framework_scores))]
            greedy_text = tokenizer.decode(
                [framework_greedy], skip_special_tokens=True
            ).strip().upper()
            greedy_choice = greedy_text if greedy_text in LETTERS else None
            result = {
                "index": index,
                "question_id": case["question_id"],
                "category": case["category"],
                "answer": case["answer"],
                "answer_index": case["answer_index"],
                "prompt_tokens": len(case["input_ids"]),
                "prompt_sha256": case["prompt_sha256"],
                "reference_choice": reference_choice,
                "framework_choice": framework_choice,
                "choice_correct": framework_choice == case["answer"],
                "choice_scores": {
                    letter: float(score) for letter, score in zip(LETTERS, framework_scores)
                },
                "reference_greedy_token": reference_greedy,
                "framework_greedy_token": framework_greedy,
                "framework_greedy_text": greedy_text,
                "framework_greedy_choice": greedy_choice,
                "greedy_correct": greedy_choice == case["answer"],
                "greedy_is_choice": greedy_choice is not None,
                "full_logits": comparison,
                "choice_scores_bitwise_equal": bool(
                    np.array_equal(reference_scores, framework_scores)
                ),
                "reference_seconds_including_compile": reference_seconds,
                "framework_seconds_including_compile": framework_seconds,
                "pjit_cache_misses": misses,
            }
            report["cases"].append(result)
            emit(
                {
                    "event": "case_checked",
                    "index": index,
                    "question_id": case["question_id"],
                    "answer": case["answer"],
                    "prediction": framework_choice,
                    "correct": result["choice_correct"],
                    "greedy_text": greedy_text,
                    "logits_bitwise_equal": comparison["bitwise_equal"],
                    "reference_seconds": reference_seconds,
                    "framework_seconds": framework_seconds,
                }
            )

        total = len(report["cases"])
        choice_correct = sum(case["choice_correct"] for case in report["cases"])
        greedy_correct = sum(case["greedy_correct"] for case in report["cases"])
        greedy_compliant = sum(case["greedy_is_choice"] for case in report["cases"])
        report["summary"] = {
            "total": total,
            "choice_logit_correct": choice_correct,
            "choice_logit_accuracy": choice_correct / total,
            "length_conditioned_wilson_95": wilson_interval(choice_correct, total),
            "one_token_greedy_correct": greedy_correct,
            "one_token_greedy_accuracy": greedy_correct / total,
            "one_token_choice_compliance": greedy_compliant / total,
            "deployment_reference_prediction_matches": sum(
                case["framework_choice"] == case["reference_choice"]
                for case in report["cases"]
            ),
            "deployment_reference_prediction_match_rate": sum(
                case["framework_choice"] == case["reference_choice"]
                for case in report["cases"]
            )
            / total,
            "full_vocab_logits_bitwise_matches": sum(
                case["full_logits"]["bitwise_equal"] for case in report["cases"]
            ),
            "full_vocab_logits_bitwise_match_rate": sum(
                case["full_logits"]["bitwise_equal"] for case in report["cases"]
            )
            / total,
            "accuracy_delta_deployment_minus_reference": 0.0,
        }
        report["hbm_after_benchmark"] = memory_snapshot()
        report["complete"] = True
        emit({"event": "benchmark_complete", **report["summary"]})
    except BaseException:
        report["error"] = traceback.format_exc()
        emit({"event": "benchmark_failed", "error": report["error"]})
        raise


if __name__ == "__main__":
    main()
