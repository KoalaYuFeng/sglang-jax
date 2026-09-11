"""Descriptive full-vocabulary comparison, with no inferred acceptance gate."""

import numpy as np


def log_softmax64(value):
    value = np.asarray(value, np.float64)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError("expected finite nonempty [positions,vocabulary]")
    shifted = value - value.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def compare_distributions(cpu_logits, http_logprobs, teacher_ids):
    cpu, tpu = log_softmax64(cpu_logits), log_softmax64(http_logprobs)
    if cpu.shape != tpu.shape or len(teacher_ids) > len(cpu):
        raise ValueError("incompatible distribution/teacher shapes")
    if any(type(t) is not int or not 0 <= t < cpu.shape[1] for t in teacher_ids):
        raise ValueError("invalid teacher token")
    p, q = np.exp(cpu), np.exp(tpu)
    rows = []
    topk = min(5, cpu.shape[1])
    for i in range(len(cpu)):
        c = np.argsort(-cpu[i], kind="stable")
        t = np.argsort(-tpu[i], kind="stable")
        row = {
            "position_index": i,
            "cpu_top1": int(c[0]),
            "tpu_top1": int(t[0]),
            "top1_equal": bool(c[0] == t[0]),
            "top5_overlap": len(set(c[:topk]) & set(t[:topk])) / topk,
            "kl_cpu_to_tpu_nats": float(np.sum(p[i] * (cpu[i] - tpu[i]))),
            "total_variation": float(np.abs(p[i] - q[i]).sum() / 2),
            "cpu_top1_probability": float(p[i, c[0]]),
            "tpu_top1_probability": float(q[i, t[0]]),
            "cpu_top1_margin_nats": float(cpu[i, c[0]] - cpu[i, c[1]])
            if len(c) > 1
            else None,
            "tpu_top1_margin_nats": float(tpu[i, t[0]] - tpu[i, t[1]])
            if len(t) > 1
            else None,
        }
        if i < len(teacher_ids):
            target = teacher_ids[i]
            row.update(
                teacher_token=target,
                cpu_teacher_nll=float(-cpu[i, target]),
                tpu_teacher_nll=float(-tpu[i, target]),
            )
        rows.append(row)
    result = {
        "positions": len(cpu),
        "vocab_size": cpu.shape[1],
        "rows": rows,
        "top1_equal_count": sum(r["top1_equal"] for r in rows),
        "mean_kl_cpu_to_tpu_nats": float(
            np.mean([r["kl_cpu_to_tpu_nats"] for r in rows])
        ),
        "max_total_variation": max(r["total_variation"] for r in rows),
        "mean_top5_overlap": float(np.mean([r["top5_overlap"] for r in rows])),
        "acceptance_gate_applied": False,
        "comparison": "CPU raw logits and HTTP logprobs each normalized in FP64 for descriptive distribution metrics only",
    }
    if teacher_ids:
        result["cpu_mean_teacher_nll"] = float(
            np.mean([r["cpu_teacher_nll"] for r in rows[: len(teacher_ids)]])
        )
        result["tpu_mean_teacher_nll"] = float(
            np.mean([r["tpu_teacher_nll"] for r in rows[: len(teacher_ids)]])
        )
    return result
