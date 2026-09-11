"""Independent CPU measurements for V4 numerical acceptance design.

Boundary attribution is NOT a pass/fail rule. It cannot excuse an incorrect
quantizer, large upstream error, changed block scale, or downstream divergence.
"""

import numpy as np


def tensor_metrics(actual, reference):
    a, b = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    if a.shape != b.shape or a.ndim < 1 or not a.size:
        raise ValueError("requires nonempty equal-shaped tensors")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("nonfinite tensor")
    delta = a - b
    return {
        "shape": list(a.shape),
        "equal": bool(np.array_equal(a, b)),
        "nrmse": float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-12)),
        "max_vector_nrmse": float(
            np.max(
                np.linalg.norm(delta, axis=-1)
                / np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
            )
        ),
        "max_abs": float(np.max(np.abs(delta))),
    }


def _fp4_cpu(value, block_size):
    # Independent E2M1/UE8M0 arithmetic, including ties-to-even and signed zero.
    a = np.asarray(value, np.float64)
    if a.ndim < 1 or a.shape[-1] % block_size or not a.size or not np.isfinite(a).all():
        raise ValueError("invalid FP4 input")
    grouped = a.reshape(*a.shape[:-1], -1, block_size)
    amax = np.maximum(np.max(np.abs(grouped), axis=-1, keepdims=True), 6 * 2.0**-126)
    scale = np.exp2(np.ceil(np.log2(amax / 6)))
    normalized = grouped / scale
    levels = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)
    priority = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    distances = np.abs(np.abs(normalized[..., None]) - levels)
    codes = priority[np.argmin(distances[..., priority], axis=-1)]
    signed = levels[codes] * np.where(np.signbit(normalized), -1, 1)
    return (signed * scale).reshape(a.shape).astype(np.float32), np.broadcast_to(
        scale, grouped.shape
    ).reshape(a.shape)


def fp4_boundary_report(
    actual_pre, reference_pre, actual_post, reference_post, block_size=32
):
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    pre = tensor_metrics(actual_pre, reference_pre)
    post = tensor_metrics(actual_post, reference_post)
    a, b = np.asarray(actual_pre, np.float64), np.asarray(reference_pre, np.float64)
    qa, qb = np.asarray(actual_post, np.float32), np.asarray(reference_post, np.float32)
    if qa.shape != a.shape or qb.shape != b.shape:
        raise ValueError("pre/post shapes differ")
    oracle_a, sa = _fp4_cpu(a, block_size)
    oracle_b, sb = _fp4_cpu(b, block_size)
    bad_a = qa.view(np.uint32) != oracle_a.view(np.uint32)
    bad_b = qb.view(np.uint32) != oracle_b.view(np.uint32)
    changed = qa != qb
    same_scale = sa == sb
    grid = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6])
    ca = np.argmin(np.abs(oracle_a[..., None] / sa[..., None] - grid), axis=-1)
    cb = np.argmin(np.abs(oracle_b[..., None] / sb[..., None] - grid), axis=-1)
    midpoint = (oracle_a.astype(np.float64) + oracle_b.astype(np.float64)) / 2
    crossing = (np.minimum(a, b) <= midpoint) & (midpoint <= np.maximum(a, b))
    boundary = (
        changed & same_scale & (np.abs(ca - cb) == 1) & crossing & ~bad_a & ~bad_b
    )
    unexplained = changed & same_scale & ~boundary
    examples = []
    for index in np.argwhere(changed)[:32]:
        key = tuple(index)
        examples.append(
            {
                "index": index.tolist(),
                "actual_pre": float(a[key]),
                "reference_pre": float(b[key]),
                "actual_post": float(qa[key]),
                "reference_post": float(qb[key]),
                "actual_scale": float(sa[key]),
                "reference_scale": float(sb[key]),
                "adjacent_boundary_crossing": bool(boundary[key]),
            }
        )
    return {
        "classification_only": True,
        "prequant": pre,
        "postquant": post,
        "actual_quantizer_mismatches": int(np.count_nonzero(bad_a)),
        "reference_quantizer_mismatches": int(np.count_nonzero(bad_b)),
        "changed_values": int(np.count_nonzero(changed)),
        "changed_scale_blocks": int(
            np.count_nonzero((sa != sb).reshape(*a.shape[:-1], -1, block_size)[..., 0])
        ),
        "same_scale_adjacent_boundary_changes": int(np.count_nonzero(boundary)),
        "same_scale_unexplained_changes": int(np.count_nonzero(unexplained)),
        "changed_values_in_changed_scale_blocks": int(
            np.count_nonzero(changed & ~same_scale)
        ),
        "examples": examples,
    }
