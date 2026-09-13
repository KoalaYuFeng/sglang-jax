"""Shared native execution/profile source identity, without JAX imports.

The bring-up oracle retains its independent source_fingerprint implementation.
Tests compare its digest to reference_fingerprint below without changing oracle
arithmetic. Historical receipts must use their original source and manifest.
"""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "python/sgl_jax/srt"


def reference_fingerprint():
    paths = [
        ROOT / "model_executor/deepseek_v4_reference.py",
        ROOT / "model_loader/deepseek_v4_checkpoint.py",
    ]
    for directory in ("kernels/low_bit", "kernels/mhc"):
        paths.extend((ROOT / directory).glob("*.py"))
    reference = hashlib.sha256()
    for path in sorted(paths):
        reference.update(
            str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes()
        )
    return reference.hexdigest()


def framework_fingerprint():
    digest = hashlib.sha256(reference_fingerprint().encode())
    names = [
        "configs/deepseek_v4.py",
        "configs/deepseek_v4_execution.py",
        "configs/model_config.py",
        "hf_transformers_utils.py",
        "models/deepseek_v4.py",
        "model_loader/deepseek_v4_native.py",
        "model_loader/deepseek_v4_packing.py",
        "layers/attention/deepseek_v4_paged_backend.py",
        "mem_cache/deepseek_v4_paged_pool.py",
        "model_executor/model_runner.py",
        "model_executor/model_runner_kv_cache_mixin.py",
        "model_executor/compilation_manager.py",
        "kernels/gmm/routing.py",
        "kernels/gmm/megablox_gmm_kernel/gmm.py",
        "kernels/gmm/megablox_gmm_kernel/common.py",
        "kernels/gmm/megablox_gmm_kernel/tuned_block_sizes.py",
    ]
    names.extend(
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "kernels/deepseek_v4").glob("*.py"))
    )
    names.extend(
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "layers/deepseek_v4").glob("*.py"))
    )
    names.extend(
        str(p.relative_to(ROOT)) for p in sorted((ROOT / "kernels/hca").glob("*.py"))
    )
    names.extend(
        str(p.relative_to(ROOT)) for p in sorted((ROOT / "kernels/csa").glob("*.py"))
    )
    names.append("kernels/dsa/streamindex_topk.py")
    for name in names:
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    # The manifest implementation itself is evidence, not an untracked policy.
    digest.update(b"scripts/deepseek_v4_source.py\0" + Path(__file__).read_bytes())
    return digest.hexdigest()
