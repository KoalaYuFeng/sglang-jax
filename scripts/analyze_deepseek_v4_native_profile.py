"""Native V4 source attribution using the existing auditable XProf exporter.

No JAX import is needed in the separate profile-tools environment. Latency is
measured; XProf FLOP/byte values remain compiler estimates. FP4/FP8 Pallas GEMM
times include their fused online dequantization, not a separately measured cost.
"""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import analyze_deepseek_v4_profile as base

ROOT = Path(__file__).resolve().parents[1] / "python/sgl_jax/srt"


def fingerprint():
    paths = [
        ROOT / "model_executor/deepseek_v4_reference.py",
        ROOT / "model_loader/deepseek_v4_checkpoint.py",
    ]
    for directory in ("kernels/low_bit", "kernels/mhc"):
        paths.extend((ROOT / directory).glob("*.py"))
    reference = hashlib.sha256()
    for path in sorted(paths):
        reference.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    digest = hashlib.sha256(reference.hexdigest().encode())
    names = [
        "configs/deepseek_v4.py",
        "configs/model_config.py",
        "hf_transformers_utils.py",
        "models/deepseek_v4.py",
        "model_loader/deepseek_v4_native.py",
        "layers/attention/deepseek_v4_backend.py",
        "layers/attention/deepseek_v4_paged_backend.py",
        "mem_cache/deepseek_v4_pool.py",
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
        str(p.relative_to(ROOT)) for p in sorted((ROOT / "kernels/deepseek_v4").glob("*.py"))
    )
    names.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "kernels/hca").glob("*.py")))
    names.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "kernels/csa").glob("*.py")))
    names.append("kernels/dsa/streamindex_topk.py")
    for name in names:
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    return digest.hexdigest()


def build_ranges():
    result = {}
    for name in ("compressor", "moe", "attention", "numerics"):
        path = ROOT / f"kernels/deepseek_v4/{name}.py"
        result[name] = [
            (node.lineno, node.end_lineno, node.name)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
        ]
    return result


ORIGINAL_STAGE = base.event_stage
ORIGINAL_HLO_SUMMARY = base.summarize_hlo_table
NATIVE_RANGES = build_ranges()


def native_stage(event, ranges):
    info = event.get("args", {})
    stack = info.get("source_stack", info.get("source", "")) or ""
    category = info.get("hlo_category", "")
    if category == "all-reduce":
        return (
            "moe_all_reduce_including_wait"
            if any(f"/deepseek_v4/{name}.py:" in stack for name in ("moe", "moe_gmm"))
            else "non_moe_collectives_including_wait"
        )
    if category == "all-gather":
        return (
            "attention_tp_all_gather_including_wait"
            if "/deepseek_v4/attention.py:" in stack
            else "non_moe_collectives_including_wait"
        )
    if "/deepseek_v4/moe_gmm.py:" in stack or "/kernels/low_bit/gmm.py:" in stack:
        return "routed_fp4_experts_including_online_dequant"
    if "/kernels/mhc/" in stack:
        return "mhc"
    if "/kernels/hca/compressor.py:" in stack:
        return "compressor_and_paged_state"
    if "/kernels/hca/attention.py:" in stack:
        return "sparse_attention"
    if "/kernels/csa/compressor.py:" in stack:
        return "compressor_and_paged_state"
    if "/kernels/csa/joint_attention.py:" in stack:
        return "sparse_attention"
    if "/kernels/csa/indexer.py:" in stack or "/kernels/dsa/streamindex_topk.py:" in stack:
        return "attention_topk" if category == "sort" else "attention_index_scores"
    # Attribute the actual attention reduction before its outer projection
    # caller; the generic normalization/linear helpers must not win here.
    for match in re.finditer(r"/deepseek_v4/numerics\.py:(\d+)", stack):
        if any(
            name == "_single_query_attention" and start <= int(match[1]) <= end
            for start, end, name in NATIVE_RANGES["numerics"]
        ):
            return "sparse_attention"
    for module, source_ranges in NATIVE_RANGES.items():
        for match in re.finditer(r"/deepseek_v4/" + module + r"\.py:(\d+)", stack):
            line = int(match[1])
            names = [name for start, end, name in source_ranges if start <= line <= end]
            if module == "moe":
                if "grouped_fp4_experts" in names:
                    return "routed_fp4_experts_including_online_dequant"
                return "router" if "route" in names else "shared_experts_and_moe_combine"
            if module == "compressor":
                return "compressor_and_paged_state"
            if module == "attention":
                return "attention_topk" if category == "sort" else "attention_projection_and_index"
            if "_single_query_attention" in names:
                return "sparse_attention"
            if "official_head_collapse" in names:
                return "lm_head_collapse"
            if any("norm" in name or "mean" in name for name in names):
                return "normalization"
            if "rope" in names:
                return "rotary_embedding"
    if "/models/deepseek_v4.py:" in stack:
        return "model_residual_or_head"
    if "/low_bit/matmul.py:" in stack:
        return "lowbit_gemm_including_online_dequant_unattributed_caller"
    return ORIGINAL_STAGE(event, ranges)


def native_hlo_summary(table, *, cores, steps, ranges):
    result = ORIGINAL_HLO_SUMMARY(table, cores=cores, steps=steps, ranges=ranges)
    columns = [column["id"] for column in table["cols"]]
    moe = attention_gather = other_gather = 0
    for record in table["rows"]:
        row = dict(zip(columns, [cell.get("v") for cell in record["c"]]))
        if row["category"] == "all-reduce" and any(
            f"/deepseek_v4/{name}.py:" in (row.get("source_info") or "")
            for name in ("moe", "moe_gmm")
        ):
            moe += int(row["occurrences"])
        if row["category"] == "all-gather":
            if "/deepseek_v4/attention.py:" in (row.get("source_info") or ""):
                attention_gather += int(row["occurrences"])
            else:
                other_gather += int(row["occurrences"])
    result.pop("expected_all_reduce_occurrences")
    result.update(
        moe_all_reduce_occurrences=moe,
        expected_moe_all_reduce_occurrences=43 * cores * steps,
        attention_tp_all_gather_occurrences=attention_gather,
        other_all_gather_occurrences=other_gather,
        additional_collective_occurrences=(
            result["all_reduce_occurrences"] - moe + attention_gather + other_gather
        ),
        complete_collective_coverage=moe == 43 * cores * steps,
        collective_coverage_scope=(
            "43 layer MoE sums only; attention TP gathers and output/sampling "
            "collectives are counted separately, not certified by this coverage flag"
        ),
    )
    return result


def main():
    # Keep the existing exporter CLI and its complete raw/XPlane evidence.
    index = sys.argv.index("--profile")
    profile = Path(sys.argv[index + 1])
    report = json.loads((profile / "report.json").read_text())
    if report["framework_source_fingerprint"] != fingerprint():
        raise ValueError("native source changed; cannot attribute captured source lines")
    base.event_stage = native_stage
    base.summarize_hlo_table = native_hlo_summary
    base.main()


if __name__ == "__main__":
    main()
