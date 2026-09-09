"""Export the compiled ModelRunner entry used by a real V4 worker session."""

import hashlib
import re
from collections import defaultdict

MARKERS = {
    "mhc_pre": "mhc-collapse-pre",
    "mhc_sinkhorn": "mhc-sinkhorn-gates",
    "mhc_post": "mhc-post",
    "mhc_head": "mhc-collapse-head-fp32-post",
    "hca_projection": "hca-state-project",
    "hca_emission": "hca-boundary-snapshot",
    "hca_attention": "hca-paged-stream",
    "csa_projection": "csa-compressor-project",
    "csa_main_emission": "csa-compressor-snapshot-d512",
    "csa_index_emission": "csa-compressor-snapshot-d128",
    "csa_indexer": "StreamIdxTC",
    "csa_attention": "csa-joint-attention",
    "fp8_linear": "gmm_checkpoint_fp8",
    "exact_norm": "v4_exact_rms_norm",
    "qnorm_rope": "v4_exact_qnorm_rope",
    "fused_wo_a": "v4_inverse_rope_checkpoint_fp8_wo_a",
}


def _hlo_graph(text):
    """Parse instruction-level SSA edges, confined to each HLO computation."""
    nodes, consumers, computation = {}, defaultdict(list), None
    for line in text.splitlines():
        header = re.match(r"^(?:ENTRY )?%([-\w.]+)\b", line)
        if header:
            computation = header[1]
            continue
        line = line.partition("backend_config=")[0]
        instruction = re.match(
            r"^\s+(?:ROOT )?%([-\w.]+) = .*? ([a-z][-\w]*)\(([^)]*)\)", line
        )
        if instruction is None or computation is None:
            continue
        name, opcode, operands = instruction.groups()
        identity = (computation, name)
        metadata = re.search(r'op_name="([^"]*)"', line)
        scope = metadata[1] if metadata else ""
        nodes[identity] = {
            "opcode": opcode,
            "scope": scope,
            "layers": set(map(int, re.findall(r"v4_layer_(\d+)(?:/|$)", scope))),
        }
        for operand in re.findall(r"%([-\w.]+)", operands):
            consumers[(computation, operand)].append(identity)
    return nodes, consumers


def _owner(identity, nodes, consumers):
    """Keep a witness path when inlining has stripped a custom call's layer.

    Use the nearest uniquely scoped *consumer*, never ordinal kernel counts,
    common scalar inputs or an arbitrary upstream layer. Stop at tuple/root
    aggregation; do not infer ownership across an HLO computation boundary.
    """
    frontier, seen = [(identity, [identity])], {identity}
    for distance in range(9):
        witnesses = []
        for current, path in frontier:
            for layer in nodes[current]["layers"]:
                witnesses.append((layer, current, path))
        if witnesses:
            layers = {layer for layer, _, _ in witnesses}
            if len(layers) != 1:
                raise AssertionError(
                    f"ambiguous CSA layer ownership: {identity}: {layers}"
                )
            layer, current, path = witnesses[0]
            return {
                "layer": layer,
                "method": "op_name" if distance == 0 else "scoped_consumer",
                "path": [f"{scope}/%{name}" for scope, name in path],
                "consumer_op_name": nodes[current]["scope"],
            }
        following = []
        for current, path in frontier:
            for consumer in consumers[current]:
                if consumer not in seen and nodes[consumer]["opcode"] != "tuple":
                    seen.add(consumer)
                    following.append((consumer, [*path, consumer]))
        frontier = following
    raise AssertionError(f"cannot establish CSA layer ownership: {identity}")


def compiled_kernel_evidence(
    text,
    *,
    mhc_backend,
    hca_backend,
    csa_backend,
    ratios,
    attention_tp=False,
    csa_decode_batch=False,
    fp8_backend="legacy",
    fused_norm=False,
    fused_wo_a=False,
):
    nodes, consumers = _hlo_graph(text)
    # Require named custom-call SSA instructions, not an import/wrapper name
    # occurring in metadata or an embedded backend-config string.
    calls = {
        key: [
            identity
            for identity, node in nodes.items()
            if node["opcode"] == "custom-call"
            and marker in identity[1]
            and (
                not (key.startswith("csa_") or key in ("hca_emission", "hca_attention"))
                or "-v4" in identity[1]
            )
        ]
        for key, marker in MARKERS.items()
    }
    counts = {key: len(values) for key, values in calls.items()}
    required = ("mhc_pre", "mhc_sinkhorn")
    if mhc_backend == "pallas":
        required += ("mhc_post", "mhc_head")
    if hca_backend == "pallas":
        required += ("hca_projection", "hca_emission", "hca_attention")
    csa_layers = [i for i, ratio in enumerate(ratios) if ratio == 4]
    witnesses = {
        key: [_owner(identity, nodes, consumers) for identity in instructions]
        for key, instructions in calls.items()
        if key.startswith("csa_")
    }
    csa_coverage = {
        key: sorted({row["layer"] for row in rows}) for key, rows in witnesses.items()
    }
    if csa_backend == "pallas":
        required += tuple(csa_coverage)
        if any(layers != csa_layers for layers in csa_coverage.values()):
            raise AssertionError(
                f"original CSA dispatch missing layers: {csa_coverage}"
            )
    if any(counts[key] == 0 for key in required):
        raise AssertionError(f"missing compiled original V4 kernel: {counts}")
    dense_required = []
    if fp8_backend == "gmm":
        dense_required.append("fp8_linear")
    if fused_norm:
        dense_required.extend(("exact_norm", "qnorm_rope"))
    if fused_wo_a:
        dense_required.append("fused_wo_a")
    if any(counts[key] < len(ratios) for key in dense_required):
        raise AssertionError(f"missing compiled V4 dense kernels: {counts}")
    if attention_tp:
        for key, ratio in (("csa_attention", 4), ("hca_attention", 128)):
            if len(calls[key]) != ratios.count(ratio) or any(
                "-h16-d512-v4" not in identity[1] for identity in calls[key]
            ):
                raise AssertionError(
                    f"head TP requires actual h16 attention on every r{ratio} layer"
                )
    if csa_decode_batch and (
        len(calls["csa_projection"]) != 2 * len(csa_layers)
        or any(
            "decode-batched" not in identity[1] for identity in calls["csa_projection"]
        )
    ):
        raise AssertionError(
            "decode requires batched main/index CSA projection on every layer"
        )
    return {
        "mhc_backend": mhc_backend,
        "hca_backend": hca_backend,
        "csa_backend": csa_backend,
        "attention_tp": attention_tp,
        "csa_decode_batch": csa_decode_batch,
        "fp8_backend": fp8_backend,
        "fused_norm": fused_norm,
        "fused_wo_a": fused_wo_a,
        "hca_layer_count": sum(ratio == 128 for ratio in ratios),
        "csa_layer_count": len(csa_layers),
        "csa_layer_coverage": csa_coverage,
        "csa_layer_witnesses": witnesses,
        "compiled_custom_call_counts": counts,
        "hlo_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scope": "actual ModelRunner JIT entry for an executed decode shape; no timing profile",
        "not_yet_replaced": (
            ["CSA attention/indexer/compressor"] if csa_backend == "reference" else []
        )
        + (["HCA attention/compressor"] if hca_backend == "reference" else []),
        "count_semantics": "custom-call instructions including conditional projection branches; not runtime invocation counts",
    }


def export_mhc_evidence(session, output):
    import jax
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
    from sgl_jax.srt.model_loader.deepseek_v4_native import HEAD_SHARDED_WEIGHTS
    from sgl_jax.srt.models.deepseek_v4 import attention_uses_tp

    runner, batch = session.runner, session.last_batch
    if not batch.forward_mode.is_decode():
        raise ValueError("kernel evidence requires an already executed decode batch")
    forward_batch = ForwardBatch.init_new(batch, runner)
    runner.attn_backend.forward_metadata = runner.attn_backend.get_forward_metadata(
        batch
    )
    logits = LogitsMetadata.from_model_worker_batch(batch, runner.mesh)
    with jax.set_mesh(runner.mesh):
        compiled = runner._jitted_run_model.lower(
            runner._model_def,
            runner._model_state_def,
            runner.model_state_leaves,
            forward_batch,
            runner.memory_pools,
            logits,
        ).compile()
    text = compiled.as_text()
    path = output / "decode_optimized_hlo.txt"
    path.write_text(text)
    ownership = []
    for layer_id, (config, weights) in enumerate(
        zip(runner.model.configs, runner.model.layers.get_value(), strict=True)
    ):
        enabled = attention_uses_tp(config, runner.model.attention_tp)
        shards = {}
        for key in sorted(HEAD_SHARDED_WEIGHTS):
            value = weights[key]
            expected = (
                value.shape[0] // 4 if enabled else value.shape[0],
                *value.shape[1:],
            )
            actual = [tuple(part.data.shape) for part in value.addressable_shards]
            if len(actual) != 4 or any(shape != expected for shape in actual):
                raise AssertionError(
                    f"layer {layer_id} has wrong physical head-weight ownership: {key}"
                )
            shards[key] = {"dtype": str(value.dtype), "local_shapes": actual}
        ownership.append(
            {
                "layer": layer_id,
                "ratio": config.ratio,
                "head_tp": enabled,
                "weights": shards,
            }
        )
    moe_evidence = None
    if runner.model.moe_backend in ("gmm", "gmm_tuned"):
        from deepseek_v4_moe_evidence import compiled_moe_evidence

        moe_evidence = compiled_moe_evidence(
            text, layers=len(runner.model.configs), backend=runner.model.moe_backend
        )
    return {
        **compiled_kernel_evidence(
            text,
            mhc_backend=runner.model.mhc_backend,
            hca_backend=runner.model.hca_backend,
            csa_backend=runner.model.csa_backend,
            ratios=[config.ratio for config in runner.model.configs],
            attention_tp=runner.model.attention_tp,
            csa_decode_batch=runner.model.csa_decode_batch,
            fp8_backend=getattr(runner.model, "fp8_backend", "legacy"),
            fused_norm=getattr(runner.model, "fused_norm", False),
            fused_wo_a=getattr(runner.model, "fused_wo_a", False),
        ),
        "head_weight_ownership": ownership,
        "moe_backend": runner.model.moe_backend,
        "moe_evidence": moe_evidence,
        "batch_size": int(batch.real_bs),
        "hlo_file": path.name,
    }
