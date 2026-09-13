"""Checkpoint-native V4 config and the four-chip integration contract."""

from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"


V4_CHUNK_SIZE = 128
V4_MAX_CONTEXT = 8192


def validate_v4_serving_args(args, model_config):
    """Paged native serving contract; reject unsupported features before loading."""
    required = {
        "tp_size": 4,
        "dp_size": 1,
        "ep_size": 4,
        "moe_dp_size": 1,
        "page_size": 128,
        "attention_backend": "deepseek_v4",
        "disable_hybrid_swa_memory": True,
    }
    for name, value in required.items():
        if getattr(args, name, None) != value:
            raise ValueError(f"V4 paged serving requires {name}={value!r}")
    context = model_config.context_len
    if not V4_CHUNK_SIZE <= context <= V4_MAX_CONTEXT or context % V4_CHUNK_SIZE:
        raise ValueError("V4 context must be page-aligned and between 128 and 8192")
    if args.max_running_requests is None:
        args.max_running_requests = 4
    if args.max_running_requests <= 0:
        raise ValueError("V4 max_running_requests must be positive")
    if args.max_total_tokens is not None and (
        args.max_total_tokens < context or args.max_total_tokens % V4_CHUNK_SIZE
    ):
        raise ValueError("V4 max_total_tokens must be page-aligned and fit one full context")
    chunk = args.chunked_prefill_size
    if chunk is not None and chunk > 0 and args.max_prefill_tokens < chunk:
        raise ValueError("V4 max_prefill_tokens must cover chunked_prefill_size")
    if args.speculative_algorithm or args.enable_lora or args.enable_sequence_parallel:
        raise ValueError("V4 does not support speculative decoding, LoRA, or sequence parallel")
    if (
        args.pd_disaggregation
        or args.ep_dispatch_algorithm
        or getattr(args, "disaggregation_mode", "null") != "null"
    ):
        raise ValueError("V4 does not support PD disaggregation or dynamic expert placement")
    if args.enable_return_routed_experts or args.enable_expert_balance_debug:
        raise ValueError("V4 routed-expert capture/balance debugging is not integrated")
    if getattr(args, "hicache_storage", "disable") != "disable":
        raise ValueError("V4 supports device RadixCache; HiCache offloading is not yet validated")
    if args.kv_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("V4 cache uses BF16 storage with the original FP4/FP8 activation QAT")
    if str(args.load_format) not in ("auto", "safetensors"):
        raise ValueError("V4 requires the original safetensors checkpoint")
