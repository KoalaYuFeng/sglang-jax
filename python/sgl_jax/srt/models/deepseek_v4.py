"""Checkpoint-native V4 on the standard packed-batch ModelRunner interface.

Only V4 kernels/cache semantics are specialized. Scheduling, page ownership,
logits selection, sampling and the whole-model JIT remain framework-owned.
"""

import logging
from dataclasses import replace

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.deepseek_v4.attention import attention
from sgl_jax.srt.kernels.deepseek_v4.csa import validate_backend as validate_csa_backend
from sgl_jax.srt.kernels.deepseek_v4.dense import DenseKernels
from sgl_jax.srt.kernels.deepseek_v4.hca import validate_backend as validate_hca_backend
from sgl_jax.srt.kernels.deepseek_v4.mhc import head_collapse, post as mhc_post, validate_backend
from sgl_jax.srt.kernels.deepseek_v4.moe import moe, validate_backend as validate_moe_backend
from sgl_jax.srt.kernels.deepseek_v4.numerics import (
    config_for_layer,
)
from sgl_jax.srt.kernels.mhc import mhc_pre_fused
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsProcessor
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.model_loader.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from sgl_jax.srt.model_loader.deepseek_v4_native import load_layer, weight_specs

logger = logging.getLogger(__name__)


def _boolean_option(config, name):
    value = getattr(config, name, False)
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def attention_uses_tp(config, enabled):
    # The first two SWA layers retain their verified full-head XLA reduction
    # shape. Only the independently exact CSA/HCA Pallas paths are selected.
    return enabled and config.ratio in (4, 128)


def local_attention_config(config, enabled):
    if not attention_uses_tp(config, enabled):
        return config
    if config.heads % 4 or config.groups % 4:
        raise ValueError("V4 head TP4 must keep complete heads and wo_a groups")
    return replace(config, heads=config.heads // 4, groups=config.groups // 4)


def use_batched_csa_decode(enabled, forward_mode):
    return enabled and forward_mode.is_decode()


class DeepseekV4DecoderLayer(nnx.Module):
    """V4 residual/attention/MoE wrapper; packed weights are dynamic arguments."""

    def __init__(
        self,
        config,
        *,
        mhc_backend="pallas",
        hca_backend="pallas",
        csa_backend="pallas",
        moe_backend="legacy",
        attention_tp=False,
        csa_decode_batch=False,
        dense_kernels=DenseKernels(),
    ):
        self.config = local_attention_config(config, attention_tp)
        self.output_tensor_axis = "tensor" if attention_uses_tp(config, attention_tp) else None
        self.csa_decode_batch = csa_decode_batch
        self.dense_kernels = dense_kernels
        self.mhc_backend = validate_backend(mhc_backend)
        self.hca_backend = validate_hca_backend(hca_backend)
        self.csa_backend = validate_csa_backend(csa_backend)
        self.moe_backend = validate_moe_backend(moe_backend)

    def __call__(self, streams, positions, token_ids, weights, cache, metadata, locations):
        config = self.config
        for sublayer in ("attn", "ffn"):
            residual = streams
            x, post, comb = mhc_pre_fused(
                streams,
                weights["hc_" + sublayer + "_fn"],
                weights["hc_" + sublayer + "_scale"],
                weights["hc_" + sublayer + "_base"],
                hc_mult=config.hc,
                sinkhorn_iters=config.sinkhorn_iters,
                norm_eps=config.eps,
                hc_eps=config.hc_eps,
                dot_precision=jax.lax.Precision.HIGHEST,
            )
            x = self.dense_kernels.norm(x, weights[sublayer + "_norm.weight"], config.eps)
            if sublayer == "attn":
                x, cache = attention(
                    x,
                    positions,
                    weights,
                    cache,
                    config,
                    metadata,
                    locations,
                    hca_backend=self.hca_backend,
                    csa_backend=self.csa_backend,
                    output_tensor_axis=self.output_tensor_axis,
                    csa_decode_batch=self.csa_decode_batch,
                    dense_kernels=self.dense_kernels,
                )
            else:
                x = moe(
                    x,
                    token_ids,
                    weights,
                    config,
                    metadata,
                    backend=self.moe_backend,
                    dense_kernels=self.dense_kernels,
                )
            streams = mhc_post(x, residual, post, comb, backend=self.mhc_backend)
        return jnp.where(metadata.token_valid[:, None, None], streams, 0), cache


def whole_model_forward(
    embedded,
    positions,
    ids,
    shared,
    layers,
    caches,
    metadata,
    locations,
    *,
    configs,
    mhc_backend="pallas",
    hca_backend="pallas",
    csa_backend="pallas",
    moe_backend="legacy",
    attention_tp=False,
    csa_decode_batch=False,
    dense_kernels=DenseKernels(),
):
    """All layers and request-local cache updates in one device program."""
    streams = jnp.repeat(embedded[:, None, :], configs[0].hc, axis=1)
    updates = []
    for layer_id, (config, weights, cache) in enumerate(zip(configs, layers, caches, strict=True)):
        with jax.named_scope(f"v4_layer_{layer_id}"):
            streams, cache = DeepseekV4DecoderLayer(
                config,
                mhc_backend=mhc_backend,
                hca_backend=hca_backend,
                csa_backend=csa_backend,
                moe_backend=moe_backend,
                attention_tp=attention_tp,
                csa_decode_batch=csa_decode_batch,
                dense_kernels=dense_kernels,
            )(streams, positions, ids, weights, cache, metadata, locations)
        # Retain the independently verified BF16 layer boundary without host
        # dispatch/synchronization or a separate compiled executable per layer.
        streams = jax.lax.optimization_barrier(streams)
        updates.append(cache)
    config = configs[0]
    with jax.named_scope("v4_head_collapse"):
        collapsed = head_collapse(
            streams,
            shared["hc_head_fn"],
            shared["hc_head_scale"],
            shared["hc_head_base"],
            eps=config.eps,
            hc_eps=config.hc_eps,
            backend=mhc_backend,
        )
        hidden = dense_kernels.norm(collapsed, shared["norm.weight"], config.eps)
    return hidden, tuple(updates)


class V4LogitsProcessor(LogitsProcessor):
    def _get_logits(self, hidden_states, lm_head):
        # Reuse standard pruning/logprobs/hidden capture, but retain the
        # official head's FP32 accumulator/output instead of rounding to BF16.
        hidden_states = jax.reshard(hidden_states, NamedSharding(self.mesh, P("data", None)))
        logits = jnp.matmul(
            hidden_states,
            lm_head.embedding.get_value().T,
            preferred_element_type=jnp.float32,
            out_sharding=NamedSharding(self.mesh, P("data", "tensor")),
        )
        return logits[:, : self.vocab_size]


class DeepseekV4ForCausalLM(nnx.Module):
    @classmethod
    def patch_model_config(cls, mc):
        from sgl_jax.srt.configs.deepseek_v4 import V4_CHUNK_SIZE, V4_MAX_CONTEXT

        if not V4_CHUNK_SIZE <= mc.context_len <= V4_MAX_CONTEXT or mc.context_len % V4_CHUNK_SIZE:
            raise ValueError(
                f"V4 context length must be a multiple of {V4_CHUNK_SIZE} up to {V4_MAX_CONTEXT}"
            )
        mc.sliding_window = None  # the V4 pool owns window + compressed KV
        mc.hf_config.v4_max_context = mc.context_len
        if mc.dtype != jnp.bfloat16:
            raise ValueError("V4 requires BF16 activations with original low-bit weights")

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        self.mesh, self.dtype = mesh, dtype
        self.mhc_backend = validate_backend(getattr(config, "v4_mhc_backend", "pallas"))
        self.hca_backend = validate_hca_backend(getattr(config, "v4_hca_backend", "pallas"))
        self.csa_backend = validate_csa_backend(getattr(config, "v4_csa_backend", "pallas"))
        self.moe_backend = validate_moe_backend(getattr(config, "v4_moe_backend", "legacy"))
        self.attention_tp = _boolean_option(config, "v4_attention_tp")
        self.csa_decode_batch = _boolean_option(config, "v4_csa_decode_batch")
        self.fp8_backend = getattr(config, "v4_fp8_backend", "legacy")
        self.fused_norm = _boolean_option(config, "v4_fused_norm")
        self.merged_projections = _boolean_option(config, "v4_merged_projections")
        self.fused_wo_a = _boolean_option(config, "v4_fused_wo_a")
        self.dense_kernels = DenseKernels(
            self.fp8_backend, self.fused_norm, self.merged_projections, self.fused_wo_a
        )
        logger.info("V4 experimental dense kernels: %s", self.dense_kernels)
        if self.csa_decode_batch and self.csa_backend != "pallas":
            raise ValueError("batched CSA decode requires the original Pallas projection")
        if self.attention_tp and (self.csa_backend != "pallas" or self.hca_backend != "pallas"):
            raise ValueError("V4 attention TP requires the independently validated Pallas paths")
        logger.info(
            "V4 attention TP4=%s (CSA/HCA only; SWA/index/compressor/KV replicated); "
            "batched CSA decode=%s",
            self.attention_tp,
            self.csa_decode_batch,
        )
        logger.info(
            "V4 MoE backend=%s (GMM remains opt-in pending full-model gates)", self.moe_backend
        )
        logger.info(
            "V4 mHC backend=%s; HCA backend=%s; CSA backend=%s (explicit BF16/QAT KV)",
            self.mhc_backend,
            self.hca_backend,
            self.csa_backend,
        )
        self.configs = tuple(
            config_for_layer(config.to_dict(), i, config.v4_max_context)
            for i in range(config.num_hidden_layers)
        )
        for layer_config in self.configs:
            local_attention_config(layer_config, self.attention_tp)
        if (
            config.expert_dtype != "fp4"
            or config.scoring_func != "sqrtsoftplus"
            or config.n_shared_experts != 1
            or config.n_routed_experts != 256
            or config.num_hidden_layers != 43
            or mesh.shape.get("tensor") != 4
            or mesh.shape.get("data", 1) != 1
            or mesh.size != 4
        ):
            raise ValueError(
                "unsupported V4-Flash architecture or mesh; expected 43 layers / EP=4 / DP=1"
            )
        self.embed_tokens = Embed(
            config.vocab_size, config.hidden_size, dtype=dtype, kernel_axes=(None, None), mesh=mesh
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, dtype=dtype, mesh=mesh)
        self.logits_processor = V4LogitsProcessor(config.vocab_size, mesh)
        # Nested raw-format pytrees stay dynamic under nnx.split/merge; Flax's
        # shape-only constructor must not create an empty Param on an Explicit mesh.
        self.shared, self.layers = nnx.data(None), nnx.data(None)

    def load_weights(self, model_config):
        if jax.default_backend() != "tpu":
            raise RuntimeError("V4 real-checkpoint loading requires four TPU devices")
        if model_config.quantization_config is not None:
            raise ValueError("V4 native mixed-format loader cannot use generic quantization rules")
        checkpoint = DeepSeekV4Checkpoint(model_config.model_path)
        expected = tuple(
            config_for_layer(checkpoint.config, i, self.configs[0].max_context)
            for i in range(checkpoint.config["num_hidden_layers"])
        )
        if self.configs != expected:
            raise ValueError("V4 architecture overrides must not change the original checkpoint")
        self.embed_tokens.embedding = nnx.Param(
            jax.device_put(checkpoint.read_tensor("embed.weight"), NamedSharding(self.mesh, P()))
        )
        self.lm_head.embedding = nnx.Param(
            jax.device_put(
                checkpoint.read_tensor("head.weight"), NamedSharding(self.mesh, P("tensor", None))
            )
        )
        shared = {
            name: jax.device_put(checkpoint.read_tensor(name), NamedSharding(self.mesh, P()))
            for name in checkpoint.weight_map
            if not name.startswith(("layers.", "mtp."))
            and name not in ("embed.weight", "head.weight")
        }
        layers = []
        for i in range(len(self.configs)):
            layers.append(
                load_layer(
                    checkpoint,
                    i,
                    self.mesh,
                    attention_tp=attention_uses_tp(self.configs[i], self.attention_tp),
                    merged_projections=self.merged_projections,
                    fp4_scale_transposed=self.moe_backend == "gmm_tuned",
                )
            )
            logger.info("V4 loaded original FP4/FP8 layer %d/%d", i + 1, len(self.configs))
        self.shared, self.layers = nnx.Param(shared), nnx.Param(tuple(layers))
        logger.info("V4 loaded all 43 layers, 256 experts/layer; no BF16 weight expansion")

    def __call__(self, forward_batch, memory_pools, logits_metadata):
        if forward_batch.forward_mode not in (
            ForwardMode.EXTEND,
            ForwardMode.DECODE,
            ForwardMode.MIXED,
        ):
            raise ValueError(f"V4 does not support {forward_batch.forward_mode}")
        if self.shared is None or self.layers is None:
            raise ValueError("V4 weights have not been loaded")
        layers = self.layers.get_value()
        embedded = self.embed_tokens(forward_batch.input_ids)

        def forward(*args):
            return whole_model_forward(
                *args,
                configs=self.configs,
                mhc_backend=self.mhc_backend,
                hca_backend=self.hca_backend,
                csa_backend=self.csa_backend,
                moe_backend=self.moe_backend,
                attention_tp=self.attention_tp,
                csa_decode_batch=use_batched_csa_decode(
                    self.csa_decode_batch, forward_batch.forward_mode
                ),
                dense_kernels=self.dense_kernels,
            )

        replicated = NamedSharding(self.mesh, P())
        hidden, updates = jax.shard_map(
            forward,
            mesh=self.mesh,
            in_specs=(
                P(),
                P(),
                P(),
                P(),
                tuple(
                    weight_specs(w, attention_tp=attention_uses_tp(c, self.attention_tp))
                    for w, c in zip(layers, self.configs, strict=True)
                ),
                P(),
                P(),
                P(),
            ),
            out_specs=(P(), P()),
            check_vma=False,
        )(
            jax.reshard(embedded, replicated),
            jax.reshard(forward_batch.positions, replicated),
            jax.reshard(forward_batch.input_ids, replicated),
            self.shared.get_value(),
            layers,
            memory_pools.token_to_kv_pool.layers,
            forward_batch.attn_backend.forward_metadata,
            jax.reshard(forward_batch.out_cache_loc, replicated),
        )
        output = self.logits_processor(hidden, self.lm_head, logits_metadata)
        return output, {"token_to_kv_pool": updates}, True, []


EntryClass = DeepseekV4ForCausalLM
