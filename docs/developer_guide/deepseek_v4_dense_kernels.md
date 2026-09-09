# Checkpoint-native V4 dense kernels

## Scope and gates

This work adds kernels to the V4-specific SGLang-JAX path. It does not change
scheduling, page allocation, KV ownership, expert routing, or collectives.
Existing CSA/HCA/mHC and checkpoint FP4 GMM kernels are retained. The previous
low-bit/reference source files remain unchanged for numerical comparison.

Implementation order:

1. Standalone exact-tree RMSNorm, QNorm/RoPE and KV conversion tests on v5p.
2. Raw-checkpoint FP8 Linear on the existing SGLang GMM storage-adapter API;
   same-input merged projections packed once during checkpoint loading.
3. Inverse-RoPE + grouped FP8 `wo_a`, with complete groups local to each TP shard.
4. Explicit model switches, loader/receipt contracts, four-chip numerical gates.
5. Real-weight microbenchmarks, then 43-layer ModelWorker numerical checks.
   Complete 8K/Engine/HTTP acceptance is a separate gate before changing defaults.

## Arithmetic and storage contract

| Kernel | Preserved arithmetic | Storage / execution |
| --- | --- | --- |
| FP8 Linear | A8 power-of-two scale roundtrip; scaled BF16 weight; full-K FP32 dot; BF16 output | Raw E4M3 bytes + compact 128×128 E8M0 scales; tile dequant in VMEM; shared GMM scheduling/DMA |
| Merged QA/KV and shared gate/up | Same component output BF16 boundaries | Byte concatenation on host at load time, replacing original arrays; no extra persistent weight bytes or runtime packing |
| RMSNorm | Adjacent-pair FP32 square reduction; original epsilon and BF16 output boundary | Tree intermediates in Pallas VMEM instead of outer XLA slice programs |
| QNorm/RoPE | BF16 square, mean, epsilon-add, rsqrt, normalized Q, then BF16 RoPE | One Pallas program; shared FP32 phase from the encompassing model JIT |
| KV norm/RoPE | Original weighted norm, BF16 rotation, A8 roundtrip on non-RoPE channels with block 64 | One Pallas program |
| Inverse-RoPE/wo_a | BF16 inverse rotation, **no activation quantization**, scaled BF16 weight/full-K FP32 dot | One grouped program; no HBM inverse-RoPE/group staging tensor |

QA/KV/shared weights remain replicated. Q heads, sink and complete `wo_a`
groups retain the existing TP4 placement. BF16 projected results are gathered
before the unchanged full-K `wo_b`; there is no new floating-point collective.

The normalization kernel uses the exact adjacent-pair tree, not a generic mean.
Its two-dimensional rotate/add network returns lane zero of the same tree;
it avoids both unsupported stride-2 vector slicing and oversized padding of a
new minor dimension of length two. QNorm batches token/head rows in VMEM;
rotary pairs reuse the existing FP4 adapter's BF16 word-interleave primitive.
YaRN phase generation must be compared inside the same enclosing JIT: eager
primitive-by-primitive FP32 phase evaluation can itself change rotary outputs.
No test tolerance or independent oracle is relaxed for these kernels.
The full YaRN/RoPE bit-exact test is TPU-specific: CPU interpreter/LLVM FMA
contraction differs at two BF16 outputs in the million-element fixture.
CPU tests separately check the exact Q normalization with identity rotation;
all full rotary paths retain strict bitwise checks on real v5p.

## Opt-in integration

Defaults remain unchanged. The existing acceptance scripts now accept:

```text
--fp8-backend gmm
--merged-projections
--fused-norm
--fused-wo-a
```

The corresponding `json_model_override_args` keys are `v4_fp8_backend`,
`v4_merged_projections`, `v4_fused_norm` and `v4_fused_wo_a`. Merging requires
the FP8 GMM backend; invalid selections fail instead of silently falling back.
Receipts bind every selection to the actual model. Read-only diagnostic
unpacking restores original raw-byte names for the independent oracle only.

Standalone tests:

```bash
python -m pytest -q python/sgl_jax/test/kernels/test_deepseek_v4_projection_kernels.py
```

Real-weight microbenchmark (idle four-chip TPU):

```bash
python scripts/benchmark_deepseek_v4_projection_kernels.py \
  --checkpoint /mnt/disks/deepseek-models/models/deepseek-ai--DeepSeek-V4-Flash/60d8d70770c6776ff598c94bb586a859a38244f1 \
  --output /home/koala/sglang-jax-results/NEW_DIRECTORY \
  --tokens 4 128 --tiles 8 32 --repeats 30
```

This benchmark uses official raw weights and deterministic random activations,
not captured hidden states. It measures warmed host dispatch through completion
on four replicas of each local TP4 shard shape. Compilation, numerical checks,
collectives and the complete serving path are excluded. HLO and compiler HBM
temporary estimates accompany each result; they are not measured HBM traffic.

## Validation status

The revised layout passes 35 standalone v5p cases including actual head-TP4
(`v4-dense-tpu-final.xml`). CPU kernel/receipt/accounting tests pass 82 cases
with six explicitly hardware-specific skips (`v4-dense-cpu-accepted.xml`).
Separate model/parallel/mHC option contracts pass 61 cases with 22 TPU-only
skips. Original-weight A/B receipt
`v4-projection-microbench-20260909-03/report.json` passes all 44 comparisons
bitwise. A failed first fusion experiment and all failed compiler attempts
are retained rather than relabeled as successful runs.

Selected warmed host-inclusive microbenchmarks, 128 tokens, local TP4 shapes:

| Operation | Baseline ms | New ms | Speedup |
| --- | ---: | ---: | ---: |
| RMSNorm, width 4096 | 0.5011 | 0.2332 | 2.15× |
| QNorm + RoPE, 16 heads | 0.2514 | 0.2436 | 1.03× |
| FP8 Q expansion | 0.5433 | 0.3696 | 1.47× |
| Merged QA + KV | 0.3511 | 0.3096 | 1.13× |
| Merged shared gate + up | 0.5724 | 0.3959 | 1.45× |
| Inverse RoPE + grouped wo_a | 0.3861 | 0.3299 | 1.17× |

These are isolated comparisons, not additive whole-model speedups. Decode
differences for small projections are close to the host-dispatch floor, so no
large decode improvement is inferred. Eight-row prefill variants can regress:
the experimental model path uses M8 below 32 tokens and M32 otherwise. Larger
tiles remain available only for explicit independent tests/experiments.

The 43-layer ModelWorker gate is complete:
`GCP_login/results/v4-dense-paged-20260909-01/report.json`. All ten execution
options were checked against the actual model, including all four new choices.
The bounded workload uses context capacity 384, two 131/132-token prompts,
ragged chunks, B1/B2/B4, reordered continuation/decode and freed-slot reuse.
All 14 full-vocabulary comparisons pass the unchanged 0.005 gate, with finite
outputs and matching top-1. Maximum NRMSE is 0.0000488059. Every comparison's
recorded metrics is identical to the prior accepted
`v4-parallel-paged-20260909-01` report. This is **not** a claim that all logits
are bitwise equal to the independent reference or that 8K has been retested.

The executed B4 decode HLO contains 236 checkpoint-FP8 GMM calls, 173 exact
RMSNorm calls, 43 fused QNorm/RoPE calls and 43 fused inverse-RoPE/wo_a calls.
It retains 129 original FP4 GMM calls (W1/W3/W2 in every layer), all 21 CSA and
20 HCA layers, and the original mHC kernels. These are compiled instruction
counts, not host launches or measured dynamic program invocations. The local
HLO file's SHA256 matches the receipt:
`545c3c8c3127dbc579c4cde85b123b96b6d6e94a361e7c73d2958b1496e869e6`.

First calls for new batch/mode shapes take roughly 97–120 seconds, including
cold compilation. They are not warm serving latency. The diagnostic process
also holds independent reference views and several executables; its final
HBM allocation is not a clean serving-memory measurement. All test processes
exited, and code, XML, JSON and HLO artifacts were mirrored to the local
Git-ignored `GCP_login/results/` directory. No Git commit/push was performed.

Defaults remain unchanged. A subsequent explicitly enabled
[same-machine whole-model A/B profile](deepseek_v4_dense_profile.md) now covers
two 7936-token prompts and B1/B2/B4 decode through position 8191. All 3832
candidate output checks are bitwise equal to that run's control. Warm prefill
improves about 23%, and aggregate decode about 24%/17%/17% for B1/B2/B4.
That ModelWorker experiment does not replace cold concurrent-prefill/internal
state or Engine/HTTP acceptance; those remain gates before default promotion.

Production source fingerprint:
`ee32d17da94f49d7ae035d8df662b6588b32e006d109e63775c1d374b4fcb4fc`.
Unchanged independent reference fingerprint:
`9ef78243dcc50567444feb4bc4dae313000b15651401b46a7ebe6e89d96727d5`.
