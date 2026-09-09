# DeepSeek-V4-Flash correctness-first bring-up on four TPU v5p chips

The milestone is a complete official-checkpoint forward and autoregressive
decode on four chips. Passing synthetic tests is **not** this milestone.
The existing baseline is `integration/deepseek-v4` at `d5e58ee6`, plus the
uncommitted v5p mHC/HCA/CSA adaptations. Local source remains authoritative;
every remote test is pushed with a local snapshot and its results pulled back.

## Gates (in order)

1. Preserve the four existing vertical-slice cases (TP1/TP4 prefill/decode and
   ragged equivalence) and their original NumPy-oracle tolerances. Add byte-level
   FP4 E2M1, FP8 E4M3FN, E8M0 block-scale and BF16 loading/conversion tests, and
   real TPU low-bit matmul correctness. Verify four-chip sharding as well as
   values. Do not download or run the complete checkpoint before this gate.
2. Attach a separate 500 GiB balanced persistent data disk, retained independently
   of the Spot VM. Download the entire official `deepseek-ai/DeepSeek-V4-Flash`
   repository at one pinned revision, including tokenizer/encoding files and
   original safetensors. Check every file's size and published digest. Do not
   modify the official checkpoint or store a full BF16 conversion beside it.
3. Validate representative real layers, tracing where each conversion occurs.
   Separate measured latency/HBM use from analytical traffic/VMEM estimates.
   Never label estimates as hardware profiler measurements.
4. Connect all main-model layers, prefill, cache updates and repeated decode.
   Compare checkpoint-native online dequant against a per-layer preconverted
   baseline (the whole model cannot reside in HBM as BF16). Only after
   correctness report GEMM/dequant latency, HBM and throughput trade-offs.

## Format contract

- Routed expert FP4 is E2M1, two values per I8 byte, low nibble first; scales are
  unsigned E8M0 bytes, one per output row and 32 input channels.
- FP8 weights are E4M3FN with compact E8M0 scales per 128x128 block. Shared
  experts are FP8, unlike routed experts.
- The low-bit HBM buffers retain the original bytes and compact scales. The
  correctness-first Pallas matmul decodes only an output-channel tile in VMEM.
- Explicit standalone dequantization materializes BF16 and is only a reference
  or controlled per-layer comparison, never the default checkpoint loader.
- "BS16" in the request is interpreted as BF16.

## Current status

Initial Gate 1 passed on 2026-09-06: **97 passed, 0 failures, 0 skips**, 189 seconds.
This includes the original 71 tests plus 26 low-bit/loader/GEMM/EP-MoE tests.
The four original vertical-slice tests retain their original NumPy tolerances.
This synthetic gate alone did not constitute real-checkpoint validation;
Gates 2–4 below were completed subsequently.

Local report: `GCP_login/results/pytest-20260906T100244Z.xml`.
Local log: `GCP_login/logs/v5p-test-20260906T100233Z.log`.

Final unified regression after Gates 3–4: **101 passed, 0 failures, 0 skips**,
186.85 seconds (original 71 + 29 low-bit + 1 query-batch invariance regression).
Only the existing Flax `.value` deprecation warning remains. Log:
`GCP_login/logs/v5p-test-20260906T114605Z.log`.

New correctness findings:

- A fused BF16 -> FP8 -> BF16 cast round-trip on this JAX/v5p stack failed to
  reproduce FP8 rounding. Explicit integer E4M3 encoding now passes an
  independent CPU oracle, including rounding ties and saturation boundaries.
- FP4 nibble interleaving requires TPU's second-minor-dimension bitcast in
  VMEM; the ordinary JAX stack/reshape does not lower to the v5p vector layout.
- Compact FP8 scale rows cannot be dynamically read at arbitrary byte-row
  alignment. The baseline reads the small scale array, decodes it in VMEM,
  then selects the output block without expanding scales in HBM.
- Routed expert weights are applied before the down projection's activation
  quantization, matching the official V4 inference implementation.

Compiler-reported temporary HBM per device (not a hardware traffic profile):

| Format | Global weight shape N x K | Temporary HBM bytes/device |
| --- | --- | ---: |
| FP4 | 2048 x 4096 | 122,848 |
| FP4 | 4096 x 2048 | 196,576 |
| FP8 | 4096 x 8192 | 147,456 |
| BF16 | 1024 x 4096 | 94,208 |

All four are below a complete BF16 weight shard; the Pallas kernel receives
packed bytes/compact scales and dequantizes its 128-channel tile in VMEM.
These values are shape/compiler-specific, not full-model peak-memory estimates.

Gate 2 passed: all 73 files (159,630,041,626 bytes), including 46 weight
shards and tokenizer/encoding/reference code, were downloaded at revision
`60d8d70770c6776ff598c94bb586a859a38244f1`. Every LFS file matched its published
SHA256; every ordinary file matched its Git blob hash. The download plus
verification took 313 seconds. The original snapshot is unmodified.

The snapshot lives on a separate 500 GiB `pd-balanced` disk mounted at
`/mnt/disks/deepseek-models`; disk UUID is
`6d079512-bcbc-48c5-b510-b5bf17b26eb0`. After reattachment/restart, mount the
existing filesystem by UUID; never reformat it. The boot disk was not changed.

Gate 3 passed the pinned official PyTorch layer oracle on 2026-09-06. GPU-only
kernel entry points are replaced with independent CPU implementations; the
official Attention/Compressor/Indexer/Gate/Expert/mHC Python layer code is used
unchanged. Tests cover SWA/CSA/HCA with 132-token prefill, four cached decode
steps, an additional HCA decode crossing 128 tokens, and complete real-weight
blocks 0/2/3 with all 256 FP4 experts distributed across four chips.

- Attention output NRMSE is 0.12–0.49%, including compression-boundary decode.
- Complete block output NRMSE is 0.307%, 1.056%, 0.583% for layers 0/2/3.
- On identical FFN inputs, routes agree and MoE output NRMSE is 0, 0, 0.0011%.
- End-to-end layer 3 has some different selected experts after upstream BF16
  numerical differences; its propagated FFN-stage NRMSE is 10.97%, while the
  full block output remains 0.583%. This is explicitly reported, not called
  bit-exact model equivalence. Local operator checks use identical inputs.
- FP4 indexer QAT is checked bit-for-bit against CPU on identical inputs.
- The official mHC head collapse, normalization and full 129,280-vocabulary
  logits pass a separate CPU oracle, NRMSE 0.00469%. The head uses the official
  FP32 projection/RMS order, without an extra BF16 normalized-input rounding.

Additional correctness fixes: v5p fused execution did not reliably preserve
plain BF16 casts at quantization boundaries. Explicit integer RNE boundaries
are now used for normalization, RoPE and Hadamard results. A BF16 midpoint/
adjacent-value test passes; low-bit tests now total 29. Vectorized prefill and
explicit decode pooling avoid the failed v5p scan/sliced-reduction lowering.
Short prefill window-index width now matches the official 64-key softmax
block grouping. The diagnostic script retains the legacy scan pooling repro.

Gate 3 report: `GCP_login/results/deepseek-v4-real-layers.json`. The full runner
refuses a gate report from a different numerical-source fingerprint.

**Gate 4 passed on 2026-09-06:** all 43 main layers and all 256 routed experts
per layer are resident on four v5p chips. A 132-token prefill followed by four
cached decode steps agrees with an independent 136-token full-prefill replay:

- Logits: NRMSE **0**, maximum absolute error **0**, identical argmax.
- Visible KV/compressed caches: maximum NRMSE **0** across all 43 layers.
- Last-token residual streams, attention outputs and MoE outputs: identical
  across all 43 layers in the diagnostic trace.
- HBM after load: about **42.52 GiB per chip**, including the small reference
  caches. Allocator-reported peak after decode: **42.66 GiB per chip**.
- Checkpoint load: 100.75 seconds. First prefill and first decode include
  compilation. Diagnostic-run timing includes heavy trace transfer and JSON
  writes, so it must not be presented as clean inference throughput.

Making attention's fixed-size single-query arithmetic independent of prefill
batch shape eliminated an initial 15% full-logit drift. The new regression
requires bitwise batch/single-query equality and an independent CPU oracle.
Cache self-consistency does not establish bitwise equivalence to the entire
official GPU model; the independent CPU comparisons above are representative
operator/block/head checks.

Local full report: `GCP_login/results/deepseek-v4-full-inference.json`.
Trace: `GCP_login/results/deepseek-v4-full-inference.trace.npz`.
Log: `GCP_login/logs/deepseek-v4-gated-full-inference-20260906T113100Z.log`.

A second full run without diagnostic tracing passed the same replay check and
three repeated executions with identical generated IDs/final logits. With
per-layer trace transfers and JSON/file writes disabled, the warmed batch-one
reference measured:

- 132-token prefill: median **2.686 s**, about **49.15 input tokens/s**.
- Cached decode: median **63.90 ms**, about **15.65 output tokens/s** over 12
  steps (three repetitions of the same four-step fixture).
- Warm-run allocator peak: about **42.73 GiB/chip**.
- Official chat encoder smoke: `What is 2 + 2? Reply with just the numeral.`
  generated token IDs `[22, 1]`, parsed content **`4`**, then official EOS.

These timings exclude checkpoint load, compilation and cache reset; decode
includes host greedy selection and all 43 layers. The reference still
synchronizes between layers. This is not a production-serving, large-batch,
long-output or general model-quality benchmark. Chat compilation used a new
18-token prefill shape and is not included in the warmed fixture timings.

Report: `GCP_login/results/deepseek-v4-full-inference-warm-chat.json`.
Log: `GCP_login/logs/deepseek-v4-full-inference-warm-chat-20260906T115200Z.log`.

Detailed execution profiling is now available in
[the v5p profile report](deepseek_v4_v5p_profile.md): per-layer timings,
Attention/MoE/mHC/collective attribution, online conversion regions, and
allocator versus compiler HBM/VMEM accounting. The numerical implementation
is unchanged. Profiler overhead and unavailable hardware traffic counters
are explicitly separated from the warmed inference result.

The reference complete-model runner uses EP=4 (64 experts/chip), replicated non-expert
weights and attention, batch one and a default maximum context of 256. It is
an unoptimized, inspectable baseline, not an HTTP/SGLang serving registration.
There is no CPU weight offload or replacement with synthetic/repeated layers.
MTP files are preserved but speculative decoding is not executed. Long-context
top-k selection, batching and production scheduling are not full-model-tested
by this bounded-context run.

## Completed next milestone: native SGLang-JAX integration

The [native V4 framework integration and whole-model compilation
milestone](deepseek_v4_framework_integration.md) is now complete for its bounded
B1/context=256/4×v5p scope. It keeps checkpoint-native FP4/FP8 storage and online
dequant, and places all 43 layers, cache updates and the output head in one
`ModelRunner` forward executable. The retained per-layer runner above remains
the numerical oracle: full prefill/decode logits and cache/state passed bitwise
comparison. Engine/scheduler smoke, the final 128-test TPU regression and the
post-integration [A/B profile](deepseek_v4_v5p_profile.md) also passed.

## Conversion and intermediate-tensor inventory

| Path | Load-time representation in HBM | Execution-time work |
| --- | --- | --- |
| Routed experts | Original packed FP4 and compact E8M0 scales, expert-sharded | Select one packed expert; unpack/dequant a 128-output-channel tile in VMEM; A8 activation QAT, BF16 GEMM with FP32 accumulation |
| Attention/shared-expert FP8 linears | Original FP8 bytes and compact 128x128 block scales | VMEM tile dequant; activation QAT for the official A8 paths |
| Attention `wo_a` | Original FP8 bytes/scales, replicated | Online dequant without activation QAT; official reference instead preconverts this weight to BF16 |
| Original BF16/F32 tensors | Original dtype, replicated | Normalization/projection arithmetic and explicit BF16 rounding boundaries |
| Main KV cache | Newly allocated BF16 cache | Official FP8-QAT round-trip on 64-value NoPE blocks; RoPE channels stay BF16 |
| Indexer cache/query | Newly allocated BF16 buffers | Hadamard rotation and FP4-QAT round-trip on 32-value blocks |

The loader only copies/selects original weight bytes; no low-bit weight is
fully expanded to BF16. The integer token-to-expert table is range-checked and
losslessly cast I64 to I32. Expert dispatch can create one packed-weight HBM
copy and BF16 activation intermediates; its output is summed across chips.
QAT means quantize then dequantize for the official numerical recipe: it does
not imply packed FP8/FP4 KV storage in this reference implementation.

The baseline Pallas tile is M=8, N=128 and full K. Each M tile rereads and
dequantizes weights; each N tile repeats activation QAT. A logical BF16 weight
tile occupies `128 * K * 2` bytes (1 MiB for K=4096, 2 MiB for K=8192), plus
raw input/scales, FP32 temporaries and other compiler buffers. This logical
size is not peak VMEM. Full HBM traffic and conversion instruction counts
have **not** been measured with a device profiler.

## Initial measured low-bit trade-off (not an optimization result)

Eight real-weight cases passed both BF16 comparison paths. Online vs same-tile
preconverted BF16 is bitwise equal in every case; online vs direct JAX matmul
has NRMSE 0 except FP8 `wq_a` decode at about 1e-8. All paths preserve the same
official activation-quantization recipe. Only one matrix is preconverted at
a time; the official checkpoint remains unchanged.

The following are p50 milliseconds over 20 warmed host-dispatch-to-ready
samples. Each of the four chips runs the whole matrix, matching the local
matrix dimensions in the reference runner, but excluding EP routing/collective
cost. The separate older `deepseek-v4-low-bit-overhead.json` experiment used
output sharding; do not mix those shapes/timings with this replicated run.

| Real layer-2 matrix (N x K) | Tokens | Online | BF16, same Pallas tile | BF16, direct JAX |
| --- | ---: | ---: | ---: | ---: |
| FP4 expert `w1` (2048 x 4096) | 1 | 0.250 | 0.226 | 0.270 |
| FP4 expert `w1` (2048 x 4096) | 128 | 0.791 | 0.348 | 0.261 |
| FP4 expert `w2` (4096 x 2048) | 1 | 0.250 | 0.226 | 0.242 |
| FP4 expert `w2` (4096 x 2048) | 128 | 0.818 | 0.400 | 0.236 |
| FP8 `wq_a` (1024 x 4096) | 1 | 0.223 | 0.217 | 0.263 |
| FP8 `wq_a` (1024 x 4096) | 128 | 0.294 | 0.275 | 0.248 |
| FP8 `wo_b` (4096 x 8192) | 1 | 0.260 | 0.249 | 0.356 |
| FP8 `wo_b` (4096 x 8192) | 128 | 0.770 | 0.637 | 0.287 |

Each FP4 matrix is **4.25 MiB including scales vs 16 MiB BF16** (3.765x storage
ratio). FP8 saves approximately 2x: `wq_a` is 4 MiB + 256 bytes vs 8 MiB;
`wo_b` is 32 MiB + 2048 bytes vs 64 MiB. Whole-model preconversion is not a
resident baseline on this machine: the 43 layers' routed experts alone would
require **516 GiB BF16**, before non-expert weights, KV or temporary buffers.

The FP4 128-token online path is 2.05–2.27x the same-tile BF16 latency; the
corresponding FP8 ratios are 1.07–1.21x. Small decode differences are close to
the host-dispatch floor and should not be interpreted as isolated instruction
latencies. Changing weight format also changes memory movement. The direct
JAX comparison changes scheduling too, so its difference is not pure dequant
cost. These results support profiling repeated tile loading/conversion next,
not selecting an optimization prematurely.

Example analytical HBM input volume: FP4 `w1`, M=128, repeats its 4.25 MiB
packed weight/scales for 16 M tiles, giving **68 MiB/chip**, plus 16 MiB
activation input loads under the baseline schedule. The same executable's
compiler temporary HBM is zero (inputs/outputs are reported separately);
standalone full-matrix dequant instead produces a 16 MiB BF16 output and
about 65.09 MiB compiler temporary HBM. These are compiler accounting and
explicit schedule estimates, not physical traffic/profiler measurements.

Report: `GCP_login/results/deepseek-v4-low-bit-replicated-overhead.json`.
Log: `GCP_login/logs/deepseek-v4-low-bit-replicated-overhead-20260906T115100Z.log`.

## Reproduce on the TPU host

After activating the existing venv and entering the remote source directory:

```bash
source /home/koala/.venvs/sglang-jax/bin/activate
cd /home/koala/sglang-jax
V4_CHECKPOINT=/mnt/disks/deepseek-models/models/deepseek-ai--DeepSeek-V4-Flash/60d8d70770c6776ff598c94bb586a859a38244f1
python scripts/validate_deepseek_v4_checkpoint.py --phase all \
  --checkpoint "$V4_CHECKPOINT" \
  --report /home/koala/sglang-jax-results/deepseek-v4-real-layers.json
python scripts/run_deepseek_v4_reference.py --checkpoint "$V4_CHECKPOINT" \
  --layer-gate /home/koala/sglang-jax-results/deepseek-v4-real-layers.json \
  --report /home/koala/sglang-jax-results/deepseek-v4-full-inference.json
python scripts/benchmark_deepseek_v4_low_bit.py --checkpoint "$V4_CHECKPOINT" \
  --layout replicated \
  --report /home/koala/sglang-jax-results/deepseek-v4-low-bit-replicated-overhead.json
```

Run only one TPU process at a time. Optional full-run flags `--diagnostic-trace`,
`--warm-repeats 3`, and `--chat-prompt 'What is 2 + 2? Reply with just the numeral.'`
produce additional trace, reference timing and official-chat-encoding checks.
On the Mac, `./GCP_login/v5p.sh test` pushes/tests/pulls automatically;
`./GCP_login/v5p.sh pull` saves custom-run results locally.
