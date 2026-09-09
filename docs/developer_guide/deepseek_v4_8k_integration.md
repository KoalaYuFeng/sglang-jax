# DeepSeek-V4-Flash 8K chunked-prefill integration

Date: 2026-09-07. Branch: `integration/deepseek-v4`.

## Status and supported contract

The bounded 8K milestone is complete on one four-chip TPU v5p slice. A normal
`Engine.generate` request now travels through the tokenizer manager, standard
SGLang scheduler, 128-token chunked prefill, native `ModelWorker`, all 43 model
layers, cache update, sampler, and detokenizer. The checkpoint stays in its
official packed FP4/FP8/block-scale representation and is dequantized online;
the loader does not materialize a full BF16 weight copy.

The supported serving contract is deliberately narrow:

- batch one, TP=4, EP=4, DP=1, one request slot;
- BF16 activations and cache, context capacity 8192, page size one;
- `chunked_prefill_size=128` and `max_prefill_tokens>=128` for contexts above
  256;
- radix/prefix reuse, overlap scheduling, hybrid SWA allocation, speculative
  decoding, LoRA, PD disaggregation, and dynamic expert placement disabled.

`context_length=8192` creates an 8192-row model cache. The ordinary worker
reserves one request token and five safety slots, so its advertised maximum
prompt is 8186 tokens. The input validator now treats that maximum as inclusive
instead of rejecting equality. At that prompt length the scheduler separately
clips the allowed generated-token count according to its request-length rule.

## Correctness evidence

The final gates use checkpoint revision
`60d8d70770c6776ff598c94bb586a859a38244f1`.

- A 271-token real-weight whole prefill and scheduler-shaped
  `[128, 128, 15]` prefill have bitwise-identical final logits and every
  relevant cache/scratch field across all 43 layers.
- Separate per-layer diagnostics find no first divergent layer. This covers
  both ratio-4 CSA/index compression and ratio-128 HCA compression.
- The ModelWorker gate executes 8186 tokens as 64 chunks. All active cache rows
  are finite, untouched suffix rows remain zero, and every one of the 21
  ratio-4 index layers reaches 2046 compressed candidates, exceeding
  `index_topk=512`.
- The Engine gate submits one unsplit 8186-token input. Scheduler logs show
  `63 x 128 + 122`; generated IDs `[377, 48159]` exactly match the direct
  ModelWorker oracle. A subsequent 128-token request in the same slot also
  matches `[1465, 721]`, proving reset/reuse after the 8K request.
- The independent official-PyTorch gate passes attention layers 0, 2, and 3,
  ratio-128 decode across positions 127 through 131, representative complete
  layers, and the 129280-vocabulary head. Attention output NRMSE is at most
  0.00327 and the output-head NRMSE is 0.0000469. Layer 3 has one end-to-end
  expert-route flip after an upstream BF16 perturbation; with identical FFN
  input, routes match and FFN NRMSE is 0.0000108. The gate records this rather
  than hiding it behind a text-only comparison.
- The final TPU suite is **141 passed, 0 failed**. It covers low-bit operators,
  mHC/HCA/CSA, KV update, TP/attention, framework metadata/cache, the vertical
  slice, and the inclusive input-boundary regression. The profile analyzer has
  an additional 17 passing CPU unit tests.

## Performance and HBM

Three complete ModelWorker runs agree on the steady full-chunk rate. The final
profile run reports:

| Measurement | Result |
| --- | ---: |
| 128-token steady chunk p50 | 2765.98 ms |
| 128-token steady chunk p95 | 2837.91 ms |
| Steady prefill throughput | 46.28 token/s |
| Full chunks recompiled as position changed | no |
| 8K hot decode, uninstrumented step | 64.81 ms/token |
| Hot decode cache misses | 0 |
| Packed model HBM after load, per chip | 42.90 GiB |
| HBM after 8186-token prefill, per chip | 43.26 GiB |
| Peak HBM, per chip | 43.62 GiB of 95.73 GiB |
| HBM after decode, per chip | 43.32 GiB |

The first Engine request took 560.29 seconds because it included cold
executable creation for the first prefill shape, the 122-token tail, and
decode. That number is not steady serving throughput. After warmup, the short
reset/reuse request took 3.01 seconds including its 128-token prefill and two
generated tokens.

XPlane/XProf captured one early full chunk, one near-8K full chunk, and a hot
decode after a separate decode warmup. Values below are HLO self-time averaged
per TensorCore; they are non-overlapping device attribution, not physical HBM
traffic counters.

| Device stage | Early 128 | Late 128 | Hot 8K decode |
| --- | ---: | ---: | ---: |
| FP4 routed experts | 1920.89 ms | 1784.74 ms | 14.65 ms |
| MoE all-reduce including wait | 312.68 ms | 307.46 ms | 8.83 ms |
| Sparse attention | 323.64 ms | 323.64 ms | 2.62 ms |
| Attention projection/index | 128.34 ms | 128.34 ms | 17.33 ms |
| Compressor | 4.18 ms | 4.17 ms | 2.64 ms |
| Attention top-k sort | 0.97 ms | 0.97 ms | 0.13 ms |
| Norms | 22.87 ms | 22.87 ms | 4.17 ms |
| All attributed and unattributed HLO | 2786.06 ms | 2644.67 ms | 56.35 ms |

The hot decode trace has 66.34 ms instrumented host wall time, 57.51 ms device
module union, and 8.83 ms outside the device module. All eight TensorCore
timelines contain one complete whole-model dispatch, and all 344 expected
all-reduces are present.

The early and late attention costs are effectively identical. The current
static 8K executable scores/masks its fixed candidate capacity, so cache growth
does not trigger a late-context latency cliff. It also means short prefixes do
unnecessary fixed-capacity work. At 128-token prefill, however, top-k itself is
under 1 ms; the dominant target is the FP4 routed-expert online-dequant matmul,
followed by collective wait. At hot 8K decode, attention projection/index is
the largest single stage, followed by routed experts and all-reduce.

XProf's aggregate roofline row labels these programs HBM-bound, but the main
FP4/FP8 custom calls publish zero FLOP/byte counters to that table. Therefore
its aggregate FLOP rate, bandwidth, and efficiency are incomplete compiler
estimates and are not used to claim hardware utilization. The measured
timeline, cache-miss counts, HBM allocator values, and HLO self-times above are
the defensible optimization evidence.

## Implementation details

- `extend_prefix_lens[0]` is the committed absolute prefix and
  `extend_seq_lens[0]` is the current chunk. Continuations must begin on a
  128-token compressor boundary; the final chunk can be shorter.
- Cache reset is a dynamic `positions[0] == 0` conditional. A continuation
  inherits window KV, compressed KV, ratio-4 main/index scratch, and ratio-128
  scratch without specializing on its absolute position.
- Compression writes at absolute `position // ratio`. Ratio-4 overlap pooling
  consumes the prior completed group, ratio-128 partial state survives into
  decode, and short final chunks still see the full 128-token window.
- Router dot products and reductions use fixed trees/lanes so whole, chunked,
  and decode shapes do not introduce TPU reduction-order drift.
- Embedding, all layers, cache updates, mHC collapse, norm, and output head are
  one top-level compiled model call per chunk/token. There is no per-layer host
  dispatch.

## Known performance limitation and next work

Full 128-token chunks reuse one executable at every absolute position. The
model intentionally slices a final partial chunk to its true length so padding
cannot mutate compressor state. Consequently, the first occurrence of each
distinct tail length still compiles a separate executable. Production startup
should precompile the chosen request shapes; the structural fix is a masked
128-token tail path (or a small set of tail buckets) that commits only valid
cache and scratch updates. That change needs the same bitwise whole/chunked
gate before it can replace true-length specialization.

After tail bucketing, optimization priority from the measured profile is:

1. FP4 routed-expert kernel/layout and per-chip expert-load balance;
2. reduce MoE collective wait without changing expert semantics;
3. fuse or reorganize the 8K decode projection/index/compressor path;
4. only then consider adaptive early-context top-k capacity, because the
   measured sort itself is not a material bottleneck.

Concurrency, prefix sharing, radix cache, overlap scheduling, speculative/MTP,
and larger contexts remain separate milestones, not implied support.

## Reproduction and retained artifacts

Run one full-checkpoint process at a time on the four-chip slice:

```bash
python scripts/run_deepseek_v4_8k_worker.py \
  --checkpoint /path/to/original/checkpoint \
  --output /path/to/new-worker-run --profile

python scripts/run_deepseek_v4_8k_engine.py \
  --worker-report /path/to/new-worker-run/report.json \
  --output /path/to/new-engine-run

python scripts/validate_deepseek_v4_checkpoint.py \
  --checkpoint /path/to/original/checkpoint \
  --report /path/to/new-official-gate.json --phase all
```

The final ignored local artifacts are under `GCP_login/results`:

- `deepseek-v4-8k-worker-20260907-run03/` and
  `deepseek-v4-8k-engine-20260907-run03/` for the matched Engine gate;
- `deepseek-v4-real-gate-8k-20260907-run01.json` for the official oracle;
- `deepseek-v4-8k-worker-20260907-run04-profile/` for early/late XPlane and
  XProf exports;
- `deepseek-v4-8k-worker-20260907-run05-hot-profile/` for the clean hot-decode
  trace;
- `pytest-20260906T181942Z.xml` for the 141-test TPU regression.

The original checkpoint remains on the independent model data disk. Code,
reports, raw traces, and XProf exports have been synchronized back locally so
Spot preemption does not lose the only copy.
