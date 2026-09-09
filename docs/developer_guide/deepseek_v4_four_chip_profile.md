# V4 four-chip acceptance and final profile (2026-09-09)

The selected combined path passes independent TP state tests, full 43-layer
8K numerical regression, Engine lifecycle/natural KV pressure and one complete
HTTP workload/soak receipt. Final native profiling also passes its numerical
checks. This is bounded fixture-based acceptance, not proof of official GPU/API
benchmark accuracy or a production latency SLA.

## Configuration and evidence

- One `v5p-8` Spot VM: four physical v5p chips, eight TensorCores. The original
  independent model disk and official snapshot are reused without re-download.
- Original 43-layer checkpoint, 7936-token prompts, 256 native decode steps
  through position 8191; public serving uses its documented 254-token headroom.
- Raw FP4/FP8/E8M0 weights and online conversion remain selected. No full
  BF16 expert-weight copy is retained.
- MoE EP4/GMM, DP1; attention head-TP4 on CSA/HCA layers 2..42. SWA layers
  0/1, compressor/index state and KV remain replicated. CSA batched projection
  is selected for pure decode, not mixed/prefill. GMM remains explicit opt-in.
- Production fingerprint:
  `4fddb359ae74277a74fdd82d84786249c1c8419122560641ea8a13bbbeb01653`.
  Reference fingerprint:
  `9ef78243dcc50567444feb4bc4dae313000b15651401b46a7ebe6e89d96727d5`.
- Pinned inference freeze is unchanged: Python 3.12.14, JAX/JAXlib 0.11.1,
  libtpu 0.0.46.1. Analysis uses separate CPU-only tools with XProf 2.23.1.

All receipts are under ignored local `GCP_login/results/`, mirrored from the
persistent disk's `profiles/`. Source, tolerances and the independent reference
are unchanged across these model/serving gates. Recovery adds only private
helpers, documentation and an opt-in profiling-harness B1 case with tests;
it does not change a model/kernel or the core scheduler.

| Gate / receipt suffix | Verified result |
| --- | --- |
| Independent TP chunks: `v4-attention-tp-chunks-20260909-02`, expanded `20260909-01` | 3691 comparisons pass; all 3498 selected CSA/HCA comparisons bitwise |
| `v4-rebuild-integration-20260909-02.xml` | 95 TPU tests pass, no skips |
| `v4-rebuild-contracts-20260909-02.xml` | 143 CPU tests pass, 7 hardware-only skips |
| `v4-parallel-native-cold-20260909-02` | 2424 full-logit checks pass; max NRMSE 1.61848e-7 |
| `v4-parallel-oracle-20260909-01` | 514 independently recomputed rows pass; max NRMSE 0.000105272; all top-1 equal |
| `v4-parallel-paged-engine-20260909-01` | 10 overlap/lifecycle cases pass |
| `v4-parallel-engine-8k-20260909-01` | 8 scenarios, 25 requests, 6350 output tokens pass |
| `v4-parallel-engine-pressure-20260909-01` | Natural free-KV=0, one retraction, zero aborts; 5 requests / 1270 tokens pass, including eviction/recompute |
| `v4-parallel-http-20260909-01` | Full 15-case receipt; seven warm B8 rounds / 56 soak requests over 314.53 seconds; protocols, cancellation, disconnect/reuse and final idle pass |
| `v4-parallel-final-profile-contracts-20260909-01.xml` | 43 profiling/reporting tests pass |
| `v4-parallel-native-profile-20260909-01` | 1794 checks pass; max NRMSE 1.61657e-7; six complete device captures |

## Warm native timing

Each batch has 252 unprofiled, no-cache-miss calls. One cold call and three
profiled calls are excluded. Prefix reconstruction/restoration is outside
decode timing. The timed native call includes worker dispatch, device wait,
full-vocabulary logits transfer and host finite/argmax checks; it is neither
pure device latency nor HTTP end-to-end throughput. B1 uses prompt fixture 0;
B2/B4 alternate the two accepted fixtures and reverse request order.

| Batch | Median call (ms) | p95 (ms) | Aggregate output tokens/s |
| --- | ---: | ---: | ---: |
| B1 | 54.42 | 55.28 | 18.38 |
| B2 | 68.78 | 70.67 | 26.98 |
| B4 | 75.00 | 76.60 | 53.39 |

B2 includes a **1.42235-second unprofiled outlier at position 7962**, with no
reported compile miss. Its cause is not determined by the selected traces.
It remains in the aggregate throughput; no outlier trimming was performed.
Do not replace average throughput with `batch / median latency`.

Historical same-workload baseline `v4-csa-overlap-native-profile-20260908-01`
used GMM but not attention TP/batched CSA projection. B4 median falls from
85.48 to 75.00 ms (12.25%); aggregate throughput rises from 46.83 to 53.39
tokens/s (14.02%). B2 median falls 11.44%, while aggregate throughput improves
only 4.71%, retaining the long tail. This is a historical comparison across
replacement physical VMs, not an interleaved same-machine A/B experiment.

The reconstructed serial prefill's warm 128-token chunks have medians
486.50/481.61 ms for the two fixtures, or 263.16/265.71 input tokens/s.
These are native chunk measurements, not full cold prompt TTFT or a batched
scheduler-prefill throughput claim.

## B4 device-time breakdown

The interior trace samples positions 8064/8065. Values below are exclusive
HLO self-time, averaged across eight TensorCores and two calls. They include
collective waits; they are not MXU utilization, HBM traffic or isolated
in-kernel dequantization measurements.

| Stage | ms per call/core | Share of HLO self-time |
| --- | ---: | ---: |
| Attention projection / surrounding index path | 12.955 | 19.89% |
| Compressor and paged-state operations | 11.538 | 17.72% |
| FP4 gate/up GMM, including online conversion | 8.584 | 13.18% |
| MoE all-reduce, including wait | 8.106 | 12.45% |
| Normalization | 4.580 | 7.03% |
| FP4 down GMM, including online conversion | 4.481 | 6.88% |
| Shared GMM metadata/zeroing | 3.236 | 4.97% |
| mHC | 1.961 | 3.01% |
| Shared experts / MoE combine | 1.771 | 2.72% |
| GMM scale-layout copy | 1.324 | 2.03% |
| Other MoE glue, including SwiGLU/combine | 1.270 | 1.95% |
| Other/unattributed HLO | 1.043 | 1.60% |
| Attention index-score kernels | 0.927 | 1.42% |
| Router | 0.855 | 1.31% |
| Sparse attention | 0.720 | 1.11% |
| MoE route/pack/gather | 0.692 | 1.06% |
| Model residual/output-head-attributed operations | 0.688 | 1.06% |
| Attention TP all-gather, including wait | 0.229 | 0.35% |
| Attention top-k | 0.142 | 0.22% |
| Other collectives, including wait | 0.019 | 0.03% |
| **Total HLO self-time** | **65.122** | **100%** |

Measured mean device module-union time is 65.901 ms; the *profiled* host call
averages 82.366 ms. Their difference is not all scheduler idle: host dispatch,
full-logit copying/checks, profiler overhead and timeline accounting differ.
It must not be combined with the unprofiled 75.00 ms median as an additive
breakdown. The model itself is one compiled dispatch per decode, followed by
the existing sampler, not 43 host layer calls.

Against the historical B4 trace, projection/index drops from 19.878 to 12.955
ms, and compressor/state from 13.892 to 11.538 ms. FP4 GMM stays approximately
13.07 ms combined. Attention TP communication is small in this capture;
MoE communication and the still-replicated compressor/state path remain
material. Existing mature kernel reuse alone does not establish that a stage
is no longer a bottleneck. Future changes should be tested independently first.

## Boundaries, coverage and measurement limits

Position 8063 (r4/r128 boundary) has a separate one-call capture. B4 boundary
HLO self-time is 63.969 ms, versus 65.122 ms for interior; compressor/state is
11.605 versus 11.538 ms. The boundary routes different tokens/experts, so total
time differences cannot be interpreted as isolated compressor emission cost.

All six captures have exactly eight active TensorCore timelines. Each call
has one model dispatch and one sampler dispatch on every TensorCore. Static
B4 ownership verifies 21 CSA + 20 HCA head-TP calls, 42 batched CSA projections
and all 129 FP4 GMM calls. Dynamic trace counts per TensorCore/call confirm
129 GMMs, 43 MoE sums and 41 attention all-gathers. Empty SparseCore planes
are explicitly excluded rather than used to dilute averaged times.

XProf logs HLO parsing warnings about an async-update operand arity; the raw
XPlane/Chrome timelines and measured HLO tables remain available, and the
above dispatch/count coverage is checked independently. No hardware memory
counter export is available. Do not infer achieved HBM bandwidth, VMEM
traffic, MXU utilization or a standalone FP4 conversion cost from these data.
Online dequantization remains fused inside the measured GMM calls.

## HBM and serving observations

Allocator statistics, maximum across the four chips:

- Post-load, including the configured 8K cache pool: **44.446 GiB/chip**.
- B4 completion snapshot: **44.864 GiB/chip**.
- Process high-water mark: **49.328 GiB/chip**, including prefix restoration
  and temporary allocations; this is not an isolated decode-intermediate size.
- Historical post-load / peak: 46.368 / 51.176 GiB. B1 is newly included in
  this timing process, so the process-lifetime workload is not identical.

Normal 8K Engine B4's common active-decode window is approximately 49.85 total
tokens/s. HTTP B8 is **queueing up to four active requests**, not eight-way
decode. Across seven warm queued rounds it produces 14224 tokens at 45.57
tokens/s over generation-case time, or 45.22 including soak checks/intervals.
These client/window metrics are not interchangeable with the native table.

Cold responsiveness remains open: the first post-abort logprobs shape logs
compile misses and a 165.41-second status-query wait. It passes the explicitly
declared 1800-second cold observer budget, **not a 60-second SLA**. Warm B8
rounds have no logged compile misses and a maximum observed status query of
0.533 seconds. A next deployment task is to validate the existing prewarm/cache
path for supported serving shapes, without changing scheduler algorithms.

HTTP ends with all four request slots and 33280 KV tokens free. The owned
server is reaped; cleanup reports `returncode=-9, forced_kill=false`, not a
normal zero-exit assertion. No serving/test process is intentionally left up.
The Spot VM and data disk remain provisioned for the user's next task.

## Reproduction and integrity

Run the ordered prerequisites in [four-chip acceptance](deepseek_v4_four_chip_acceptance.md).
The final native driver uses the accepted **same-source** cold receipt with:

```text
--moe-backend gmm --attention-tp --csa-decode-batch
--reuse-goldens <v4-parallel-native-cold-20260909-02/report.json>
--profile --profile-boundary --profile-reused-b1 --check-kernel-dispatch
```

Do not add `--cold-prefill` to this timing workload. CPU analysis uses
`analyze_deepseek_v4_fp4_moe.py --profile <capture-directory> --export` in the
separate analysis environment after the TPU process exits. Retain the raw
traces, HLO tables, grouping witnesses and each failed/incomplete attempt.

Final profile report SHA256:
`1278c6a75586f6d71084e470e2ec4ed356509facb5b754e41368e6d46a104ab5`.
Final B4 optimized HLO SHA256 (identical to the cold numerical gate):
`6e48cbe3a9d416a21bc1d5a47da11770eee79ca49d22d4ab76bd4260b53c77b6`.
Full HTTP report SHA256:
`453a01aa3ebb6ef0bce0624ff77bf1c8abc5a2d502c0b8f74d9b326401d40005`.

All 64 final-profile files (3,756,986,445 bytes), including six XPlanes, six
Chrome traces and every exported analysis file, match the remote copy by size
and SHA256. Local/remote manifests are preserved beside the local result folder.
