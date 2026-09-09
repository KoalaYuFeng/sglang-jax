# V4 prefill chunk experiment and profile (2026-09-09)

Completed: the original 43-layer model passes the bounded B1 8K numerical
gate for chunks 128, 256 and 512. Warm native prefill measures 263.97, 300.18
and 330.02 input tokens/s respectively. All 2081 full-vocabulary comparisons
are bitwise equal to the accepted fixture. This is an experiment, not a
serving-default change or new Engine/HTTP acceptance at larger chunks.

## Plan and execution chronology

User approved the next diagnostic/optimization step after the four-chip
acceptance profile. Start with measurements, not a scheduler rewrite.

1. Preserve the accepted production/reference fingerprints, all six selected
   backend options and original FP4/FP8/block scales. Reuse the existing VM,
   checkpoint and ordinary paged ModelWorker; one process owns the TPU.
2. Extend only the diagnostic session's optional packed token capacity.
   Existing callers retain the 128-token default. Test invalid capacities
   before allocating KV pages, including padding and BF16 state hashing.
3. On the same loaded model, validate B1 chunks 128, 256 and 512 independently.
   Each processes both accepted 7936-token prompts from empty cache and then
   all 256 teacher-forced decode steps through position 8191. Compare every
   available end-of-chunk full-vocabulary golden using unchanged tolerance.
   Compare all layer KV/pages/snapshots/live compressor state at prompt end
   bitwise against the fresh 128-chunk baseline. Preserve the first failure.
4. Only after that chunk passes, run three fresh-prefix warm prefill rounds.
   Trace one chunk at prefix 1024 and one at 7168 in the final round. Exclude
   these profiled calls from throughput. Count only real tokens in the padded
   512-bucket tail. Report cold compilation and diagnostic memory separately.
5. Export raw traces/HLO and attribute prefill time with the existing FP4-MoE
   analyzer. Do not reuse decode percentages or equate HLO byte estimates
   with hardware traffic. Compare chunk measurements and decide which kernel
   merits an independent experiment; do not change service defaults from a
   B1-only test or skip later Engine/concurrency regression.

Each chunk receives a separate complete receipt. If a later chunk fails, the
parent stays incomplete while earlier passed children retain their own scope.
This first sweep is sequential on one VM, not randomized/interleaved A/B, and
is not a serving-throughput or official-accuracy benchmark.

Status (about 06:35 UTC): TPU experiment running in
`profiles/v4-prefill-chunks-20260909-01`. CPU receipt
`v4-prefill-chunk-contracts-20260909-02.xml` has 94 passes and six hardware-only
skips. The first CPU attempt caught four list-versus-array helper errors;
those were fixed without touching production and its failed receipt is retained.
Both attempts' XML/logs are local. No new TPU chunk result is claimed yet.

At about 06:41 UTC, child `chunk128` completed both correctness sequences and
three fresh-prefix timing rounds. The 184 unprofiled calls measure 263.9673
input tokens/s (median 484.8063 ms per 128-token call); both early and late
prefill traces and the executed prefill HLO are saved. This is the fresh state
baseline, not an independent state oracle. `chunk256` is starting; the parent
experiment and larger-chunk acceptance remain incomplete.

At about 06:50 UTC, child `chunk256` is complete: both full decode trajectories
pass and all 706 prompt-end state arrays are bitwise equal to the fresh 128
baseline. Its 91 unprofiled calls measure 300.1765 input tokens/s (median
852.4626 ms per 256-token call). The first cold call took 149.5170 s and is
excluded from these warm measurements. Chunk 512 is in correctness testing;
the parent is not yet complete. No serving default has changed.

At about 06:54 UTC, all three children and the parent completed; the TPU
driver exited zero. Only then did three separate CPU-only XProf exporters
run, each writing a different child directory. All exporters exited zero.
The following sections supersede the in-progress status above.

## Configuration and numerical gate

Receipt: `GCP_login/results/v4-prefill-chunks-20260909-01/report.json`, mirrored
from the independent data disk's `profiles/` directory. The same loaded model
and four-chip v5p Spot VM were used throughout this sequential sweep. The
production fingerprint remains
`4fddb359ae74277a74fdd82d84786249c1c8419122560641ea8a13bbbeb01653`;
the reference remains
`9ef78243dcc50567444feb4bc4dae313000b15651401b46a7ebe6e89d96727d5`.
The pinned inference environment, raw FP4/FP8/E8M0 checkpoint, GMM/EP4,
CSA/HCA head-TP4, DP1 and pure-decode batched CSA selections are unchanged.

Each candidate processes two 7936-token prompts from empty cache, then all
256 teacher-forced decode steps through position 8191. End-of-chunk logits
use the matching accepted 128-token boundary; internal intermediate logits
inside a larger chunk are not separately observed. The accepted native fixture
is `v4-parallel-native-cold-20260909-02`, linked to the previously completed
independent `v4-parallel-oracle-20260909-01` receipt. This does not establish
official GPU/API benchmark accuracy.

| Chunk | Full-logit comparisons, including three timing rounds | Bitwise equal | Prompt-end KV / snapshot / live compressor observations |
| --- | ---: | ---: | --- |
| 128 | 822 | 822 | 706 arrays recorded as fresh self-baseline |
| 256 | 667 | 667 | All 706 bitwise equal to chunk 128 |
| 512 | 592 | 592 | All 706 bitwise equal to chunk 128 |

All comparisons are finite, top-1 equal, max NRMSE zero, without changing the
original 0.005 gate. The 353 arrays per prompt cover all layer-owned pages and
live compressor scratch at the prompt end; these are not observations at
every internal token. Chunk 128 is explicitly not an independent state oracle.
Larger chunks were timed only after both state checks and full decode paths
passed. No TPU numerical failure occurred in this sweep.

Only diagnostic harness/test/docs code changed in this turn. The optional
`PagedWorkerSession.step(..., token_bucket=...)` retains default 128 and rejects
invalid capacities before allocating pages. CPU contracts: 94 passed, six
hardware-only skips; the earlier four list-normalization failures and their
fixed retry receipts are both retained. Core scheduler and production kernels
were not changed.

## Unprofiled warm prefill

Three fresh-prefix rounds alternate fixture 0, 1, 0. Two traced calls in the
last round are excluded; all included calls have zero compile-cache misses.
Throughput is total real input tokens divided by total measured call seconds,
not chunk size divided by median latency. Native timing includes dispatch,
device wait, logits transfer and in-session host checks, not HTTP TTFT.

| Chunk | Included calls | Real input tokens | Median call (ms) | p95 (ms) | Input tokens/s | Gain over 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 184 | 23552 | 484.81 | 490.89 | 263.97 | baseline |
| 256 | 91 | 23296 | 852.46 | 860.74 | 300.18 | 13.72% |
| 512 | 46 | 22784 | 1516.97 | 1527.66 | 330.02 | 25.02% |

Individual warm-round throughput is 263.15/265.74/263.00 for 128,
299.33/302.07/299.08 for 256, and 329.63/331.89/328.35 for 512. This is a
same-machine sequential sweep, not randomized or interleaved A/B.

The 7936-token prompt has a 256-real-token tail in a 512-token bucket. The
three tail calls take 1.2553/1.2544/1.2660 seconds and remain in the 330.02
result. Full 512-token calls alone measure 337.35 input tokens/s; do not use
that more flattering number as whole-prompt throughput. The first two fully
unprofiled prompt rounds spend 30.16/29.86 seconds in calls for 128,
26.51/26.27 for 256, and 24.08/23.91 for 512; these omit host work between
calls and are not complete request latency.

Cold/cache-miss calls are separate: 128 takes 20.3599 s then 0.5865 s;
the first new 256/512 shape takes 149.5170/154.3159 s. These are entire cold
call durations, not separately measured compiler-only time. Historical
persistent cache entries can retain the originating harness stack metadata;
production source identity remains verified.

## Measured prefill breakdown

Each of six captures is one full chunk at prefix 1024 or 7168, on eight active
TensorCores. Values below are exclusive HLO self-time, averaged per TensorCore
per chunk, including waits where indicated. They are not hardware traffic,
MXU utilization or isolated dequantization timing.

Near-8K capture (prefix 7168):

| Source-attributed stage | 128 (ms/chunk) | 256 (ms/chunk) | 512 (ms/chunk) | Share at 512 |
| --- | ---: | ---: | ---: | ---: |
| FP4 gate/up/down GMM including online conversion | 205.51 | 330.76 | 569.12 | 37.74% |
| Attention projection / surrounding index path | 93.74 | 195.36 | 347.07 | 23.02% |
| MoE all-reduce including wait | 52.67 | 91.23 | 165.76 | 10.99% |
| Shared experts / combine | 20.93 | 43.40 | 82.98 | 5.50% |
| Sparse attention | 20.42 | 40.85 | 81.65 | 5.41% |
| Compressor and paged-state operations | 23.28 | 43.11 | 80.67 | 5.35% |
| Other normalization | 23.46 | 41.79 | 79.39 | 5.26% |
| Remaining stages | 35.62 | 56.39 | 101.39 | 6.72% |
| Total HLO self-time | 475.64 | 842.91 | 1508.02 | 100% |

The early-prefix total is 478.68/845.69/1508.99 ms, so these sampled late
chunks do not show a large overall 8K slowdown. The index-score component
does increase: 0.88 to 3.18 ms for 128, 1.52 to 6.13 for 256, and 2.77 to
11.91 for 512. Different chunk shapes include different token/route ranges;
this is not an isolated index-length or expert-route counterfactual.

On a per-token basis, near-8K FP4 GMM falls from 1.6056 to 1.2920 to 1.1116 ms.
This accounts for about 64% of the decrease in sampled HLO self-time per
token between 128 and 512; it is an observed attribution, not proof that all
of that saving is dequantization. Other examples: MoE sums fall from 0.4115
to 0.3238 ms/token, shared GMM metadata from 0.0396 to 0.0112, and scale-layout
copies from 0.0103 to 0.0029. Sparse attention stays about 0.1595 ms/token.

The broad attention bucket must not be described as only CSA/HCA kernels.
The HLO tables retain precise source witnesses. At chunk 512 / prefix 7168:

- FP8 `attn.wo_b` projection is 93.3905 ms and FP8 `attn.wq_b` is 49.1534 ms
  (`deepseek_v4/attention.py:204` and `:52`). Grouped `wo_a` adds 21.4487 ms.
- Q-normalization fixed-tree slices alone account for 70.9488 ms
  (`numerics.py:53`, called from `attention.py:54`). They belong to this broad
  attention bucket, not the separate generic-normalization bucket above.
- The compressor/paged-state bucket includes 27.2375 ms of logical-page
  address gathering (`compressor.py:17`, called from `attention.py:142`).
  This is mapping selected compressed positions to physical pages, not the
  compressor's numerical pooling and not the historical channel-select
  gather. Broad source ownership is not a standalone kernel cost.
- The actual CSA joint-attention Pallas calls total 38.4686 ms; HCA paged
  attention calls total 23.1277 ms. CSA compressor main/index projection calls
  total 12.0179/4.9867 ms. These figures come from all matching measured HLO
  rows, not multiplying a single-layer witness by the layer count.

Coverage checks pass in every capture: 1032 dynamic GMM occurrences
(129 x 8 TensorCores), 344 MoE sums (43 x 8), and 328 attention all-gathers
(41 x 8). Every TensorCore has one `jit_jitted_run_model` and one sampler
dispatch. There are no 43 separate host model-layer launches. Four empty
SparseCore planes are excluded. Executed prefill HLO independently verifies
all 129 GMM call owners; chunk 512's exported final shape has 256 real tokens
in its 512 bucket, while both traces use full 512-token chunks.

Near-8K mean device module-union is 476.40/843.71/1508.86 ms; profiled host
calls are 488.66/857.89/1523.28 ms. Do not mix these with unprofiled medians
as an additive breakdown. XProf emits the known async-update operand-arity
parsing diagnostics; raw timelines and measured HLO tables are preserved,
and the dispatch/dynamic-count checks above pass independently. No hardware
memory-counter export is available. FP4/FP8 conversion remains inside its
GMM/matmul timing and is not separately measured here.

## Memory and next bounded work

Maximum across four chips at each child's end is 44.658/44.790/44.925 GiB.
Allocator high-water is 44.689/44.820/44.956 GiB, respectively. This single
process accumulates compiled variants, and these observations include the
diagnostic KV/state readbacks; they are not isolated serving peaks or proof
of a larger chunk's incremental workspace size. No full BF16 expert copy
was introduced.

The next kernel priority is the FP4 storage adapter inside the existing mature
GMM structure, not a scheduler rewrite. `low_bit/gmm.py` deliberately retains
an 8/full-K/128 tile, fused unpack/scale/BF16 conversion and activation FP8
roundtrip inside each tile. Test tile reuse and conversion placement
independently with real routed inputs and the accepted arithmetic first.
Do not interpret the entire 569.12 ms as removable dequant overhead.

Second, independently test FP8 projection and fixed-tree normalization paths;
`low_bit/matmul.py` is explicitly a correctness-first full-K baseline. Preserve
the rounding/reduction contract, especially the compressor state involved in
the historical position-8023 failure. MoE sum time includes route imbalance
and wait, so it does not establish saturated inter-chip bandwidth.

Chunk 512 is a measured prefill candidate, not automatically the serving
default: one native chunk now takes about 1.52 s versus 0.485 s for 128.
Next serving gate must compare 256/512 in standard Engine mixed/concurrent
prefill+decode, including decode inter-token tails, prefix reuse, cancellation,
KV pressure and HTTP regression. This turn does not claim those new gates
passed and leaves defaults unchanged. The earlier accepted 128-chunk
Engine/HTTP/decode receipts retain their original scope.

## Durable handoff

The complete result tree is local and on the original independent disk:
70 manifested files, 4,407,900,864 bytes, plus the manifest itself. Local
`shasum -a 256 -c SHA256SUMS` passes for every file; the file set is exact,
and the manifest SHA-256 matches the remote value:
`d705942e5bec6b9ec3e15f91de71dd4a5fb162af77d4860bbf9672bced394ae0`.
Parent report SHA-256:
`f2a1b6ce40ed583b9d4c22bac78cb0a8de30abb7ae957727479f73eb30ec35f3`.
Child report hashes are recorded and verified against the parent. The remote
run log is also copied locally; the original live local log remains separate.

At 07:02 UTC the original disk UUID is still mounted, with 326 GiB free;
the source fingerprint remains unchanged on both sides. No experiment,
exporter or HTTP server process is left running. The Spot VM and disk remain
provisioned and billable. `GCP_login` stays ignored, and no GitHub commit or
push was performed.
