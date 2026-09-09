# V4 FP4 MoE: standalone optimization gate — 2026-09-09

This is the historical isolated phase. The recovery and subsequent opt-in
model integration are tracked separately in `deepseek_v4_fp4_integration.md`;
do not read the scope statements below as the latest deployment status.

## Status and scope

The first isolated optimization gate passes on **four physical TPU v5p chips,
EP4**. The serving model, loader, shared GMM driver, scheduling, routing,
collectives and backend defaults are **unchanged**. These candidates are not
yet active in the 43-layer ModelWorker/Engine/HTTP path. There is no new
end-to-end tokens/s or 8K/position-8023 acceptance claim in this report.

The new `low_bit/fp4_tuning.py` adapter reuses the existing MegaBlocks GMM
driver. The benchmark uses the existing isolated V4 MoE diagnostic glue and
compares against a freshly compiled, unchanged production `gmm_fp4_experts`.
This is routed MoE only: router, shared expert, attention and KV are excluded.

## Changes actually tested

1. **Compact scale layout:** prepare E8M0 bytes as `[E,K/32,N]` once before
   inference, with the inverse transpose inside the tile's VMEM. The original
   packed FP4 weights are untouched. Optimized HLO has **zero** full W2 scale
   copies per isolated layer, versus **one** in the control. This does not mean
   the 43 copies have already disappeared from the unchanged serving model.
2. **Scale before nibble interleave:** decode low/high FP4 nibble streams,
   perform the same FP32 scale multiplication and BF16 rounding, then interleave
   the resulting BF16 words. Expanded FP32 scales have K/2 rather than K columns.
   This is a dataflow change, not a BF16-scale approximation or a different
   FP4 encoding. All of it stays inside the Pallas tile.
3. **Shape-specific M/N tiles:** the best tested small-token configuration is
   M8/full-K/N256; at M128 it is M32/full-K/N256. Larger M is not universally
   faster. No split-K, altered accumulation tree or precision relaxation was
   introduced. Intermediate token counts and other model dimensions still need
   tuning/gates before defining a general serving selection policy.

Preserved semantics include duplicate expert coalescing, route scaling before
W2 activation FP8 roundtrip, BF16 projection boundaries, and FP32 local expert
combination order. No BF16 expert collection is created in HBM.

## Evidence and numerical acceptance

Git-ignored local evidence, also on the independent TPU data disk:

- `GCP_login/results/v4-fp4-tuning-20260909-01/`: layout-only and initial tile sweep.
- `GCP_login/results/v4-fp4-tuning-20260909-02/`: packed-scale and combined tile sweep.
- `GCP_login/results/v4-fp4-tuning-20260909-03/`: repeat A/B, adversarial routes,
  optimized HLO, 12 raw traces and eight selected exported device profiles.

Official checkpoint revision: `60d8d70770c6776ff598c94bb586a859a38244f1`.
All 256 experts of layer 3 are loaded, 64 per chip. Activations and routes are
the same 128 selected rows of the previously accepted 7,936-token layer capture.
M1/M2/M4 use its first rows; they are not measured concurrent HTTP requests.

After Spot recovery the historical capture was restored from local backup.
`prepare_deepseek_v4_fp4_capture.py` copies only `ffn.norm.npy`, `expert_ids.npy`
and `routing_weights.npy`, **byte-for-byte**, from the original NPZ. Original
archive, member, subset and unchanged schema hashes are saved in the manifest.
The original local 1.5 GB layer-state archive is not modified.

Final isolated report:

- **97 bitwise comparisons pass**, zero rejected candidates or numerical failures.
- Includes a fresh independent NumPy M1 oracle, normal captured routes,
  reordered rows, all-inactive routes, only chip 0/3 active, repeated/cancelled
  expert choices, post-timing outputs and traced replays.
- **29 TPU unit cases pass**, covering partial tiles, empty/non-local groups,
  EP offsets, M8/M16/M32, N128/N256, independent NumPy matmuls and every FP4 code
  with every E8M0 exponent byte, including signed zeros and non-finite cases.
- CPU adapter plus original GMM regression: **48 passed, 8 explicitly skipped**
  hardware cases. Separate byte-subset/accounting tests: **14 passed** locally.
- Full-K and baseline tile-contract rejection tests remain strict.

The candidate changes the broad source fingerprint because the fingerprint
includes newly added low-bit files. Each report records current/historical
fingerprints and candidate/diagnostic source hashes. Old full-model receipts
are not relabeled as receipts for these candidates.

## Repeated warm latency

Final run: 120 samples per variant, rotated execution order, five untimed
warmups. Compilation, setup/transposition, checks and profiler calls are
excluded; no timing outliers are dropped. The preceding independent 80-sample
run reproduced the gains (M4 1.295 → 1.072 ms, M128 5.809 → 3.491 ms).

| Routed tokens | Unchanged production | Selected candidate | Latency reduction |
| --- | ---: | ---: | ---: |
| M1 | 0.5931 ms | 0.5128 ms | 13.5% |
| M2 | 0.8570 ms | 0.7279 ms | 15.1% |
| M4 | 1.3072 ms | 1.0826 ms | 17.2% |
| M128 | 5.8111 ms | 3.5006 ms | 39.8% |

These are warmed host-dispatch-to-ready **single-layer routed-MoE** latencies,
not full-model decode/prefill throughput. Do not multiply them by 43 and claim
a measured model latency or apply the percentage directly to Engine tokens/s.

## Device profile: the savings are in compute and associated waiting

HLO self-time, ms/call averaged over eight TensorCores, eight traced calls per
capture. Each exported capture has all **192 GMM occurrences** and **64 MoE
all-reduce occurrences**. Empty SparseCore planes are excluded.

| Component | M4 control | M4 candidate | M128 control | M128 candidate |
| --- | ---: | ---: | ---: | ---: |
| W1/W3 GMM, including DMA/conversions/dot | 0.3615 | 0.2844 | 2.9157 | 1.6784 |
| W2 GMM, including DMA/conversions/dot | 0.1895 | 0.1495 | 1.5338 | 0.8643 |
| All-reduce, **including wait** | 0.3312 | 0.2595 | 0.7816 | 0.3578 |
| Full W2 scale layout copy | 0.0348 | 0 | 0.0308 | 0 |
| Total HLO self-time, including other glue | 1.0378 | 0.8155 | 5.5327 | 3.2217 |

M128 logical active M-group tiles per projection change from **43/33/48/40**
to **29/22/27/25** across chips. Larger M reduces group-boundary splits and
repeated weight-tile processing. These are logical schedule counts, **not**
measured MXU utilization or physical DMA bytes. M4 keeps counts 4/6/8/2.

Collective code and output shape are unchanged, yet its measured duration falls.
That supports treating collective time as including compute imbalance/waiting,
not as a pure network-bandwidth bill. It does not isolate wire transfer time.
Conversion, DMA and matrix math are fused in Pallas; no independent dequant
instruction time or measured HBM bandwidth is claimed.

XProf logs a reconstructed-HLO `async-update` arity warning. GMM/collective
coverage is complete; `audit_deepseek_v4_fp4_profiles.py` independently verified
all eight exported profiles' durations/counts directly from raw TensorCore XLA
Ops events, excluding duplicate Async Ops. Results match XProf within 1 ns/call.
This audit completed locally using JAX 0.11.1 after the Spot interruption below.

## Memory

Logical raw routed-weight-plus-scale storage is unchanged: **816 MiB/chip**
for this layer. The candidate's scale permutation is compact; it is not a
predecoded BF16 checkpoint. The diagnostic process deliberately retains both
control and candidate scale arrays for alternating comparisons; a future
loader should replace rather than duplicate those arrays.

Compiler-estimated temporary HBM, **not measured serving peak HBM**:

| Shape | Control | Selected candidate |
| --- | ---: | ---: |
| M1 | 32.97 MiB | 1.00 MiB |
| M4 | 33.49 MiB | 2.72 MiB |
| M128 | 43.27 MiB | 53.34 MiB |

The M128 temporary estimate increases by about 10 MiB despite the speedup.
It must be checked again under the complete model allocator/lifetimes. Neither
single-layer temporaries nor tile sizes establish full-model peak VMEM/HBM.

## Next gate: thin model integration, then complete regressions

1. Add an explicit opt-in V4-only loader/adapter choice. Prepare compact scales
   once, preserve a lossless original-format diagnostic view, and leave the
   existing `legacy`/`gmm` defaults unchanged. Do not modify core scheduling.
2. Test intermediate/ragged prefill and decode buckets, all-empty/duplicate
   routes, tensor shapes and dtype/weight-format validation. Choose schedules
   from measured route distributions, not only total M.
3. Run actual 43-layer ModelWorker prefill/decode comparisons and confirm the
   43 runtime scale copies disappear in the **model's** HLO. Revisit position
   8023 and full 8K state/cache correctness with the frozen candidate.
4. Only after numerical acceptance: Engine/HTTP stress, end-to-end throughput,
   memory and final profile. Compressor/state optimization remains a subsequent
   independent task; it was not changed here.

## Post-test Spot interruption

The VM became **PREEMPTED after all tests, benchmarks, traces and XProf exports
completed**. The three result directories, all 12 raw traces, eight selected
exports and current source were already local. No numerical run was interrupted.
The subsequent source push could not connect; the final report, new audit
utility and local audit results remain local-first and must be included on
recovery. Process/exporter logs remain on the independent data disk.

At the final control-plane check around 14:41 UTC, the original 500 GB data disk
was **READY with no attached user**. No VM/disk was deleted or recreated during
this optimization task. Model integration and 8K/Engine/HTTP gates require
restoring the same approved four-chip Spot configuration and reattaching it.
