# DeepSeek V4 FP4 MoE breakdown — 2026-09-08

## Scope and conclusion

Measured on the existing v5p-8 Spot VM: **four physical v5p chips**, EP4,
eight TensorCore execution lanes. This is the accepted **opt-in `gmm`** V4
backend, not the retained `legacy` default and not an Engine/HTTP benchmark.
No production model, kernel, backend default or framework code was changed
for this diagnostic. Only diagnostic scripts, accounting tests and this report
were added.

The important distinction is:

- Within FP4 routed MoE, the online **weight-processing/tiled-compute path**
  is a larger opportunity than just hoisting activation FP8 roundtrip.
- Across the full current model, MoE is no longer the old two-thirds bottleneck.
  Including its helpers and collective waits, it accounts for about **37%**
  of profiled TensorCore HLO self-time; attention/compressor work accounts
  for about **52%**.
- Predecoded BF16 is a diagnostic counterfactual, not a deployable full-model
  recommendation. The routed weights alone would require **129 GiB/chip**.

## Evidence and acceptance

Local evidence is under the Git-ignored `GCP_login/results/`:

- `v4-moe-breakdown-native-20260908-01/`: unchanged full-model profiles,
  full logits, optimized HLO, numerical receipt and GMM dispatch proof.
- `v4-fp4-moe-diagnostic-20260908-01/`: single-layer counterfactuals,
  alternating timing samples, optimized HLO, profiles and bitwise checks.
- Each capture has `analysis/<label>/fp4_moe_breakdown.json`, full
  `hlo_stats.json`, per-core timelines and raw XPlane traces.

Remote originals are on the independent data disk at
`/mnt/disks/deepseek-models/profiles/`, not the nearly-full boot disk.

Provenance:

- Framework fingerprint:
  `a8b53fea756d23a8e714b4c5d79814c6fef1b5e499fdb9d8f117f58ba199cb4c`.
- Official checkpoint revision:
  `60d8d70770c6776ff598c94bb586a859a38244f1`.
- Native optimized HLO SHA256:
  `eeffad3ef0b4b6f645f720fa3f5b1563a08d844bf664924ff8e5b0f767420c05`,
  identical to the earlier accepted GMM 8K gate.
- Native receipt SHA256:
  `aa1876bdc52d641b3a61ed55e7a6b17b6bf3b1b925ef8d83c0e49c93e5326f42`.
- This native replay passed **1,538 numerical checks**, zero failures:
  two reconstructed B1 prefill endpoints and 1,536 B2/B4 decode-row checks
  through position 8191. It reused the earlier accepted same-source B1 full
  logits; this is not a newly independent model-accuracy benchmark.
- All **28 single-layer comparisons are bitwise**, including an independent
  NumPy M1 oracle, diagnostic variants and profiled replays.
- Eight CPU accounting unit tests pass. Twelve additional CPU cases compare
  analytical routing/tile counts with the real shared grouping helpers,
  including duplicate, cancelled and zero-weight routes.

## Current execution and storage

`deepseek_v4/moe_gmm.py` groups routes with the shared stable permutation,
then invokes shared MegaBlocks GMM for W1, W3 and W2. Each call uses
`low_bit/gmm.py::CheckpointFP4Rhs`, with an **8/full-K/128** tile.

The weight path is raw packed FP4/E8M0 in HBM, tile DMA, FP4 unpack in VMEM,
E8M0 decode and per-32 scale expansion/multiplication, conversion to BF16,
then BF16 dot with FP32 accumulation. No whole BF16 expert is stored in HBM
by this online path. GMM output rounds to BF16.

Activation FP8 roundtrip is also inside the tile: per-128 amax, power-of-two
scale, explicit E4M3 rounding/encoding and BF16 reconstruction. It is not
merely an inexpensive dtype cast. W1/W3 share the same mathematical input,
but the current tile body computes its roundtrip separately. W2 roundtrip
must remain **after** SwiGLU and route-weight scaling for numerical fidelity.

W1/W3 have 16 N tiles each; W2 has 32. This gives repeated conversion work
in the source-level tile schedule. The controlled experiment below measures
the benefit of moving it outside GMM; it does not assume the compiler executes
every source operation literally.

Logical storage/working-set sizes, **not measured traffic or simultaneous
peak VMEM allocation**:

| Item | FP4 + E8M0 | Predecoded BF16 |
| --- | ---: | ---: |
| One expert's W1/W3/W2 | 12.75 MiB | 48 MiB |
| 64 experts, one layer, one chip | 816 MiB | 3 GiB |
| Routed experts, 43 layers, one chip | 34.266 GiB | 129 GiB |

A decoded W1/W3 weight tile is 1 MiB in BF16; a W2 tile is 0.5 MiB.
Intermediate FP32 values/scales add VMEM work, but their actual live ranges
are compiler-dependent. Full-model loaded HBM was about 46.37 GiB/chip;
profiling buffers raised this run's peak to about 51.20 GiB/chip. That peak
must not replace the earlier unprofiled inference memory measurement.

## Full-model decode breakdown

Native 43-layer, 7936-token prefill plus decode to 8191. B2/B4 restore real
B1 prefix pages outside timing. Warm decode medians are **82.189 ms (B2)**
and **92.544 ms (B4)**, excluding compilation and profiled calls.

The following B4 table uses the two interior profiled steps at positions
8064/8065. It is **HLO self-time in ms per call, averaged over eight
TensorCores**, not host latency, not a sum across chips and not pure MXU time.

| Work | ms/call | Share of device HLO self-time |
| --- | ---: | ---: |
| MoE W1/W3 GMM, including online conversions/DMA/dot | 8.617 | 10.4% |
| MoE W2 GMM, including online conversions/DMA/dot | 4.491 | 5.4% |
| MoE all-reduce, **including waiting** | 8.208 | 9.9% |
| Shared GMM metadata, scalar prefetch and zeroing | 3.342 | 4.0% |
| W2 scale layout copies | 1.330 | 1.6% |
| Route packing/gather, SwiGLU, unpermutation and local combine | 1.891 | 2.3% |
| Router and shared-expert/remaining MoE combine | 2.676 | 3.2% |
| **MoE subtotal** | **30.556** | **36.9%** |
| Attention/compressor/paged state/index/top-k | 42.639 | 51.5% |
| Norm, mHC, head/residual and remaining work | 9.521 | 11.5% |
| Total | 82.715 | 100% |

In these profiled calls, mean module-union time is 83.533 ms and host time is
100.000 ms. The difference includes harness/host activity and profiler effects;
it is not 43 per-layer Python dispatches. Do not subtract profiled device time
from a different, unprofiled median to manufacture a host-overhead estimate.

Both interior profiles have all **2,064 GMM occurrences**
(`43 × 3 × 8 TensorCores × 2 calls`) and **688 MoE collectives**. The boundary
capture at position 8063 also has full coverage; its total is 81.420 ms and is
kept separate rather than averaged into the interior profile.

The exporter emits four empty SparseCore planes as well as eight TensorCore
planes, even with SparseCore capture disabled. The accounting explicitly
excludes those empty planes; counting twelve cores would understate every
absolute time by one third. Unit tests guard this and SSA copy attribution.

### A concrete remaining layout cost

There are **43 runtime copies of W2 scale arrays**, directly consumed as scale
operands of the actual GMM custom calls. Each is logical `u8[64,4096,64]` and
changes minor-to-major layout from `{1,2,0}` to `{2,1,0}`. These copies cover
the local scale array, not only selected experts, and total **1.330 ms/call**.
Their logical outputs total 688 MiB/chip/call, not a hardware byte counter.

This is different from the removed legacy whole-expert weight selection loop.
The earlier proof of no whole-expert `fusion`/`dynamic-slice` does **not** prove
that the compiler inserts no layout copies. The new report retains direct
SSA consumer witnesses for all these copies instead of attributing them only
to their broad outer model source line.

## Controlled single-layer experiments

Real layer 3 FFN inputs/routes from the accepted 7,936-row capture, using the
same 128 evenly spaced selected rows as the earlier gate. M1/M4 use prefixes
of that selection; M4 is **not** the native two-prompts-repeated B4 workload.
All 256 experts are present, partitioned over four chips. The scope is routed
MoE only; router/shared expert/attention are excluded.

Forty alternating warmed dispatch-to-ready samples per variant; loading,
predecode, compilation, oracle and tracing are excluded. All variants retain
the same GMM driver, M8/full-K/N128 tile, projection BF16 rounding and output
combination order.

| Routed tokens M | Original FP4 | FP4, activation roundtrip once outside GMM | BF16 weights, original in-tile activation roundtrip | BF16 + roundtrip once |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.594 ms | 0.597 ms | 0.406 ms | 0.416 ms |
| 4 | 1.299 ms | 1.247 ms | 0.630 ms | 0.635 ms |
| 128 | 5.808 ms | 5.370 ms | 1.698 ms | 1.643 ms |

The metadata-only diagnostic copy differs from the unmodified production
median by less than 1% for all three shapes and is bitwise identical.

Interpretation:

- Hoisting activation conversion gives no host-latency benefit at M1, about
  **4.0%** at M4, and **7.5%** at M128. Extra launch/dataflow costs can outweigh
  saved tile work in tiny decode shapes.
- Predecoded BF16 with unchanged activation conversion reduces measured
  routed-MoE latency by **31.7%, 51.5%, 70.8%**, respectively. It removes
  online FP4/scale processing but increases weight storage/logical loads by
  **3.765×**. This is strong evidence that the current FP4 weight-processing
  path warrants work, **not** evidence of HBM-bandwidth saturation.
- Those deltas include changed VMEM/DMA/compiler scheduling and changed
  collective waits. They are **not additive, separately measured unpack,
  scale, cast, dot or communication costs**.
- FP4 unpack, scale operations and dot are fused inside Pallas. This XProf
  capture cannot truthfully assign an independent millisecond number to each
  internal instruction family. No physical bandwidth/perf-counter tool was
  available; compiler-estimated bytes/FLOPs are not a substitute.

For context, production isolated M128 HLO self-time is 5.565 ms: W1/W3 GMM
2.916, W2 GMM 1.534, all-reduce/wait 0.813, and other glue 0.302. Do not multiply
these single-layer fixture times by 43 and label them a full-model measurement.

## Routing and tile work

Analytical counts agree with shared `make_group_metadata` boundaries, including
partially overlapping M8 groups. They measure logical useful rows, **not MXU
utilization**:

| M | Active group tiles by chip, per projection | Useful rows / M8 rows, global |
| --- | --- | ---: |
| 1 | 1 / 2 / 2 / 1 | 12.5% |
| 4 | 4 / 6 / 8 / 2 | 15.0% |
| 128 | 43 / 33 / 48 / 40 | 58.5% |

M4's busiest chip has four times the tile work of the lightest chip. This does
not mean four times overall speedup is available; it shows why all-reduce time
includes load imbalance, not only network transfer. M8 masking and repeated
weight-tile processing also limit low-bit work amortization in decode.

## Suggested next experiments, not implemented here

1. Retain raw FP4/E8M0 and the shared GMM driver. Prioritize the adapter's
   unpack/scale/BF16 conversion dataflow and tile reuse. Any full-K/tile/layout
   change must preserve or re-establish the existing numerical gate.
2. Test preparing static W2 scales in the required device layout at load time,
   removing the observed per-decode copies without expanding full weights.
   The measured opportunity is 1.33 ms in this B4 trace, not an order-of-magnitude
   end-to-end improvement.
3. Investigate paired gate/up execution and input reuse, and decode-specific
   tile scheduling. Preserve route scaling before W2 activation roundtrip.
4. Treat standalone activation hoisting as a secondary, shape-dependent
   option; the experiment does not justify making it a blanket default.
5. Validate collective/load imbalance with per-chip timelines before proposing
   communication-only changes. Full-model optimization must also address the
   now-larger attention/compressor share.

The BF16 and hoisted-activation variants exist only in the diagnostic script;
they have not been integrated into the full 43-layer model or Engine path.
