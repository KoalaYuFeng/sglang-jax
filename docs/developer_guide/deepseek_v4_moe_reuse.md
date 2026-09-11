# V4 MoE: reuse the shared GMM execution structure

Status: opt-in implementation; default serving remains `legacy`. Real-weight
and full 43-layer native cold-8K correctness gates pass on four TPU v5p chips.
Engine/HTTP acceptance and new device-timeline profiling are still pending.

## Scope and execution contract

The previous B4 profile found whole-expert packed-weight slices outside the
active-token loop. The new path uses the existing MegaBlocks `gmm()` itself,
not a separately copied grouped-matmul driver:

- `kernels/gmm/routing.py::expert_permutation` extracts EPMoE's existing stable
  sorting and expert histogram. EPMoE and the V4 adapter both call it.
- `kernels/gmm/megablox_gmm_kernel/gmm.py` gains an optional, static RHS adapter.
  Its ordinary path remains the default. The same nonempty-group metadata,
  expert-shard offsets, Pallas grid, scalar prefetch, tile transfers and masked
  output stores execute both ordinary weights and V4 packed FP4 weights.
- `kernels/low_bit/gmm.py::CheckpointFP4Rhs` owns only format validation,
  checkpoint-shaped BlockSpecs, VMEM unpack/E8M0 decode and tile arithmetic.
- `kernels/deepseek_v4/moe_gmm.py` preserves V4's activation, route scaling and
  combination semantics. There is no JAX loop slicing a complete expert weight.

The storage contract stays uint8 `[E,N,K/2]` plus compact uint8 `[E,N,K/32]`
E8M0 scales. Only a weight tile is expanded in VMEM. There is no persistent
BF16 copy, all-expert pre-dequantization or checkpoint repacking.

This first stage deliberately preserves an 8-row / full-K / 128-output tile,
FP32 dot accumulation and the existing BF16 projection boundaries. It does
not yet tune K tiling or implement the fused-MoE v2 cross-device pipeline.

## Numerical invariants

1. The existing V4 router is unchanged, including hash routing and sqrt-softplus.
2. Gate clipping precedes SiLU; the up branch is clipped symmetrically.
3. Routing weights multiply the intermediate activation before W2's FP8 QAT.
4. Duplicate expert choices coalesce before that quantization, as in legacy.
5. Local expert outputs sum in ascending expert order, then use V4's
   deterministic ascending-rank FP32 reduction. See the later
   [EP reduction acceptance](deepseek_v4_ep_reduction_correctness.md) for
   the B16/B32 defect, retained old baseline and new independent-reference gates.
   Inactive routes belong to a sentinel group outside all device-owned expert
   ranges, including entirely idle chips.

## Rollout gates

1. Synthetic and independent byte/NumPy gates, one/four-chip execution, sparse
   groups, duplicate/inactive routes, and shared-driver regressions.
2. Captured real activations/routes plus official checkpoint experts: require
   bitwise agreement with legacy before timing, and an independent NumPy check.
   Inspect actual optimized HLO for the shared-GMM call and disappearance of
   whole-expert slice instructions.
3. Full 43-layer forward/decode and 8K B1/B2/B4 numerical regression, then
   Engine/HTTP acceptance and end-to-end profiling. Only then consider changing
   the default. A single-layer timing improvement is not an end-to-end claim.

## Activation and reproducibility

Use the standard model override mechanism:

```json
{"v4_moe_backend": "gmm"}
```

The default is `legacy`; neither backend automatically falls back. The native
8K runner accepts `--moe-backend gmm`. Saved-golden reuse checks the MoE backend
as well as the source fingerprint, so a legacy golden cannot be mislabeled as
a same-backend GMM golden. The framework fingerprint now covers the shared GMM
driver/routing dependencies. Historical comparison remains an explicit gate.
The model override is separate from the serving framework's `--moe-backend`;
that framework flag stays `epmoe` for this V4 integration.

The isolated real-input driver is `scripts/validate_deepseek_v4_moe_gmm.py`.
It records capture provenance, source/script fingerprints, all timing samples,
numerical metrics and optimized HLO. Its historical capture predates a direct
checkpoint field; the report explicitly records when checkpoint identity is
inferred from the capture's fixture report.

## Results so far

- `v4-moe-gmm-unit-20260908-02.xml`: 27 passed on the actual v5p. Includes
  B1/B2/B4/M17/M128, one/four-chip EP, duplicated routes, an idle shard and an
  entirely inactive batch. New vs retained MoE outputs agree exactly.
  The final run also rejects an inconsistent local/global EP expert count.
- `v4-moe-gmm-integration-20260908-01.xml`: 55 passed, covering existing V4
  framework/mHC contracts and shared MoE weight-loader sharding tests.
- `v4-moe-gmm-shared-regression-20260908-02.xml`: 48 pre-existing ordinary GMM
  tests passed (BF16, bias/no bias, different expert offsets and shapes).
  The 930 unrelated/quantized variants were deselected, not counted as passed.
- `v4-moe-gmm-evidence-cpu-20260908-02.xml`: 33 reporting/evidence tests passed
  with the CPU backend, including shared-GMM attribution, per-layer compiled
  call ownership and matching runtime/offline source fingerprints.
- `v4-moe-gmm-real-20260908-02/report.json`: complete. Historical real layer-3
  activations/routes, original 256-expert checkpoint weights, four-chip EP.
  B1/B2/B4/M128 agree bitwise with the retained path; the independent NumPy
  B1 result is also bitwise equal. Both runs keep original FP4/E8M0 storage.

Isolated routed-MoE wall-clock timings (30 alternating, warmed samples; excludes
loading/compilation and does not include shared expert, router or the model):

| Tokens | Legacy p50, ms | Shared GMM p50, ms | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 2.6902 | 0.5931 | 4.54x |
| 2 | 3.5175 | 0.8641 | 4.07x |
| 4 | 3.9901 | 1.3018 | 3.07x |
| 128 | 8.3514 | 5.8103 | 1.44x |

The actual optimized HLO includes `gmm_checkpoint_fp4`; the candidate has no
whole-expert packed-weight slice instructions in any of those four cases.
The old path still has the corresponding slices. This is structural compiler
evidence, not a physical HBM-bandwidth counter or a full-model speedup estimate.

`scripts/deepseek_v4_moe_evidence.py` checks the completed native receipt's
hash-bound optimized HLO separately. It requires three actual GMM custom-call
instructions per layer (two 4096-to-2048 projections and one 2048-to-4096),
with unambiguous layer ownership through SSA consumer paths. Marker strings
or the right total count with missing layers do not pass. It also rejects
whole-expert packed-weight and scale slice/fusion outputs. The offline native
profile analyzer recognizes the new routed-MoE and EP collective call stacks;
no new device timeline profile has been collected in this stage.

## Full-model native acceptance

`v4-moe-gmm-8k-20260908-01/report.json` is complete: 43 layers, four physical
v5p chips, 7936-token cold prefill and 256 decode steps through position 8191.
All 2424 full-logit comparisons pass with the original tolerance unchanged:

- 514 B1 checks against the immutable previous Pallas-CSA/legacy-MoE report.
- Two retained-reference decode checks at position 8191. These use the native
  prefix cache; they are not an independently reconstructed 8K reference prefill.
- 372 cold B2/B4 chunk-boundary checks and 1536 B2/B4 decode-row comparisons,
  including alternating request order, against the independently checked B1 runs.

All results are finite and top-1 IDs agree. The 516 B1/reference checks are
bitwise equal. Batched comparisons are not bitwise equal: maximum NRMSE across
all checks is `1.61848e-7`, maximum absolute logit difference `1.90735e-5`.
This is execution-path numerical acceptance, not an official accuracy benchmark.

`moe_kernel_evidence.json` passes: 129 actual shared-GMM custom-call instructions,
W1/W3/W2 coverage in every one of the 43 layers, and no whole-expert weight or
scale slice/fusion outputs of the checked checkpoint shapes. The same executed
B4 entry also passes the existing original mHC/HCA/CSA kernel-dispatch checks.
Evidence binds the native receipt and HLO by SHA256; accepted framework source:
`a8b53fea756d23a8e714b4c5d79814c6fef1b5e499fdb9d8f117f58ba199cb4c`.

Native full-model warm decode measurements near 8K (not Engine/HTTP throughput):

| Requests | Previous legacy p50, ms | GMM p50, ms | GMM p95, ms | GMM aggregate tokens/s |
| --- | ---: | ---: | ---: | ---: |
| B1, case 0 | 157.182 | 64.828 | 65.790 | 15.414 |
| B2 | 197.309 | 81.797 | 83.601 | 24.473 |
| B4 | 206.835 | 92.838 | 94.603 | 43.101 |

The previous measurements are from `v4-csa-performance-8k-20260908-01`, not
alternating measurements in this new process. Warm statistics exclude compilation
and profiled calls. The new first decode calls took 137.870 / 147.798 / 148.415
seconds for B1/B2/B4; these are recorded separately, not included in the table.
Native wall time includes submission, completion wait and output transfer.
The observed B4 p50 ratio is about 2.23x; it is not a serving-speedup claim.

Allocator snapshots show 46.368 GiB per chip after loading, about 47.112 GiB
after B4, and a largest recorded per-chip peak of 48.033 GiB. These include
the validation process's caches/reference work and are not just weight bytes
or a steady-state serving memory measurement. No BF16 expert copy was introduced.

Next gate: run the selected GMM backend through Engine/HTTP lifecycle, pressure
and end-to-end measurements before switching the production default. First
extend those acceptance harnesses to propagate and verify the selected model
backend from their prerequisite receipts; the existing HTTP launcher explicitly
sets only mHC/HCA/CSA, so rerunning it unchanged would still measure legacy MoE.
Fused-MoE
v2 cross-device execution remains a later reuse opportunity, not an implemented
part of this first stage.

Reports are retained in ignored local `GCP_login/results/` and on the existing
Spot VM. Code is edited locally first and synchronized with conflict checks and
local source snapshots; no commit or GitHub push is performed by this task.
