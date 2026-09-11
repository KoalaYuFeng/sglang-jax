# V4 expert-parallel reduction: request-layout consistency

Status: **4K B1/B16/B32 numerical acceptance passed** with the new independent
EP reference. The final saved-array audit and four sequential HTTP smoke
requests also passed. The candidate is retained locally and on the TPU, with
HTTP restored to its original service settings.
After the two historical-compatibility failures below, the user approved adding
a deterministic numerical reference while retaining the old baseline as a
historical comparison. Candidate wiring was retested in
`v4-ep-accept-20260910-01`. No commit or push was made.

## Reproduced defect

The 2026-09-10 fixed-batch run used four physical TPU v5p chips (TP4/EP4),
the full 43-layer checkpoint, original FP4/FP8 storage, tuned GMM MoE and
128 prefill tokens per request per call. B16 and B32 with 4096 input tokens
failed the existing full-vocabulary NRMSE gate of 0.005 despite identical
generated token IDs. This is not an OOM or a scheduler result.

For B16, faithful full-model replay localized the first hidden-state difference
at position 4100 to zero-based layer 28. Matching requests had identical
attention/compressor outputs, routes, FP4 projection results, local expert
partials and shared-expert outputs. The first difference was the FP32 EP
`psum`, before the shared-expert add and BF16 cast. Cache differences first
appeared downstream in layer 30; they were a consequence, not the cause.

At channel 2847 the rank-ordered FP32 partials were:

```text
[-0.1513671875, 2**-27, 0.042236328125, 0]
shared expert = 0.1904296875
```

The collective produced two different FP32 sums for identical rows. Their
one-ULP difference crossed a BF16 midpoint after adding the shared expert:
`0.0810546875` versus `0.08154296875`. The same failure reproduced without
loading any model or executing GEMM, using only the captured partials and
the collective. It occurred at B16/B32 but not B1/B4/B8 in that reproducer.
The larger sum equals the exact sum of these already-rounded partials;
the old B1 answer is not intrinsically more mathematically accurate.

## Model-local numerical contract

`kernels/deepseek_v4/collectives.py::ordered_ep_sum` gathers partials in rank
order and performs an explicit ascending-rank FP32 tree with optimization
barriers. EP4 is `((rank0 + rank1) + rank2) + rank3`. The candidate connects
both retained V4 MoE and V4 GMM MoE to this contract. Single-rank execution returns its
FP32 input directly; non-FP32 inputs are rejected.

This specifies reproducible arithmetic for a fixed expert partition, not
exact summation, equivalence across different EP sizes, or official-runtime
bitwise equivalence. The router, local expert accumulation, FP4 GEMM,
checkpoint storage, shared expert and scheduler are unchanged. The independent
`kernels/low_bit/moe.py` reference is deliberately unchanged.

An all-gather materializes the rank partials rather than just a reduced tensor.
Its memory/communication trade-off must be measured; numerical correctness
alone is not a claim of performance neutrality.

## Verification and evidence

The source test `test_deepseek_v4_collectives.py` checks an independent NumPy
FP32 tree, the real halfway counterexample over multiple batch sizes,
cancellation, reordered/duplicated rows, padding and all output replicas.
The existing V4 MoE/GMM tests additionally check FP4 projection arithmetic and
legacy/GMM agreement. Because both V4 backends now share the reduction helper,
their agreement alone is not an independent reduction oracle.

Historical candidate experiment: `v4-ep-fix-20260910-02`. Diagnosis artifacts:
`v4-b16-debug-20260910-02` through `-05`. Reports and frozen inputs remain
in ignored local `GCP_login/results/` and on the model data disk; credentials
and checkpoint weights are not tracked in Git.

First candidate (`-01`, balanced adjacent-pair tree; rejected) gates:

- CPU collective tests: 11 passed.
- Actual four-chip v5p collective plus existing MoE/GMM tests: 40 passed.
- Frozen real partials, all 4096 channels, B1/4/8/16/32/128 with repeated and
  mixed rows: 12 cases match the independent NumPy tree exactly.
- CPU evidence/fingerprint tests: 18 passed. The existing source fingerprint
  includes the new helper through the V4 kernel-directory file list.

- Full-model B1 repeatability passed. The first B16 wave passed against fresh
  B1, maximum NRMSE `1.5464151e-7` across all 32 output positions.
- Immutable old B1 comparison failed: prompts 0/1/2 were bitwise equal;
  prompt 3 had maximum NRMSE `0.14854911`, with unchanged output token IDs.
  Old B32 prompt-3 rows 3/15/23/31 instead matched the new B1 at about
  `1.54e-7`. This is evidence of two historical numerical paths, not permission
  to relax the acceptance gate or relabel the old baseline.
- The owned B16 process was intentionally terminated after its first passing
  numerical wave, to avoid further performance rounds for a rejected recipe.
  B32 was not run with this candidate. The controller rolled back the remote
  candidate and restored the original HTTP service. Failed evidence is retained.

The second candidate (`-02`, ascending-rank left-deep accumulation) also passed
the 11 CPU tests, 40 TPU tests and 12 real-partial cases, but failed the same
immutable B1 prompt-3 check, with the same NRMSE. It was rejected before
B16/B32, and the remote source and HTTP service were restored again. Neither
candidate was accepted at that stage. The later user-approved gate is recorded
separately below; isolated passing tests were not used as full-model acceptance.

## Faithful prompt-3 prefill investigation

`v4-prefill-ep-debug-20260910-03` completed: the original full ModelWorker,
with observational shadow reductions, reproduced all 32 immutable old B1
full-vocabulary logits **exactly** (NRMSE zero). Three MoE BF16 differences
between the original collective and the shadow ascending-rank recipe were
captured along that same original trajectory:

| Token position | Layer, zero-based | Channel | Original MoE BF16 | Shadow MoE BF16 |
| --- | --- | --- | --- | --- |
| 882 | 5 | 1788 | -0.04248046875 | -0.042236328125 |
| 2369 | 30 | 3330 | 0.045166015625 | 0.044921875 |
| 2539 | 19 | 2643 | 9.98377799987793e-7 | 1.0132789611816406e-6 |

The first is row 114 of the chunk beginning at 768. Its local FP32 partials
are `[-0.0242919921875, -0.0279540978372097, 0, 0.1051025390625]`, with
shared expert `-0.09521484375`. Original sum `0.0528564453125` versus
shadow `0.0528564490377903` crosses the following BF16 midpoint. These are
observations on the old trajectory, not a claim that the whole changed model
has only three differing tensors or that either recipe is always more accurate.

The failed `-01` diagnostic lacked a CPU callback backend; `-02` enabled it
through an environment value rejected by ServerArgs. Neither is a numerical
test failure. `-03` leaves the environment at `JAX_PLATFORMS=tpu` and enables
`tpu,cpu` through process-local diagnostic JAX configuration. Default compute
remains the same four TPU chips. The production configuration is unchanged.

This establishes a compatibility conflict with the historical collective's
rounding, not an official-runtime accuracy verdict. It motivated the separate
deterministic reference and user-approved acceptance policy below. No old
baseline, tolerance or independent reference file was replaced.

## Independent deterministic acceptance (user approved)

`scripts/deepseek_v4_ep_reference.py` is test/reference-only. It never imports
the candidate helper: `ring_ep_sum_reference` passes an FP32 accumulator from
rank 0 through rank 3 using point-to-point permutations, then broadcasts the
result without arithmetic. The candidate instead gathers rank partials before
its local additions. Both are checked bitwise against an independent NumPy
ascending-rank implementation, including all four real halfway cases.

The real-weight gate also decodes selected official FP4 bytes/scales and
computes activation quantization and expert projections in NumPy. A separate
gate executes the unchanged low-bit reference's per-expert loop with the new
rank-passing reduction injected only during diagnostic tracing. No old
reference file is edited and no whole-checkpoint BF16 expansion is performed.

Numerical results of `v4-ep-accept-20260910-01`:

- CPU: 24 passed; four-chip v5p: 53 passed.
- All four real failure inputs (decode 4100, prefill 882/2369/2539) pass.
  For each selected token, all four independently computed CPU expert partials
  are exactly equal to the capture. The candidate GMM result is exactly equal
  to the independent per-expert loop and the NumPy result.
- Both reduction transports match NumPy exactly for every channel in the
  captured 16/128-row arrays, including the shared-add/BF16 boundary.
- The independently reduced full-model B1 reference completed all four
  4096-token prompts and 32 outputs each. Candidate B1 matches all 128
  full-vocabulary output vectors exactly (NRMSE zero, identical top-1).
  B16 also passes all 512 full-vocabulary output vectors against the independent
  reference, maximum NRMSE `1.5464151204014343e-7`, matching top-1 throughout.
  B32 passes all 1024 vectors with the same maximum NRMSE. All outputs are
  finite and all top-1 IDs agree.

Full-model reference B1 uses the independent rank-passing EP implementation.
It still shares the **unchanged** attention, dense, GMM and framework paths
with the candidate: this is an independent reference for the modified EP
operation, not a second independent full-model or official GPU runtime.
Candidate B1/B16/B32 must compare all 32 full-vocabulary outputs directly to
that reference (finite, matching top-1, NRMSE <= 0.005). Old B1 differences
remain recorded with their original threshold and are not relabeled as passes.
The full-model gate passed, so the controller retained the candidate and
restored HTTP with the original service settings. Accepted framework source:
`2543e8b0e98fc56658912cd2d14f05e4d4ae4e00cc961f07bc150744070e0c0f`.

| Actual batch | Input / output tokens per request | Independent full-vocabulary vectors | Maximum NRMSE |
| --- | --- | --- | --- |
| 1 (four prompts separately) | 4096 / 32 | 128 | 0 (bitwise equal) |
| 16 | 4096 / 32 | 512 | 1.5464151204014343e-7 |
| 32 | 4096 / 32 | 1024 | 1.5464151204014343e-7 |

These are correctness runs, including first-call compilation, not a new
performance benchmark. Candidate B32 is a real concurrent packed batch,
not 32 queued B1 requests. Scheduler algorithms, raw checkpoint storage,
FP4/FP8 GEMMs, routing, attention/compressor and quantization boundaries are
unchanged; only V4's EP reduction contract changed.

The final audit independently recomputed all 1664 output-vector comparisons
from the saved arrays and confirmed the unchanged historical-reference hash.
After-run allocator HBM snapshots were about 40.6 / 49.5 / 58.9 GiB per chip
for B1 / B16 / B32 respectively, with no OOM. These are allocator snapshots,
not a time-resolved physical-memory profile.

The restored HTTP service passed four sequential B1 requests, each with 4096
input tokens, 32 output tokens and zero cached input tokens. All generated
IDs match the independent EP reference. The first request included cold
compilation; this is a token-ID service smoke, not a latency benchmark or
full-vocabulary HTTP accuracy gate. Afterwards there were no queued/running
requests, all four request slots were free, and all 33280 KV tokens were free.

Full 8K/long-decode regression and Engine/HTTP pressure remain separate work.
Neither the smoke nor the saved-array audit substitutes for those broader
gates or official-runtime accuracy evaluation.

Acceptance evidence is archived on the data disk as
`profiles/v4-ep-accept-20260910-01-evidence.tar.gz` and copied to ignored local
`GCP_login/results/v4-ep-accept-20260910-01/evidence.tar.gz`.
Archive SHA-256:
`974b5ec880bdd28fff900cc01da45d2e4dc9fa0e71f5a03da61f8fe1def72ac0`.
It includes source manifests, the acceptance protocol, independent reference
and candidate output arrays, test logs, final audit and HTTP smoke receipts.

The separate 8192-input + 32-output limit is unchanged: it needs 8320-token
page-aligned capacity, beyond the current 8192 total-context guard. This
reduction change does not validate that larger cache/index path.

Follow-up validation and post-fix profiling are recorded separately in
[V4 8192-input validation and profiling](deepseek_v4_8320_validation_profile.md).
That experiment passes diagnostic 8320-capacity boundary/full-model gates
and the 4K/8K B1/4/8/16/32 timing matrix. It retains the production 8192
guard and does not relabel this historical acceptance run.
