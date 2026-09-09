# V4 position 8023 correctness repair

Status: **numerical repair milestone complete, 2026-09-07**. Original position
8023 failure reproduced and all 43 layers faithfully replayed. The fixes cover
batch-dependent compressor projections, an elided fused mHC BF16 boundary,
and phase-consistent reference pooling with safe channel gathers on v5p.

Final same-source results: **164 related tests + 35 shared CPU tests pass**;
the native cold-cache B1/B2/B4 gate passes **1910/1910 full-logit comparisons**;
the independent from-empty-cache gate passes **514/514 rows bitwise** across
two complete 7936-token prefills plus 256 decode steps each, reaching 8192.
No numerical tolerance is weakened. See the
[diagnosis and evidence report](deepseek_v4_8023_correctness_report.md).
Performance and Engine pressure revalidation remain follow-up work; old failed
measurements are not promoted to accepted results.

## Fixed contract

- Original 43-layer FP4/FP8/block-scale checkpoint, online dequant, four v5p chips.
- Preserve the existing dirty worktree and shared scheduler/allocator behavior.
- Do not weaken the 0.5% full-logit NRMSE gate or use greedy-token agreement as
  a substitute for numerical correctness. Preserve the original failure reports.
- One TPU-owning process at a time; back up diagnostics and changes locally.

## Execution plan

1. Reproduce the uninstrumented failure with the frozen failing source
   `c7ec7e1357f41b4710f5505fa76ed95fc458693998df6ebab4ec5b0fc39ce983`.
   Teacher-force the saved B1 sequence through zero-based cache position 8023.
   Save exact B1/B2 pre-step physical KV/snapshot/live-scratch buffers and
   request metadata, plus production post-step buffers and logits. Snapshotting
   must not change the model call or compiled graph.
2. Replay layers from those saved states with shared read-only real weights.
   Compare hidden states, SWA KV, CSA/HCA compressed KV, private scratch,
   snapshots, index scores/top-k and routing. Verify replay against the original
   production result before using instrumentation to identify a cause.
3. Reduce the first divergent operation to a real-weight, real-state regression.
   Establish which input/state/rounding operation causes the discrepancy.
   Apply the smallest evidence-backed V4 kernel/adapter fix, not a scheduler
   workaround and not a tolerance change.
4. Verify the reproducer and layer/cache checks, then the complete 43-layer
   8192-boundary B1/B2/B4 full-logit gate. Generate fresh post-fix goldens with
   independent reference checks; do not silently reuse old-source goldens.
   Also run B2/B4 packed cold prefill from empty caches, comparing every
   128-token boundary and every subsequent decode against B1. Restoring B1
   prefix pages alone does not test concurrent compressor state construction.
   Run the framework/low-bit tests and revalidate serving prefix/queue/pressure
   behavior only after the numerical prerequisite passes.

## Evidence collected

- Unmodified production B1 at position 8023 matches the saved baseline bitwise
  for both fixtures.
- Unmodified B2 reproduces request 1's original full-logit NRMSE
  `0.10763054341077805`, maximum absolute error `3.1207332611083984`.
  Request 0 has NRMSE `1.3686926081390993e-07` at the same position.
- Preliminary pre-step cache comparison: layers 0 and 1 match exactly; layers
  2–5 already have different FP32 compressor scratch, but their valid window
  and compressed KV match. This is an observation, not yet a root-cause claim.
- Snapshot helpers passed five CPU tests, including bit-preserving BF16
  serialization and physical-to-logical request/cache normalization.
- The complete layer replay matches all three production logits bitwise, and
  every physical post-step cache buffer is bitwise equal to its production
  snapshot. The first hidden divergence is layer 6, request 1. Its attention
  input, Q/KV, index scores/top-k and logical addresses still match exactly.
- At that layer, the main compressor's pooled BF16 channel 468 differs by
  `0.00006103515625`; normalization and RoPE turn that into a difference of
  `0.0078125` in compressed vector 2005, channel 469. The layer-6 hidden NRMSE
  is `0.004925544839352369`, amplified by subsequent layers.
- Isolated real-state compressor replay reproduces both production caches
  bitwise. Replacing B2's historical scratch with B1's, or replacing only its
  current projection with B1's, removes the visible compressed-KV difference.
  This establishes the causal path from batch-dependent FP32 projections to
  the pooling rounding boundary, not a physical-page or top-k ownership bug.
- A naive single-row `lax.map` still produces different FP32 values across
  B1/B2 on v5p. Reusing the existing position-aligned 8-row metadata produces
  bitwise-equal current-token KV and score projections in the real fixture.
- Added a stronger final gate, `scripts/validate_deepseek_v4_8k_oracle.py`:
  independently compute the full prompt from empty cache and all 256 decode
  steps. It does not import native prefix KV/scratch, unlike the older late
  single-step reference check. Its first run fails at position 7935; see the
  report for the additional fused-rounding diagnosis and follow-up results.

## Completion criteria

The original failure is reproduced and explained, a regression catches it,
the fix passes that regression and full-model B1/B2/B4 long-context numerical
checks, and no relevant existing tests regress. If new failures appear, keep
acceptance failed and continue diagnosis. Performance work stays deferred.
