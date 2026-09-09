# V4 four-chip integration completion

User-approved order (2026-09-09): independent TP prefill/chunk correctness,
thin whole-model integration plus CSA decode batching, full 43-layer/8K
numerics, complete Engine/HTTP pressure, and only then final profiling.

Current result: all five ordered stages have completed for the specified
fixtures. See [final profile and remaining limits](deepseek_v4_four_chip_profile.md)
for measured B1/B2/B4 results, device-time breakdown and the still-open cold
responsiveness issue. Defaults remain unchanged; GMM is explicitly selected.

## Boundaries

- Keep the existing core scheduler, allocator, request lifecycle and sampling
  algorithms. Add V4-specific kernel/model/weight-layout adaptation only.
- Use one `v5p-8` Spot slice: four physical chips, eight TensorCores.
  After preemption the user explicitly approved same-configuration replacement
  and reuse of the original data disk; no larger or additional slices.
  Keep source and immutable receipts locally.
- Retain raw checkpoint FP4/FP8/E8M0 and online dequantization; no full BF16
  expert copies. KV remains the explicit BF16/QAT representation.
- Do not alter the independent reference or accepted numerical tolerances.
- Keep current defaults until the selected combined path passes serving gates.
  In particular, GMM MoE remains an explicit selection during acceptance.
- Preserve failed attempts with their original scope. A passing subset is not
  a passing full HTTP run, and a single-layer result is not model throughput.

## Gates

1. **Complete for the selected CSA/HCA path: independent head-TP stateful validation.** Compare replicated
   and head-TP4 execution with official single-layer weights, including SWA,
   CSA and HCA. Cover advancing prefill/decode, ragged/padded and reordered
   B1/B2/B4, r4/r128/page boundaries, prefix-snapshot restoration and near-8K
   cache/index paths. State/cache comparisons are exact. Frozen real decode
   captures and synthetic continuation activations are explicitly distinguished
   from full-model runs. Verify actual local head ownership and collectives.
2. **Complete: thin whole-model connection.** Shard Q/sink/complete wo_a groups and
   corresponding FP8 scale blocks in the V4 loader for CSA/HCA; keep full-K wo_b behind
   the BF16 all-gather. Keep index heads/compressor/KV replicated. Select the
   unchanged-arithmetic batched CSA projection only for pure decode, retaining
   the existing mixed/prefill route and masking invalid token rows.
3. **Complete: native B1/B2/B4 and fresh independent 8K oracle pass.** Explicit combined GMM + TP4
   + batched-CSA selection, original 43 layers, cold B1/B2/B4 8K, independent
   full-vocabulary reference, chunk/page/order/prefix checks, actual executed
   kernel/backend ownership, and no source changes between prerequisite gates.
4. **Complete for the specified workload: same-source serving acceptance.** Engine lifecycle, actual page
   exhaustion/retract/resume, 8K concurrency/queueing, full HTTP workload/soak
   and lifecycle in one all-green receipt. Propagate all selected backends to
   workers and HTTP launchers. Record cold compilation/control-plane latency
   separately; do not claim responsiveness from an increased timeout.
5. **Complete: final measurements and trace audit.** Re-measure B1/B2/B4 warm latency, aggregate
   throughput, HBM allocator snapshots and raw device traces. Use comparable
   workloads and distinguish host/device timing and profiled/unprofiled calls.
   Compare the combined path, not a sum of isolated speedups.

## Starting evidence

Pre-integration source fingerprint:
`538191ea3a138fbab26fd5e48a7f313439507f12ae46970949f45ac4b56de4d2`.
Prior independent head-TP decode and CSA projection evidence is retained in
[attention parallel experiments](deepseek_v4_attention_parallel_experiments.md).
The latest accepted complete-model numerical/performance baseline is recorded
in [CSA compressor fusion](deepseek_v4_csa_compressor_fusion.md); it does not
yet select head-TP or the batched CSA decode projection.

## Independent stateful gate log

- `v4-attention-tp-chunks-cpu-20260909-01.xml`: 16 fixture/experiment CPU tests
  pass. Production sources and their fingerprint are unchanged.
- `v4-attention-tp-chunks-20260909-01`: failed/incomplete overall, retained.
  Layer 0 (SWA) and layer 2 (CSA) complete cold ragged/chunk/fork and near-8K
  suites. Layer 3 stops in the Python byte comparator: an HCA host array has
  non-contiguous last-axis strides. Normalize only the comparator's host view
  to C order; do not change kernel arithmetic or tolerance. A fresh complete
  rerun is required, not relabeling this failed receipt.
- Acceptance-driver preparation also adds explicit execution-option receipt
  validation. Historical Engine/HTTP launchers did not propagate the V4 GMM
  override; the future combined gate must check all selected options rather
  than assume `ServerArgs.moe_backend='epmoe'` implies GMM.
- `v4-attention-tp-chunks-cpu-20260909-02.xml`: 26 tests pass, including
  non-contiguous byte comparisons and explicit backend selection guards.
- `v4-attention-tp-chunks-20260909-02`: all 1359 checks pass in six suites
  (layers 0/2/3, cold and near-8K); 1317 are bitwise. The 42 non-bitwise checks
  are exclusively SWA output/attention-value comparisons: maximum output
  NRMSE 0.001229974 (0.1230%, gate 0.5%); maximum attention-value NRMSE
  0.0000425742 (gate 0.0002). Do not call SWA TP numerically identical.
- `v4-attention-tp-chunks-expanded-20260909-01`: layers 22/23/42/41, cold and
  near-8K, all 2332 comparisons pass bitwise. No full model was constructed.
- Scope choice: the first integrated TP path selects only 41 CSA/HCA layers.
  The first two SWA layers retain their full-head XLA arithmetic, avoiding
  introducing the measured SWA shape-dependent rounding difference into the
  model. This is not a claim that all 43 attention layers have head TP.
- Five V4 production files add the opt-in loader/model/kernel connection.
  No scheduler/allocator/core-framework file is changed in this stage.
  `v4-parallel-integration-cpu-20260909-01.xml`: 35 pass, 10 TPU-only skips;
  `v4-parallel-integration-tpu-20260909-01.xml`: all 95 pass, no skips/failures,
  including the ten new TPU projection/state cases and existing CSA/HCA tests.
  Whole-model and serving acceptance are still pending.

## Selected combined path

Explicit model overrides (defaults remain unchanged):

```json
{"v4_mhc_backend":"pallas","v4_hca_backend":"pallas","v4_csa_backend":"pallas","v4_moe_backend":"gmm","v4_attention_tp":true,"v4_csa_decode_batch":true}
```

Combined production fingerprint after the five-file V4-only connection:
`4fddb359ae74277a74fdd82d84786249c1c8419122560641ea8a13bbbeb01653`.
Numerical and serving drivers now propagate all six selections, reject receipt
claims inconsistent with launched JSON overrides, and check the actual worker
or reported server configuration. HLO evidence must show h16 CSA/HCA calls,
batched main/index projections on every CSA layer, and all 129 original FP4
GMM calls. Every physical head-weight/scale shard is also checked, with SWA
explicitly remaining replicated.

## Combined model gate log

- `v4-parallel-serving-contracts-20260909-01.xml`: 109 pass, seven hardware
  cases skipped in the CPU process. This validates the acceptance plumbing
  and framework interfaces, not an Engine/HTTP workload.
- `v4-parallel-paged-20260909-01`: complete original 43-layer short-context
  numerical gate, all 14 B1/B2/B4 full-logit comparisons pass. Maximum NRMSE
  `0.00004880590131506324`; all finite and top-1 equal. Unequal chunks and
  reversed request order use the ordinary paged worker. Executed B4 HLO
  confirms 21 CSA/20 HCA h16 attention calls, 42 batched CSA projections,
  all 129 original FP4 GMM calls, and physical head-weight/scale ownership.
  The first two SWA layers remain replicated. This is not the 8K or serving
  gate, and its retained independent reference increases diagnostic HBM;
  do not use that process's final allocation as serving-model memory usage.
- Final-profile preparation separates attention TP all-gather self-time and
  waits from attention projections and MoE. Collective coverage still refers
  explicitly to the 43 MoE sums; gather counts are reported separately.
  `v4-parallel-profile-accounting-20260909-01.xml`: 53 CPU tests pass.
- Short-gate receipt SHA256 (remote/local verified):
  `86977a8a5e8ba4385e217327d6396fc33b461bbd031c15aeab5979ec5647c3ee`.
  Executed optimized HLO SHA256:
  `2fff3f52eb4b7f179f52a29c3763c6d0db9029bdaf61d42d597ac9326718db21`.
  The HLO has 43 all-reduce and 41 all-gather instructions, with the gathers
  scoped precisely to layers 2 through 42. These are compiled instruction
  counts, not measured runtime communication latency.
- `v4-parallel-native-cold-20260909-01`: interrupted before its first stored
  numerical comparison. The model loaded with `47723995648` allocator bytes
  in use on chip 0 (44.45 GiB, 8K pool); no serving or peak-memory claim.
  Google reported `UNHEALTHY_MAINTENANCE` with maintenance-event timestamp
  `2026-09-08T19:21:55.203721199Z`. The SSH stream closed and reconnection
  timed out while node state still said READY. Preserve this incomplete
  attempt; it is not an observed numerical failure or a passing 8K gate.
  The local streamed log survives. A subsequent control-plane check at about
  19:25 UTC confirms terminal node state **PREEMPTED**, not just a transient
  SSH issue. The original independent 500 GiB model disk is READY and detached,
  not deleted; its contents cannot be rechecked until a VM mounts it again.
  Cloud TPU Spot nodes require recreation after preemption, not a start call
  ([Google documentation](https://docs.cloud.google.com/tpu/docs/spot)).
  No node/disk was deleted, recreated, attached or formatted before the pause.
- The user approved same-configuration Spot replacement and reuse of the
  original disk on 2026-09-09. The preempted node was verified and removed;
  a same-name/same-zone `v5p-8` Spot replacement with the original runtime,
  network, service account and data-disk reference is provisioning. Preserve
  the interrupted receipt and rerun 8K from scratch after environment and
  full checkpoint-hash verification; never format the original model disk.

## Same-configuration recovery gates

- The first replacement reached READY/HEALTHY and exposed four physical v5p devices. The
  original independent disk ID and filesystem UUID match, and it is mounted
  at the original path without formatting or ownership changes.
- Restored Python 3.12.14; the entire saved inference package freeze matches
  byte-for-byte, including JAX/JAXlib 0.11.1 and libtpu 0.0.46.1. Production
  fingerprint remains `4fddb359ae74277a74fdd82d84786249c1c8419122560641ea8a13bbbeb01653`.
  Four-device sharded BF16 matmul self-check passes.
- `v4-rebuild-integration-20260909-01.xml`: all 95 original TPU CSA/HCA and
  head-parallel/batched-projection cases pass, no skips. The frozen 7979
  fixture was restored from its verified local copy, not regenerated.
- `v4-rebuild-contracts-20260909-01.xml`: 143 CPU tests pass, seven hardware
  skips. Both XML receipts and the restored package freeze are copied locally.
- Read-only full-checkpoint SHA256 verification against the preemption-safe
  official manifest started. No snapshot file was redownloaded or rewritten.
- The interrupted cold-8K receipt is now copied locally and remains incomplete
  with zero comparisons. Its SHA256 is
  `9993d41fef59f147f424ba8228bedcdb74e888d79c28f30dbfc58658781ad07c`.
- The first replacement was itself preempted after another maintenance event,
  `2026-09-09T03:19:19.461013281Z`, before checkpoint hashing finished. TPU
  tests had already ended; only CPU read-only verification was running. The
  disk remains READY and detached, and all passing XML receipts were pulled
  before this interruption. No new 8K comparison has run yet.
- One bounded same-name/same-zone/same-size Spot retry is in progress under the
  approved single-instance recovery scope. Its checkpoint verification may
  explicitly resume prior recorded SHA256 entries only when snapshot/manifest
  identities match and file size/mtime/ctime establish no subsequent change;
  all other files are hashed anew. The prior incomplete report remains intact,
  and fresh versus reused hashes are separately reported. Five local CPU
  tests cover rejection of stale provenance, wrong hashes/sizes and changed files.
- The second replacement reached READY/HEALTHY; original disk identity and
  UUID were verified before mounting, and local source was restored. The full
  package freeze matches byte-for-byte, four-chip smoke passes, and
  `v4-rebuild-integration-20260909-02.xml` has 95 passes with no skips.
  `v4-rebuild-contracts-20260909-02.xml` adds 143 CPU contract passes and
  seven hardware-only skips on this same replacement VM.
  The new complete checkpoint receipt covers all 73 files / 159,630,041,626
  bytes: 57 unchanged prior SHA256 records plus 16 new hashes, explicitly
  distinguished rather than claiming every file was rehashed on this node.
  Both checkpoint receipts and test XMLs are copied locally. Production
  fingerprint and accepted numerical tolerances remain unchanged.
- Cold full-model regression resumed in `v4-parallel-native-cold-20260909-02`
  with the same selected options and the immutable prior GMM full-logit
  baseline. Until it and the independent from-empty-cache oracle pass,
  Engine/HTTP and final profiling remain pending.
- `v4-parallel-native-cold-20260909-02` is now complete: all 2424 full-logit
  checks pass, all finite/top-1 equal, maximum NRMSE
  `1.618478790987865e-07`. The 516 B1 comparisons are bitwise; the 1908
  concurrent prefill/decode comparisons are not claimed bitwise. All B1/B2/B4
  requests start with empty caches and reach position 8191. Original numerical
  tolerances, checkpoint, reference, and production fingerprint are unchanged.
  Executed B4 HLO confirms 21 CSA/20 HCA h16 attention calls, 42 batched CSA
  projections and all 129 FP4 GMM calls. Physical weight/scale ownership covers
  exactly CSA/HCA layers 2..42; SWA 0/1 remains replicated. There are 41
  attention all-gather and 43 MoE all-reduce instructions, not measured times.
- Full native receipt, goldens, prefill checkpoints and HLO are copied locally.
  Report SHA256 `6b22d45aa9e755ae55e4e86dcdcbb71cab5bfccda31e0b74d9c5af2c526c8457`;
  HLO SHA256 `6e48cbe3a9d416a21bc1d5a47da11770eee79ca49d22d4ab76bd4260b53c77b6`.
  Both match remote/local. Fresh `v4-parallel-oracle-20260909-01` is running
  from empty cache; do not substitute native prefix state or an old receipt.
- `v4-parallel-oracle-20260909-01` has now completed both independently
  recomputed 7936-token prompts and all 256 teacher-forced decode steps each:
  all 514 full-vocabulary rows pass, 435 bitwise, maximum NRMSE
  `0.00010527185077080503` (0.01053%), all finite/top-1 equal. It shares no
  native hidden states, prefix KV or compressor scratch. The reference and
  production fingerprints are unchanged. This is a numerical regression,
  not an official GPU/API accuracy benchmark. Full reference logits and report
  are local; report SHA256, verified against the remote copy:
  `9ec719b6fdcd6be4dcd9c2f8e12f88f5098709d6a2ddf336c82d81ef467d47eb`.
- The oracle process exited zero before starting the same-source overlap
  Engine lifecycle gate `v4-parallel-paged-engine-20260909-01`. It explicitly
  inherits all six selected options from the passing short worker receipt.
  Full 8K Engine, genuine memory pressure, HTTP/soak and final profile are
  still pending; a passing numerical gate does not certify serving lifecycle.
- `v4-parallel-paged-engine-20260909-01` now passes all ten short-context
  overlap Engine cases: cold B2, actual radix-prefix B4, streaming/logprobs,
  acknowledged cancellation and slot reuse, explicit retract/resume, and
  flush/recompute. The real server reports all six selected options. Receipt
  is local, SHA256 `5e2be54381f6b9fbcb2ff31fb50ea7ece70ad94bd762b2566ef654bda12c3392`.
  This manual pause/retract lifecycle case is not the later natural KV-pressure
  gate. Cold timings include new-shape compilation, not warm throughput.
- After clean shutdown, `v4-parallel-engine-8k-20260909-01` started with the
  complete new native 8K receipt and the same options. It is still in progress.
  Pressure, full HTTP/soak and final profiling are not yet certified.
- `v4-parallel-engine-8k-20260909-01` is now complete: all eight normal Engine
  scenarios pass (cold B1, two serial reference variants, cold B2, two cold B4
  orderings, long-prefix B4 and queued B8). The 25 requests produce 6350 output
  tokens in total. Public serving requests use the documented 254-token
  generation headroom; the native/oracle tests separately reach position 8191.
  Both cold B4 rounds observe four running requests. B8 observes four running
  and up to seven waiting; it is queueing, not eight simultaneous decodes.
  The two rotated-prompt serial runs establish Engine baselines, not additional
  independent GPU/reference accuracy tests. Receipt is local, SHA256
  `14a7ef60567efddb24c6c2e75fedc35f52ecdbbf94098ada19eeaf57630d14a3`.
- The normal Engine exited cleanly before starting
  `v4-parallel-engine-pressure-20260909-01`. It uses the existing admission
  clip of 64 and a 16256-token KV pool; no scheduler algorithm changes or
  forced test-retraction flag. Natural exhaustion/recovery, full HTTP/soak and
  final profile remain pending until their complete receipts pass.
- `v4-parallel-engine-pressure-20260909-01` is now complete: four scenarios,
  five requests and 1270 output tokens pass. Free KV tokens actually reach
  zero; the ordinary scheduler naturally retracts one request and aborts none.
  Both restored requests match their full output baseline. Two unique long
  prefixes then evict the old prefix, and its zero-cache-hit recomputation
  again matches the original output. No forced retraction or algorithm edit.
  Receipt is local, SHA256
  `285e2a28602474e49c8707962bf79d78a9e4d00ad07267197e81778a3208e55f`.
  Its cold elapsed time includes compilation after the B2-to-B1 transition;
  do not report the pressure common-window timing as steady-state throughput.
- After clean pressure-Engine shutdown, full-mode HTTP acceptance started in
  `v4-parallel-http-20260909-01`: owned loopback port 30124, 300-second soak,
  at least three warm rounds, all selected options inherited from normal 8K
  Engine. It is not lifecycle-only. This HTTP receipt and final profiling are
  still pending.

## Completed serving gate and final-profile preparation

`v4-parallel-http-20260909-01` is a complete, all-green **full** receipt:
15 workload/lifecycle cases; seven B8 soak rounds, all seven warm, 56 submitted
soak requests over 314.5276 seconds. All compared outputs match; actual
GMM/TP4/batched-CSA selections match the prerequisite. OpenAI streaming and
non-streaming completions and over-context rejection pass. Acknowledged abort,
client disconnect, subsequent slot reuse/logprobs and flush/recompute pass.
Final idle state has all four request slots and all 33280 KV tokens available,
with no waiting/running requests. The owned server is reaped and no test/server
process remains. Cleanup reports `returncode=-9, forced_kill=false`; this is
not a claim of a normal zero-exit server shutdown. The complete folder is local;
report SHA256 matches remote:
`453a01aa3ebb6ef0bce0624ff77bf1c8abc5a2d502c0b8f74d9b326401d40005`.

Cold responsiveness is **not** fully solved: the first post-abort logprobs
shape records two logged compile misses and a 165.4101-second status-query
wait (169.0058 seconds for the case). It passes the declared 1800-second cold
observer budget, not a 60-second responsiveness SLA. The warm B8 rounds have
no logged compile misses and approximately 0.53-second maximum status queries.

The final profiling harness adds opt-in `--profile-reused-b1`: B1, like B2/B4,
restores the accepted reconstructed prefix outside timing and compares every
decode row with the same-source golden. This avoids retaining the diagnostic
late-reference model in the measured process. Without this flag the historical
gate/workload is unchanged; it requires `--profile --reuse-goldens` and rejects
`--cold-prefill`. Only the test harness and its tests change; production and
reference fingerprints, arithmetic, tolerances and scheduler remain unchanged.
`v4-parallel-final-profile-contracts-20260909-01.xml`: all 43 reporting/profile
tests pass, including eight option combinations verifying unchanged defaults
and rejection of invalid B1-profile combinations. Final measurement runs in
`v4-parallel-native-profile-20260909-01`; no final results are claimed yet.

## Reproduction contract

Use the existing four-chip VM and its pinned inference environment. All new
compilation-cache files and large artifacts go on the independent model disk.
The native drivers explicitly select the combined path with:

```text
--moe-backend gmm --attention-tp --csa-decode-batch --check-kernel-dispatch
```

Run `run_deepseek_v4_paged.py` first, followed by
`run_deepseek_v4_8k_native.py --cold-prefill` and
`validate_deepseek_v4_8k_oracle.py`. The cold gate compares the immutable prior
GMM baseline through `--baseline-report`; it must not reuse different-source
goldens as if they were new independent results.

Only after these gates pass, feed the new receipts to
`run_deepseek_v4_paged_engine.py --overlap`, `run_deepseek_v4_8k_serving.py`
(normal, then genuine-pressure mode), and `run_deepseek_v4_http_stress.py`
(full workload, not `--lifecycle-only`). The serving drivers inherit and check
all six options from their prerequisite, rather than silently using legacy
MoE or replicated attention. Pressure uses admission estimate clip 64 and an
owned run log, never forced test retraction.

Finally repeat the native 8K driver with `--reuse-goldens` pointing at the new
accepted native receipt, `--profile --profile-boundary --profile-reused-b1`, and the same selected
options. This final timing workload reconstructs B1 prefixes and restores them
for B2/B4 outside timing; it is distinct from cold prefill and HTTP throughput.
Keep position 8063 boundary traces separate from 8064/8065 interior traces.
