# DeepSeek-V4-Flash experiment records

Start with the [experiment ledger](../../../docs/developer_guide/deepseek_v4_flash_experiments.md).

Latest published deployment results:

- [Full GPQA Diamond, GSM8K and HumanEval accuracy](../../../docs/developer_guide/deepseek_v4_full_benchmarks_20260911.md),
  including official Instruct comparison, frozen evaluator revision and limitations.
- [Same-runtime HTTP performance and settings](../../../docs/developer_guide/deepseek_v4_v5p_release_20260911.md#performance-provenance).
  Historical ModelWorker timings below are separate experiments, not substitutes
  for current-source HTTP measurements.

| Record | Comparison | Acceptance scope |
| --- | --- | --- |
| [20260909-v5p4-fp4-model-ab.json](20260909-v5p4-fp4-model-ab.json) | Existing `gmm` versus opt-in `gmm_tuned`; all other execution options fixed | Full 43-layer, 8K-tail ModelWorker; 3832 paired bitwise output rows; ten audited device captures |

The JSON contains exact settings, source fingerprints, report hashes, warm
latency/throughput, allocator memory and selected device-time groups. It is a
whitelisted summary, not a copy of private reports. Batch decode throughput is
aggregate, not per request. Do not compare profiled device time with warm host
latency as if they were the same measurement.

Environment snapshots:

- [Inference dependency pins](requirements-v5p-inference-20260909.txt): every
  package/version from the measured Linux TPU environment is preserved. The
  private absolute editable-checkout line is replaced by an installation
  comment. Consequently this public file does not have the private freeze's
  original hash. Install this repository separately, from its root, with
  `python -m pip install --no-deps -e ./python`.
- [CPU profile-tool pins](requirements-v5p-profile-tools-20260909.txt): separate
  XProf environment, not additions to the inference environment.

These are historical environment records, not a universal lockfile or a promise
that wheels exist for another OS/Python version. In particular `libtpu` targets
Linux and `torch==2.14.0+cpu` requires a matching CPU-wheel source. Preserve the
measured JAX/libtpu versions when reproducing a comparison.

Append a new dated record for future experiments; do not overwrite earlier
measurements. Include the Git revision plus numerical/source fingerprints,
checkpoint revision, actual backend choices, hardware, workload, timer bounds,
warmup/sample policy, numerical results and report hashes. Mark incomplete runs
explicitly and do not mix nodes, fixtures or measurement scopes inside an A/B.

Credentials, cloud account/project configuration, SSH material, checkpoints,
logits, raw traces and full logs are not published here. They remain in ignored
local storage and/or the original independent data disk.
