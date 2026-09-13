# V4 diagnostic archive index

This is a **classified index**, not a file relocation or a numerical-acceptance
claim. The 30 historical debug/probe/replay/isolate entry points remain under
`scripts/`: acceptance tools still import some of their capture/comparison
helpers. Keeping the paths preserves those imports, direct CLI invocation and
source-relative evidence handling. No historical evidence is deleted.

For supported acceptance commands and result records, use the
[benchmark index](../../../benchmark/deepseek_v4/README.md).
These tools are not serving backends or part of the default test command.

Run an applicable tool from the repository root with its original name, for
example `PYTHONPATH=python python scripts/replay_deepseek_v4_8023.py --help`.
Inspect its arguments and prerequisites before execution: some commands load
the full checkpoint, require a TPU, or emit private tensor captures. The index
does not automatically execute any diagnostic.

Old captures and source-line traces must use their matching frozen revision
and analyzer. Import migration does not retroactively revalidate an experiment.
The index-coverage test requires every matching entry point to appear once.

## FP8 projections and accumulation

- [debug_deepseek_v4_fp8_accumulator.py](../../debug_deepseek_v4_fp8_accumulator.py)
- [debug_deepseek_v4_fp8_last_coordinates.py](../../debug_deepseek_v4_fp8_last_coordinates.py)
- [debug_deepseek_v4_fp8_partials.py](../../debug_deepseek_v4_fp8_partials.py)
- [debug_deepseek_v4_fp8_residuals.py](../../debug_deepseek_v4_fp8_residuals.py)

## CSA/HCA/compressor state

- [debug_deepseek_v4_compressor.py](../../debug_deepseek_v4_compressor.py)
- [debug_deepseek_v4_paged_compressor.py](../../debug_deepseek_v4_paged_compressor.py)
- [isolate_deepseek_v4_csa_attention.py](../../isolate_deepseek_v4_csa_attention.py)
- [isolate_deepseek_v4_csa_compressor.py](../../isolate_deepseek_v4_csa_compressor.py)
- [probe_deepseek_v4_8023_compressor.py](../../probe_deepseek_v4_8023_compressor.py)
- [probe_deepseek_v4_csa_gemv.py](../../probe_deepseek_v4_csa_gemv.py)
- [probe_deepseek_v4_hca_attention.py](../../probe_deepseek_v4_hca_attention.py)
- [probe_deepseek_v4_hca_cache_swap.py](../../probe_deepseek_v4_hca_cache_swap.py)
- [probe_deepseek_v4_hca_emitter.py](../../probe_deepseek_v4_hca_emitter.py)
- [probe_deepseek_v4_hca_projection.py](../../probe_deepseek_v4_hca_projection.py)
- [probe_deepseek_v4_hca_sum.py](../../probe_deepseek_v4_hca_sum.py)
- [probe_deepseek_v4_oracle_pool.py](../../probe_deepseek_v4_oracle_pool.py)
- [probe_deepseek_v4_pool_order.py](../../probe_deepseek_v4_pool_order.py)
- [probe_deepseek_v4_projection.py](../../probe_deepseek_v4_projection.py)
- [replay_deepseek_v4_csa_divergence.py](../../replay_deepseek_v4_csa_divergence.py)

## mHC residuals

- [probe_deepseek_v4_mhc_post.py](../../probe_deepseek_v4_mhc_post.py)

## Whole-model, chunk and position diagnostics

- [debug_deepseek_v4_8023.py](../../debug_deepseek_v4_8023.py)
- [debug_deepseek_v4_chunking.py](../../debug_deepseek_v4_chunking.py)
- [debug_deepseek_v4_cpu_attention_stages.py](../../debug_deepseek_v4_cpu_attention_stages.py)
- [debug_deepseek_v4_cpu_layer_stages.py](../../debug_deepseek_v4_cpu_layer_stages.py)
- [debug_deepseek_v4_framework_head.py](../../debug_deepseek_v4_framework_head.py)
- [debug_deepseek_v4_native_layers.py](../../debug_deepseek_v4_native_layers.py)
- [debug_deepseek_v4_oracle_decode.py](../../debug_deepseek_v4_oracle_decode.py)
- [debug_deepseek_v4_position108.py](../../debug_deepseek_v4_position108.py)
- [debug_deepseek_v4_position108_origin.py](../../debug_deepseek_v4_position108_origin.py)
- [replay_deepseek_v4_8023.py](../../replay_deepseek_v4_8023.py)
