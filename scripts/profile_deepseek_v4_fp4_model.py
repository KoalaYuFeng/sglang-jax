"""Same-source full-model A/B for the opt-in tuned FP4 adapter.

Both sides enable the accepted FP8/norm/merged/wo_a kernels, attention TP4
and batched CSA decode. Only V4 MoE changes from gmm to gmm_tuned. Reuse the
existing ModelWorker profiling harness; no core scheduling modifications.
Every candidate row must match the fresh control BITWISE as well as the
independently accepted tuned-backend cold B1/B2/B4 numerical fixture. The
fixture configuration is separate from the A/B control, not relabeled as it.
This is not an Engine/HTTP test.
"""

from profile_deepseek_v4_dense import BASE_OPTIONS, main

CONTROL_OPTIONS = {
    **BASE_OPTIONS,
    "fp8_backend": "gmm",
    "fused_norm": True,
    "merged_projections": True,
    "fused_wo_a": True,
}
FIXTURE_OPTIONS = {**CONTROL_OPTIONS, "moe_backend": "gmm_tuned"}

if __name__ == "__main__":
    main(
        base_options=CONTROL_OPTIONS,
        candidate_options={"moe_backend"},
        scope=__doc__,
        entry_source=__file__,
        require_control_bitwise=True,
        fixture_options=FIXTURE_OPTIONS,
    )
