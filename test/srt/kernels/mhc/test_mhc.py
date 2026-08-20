"""Correctness and performance for the mHC kernels.

Correctness is against ``ref.py``, numpy with no JAX import, so agreement is
evidence about the semantics rather than about two programs sharing a lowering.

Tolerances follow provenance, not storage dtype.  The gate projection runs at
``Precision.DEFAULT``, which truncates its fp32 operands to bf16 on TPU, so
every gate downstream carries bf16 error (1.8e-3 to 2.3e-3) despite fp32
storage.  Only the Sinkhorn kernel is fp32 end to end, and it matches at 2.1e-7.
A JAX reference hides this by truncating in the same place.

mHC is per-token, so a token count is the only shape that matters -- a batch is
just a longer packed sequence.  The last test reports latency and unit
utilisation across those counts; run with ``-s`` to see it.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.mhc import (
    mhc_gates_sharded,
    mhc_gates_token_major,
    mhc_head_collapse_fused,
    mhc_post_fused,
    mhc_pre_fused,
)

from . import ref

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="mHC kernels require real Mosaic lowering"
)

# DeepSeek-V4-Flash shipped configuration.
HC, HIDDEN, ITERS, EPS = 4, 4096, 20, 1e-6
# Anything downstream of the bf16 projection: the repo convention for bf16
# kernels, which these clear by 5x.
PROJECTED = {"rtol": 2e-2, "atol": 1e-2}
# The Sinkhorn kernel in isolation: fp32 in, fp32 out, nothing truncated.
EXACT = {"rtol": 1e-5, "atol": 1e-6}

# Powers of two spanning the tuned Pallas tile sizes, plus counts that are a
# multiple of no block size.  The latter exercise the pad-then-slice-back path,
# where an error would corrupt only the final block.
TOKENS = [128, 256, 512, 1024, 2048, 4096, 8192]
RAGGED_TOKENS = [1, 127, 1000]


def _inputs(n, hc=HC, hidden=HIDDEN, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    mix_hc = ref.mix_hc_width(hc)
    return {
        "x": (jax.random.normal(keys[0], (n, hc, hidden), jnp.float32) * 0.1).astype(jnp.bfloat16),
        "fn": jax.random.normal(keys[1], (mix_hc, hc * hidden), jnp.float32) * 0.01,
        "head_fn": jax.random.normal(keys[2], (hc, hc * hidden), jnp.float32) * 0.01,
        "scale": jnp.asarray([0.7, 1.1, 0.9], jnp.float32),
        "base": jax.random.normal(keys[3], (mix_hc,), jnp.float32) * 0.05,
        "head_scale": jnp.asarray([0.8], jnp.float32),
        "head_base": jax.random.normal(keys[4], (hc,), jnp.float32) * 0.05,
        "block_out": (jax.random.normal(keys[5], (n, hidden), jnp.float32) * 0.1).astype(
            jnp.bfloat16
        ),
        "mixes": jax.random.normal(keys[0], (n, mix_hc), jnp.float32),
    }


def _close(got, want, label, tol=PROJECTED):
    for index, (a, b) in enumerate(zip(got, want)):
        np.testing.assert_allclose(
            np.asarray(a, np.float32),
            np.asarray(b, np.float32),
            err_msg=f"{label}[{index}]",
            **tol,
        )


@pytest.mark.parametrize("n", TOKENS + RAGGED_TOKENS)
def test_gates_match_reference(n):
    """Sinkhorn gates: [n, mix_hc] -> pre, post, comb."""
    d = _inputs(n)
    kw = {"hc_mult": HC, "sinkhorn_iters": ITERS, "eps": EPS}
    _close(
        mhc_gates_token_major(d["mixes"], d["scale"], d["base"], **kw),
        ref.sinkhorn_gates(d["mixes"], d["scale"], d["base"], **kw),
        f"gates n={n}",
        EXACT,
    )


@pytest.mark.parametrize("n", TOKENS + RAGGED_TOKENS)
def test_pre_matches_reference(n):
    """Pre-block mixing: collapse hc streams to one and emit the gates."""
    d = _inputs(n)
    args = (d["x"], d["fn"], d["scale"], d["base"])
    kw = {"hc_mult": HC, "sinkhorn_iters": ITERS, "norm_eps": EPS, "hc_eps": EPS}
    _close(mhc_pre_fused(*args, **kw), ref.pre(*args, **kw), f"pre n={n}")


@pytest.mark.parametrize("n", TOKENS + RAGGED_TOKENS)
def test_post_matches_reference(n):
    """Post-block mixing: expand one stream back to hc and remix the residual."""
    d = _inputs(n)
    kw = {"hc_mult": HC, "sinkhorn_iters": ITERS, "norm_eps": EPS, "hc_eps": EPS}
    _, post, comb = ref.pre(d["x"], d["fn"], d["scale"], d["base"], **kw)
    post = jnp.asarray(post, jnp.float32)
    comb = jnp.asarray(comb, jnp.float32)
    _close(
        [mhc_post_fused(d["block_out"], d["x"], post, comb)],
        [ref.post(d["block_out"], d["x"], post, comb)],
        f"post n={n}",
    )


@pytest.mark.parametrize("n", TOKENS + RAGGED_TOKENS)
def test_head_matches_reference(n):
    """Head collapse: the final hc -> 1 before the LM head."""
    d = _inputs(n)
    args = (d["x"], d["head_fn"], d["head_scale"], d["head_base"])
    _close(
        [mhc_head_collapse_fused(*args, hc_mult=HC, norm_eps=EPS, hc_eps=EPS)],
        [ref.head_collapse(*args, norm_eps=EPS, hc_eps=EPS)],
        f"head n={n}",
    )


@pytest.mark.parametrize("hidden", [4096, 7168])
@pytest.mark.parametrize("hc", [2, 4, 8])
def test_shapes_beyond_the_shipped_config(hc, hidden):
    """Nothing is wired to hc_mult=4 or hidden=4096: a wider model must select a
    smaller block rather than overflow scoped VMEM."""
    d = _inputs(512, hc=hc, hidden=hidden)
    args = (d["x"], d["fn"], d["scale"], d["base"])
    kw = {"hc_mult": hc, "sinkhorn_iters": ITERS, "norm_eps": EPS, "hc_eps": EPS}
    _close(mhc_pre_fused(*args, **kw), ref.pre(*args, **kw), f"pre hc={hc} hidden={hidden}")


@pytest.mark.parametrize("iters", [1, 2, 40])
def test_iteration_counts_other_than_the_shipped_twenty(iters):
    d = _inputs(512)
    kw = {"hc_mult": HC, "sinkhorn_iters": iters, "eps": EPS}
    _close(
        mhc_gates_token_major(d["mixes"], d["scale"], d["base"], **kw),
        ref.sinkhorn_gates(d["mixes"], d["scale"], d["base"], **kw),
        f"gates iters={iters}",
        EXACT,
    )


def test_comb_is_a_near_doubly_stochastic_mixing_matrix():
    """The property the Sinkhorn exists to establish.

    The schedule ends on a column pass, so columns are exact to ~1e-6 while rows
    stay approximate.  Entries must stay positive, or streams could cancel.
    """
    d = _inputs(2048)
    _, _, comb = mhc_gates_token_major(
        d["mixes"], d["scale"], d["base"], hc_mult=HC, sinkhorn_iters=ITERS, eps=EPS
    )
    comb = np.asarray(comb, np.float64)
    assert comb.min() > 0.0
    np.testing.assert_allclose(comb.sum(axis=-2), 1.0, atol=1e-5)
    np.testing.assert_allclose(comb.sum(axis=-1), 1.0, atol=0.2)


def test_sharded_gates_need_no_collective():
    """Per-token work splits over the mesh without any cross-device reduction."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs a multi-device mesh")
    mesh = jax.make_mesh((len(devices),), ("data",))
    d = _inputs(1024)
    kw = {"hc_mult": HC, "sinkhorn_iters": ITERS, "eps": EPS}
    got = mhc_gates_sharded(d["mixes"].T, d["scale"], d["base"], mesh=mesh, **kw)
    want = ref.sinkhorn_gates(d["mixes"], d["scale"], d["base"], **kw)
    _close(
        [g.T if g.ndim == 2 else g.transpose(2, 0, 1) for g in got],
        want,
        "sharded gates",
        EXACT,
    )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
_TC = "VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_"
_HBM = re.compile(
    r"^VF_CHIP_HBM_[01]_HBMC_\d+_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_"
    r"(?:RD_RESP|WR_REQ|PARTIAL_WRITE_REQ)_PS[01]$"
)
V6E_HZ, V6E_PEAK_BW = 1.75e9, 1640e9


def _profile(call, iterations=20):
    """Active ms, MXU busy % and HBM bandwidth %, over merged XLA-module spans.

    The denominator is the time a module was actually resident, so gaps between
    dispatches do not dilute the percentages.
    """
    for _ in range(3):
        jax.block_until_ready(call())
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    with tempfile.TemporaryDirectory() as tmp:
        with jax.profiler.trace(tmp, profiler_options=options):
            for _ in range(iterations):
                result = call()
            jax.block_until_ready(result)
        profile = jax.profiler.ProfileData.from_file(
            str(next(Path(tmp).glob("plugins/profile/**/*.xplane.pb")))
        )
    device = profile.find_plane_with_name("/device:TPU:0")
    counters, modules = {}, []
    for line in device.lines:
        if line.name == "XLA Modules":
            modules = [(event.start_ns, event.end_ns) for event in line.events]
        elif line.name.startswith("counters_"):
            for event in line.events:
                value = dict(event.stats).get("counter_value")
                if value is not None:
                    counters[event.name] = counters.get(event.name, 0) + int(value)
    merged = []
    for begin, end in sorted(modules):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    active_s = sum(end - begin for begin, end in merged) / 1e9
    mxu = 0.5 * counters.get(_TC + "MXU_BUSY_1", 0) + counters.get(_TC + "MXU_BUSY_2", 0)
    moved = 32 * sum(v for k, v in counters.items() if _HBM.match(k))
    return (
        active_s / iterations * 1e3,
        100.0 * mxu / (active_s * V6E_HZ),
        100.0 * moved / (V6E_PEAK_BW * active_s),
    )


def _kernels(d):
    """The four entry points, each as a no-argument callable."""
    pre_kw = {"hc_mult": HC, "sinkhorn_iters": ITERS, "norm_eps": EPS, "hc_eps": EPS}
    _, post, comb = mhc_pre_fused(d["x"], d["fn"], d["scale"], d["base"], **pre_kw)
    return [
        (
            "pre",
            jax.jit(lambda: mhc_pre_fused(d["x"], d["fn"], d["scale"], d["base"], **pre_kw)),
        ),
        (
            "gates",
            jax.jit(
                lambda: mhc_gates_token_major(
                    d["mixes"],
                    d["scale"],
                    d["base"],
                    hc_mult=HC,
                    sinkhorn_iters=ITERS,
                    eps=EPS,
                )
            ),
        ),
        ("post", jax.jit(lambda: mhc_post_fused(d["block_out"], d["x"], post, comb))),
        (
            "head",
            jax.jit(
                lambda: mhc_head_collapse_fused(
                    d["x"],
                    d["head_fn"],
                    d["head_scale"],
                    d["head_base"],
                    hc_mult=HC,
                    norm_eps=EPS,
                    hc_eps=EPS,
                )
            ),
        ),
    ]


def test_performance(capsys):
    """Latency and unit utilisation for each kernel across token counts.

    Reported, not asserted, beyond every shape completing: the numbers move with
    the toolchain, and a threshold tuned on one jaxlib fails on the next.  Run
    with ``-s`` to see the table.
    """
    with capsys.disabled():
        print(f"\n  {'tokens':>7s} {'kernel':>7s} {'ms':>9s} {'MXU':>7s} {'HBM':>7s}")
        for n in TOKENS:
            d = _inputs(n)
            for name, call in _kernels(d):
                active_ms, mxu, bandwidth = _profile(call)
                assert active_ms > 0.0
                print(f"  {n:7d} {name:>7s} {active_ms:9.4f} {mxu:6.1f}% {bandwidth:6.1f}%")
