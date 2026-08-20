"""DeepSeek-V4 multi-stream hyper-connection (mHC) kernels for TPU.

mHC replaces the single residual stream with ``hc_mult`` parallel ones.  Around
each block they collapse to one (``mhc_pre_fused``), the block runs, and the
output expands back while the residual is remixed (``mhc_post_fused``);
``mhc_head_collapse_fused`` folds them down once at the end.  The weights are
per-token gates -- ``pre`` and ``post`` are sigmoids of a projection, ``comb`` is
that projection driven near doubly-stochastic by a Sinkhorn iteration.

Four specialised Pallas programs:

* ``mhc-collapse-pre``    projection, RMS, and the per-block gated collapse
* ``mhc-sinkhorn-gates``  [N, mix_hc] -> pre / post / comb
* ``mhc-post``            expansion and residual mixing
* ``mhc-collapse-head``   projection, RMS, and the final gated collapse

The pre and head programs share one Python kernel template, specialised at
compile time.  All four sizes use Pallas; there is no automatic XLA crossover.

"""

from __future__ import annotations

import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.mhc.tune import kernel_schedule


def _device_kind() -> str:
    devices = jax.devices()
    return devices[0].device_kind if devices else "TPU v6e"


def mix_hc_width(hc_mult: int) -> int:
    """Width of the mHC mixing projection: pre(hc) + post(hc) + comb(hc*hc)."""
    return (2 + hc_mult) * hc_mult


# ---------------------------------------------------------------------------
# Multi-device wrapper
# ---------------------------------------------------------------------------


def _fit_block(budget: int, n: int) -> int:
    """Block for ``n`` tokens: the VMEM ceiling, capped by ``n`` itself.

    The kernel pads ``n`` up to a multiple of the block, so a block larger than
    the input wastes a whole grid step on padding -- at n=1 with a 128-token
    block that is 128x the work.  Capping at the sublane-aligned ``n`` keeps the
    single-token case within 1.2x of XLA instead of 5x.
    """
    return max(8, min(int(budget), -(-int(n) // 8) * 8))


def mhc_sharded(
    fn,
    *args,
    mesh,
    tokens_axis: str = "data",
    n_token_args: int = 1,
    n_out: int = 1,
    token_major: bool = True,
    output_specs=None,
    **kwargs,
):
    """Split the token axis across a mesh; every mHC op is per-token, so nothing
    is reduced.

    The first ``n_token_args`` arguments are split and the rest replicated --
    stated rather than inferred, since a weight can have an activation's rank.
    ``output_specs`` handles mixed-rank outputs whose token axes differ.
    """

    def spec(array, index):
        if index >= n_token_args:
            return P(*([None] * array.ndim))
        if token_major:
            return P(tokens_axis, *([None] * (array.ndim - 1)))
        return P(*([None] * (array.ndim - 1)), tokens_axis)

    in_specs = tuple(spec(a, i) for i, a in enumerate(args))
    # Explicit meshes need the caller sharding to match shard_map's contract.
    args = tuple(jax.sharding.reshard(a, s) for a, s in zip(args, in_specs))
    if output_specs is None:
        out = P(tokens_axis) if token_major else P(None, tokens_axis)
        output_specs = out if n_out == 1 else tuple(out for _ in range(n_out))
    return jax.shard_map(
        functools.partial(fn, **kwargs),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=output_specs,
        check_vma=False,
    )(*args)


# ---------------------------------------------------------------------------
# Sinkhorn gates: [N, mix_hc] -> pre, post, comb
# ---------------------------------------------------------------------------

# Lane width of the TPU vector unit; token blocks are padded up to a multiple.
LANE = 128
MIN_TUNED_BLOCK_TOKENS = 512
MAX_BLOCK_TOKENS = kernel_schedule("TPU v6e", hc_mult=4, hidden=4096).gates_block_tokens


def _auto_block_tokens(n: int) -> int:
    """Largest measured-optimal block without padding a short input to 512.

    Per-grid-step overhead dominates below 2048, but ``n`` is padded up to a
    multiple of the block, so an oversized block wastes work on short inputs.
    Measured on v6e, bit-identical across block sizes: 1.23x at n=2048 rising to
    1.51x at n=131072, with 4096 slower everywhere.
    """
    block = MIN_TUNED_BLOCK_TOKENS
    while block < MAX_BLOCK_TOKENS and block < n:
        block *= 2
    padded_n = -(-n // LANE) * LANE
    return max(LANE, min(block, padded_n))


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _sinkhorn_gates_kernel(
    mixes_ref,  # [mix_hc, BT] f32
    scale_ref,  # [3, 1] f32
    base_ref,  # [mix_hc, 1] f32
    pre_ref,  # [hc, BT] f32
    post_ref,  # [hc, BT] f32
    comb_ref,  # [hc*hc, BT] f32
    *,
    hc: int,
    sinkhorn_iters: int,
    eps: float,
):
    mixes = mixes_ref[...]
    base = base_ref[...]
    scale = scale_ref[...]

    # ---- pre / post gates -------------------------------------------------
    # pre adds eps after the sigmoid; post does not, and carries a factor of 2.
    pre_ref[...] = jax.nn.sigmoid(mixes[:hc] * scale[0:1] + base[:hc]) + eps
    post_ref[...] = 2.0 * jax.nn.sigmoid(mixes[hc : 2 * hc] * scale[1:2] + base[hc : 2 * hc])

    # ---- comb: [hc*hc, BT] -> [hc, hc, BT], tokens stay on lanes ----------
    # comb[j, k] == mixes[2*hc + j*hc + k], so a (hc, hc, BT) reshape puts the row
    # index j on the outer sublane group and k on the inner one; both reduction
    # axes are then leading axes with tokens on lanes.
    #
    # Measured alternative that did NOT pay off: holding all hc*hc entries as
    # separate [1, BT] vectors and fully unrolling the schedule as register adds
    # (no reduction primitive at all).  That is *slower* -- 1.16x vs 1.45x over the
    # XLA reference at N=131072 -- presumably from register pressure across the
    # unrolled iterations plus the final concatenate.  Keeping jnp.sum here.
    c = mixes[2 * hc :] * scale[2:3] + base[2 * hc :]
    c = c.reshape(hc, hc, -1)

    # iteration 0: row softmax (+eps), then one column normalisation
    c = c - jnp.max(c, axis=1, keepdims=True)
    c = jnp.exp(c)
    c = c / jnp.sum(c, axis=1, keepdims=True) + eps
    c = c / (jnp.sum(c, axis=0, keepdims=True) + eps)

    # iterations 1..sinkhorn_iters-1: row then column.  The whole loop stays in
    # VMEM; only the final result is written back.
    def body(_, cc):
        cc = cc / (jnp.sum(cc, axis=1, keepdims=True) + eps)
        cc = cc / (jnp.sum(cc, axis=0, keepdims=True) + eps)
        return cc

    # Unrolled: the iterations stay strictly ordered, so every gate is bit
    # identical, but the scheduler can overlap independent work across them
    # (measured 4-6%).
    c = jax.lax.fori_loop(0, sinkhorn_iters - 1, body, c, unroll=True)

    comb_ref[...] = c.reshape(hc * hc, -1)


@functools.partial(
    jax.jit,
    static_argnames=("hc_mult", "sinkhorn_iters", "eps", "block_tokens", "interpret"),
)
def mhc_gates(
    mixes_t: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    block_tokens: int | None = None,
    interpret: bool | None = None,
):
    """Fused gate generation, features-major so tokens land on the lane axis.

    ``block_tokens=None`` selects the measured optimum for the input length.
    """
    hc = hc_mult
    mix_hc = mix_hc_width(hc)
    if mixes_t.ndim != 2 or mixes_t.shape[0] != mix_hc:
        raise ValueError(
            f"mixes_t must be [mix_hc={mix_hc}, N] for hc_mult={hc}, got {mixes_t.shape}"
        )
    if hc_scale.shape != (3,):
        raise ValueError(f"hc_scale must be [3], got {hc_scale.shape}")
    if hc_base.shape != (mix_hc,):
        raise ValueError(f"hc_base must be [{mix_hc}], got {hc_base.shape}")
    if sinkhorn_iters < 1:
        raise ValueError(f"sinkhorn_iters must be >= 1, got {sinkhorn_iters}")

    n = mixes_t.shape[1]
    if block_tokens is None:
        block_tokens = _auto_block_tokens(n)
    bt = max(LANE, (int(block_tokens) // LANE) * LANE)
    n_pad = -(-n // bt) * bt  # round up
    if n_pad != n:
        mixes_t = jnp.pad(mixes_t, ((0, 0), (0, n_pad - n)))

    if interpret is None:
        interpret = get_interpret()

    kernel = functools.partial(
        _sinkhorn_gates_kernel, hc=hc, sinkhorn_iters=int(sinkhorn_iters), eps=float(eps)
    )
    pre, post, comb = pl.pallas_call(
        kernel,
        grid=(n_pad // bt,),
        in_specs=[
            pl.BlockSpec((mix_hc, bt), lambda i: (0, i)),
            pl.BlockSpec((3, 1), lambda i: (0, 0)),
            pl.BlockSpec((mix_hc, 1), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((hc, bt), lambda i: (0, i)),
            pl.BlockSpec((hc, bt), lambda i: (0, i)),
            pl.BlockSpec((hc * hc, bt), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((hc, n_pad), jnp.float32),
            jax.ShapeDtypeStruct((hc, n_pad), jnp.float32),
            jax.ShapeDtypeStruct((hc * hc, n_pad), jnp.float32),
        ],
        interpret=interpret,
        name="mhc-sinkhorn-gates",
    )(
        mixes_t.astype(jnp.float32),
        hc_scale.astype(jnp.float32).reshape(3, 1),
        hc_base.astype(jnp.float32).reshape(mix_hc, 1),
    )

    if n_pad != n:
        pre, post, comb = pre[:, :n], post[:, :n], comb[:, :n]
    return pre, post, comb.reshape(hc, hc, n)


def mhc_gates_sharded(mixes_t, hc_scale, hc_base, *, mesh, tokens_axis: str = "data", **kwargs):
    """``mhc_gates`` on a multi-device mesh."""
    return mhc_sharded(
        mhc_gates,
        mixes_t,
        hc_scale,
        hc_base,
        mesh=mesh,
        tokens_axis=tokens_axis,
        n_token_args=1,
        n_out=3,
        token_major=False,
        output_specs=(
            P(None, tokens_axis),
            P(None, tokens_axis),
            P(None, None, tokens_axis),
        ),
        **kwargs,
    )


def mhc_gates_token_major(
    mixes: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    block_tokens: int | None = None,
    interpret: bool | None = None,
):
    """``mhc_gates`` for the natural [..., mix_hc] layout.

    Transposes on both ends; prefer ``mhc_gates`` on hot paths and produce
    ``mixes`` features-major at the source.
    """
    lead = mixes.shape[:-1]
    mix_hc = mixes.shape[-1]
    flat = mixes.reshape(-1, mix_hc).T  # [mix_hc, N]
    pre, post, comb = mhc_gates(
        flat,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        block_tokens=block_tokens,
        interpret=interpret,
    )
    pre = pre.T.reshape(*lead, hc_mult)
    post = post.T.reshape(*lead, hc_mult)
    comb = jnp.transpose(comb, (2, 0, 1)).reshape(*lead, hc_mult, hc_mult)
    return pre, post, comb


# ---------------------------------------------------------------------------
# Collapse: hc streams -> 1, for the per-block pre and the final head
# ---------------------------------------------------------------------------

# Block sizes come from the VMEM model in tune.py rather than a hardcoded tile
# and are derived per call from the real hc_mult and hidden dimensions.


def _collapse_kernel(
    x_ref,  # [BT, hc, d]  input dtype
    fn_ref,  # [rows, hc*d] f32   (rows = hc for "head", mix_hc for "pre")
    scale_ref,  # [4] f32 (padded; only the first 1 or 3 entries are used)
    base_ref,  # [rows] f32
    *outs,  # y_ref [BT, d], followed by mixes_ref [BT, rows] in "pre"
    hc: int,
    d: int,
    mode: str,
    hc_eps: float,
    norm_eps: float,
    dot_precision,
):
    x = x_ref[...]  # [BT, hc, d]
    bt = x.shape[0]
    y_ref, *extra_outs = outs
    # The activation is already resident, so the sum of squares is free in
    # bandwidth.  Its reduction tree differs from XLA's, so a small number of
    # FP32 gate values and final BF16 outputs can move by a few ULP.
    rms = jax.lax.rsqrt(
        jnp.sum(jnp.square(x.astype(jnp.float32).reshape(bt, hc * d)), axis=-1, keepdims=True)
        / (hc * d)
        + norm_eps
    )

    if mode == "head":
        # Head normalizes in fp32, rounds to bf16, then projects.  Hoisting the
        # RMS scalar past the matmul is algebraically valid but removes that
        # deliberate bf16 boundary.
        xf = x.astype(jnp.float32).reshape(bt, hc * d)
        normalized = (xf * rms).astype(jnp.bfloat16)
        mixes = jax.lax.dot_general(
            normalized,
            fn_ref[...],
            (((1,), (1,)), ((), ())),
            precision=dot_precision,
            preferred_element_type=jnp.float32,
        )
    else:
        xf = x.astype(jnp.float32).reshape(bt, hc * d)
        # The RMS scalar lands after the matmul; a linear map is homogeneous.
        mixes = (
            jax.lax.dot_general(
                xf,
                fn_ref[...],
                (((1,), (1,)), ((), ())),
                precision=dot_precision,
                preferred_element_type=jnp.float32,
            )
            * rms
        )  # [BT, rows]

    scale = scale_ref[...].reshape(-1)
    base = base_ref[...].reshape(-1)

    if mode == "head":
        # Head has no Sinkhorn and stays token-major.
        pre = jax.nn.sigmoid(mixes * scale[0] + base[None, :]) + hc_eps
    else:
        pre = jax.nn.sigmoid(mixes[:, :hc] * scale[0] + base[None, :hc]) + hc_eps
        # Emit the whole [N, mix_hc] projection rather than pre-sliced pieces:
        # its layout selects the reduction tree for some batch shapes.
        extra_outs[0][...] = mixes

    # Keep this width-hc contraction on the VPU.  A dot_general is sent to the
    # MXU, where Mosaic rounds the FP32 gates to BF16 before multiplying.  The
    # static loop streams one residual at a time through an FP32 accumulator,
    # preserving the gates without materialising an FP32 [BT, hc, d] tensor.
    collapsed = jnp.zeros((bt, d), dtype=jnp.float32)
    for stream in range(hc):
        collapsed = collapsed + pre[:, stream, None] * x[:, stream, :].astype(jnp.float32)
    y_ref[...] = collapsed.astype(y_ref.dtype)


def _run(
    x_streams,
    hc_fn,
    hc_scale,
    hc_base,
    *,
    mode: str,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    block_tokens: int | None,
    interpret: bool | None,
    dot_precision,
):
    hc = hc_mult
    if x_streams.ndim != 3:
        raise ValueError(f"x_streams must be [N, hc, d], got {x_streams.shape}")
    n, hc_in, d = x_streams.shape
    if hc_in != hc:
        raise ValueError(f"x_streams axis 1 must be hc_mult={hc}, got {hc_in}")
    rows = hc if mode == "head" else mix_hc_width(hc)
    if hc_fn.shape != (rows, hc * d):
        raise ValueError(f"hc_fn must be [{rows}, {hc * d}] for mode={mode!r}, got {hc_fn.shape}")
    if hc_base.shape != (rows,):
        raise ValueError(f"hc_base must be [{rows}], got {hc_base.shape}")
    n_scale = 1 if mode == "head" else 3
    if hc_scale.size != n_scale:
        raise ValueError(f"hc_scale must have {n_scale} entries, got {hc_scale.shape}")
    if mode not in ("head", "pre"):
        raise ValueError(f"mode must be 'head' or 'pre', got {mode!r}")
    if mode == "pre" and sinkhorn_iters < 1:
        raise ValueError(f"sinkhorn_iters must be >= 1, got {sinkhorn_iters}")

    # VMEM scales with hc*d, so a wider model needs a smaller block than the
    # hidden=4096, hc_mult=4 shape the constants are named for.
    if block_tokens is None:
        budget = kernel_schedule(_device_kind(), hc_mult=hc, hidden=d).collapse_block_tokens
        block_tokens = _fit_block(budget, n)
    bt = max(8, int(block_tokens))
    n_pad = -(-n // bt) * bt
    if n_pad != n:
        x_streams = jnp.pad(x_streams, ((0, n_pad - n), (0, 0), (0, 0)))

    # pad the scale vector to 4 lanes so it has a sane VMEM layout
    scale_p = (
        jnp.zeros((4,), jnp.float32)
        .at[:n_scale]
        .set(hc_scale.astype(jnp.float32).reshape(-1)[:n_scale])
        .reshape(4, 1)
    )

    if interpret is None:
        interpret = get_interpret()

    out_shape = [jax.ShapeDtypeStruct((n_pad, d), x_streams.dtype)]
    out_specs = [pl.BlockSpec((bt, d), lambda i: (i, 0))]
    if mode == "pre":
        out_shape += [
            jax.ShapeDtypeStruct((n_pad, rows), jnp.float32),
        ]
        out_specs += [
            pl.BlockSpec((bt, rows), lambda i: (i, 0)),
        ]

    kernel = functools.partial(
        _collapse_kernel,
        hc=hc,
        d=d,
        mode=mode,
        hc_eps=float(hc_eps),
        norm_eps=float(norm_eps),
        dot_precision=dot_precision,
    )
    in_specs = [
        pl.BlockSpec((bt, hc, d), lambda i: (i, 0, 0)),
        pl.BlockSpec((rows, hc * d), lambda i: (0, 0)),
        pl.BlockSpec((4, 1), lambda i: (0, 0)),
        pl.BlockSpec((rows, 1), lambda i: (0, 0)),
    ]
    operands = [
        x_streams,
        hc_fn.astype(jnp.float32),
        scale_p,
        hc_base.astype(jnp.float32).reshape(rows, 1),
    ]
    res = pl.pallas_call(
        kernel,
        grid=(n_pad // bt,),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=out_shape,
        interpret=interpret,
        name=f"mhc-collapse-{mode}",
    )(*operands)

    if mode == "head":
        y = res if not isinstance(res, (tuple, list)) else res[0]
        return y[:n]
    y, mixes = res
    mixes = mixes[:n]
    # Own kernel rather than inline XLA: the [bt, hc, hc] Sinkhorn state wants a
    # 2048-token block while collapse is pinned near 128 by its FP32 projection
    # operand.  Merging the two measured 0.80x.
    _, post, comb = mhc_gates_token_major(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc,
        sinkhorn_iters=sinkhorn_iters,
        eps=hc_eps,
    )
    return y[:n], post, comb


@functools.partial(
    jax.jit,
    static_argnames=(
        "hc_mult",
        "norm_eps",
        "hc_eps",
        "block_tokens",
        "interpret",
        "dot_precision",
    ),
)
def mhc_head_collapse_fused(
    x_streams: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
    block_tokens: int | None = None,
    interpret: bool | None = None,
    dot_precision=jax.lax.Precision.DEFAULT,
):
    """Fold hc streams into one at the end of the model."""
    if x_streams.ndim < 3:
        raise ValueError(f"x_streams must be [..., hc, d], got {x_streams.shape}")
    outer_shape = x_streams.shape[:-2]
    hidden = x_streams.shape[-1]
    # Leading dimensions are merged for the kernel and restored afterwards.  On
    # TPU that reshape is a bitcast, because the tiling lives on the last two
    # axes -- measured at exactly the read-once write-once traffic.
    x_flat = x_streams.reshape(-1, x_streams.shape[-2], hidden)
    output = _run(
        x_flat,
        hc_fn,
        hc_scale,
        hc_base,
        mode="head",
        hc_mult=hc_mult,
        sinkhorn_iters=1,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        block_tokens=block_tokens,
        interpret=interpret,
        dot_precision=dot_precision,
    )
    return output.reshape(*outer_shape, hidden)


@functools.partial(
    jax.jit,
    static_argnames=(
        "hc_mult",
        "sinkhorn_iters",
        "norm_eps",
        "hc_eps",
        "block_tokens",
        "interpret",
        "dot_precision",
    ),
)
def mhc_pre_fused(
    x_streams: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    block_tokens: int | None = None,
    interpret: bool | None = None,
    dot_precision=jax.lax.Precision.DEFAULT,
):
    """Collapse hc streams to one and produce the gates the post step needs."""
    if x_streams.ndim < 3:
        raise ValueError(f"x_streams must be [..., hc, d], got {x_streams.shape}")
    outer_shape = x_streams.shape[:-2]
    hidden = x_streams.shape[-1]
    x_flat = x_streams.reshape(-1, x_streams.shape[-2], hidden)
    y, post, comb = _run(
        x_flat,
        hc_fn,
        hc_scale,
        hc_base,
        mode="pre",
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        block_tokens=block_tokens,
        interpret=interpret,
        dot_precision=dot_precision,
    )
    return (
        y.reshape(*outer_shape, hidden),
        post.reshape(*outer_shape, hc_mult),
        comb.reshape(*outer_shape, hc_mult, hc_mult),
    )


# ---------------------------------------------------------------------------
# Post: 1 stream -> hc, with the residual remixed by comb
# ---------------------------------------------------------------------------


# Named for the shipped DeepSeek-V4 shape; each call derives its own block.


def _expand(x, res, post, comb, *, precision):
    """Broadcast the block output over hc streams and add the remixed residual:
    ``y[t,j,:] = post[t,j]*x[t,:] + sum_i comb[t,i,j]*res[t,i,:]``."""
    mixed = jax.lax.dot_general(
        comb,
        res.astype(jnp.float32),
        (((1,), (1,)), ((0,), (0,))),
        precision=precision,
        preferred_element_type=jnp.float32,
    )  # [BT, hc, d]
    return post[:, :, None] * x[:, None, :].astype(jnp.float32) + mixed


# ---------------------------------------------------------------------------
# 1. mhc_post on its own
# ---------------------------------------------------------------------------


def _post_kernel(x_ref, res_ref, post_ref, comb_ref, y_ref, *, precision):
    y = _expand(x_ref[...], res_ref[...], post_ref[...], comb_ref[...], precision=precision)
    y_ref[...] = y.astype(y_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_tokens",
        "interpret",
        "precision",
    ),
)
def mhc_post_fused(
    x: jax.Array,
    residual_streams: jax.Array,
    post: jax.Array,
    comb: jax.Array,
    *,
    block_tokens: int | None = None,
    interpret: bool | None = None,
    precision=jax.lax.Precision.DEFAULT,
):
    """Expand one stream back to hc, remixing the residual by ``comb``."""
    if residual_streams.ndim != 3:
        raise ValueError(f"residual_streams must be [N, hc, d], got {residual_streams.shape}")
    n, hc, d = residual_streams.shape
    if x.shape != (n, d):
        raise ValueError(f"x must be [{n}, {d}], got {x.shape}")
    if post.shape != (n, hc):
        raise ValueError(f"post must be [{n}, {hc}], got {post.shape}")
    if comb.shape != (n, hc, hc):
        raise ValueError(f"comb must be [{n}, {hc}, {hc}], got {comb.shape}")

    if block_tokens is None:
        block_tokens = _fit_block(
            kernel_schedule(_device_kind(), hc_mult=hc, hidden=d).post_block_tokens, n
        )
    bt = max(8, int(block_tokens))
    n_pad = -(-n // bt) * bt
    if n_pad != n:
        pad = n_pad - n
        x = jnp.pad(x, ((0, pad), (0, 0)))
        residual_streams = jnp.pad(residual_streams, ((0, pad), (0, 0), (0, 0)))
        post = jnp.pad(post, ((0, pad), (0, 0)))
        comb = jnp.pad(comb, ((0, pad), (0, 0), (0, 0)))

    if interpret is None:
        interpret = get_interpret()

    y = pl.pallas_call(
        functools.partial(_post_kernel, precision=precision),
        grid=(n_pad // bt,),
        in_specs=[
            pl.BlockSpec((bt, d), lambda i: (i, 0)),
            pl.BlockSpec((bt, hc, d), lambda i: (i, 0, 0)),
            pl.BlockSpec((bt, hc), lambda i: (i, 0)),
            pl.BlockSpec((bt, hc, hc), lambda i: (i, 0, 0)),
        ],
        out_specs=pl.BlockSpec((bt, hc, d), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((n_pad, hc, d), x.dtype),
        interpret=interpret,
        name="mhc-post",
    )(
        x,
        residual_streams,
        post.astype(jnp.float32),
        comb.astype(jnp.float32),
    )
    return y[:n]


__all__ = [
    "kernel_schedule",
    "mhc_gates",
    "mhc_gates_sharded",
    "mhc_gates_token_major",
    "mhc_head_collapse_fused",
    "mhc_post_fused",
    "mhc_pre_fused",
    "mhc_sharded",
]
