"""Benchmark the end-to-end CSA kernel family from raw HBM tensors.

The timed boundary includes both compressors and their cache writes,
Lightning Top-K, selected-cache gather, sliding-window attention, and the
final online-softmax merge. Query projections remain outside that boundary.
"""

from __future__ import annotations

import argparse
import math
import re
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from sgl_jax.srt.kernels.csa.csa import build_csa_step
from sgl_jax.srt.kernels.csa.tune import (
    CSA_ATTENTION_DIM,
    CSA_ATTENTION_HEADS,
    CSA_CACHE_PACKING,
    CSA_COMPRESSION_RATIO,
    CSA_DEFAULT_PAGE_SIZE,
    CSA_DUAL_PROJECTION_DIM,
    CSA_HIDDEN_DIM,
    CSA_INDEX_DIM,
    CSA_INDEX_HEADS,
    CSA_INDEX_PROJECTED_DIM,
    CSA_INDEX_RECORD_BYTES,
    CSA_MAIN_NOPE_DIM,
    CSA_MAIN_NOPE_SCALE_COUNT,
    CSA_MAIN_PROJECTED_DIM,
    CSA_ROPE_DIM,
    CSA_ROPE_FREQUENCY_DIM,
    CSA_STATE_SLOTS,
    CSA_WINDOW_SIZE,
    TPU_V6E,
)

ROTATING_BUFFERS = 4
PREFILL_QUERY_TOKENS = CSA_WINDOW_SIZE
RAGGED_QUERY_LENGTHS = (
    1,
    TPU_V6E.sublanes,
    CSA_COMPRESSION_RATIO * TPU_V6E.sublanes,
    CSA_WINDOW_SIZE,
)
WINDOW_CACHE_PAGES_PER_REQUEST = (
    CSA_WINDOW_SIZE + CSA_DEFAULT_PAGE_SIZE - 1
) // CSA_DEFAULT_PAGE_SIZE
INPUT_WEIGHT_SCALE = 0.005
INPUT_APE_SCALE = 0.02
INPUT_CACHE_SCALE = 0.25
BENCHMARK_SEED = 31000

MEASURE_WARMUPS = 3
MEASURE_ITERATIONS = 10
PROFILE_WARMUPS = 5
PROFILE_ITERATIONS = 10
TPU_V6E_CLOCK_HZ = 1.75e9
TPU_V6E_FALLBACK_HBM_GBPS = 1640.0
HBM_COUNTER_BYTES = 32
VECTOR_COUNTER_COUNT = 4
MXU_BUSY_ONE_CYCLE_WEIGHT = 0.5
NANOSECONDS_PER_SECOND = 1e9
NANOSECONDS_PER_MILLISECOND = 1e6

TC_PREFIX = "VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_"
HBM_READ = re.compile(
    r"^VF_CHIP_HBM_[01]_HBMC_\d+_CMN_HI_FREQ_STATS_COUNTERS_"
    r"UNPRIVILEGED_RD_RESP_PS[01]$"
)
HBM_WRITE = re.compile(
    r"^VF_CHIP_HBM_[01]_HBMC_\d+_CMN_HI_FREQ_STATS_COUNTERS_"
    r"UNPRIVILEGED_(?:WR_REQ|PARTIAL_WRITE_REQ)_PS[01]$"
)


@dataclass(frozen=True)
class Measurement:
    mean_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class Resources:
    device_active_ms: float
    device_envelope_ms: float
    mxu_busy_pct: float
    vector_issue_pct: float
    hbm_gb_s: float
    hbm_pct: float
    hbm_bytes_per_call: float
    top_ops: tuple[tuple[str, float], ...]
    operations: tuple[tuple[str, float], ...]


def _ready(value):
    jax.block_until_ready(value)
    return value


def measure(call) -> Measurement:
    _ready(call())  # compile
    for _ in range(MEASURE_WARMUPS):
        _ready(call())
    samples = []
    for _ in range(MEASURE_ITERATIONS):
        start = time.perf_counter_ns()
        _ready(call())
        samples.append((time.perf_counter_ns() - start) / NANOSECONDS_PER_MILLISECOND)
    return Measurement(float(np.mean(samples)), min(samples), max(samples))


def _union_duration(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def profile(call, *, iterations: int = PROFILE_ITERATIONS) -> Resources:
    for _ in range(PROFILE_WARMUPS):
        _ready(call())
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 0
    with tempfile.TemporaryDirectory() as directory:
        with jax.profiler.trace(directory, profiler_options=options):
            for _ in range(iterations):
                _ready(call())
        traces = list(Path(directory).glob("plugins/profile/**/*.xplane.pb"))
        if len(traces) != 1:
            raise RuntimeError(f"expected one XSpace trace, found {traces}")
        data = jax.profiler.ProfileData.from_file(str(traces[0]))

    device = data.find_plane_with_name("/device:TPU:0")
    if device is None:
        raise RuntimeError("TPU device plane is absent from the XSpace trace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        stats = dict(device.stats)
    counters = {}
    modules = []
    operations = []
    for line in device.lines:
        if line.name == "XLA Modules":
            modules.extend(line.events)
        elif line.name == "XLA Ops":
            operations.extend(line.events)
        elif line.name.startswith("counters_"):
            for event in line.events:
                value = dict(event.stats).get("counter_value")
                if value is not None:
                    counters[event.name] = counters.get(event.name, 0) + int(value)
    if not modules or len(modules) % iterations:
        raise RuntimeError(
            f"cannot group {len(modules)} XLA modules into {iterations} calls"
        )
    per_call = len(modules) // iterations
    groups = [modules[i : i + per_call] for i in range(0, len(modules), per_call)]
    active_ns = _union_duration((event.start_ns, event.end_ns) for event in modules)
    active_per_call = [sum(event.duration_ns for event in group) for group in groups]
    envelope_per_call = [group[-1].end_ns - group[0].start_ns for group in groups]
    cycles = active_ns * TPU_V6E_CLOCK_HZ / NANOSECONDS_PER_SECOND
    mxu_busy = MXU_BUSY_ONE_CYCLE_WEIGHT * counters.get(
        TC_PREFIX + "MXU_BUSY_1", 0
    ) + counters.get(TC_PREFIX + "MXU_BUSY_2", 0)
    vector_issued = sum(
        counters.get(TC_PREFIX + f"VECTOR_ALU_INSTRUCTION_{lane}", 0)
        for lane in range(VECTOR_COUNTER_COUNT)
    )
    hbm_bytes = HBM_COUNTER_BYTES * sum(
        value
        for name, value in counters.items()
        if HBM_READ.match(name) or HBM_WRITE.match(name)
    )
    peak = (
        float(
            stats.get(
                "peak_hbm_bw_gigabytes_per_second",
                TPU_V6E_FALLBACK_HBM_GBPS,
            )
        )
        * NANOSECONDS_PER_SECOND
    )
    active_s = active_ns / NANOSECONDS_PER_SECOND
    op_durations = {}
    for event in operations:
        name = event.name.split(";", 1)[0]
        op_durations[name] = op_durations.get(name, 0) + event.duration_ns
    operations = tuple(
        (name, duration / iterations / NANOSECONDS_PER_MILLISECOND)
        for name, duration in sorted(
            op_durations.items(), key=lambda item: item[1], reverse=True
        )
    )
    return Resources(
        device_active_ms=float(np.median(active_per_call))
        / NANOSECONDS_PER_MILLISECOND,
        device_envelope_ms=float(np.median(envelope_per_call))
        / NANOSECONDS_PER_MILLISECOND,
        mxu_busy_pct=100.0 * mxu_busy / cycles if cycles else math.nan,
        vector_issue_pct=(
            100.0 * vector_issued / (VECTOR_COUNTER_COUNT * cycles)
            if cycles
            else math.nan
        ),
        hbm_gb_s=(
            hbm_bytes / active_s / NANOSECONDS_PER_SECOND if active_s else math.nan
        ),
        hbm_pct=100.0 * hbm_bytes / (peak * active_s) if active_s else math.nan,
        hbm_bytes_per_call=hbm_bytes / iterations,
        top_ops=operations[:5],
        operations=operations,
    )


@dataclass(frozen=True)
class Workload:
    pattern: str
    query_lengths: tuple[int, ...]
    distribution: tuple[int, int, int]

    @property
    def batch(self) -> int:
        return len(self.query_lengths)

    @property
    def tokens(self) -> int:
        return sum(self.query_lengths)

    @property
    def maximum_query(self) -> int:
        return max(self.query_lengths)


@dataclass(frozen=True)
class Benchmark:
    pattern: str
    batch: int
    tokens: int
    sequence: int
    mean_ms: float
    minimum_ms: float
    maximum_ms: float


def _workload(pattern: str, batch: int) -> Workload:
    if pattern == "decode":
        return Workload(pattern, (1,) * batch, (batch, batch, batch))
    if pattern == "prefill":
        return Workload(pattern, (PREFILL_QUERY_TOKENS,) * batch, (0, 0, batch))
    if pattern == "ragged":
        return Workload(
            pattern,
            tuple(
                RAGGED_QUERY_LENGTHS[index % len(RAGGED_QUERY_LENGTHS)]
                for index in range(batch)
            ),
            (1, 1, batch),
        )
    raise ValueError(f"unknown workload: {pattern}")


def _initial_state(batch: int, width: int) -> np.ndarray:
    state = np.zeros((batch, CSA_STATE_SLOTS, 2, width), np.float32)
    state[:, :, 1] = -np.inf
    return state


def _unit_e8m0() -> np.uint8:
    return np.uint8(-ml_dtypes.finfo(ml_dtypes.float8_e8m0fnu).minexp)


def _main_caches(rng: np.random.Generator, pages: int):
    records = pages * CSA_DEFAULT_PAGE_SIZE
    nope = np.zeros((records, CSA_ATTENTION_DIM), np.uint8)
    values = (
        INPUT_CACHE_SCALE * rng.standard_normal((records, CSA_MAIN_NOPE_DIM))
    ).astype(ml_dtypes.float8_e4m3fn)
    nope[:, : values.shape[1]] = values.view(np.uint8)
    nope[
        :,
        CSA_MAIN_NOPE_DIM : CSA_MAIN_NOPE_DIM + CSA_MAIN_NOPE_SCALE_COUNT,
    ] = _unit_e8m0()
    rope = np.asarray(
        jnp.asarray(rng.standard_normal((records, CSA_ROPE_DIM)), jnp.bfloat16)
    ).view(np.uint16)
    rope_bytes = np.concatenate(
        ((rope >> 8).astype(np.uint8), (rope & 0xFF).astype(np.uint8)), axis=-1
    )
    return (
        jnp.asarray(
            nope.reshape(
                pages,
                CSA_DEFAULT_PAGE_SIZE,
                CSA_CACHE_PACKING,
                TPU_V6E.vector_lanes,
            )
        ),
        jnp.asarray(
            rope_bytes.reshape(
                pages,
                CSA_DEFAULT_PAGE_SIZE // CSA_CACHE_PACKING,
                CSA_CACHE_PACKING,
                TPU_V6E.vector_lanes,
            )
        ),
    )


def _ordered_index_cache(pages: int, pages_per_request: int):
    """Encode row ids as exact FP8 keys for stable Top-K workloads."""
    records = pages * CSA_DEFAULT_PAGE_SIZE
    capacity = pages_per_request * CSA_DEFAULT_PAGE_SIZE
    logical_rows = np.arange(records, dtype=np.int32) % capacity
    logical = np.zeros((records, CSA_INDEX_DIM), np.float32)
    encoded_bits = min((capacity - 1).bit_length(), CSA_INDEX_DIM)
    for bit in range(encoded_bits):
        logical[:, bit] = ((logical_rows >> bit) & 1).astype(np.float32)
    fp8_records = np.zeros((records, CSA_INDEX_RECORD_BYTES), np.uint8)
    fp8_records[:, :CSA_INDEX_DIM] = logical.astype(ml_dtypes.float8_e4m3fn).view(
        np.uint8
    )
    fp8_records[:, CSA_INDEX_DIM] = _unit_e8m0()
    return jnp.asarray(
        fp8_records.reshape(
            pages,
            CSA_DEFAULT_PAGE_SIZE // CSA_CACHE_PACKING,
            CSA_CACHE_PACKING,
            CSA_INDEX_RECORD_BYTES,
        )
    )


def _metadata(workload: Workload, sequence: int):
    q_lens = np.asarray(workload.query_lengths, np.int32)
    cu_q = np.asarray((0, *np.cumsum(q_lens)), np.int32)
    seq_ids = np.repeat(np.arange(workload.batch, dtype=np.int32), q_lens)
    prefixes = sequence - q_lens
    positions = np.concatenate(
        [np.arange(prefix, sequence, dtype=np.int32) for prefix in prefixes]
    )

    compressed_pages_per_request = (
        sequence // CSA_COMPRESSION_RATIO + CSA_DEFAULT_PAGE_SIZE - 1
    ) // CSA_DEFAULT_PAGE_SIZE
    compressed_pages = np.arange(
        workload.batch * compressed_pages_per_request, dtype=np.int32
    ).reshape(workload.batch, compressed_pages_per_request)

    raw_pages_per_request = (
        sequence + CSA_DEFAULT_PAGE_SIZE - 1
    ) // CSA_DEFAULT_PAGE_SIZE
    raw_page_ids = np.arange(raw_pages_per_request, dtype=np.int32)[None, :]
    window_pages = (
        WINDOW_CACHE_PAGES_PER_REQUEST
        * np.arange(workload.batch, dtype=np.int32)[:, None]
    )
    window_pages = window_pages + np.mod(raw_page_ids, WINDOW_CACHE_PAGES_PER_REQUEST)
    return (
        cu_q,
        seq_ids,
        positions,
        compressed_pages,
        window_pages,
        np.asarray(workload.distribution, np.int32),
    )


def _operands(workload: Workload, sequence: int, seed: int):
    rng = np.random.default_rng(seed)
    (
        cu_q,
        seq_ids,
        positions,
        compressed_pages,
        window_pages,
        distribution,
    ) = _metadata(workload, sequence)
    x = jnp.asarray(
        rng.standard_normal((workload.tokens, CSA_HIDDEN_DIM)), jnp.bfloat16
    )
    weight_np = INPUT_WEIGHT_SCALE * rng.standard_normal(
        (CSA_HIDDEN_DIM, CSA_DUAL_PROJECTION_DIM)
    )
    weight = jnp.asarray(weight_np, jnp.bfloat16)
    main_ape = jnp.asarray(
        INPUT_APE_SCALE
        * rng.standard_normal(
            (CSA_COMPRESSION_RATIO, CSA_MAIN_PROJECTED_DIM), dtype=np.float32
        )
    )
    index_ape = jnp.asarray(
        INPUT_APE_SCALE
        * rng.standard_normal(
            (CSA_COMPRESSION_RATIO, CSA_INDEX_PROJECTED_DIM), dtype=np.float32
        )
    )
    main_norm = jnp.ones((CSA_ATTENTION_DIM,), jnp.float32)
    index_norm = jnp.ones((CSA_INDEX_DIM,), jnp.float32)
    angles = rng.standard_normal(
        (sequence + 1, CSA_ROPE_FREQUENCY_DIM), dtype=np.float32
    )
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)

    main_nope, main_rope = _main_caches(rng, compressed_pages.size)
    index_cache = _ordered_index_cache(
        compressed_pages.size,
        compressed_pages.shape[1],
    )
    main_state = _initial_state(workload.batch, CSA_MAIN_PROJECTED_DIM)
    index_state = _initial_state(workload.batch, CSA_INDEX_PROJECTED_DIM)
    window_cache = jnp.asarray(
        rng.standard_normal(
            (
                WINDOW_CACHE_PAGES_PER_REQUEST * workload.batch,
                CSA_DEFAULT_PAGE_SIZE,
                CSA_ATTENTION_DIM,
            )
        ),
        jnp.bfloat16,
    )

    logical_index_q = np.zeros(
        (workload.tokens, CSA_INDEX_HEADS, CSA_INDEX_DIM), np.float32
    )
    index_weights_np = np.zeros((workload.tokens, CSA_INDEX_HEADS), np.float32)
    encoded_bits = min(
        (compressed_pages.shape[1] * CSA_DEFAULT_PAGE_SIZE - 1).bit_length(),
        CSA_INDEX_HEADS,
        CSA_INDEX_DIM,
    )
    for bit in range(encoded_bits):
        logical_index_q[:, bit, bit] = 1
        index_weights_np[:, bit] = 1 << bit
    index_weights = jnp.asarray(index_weights_np)
    attention_q = jnp.asarray(
        rng.standard_normal((workload.tokens, CSA_ATTENTION_HEADS, CSA_ATTENTION_DIM)),
        jnp.bfloat16,
    )
    new_kv = jnp.asarray(
        rng.standard_normal((workload.tokens, CSA_ATTENTION_DIM)), jnp.bfloat16
    )
    sink = jnp.asarray(rng.standard_normal((CSA_ATTENTION_HEADS,)), jnp.float32)

    return [
        x,
        weight,
        main_ape,
        index_ape,
        main_norm,
        index_norm,
        jnp.asarray(cos),
        jnp.asarray(sin),
        jnp.asarray(positions),
        jnp.asarray(cu_q),
        jnp.asarray(seq_ids),
        jnp.asarray(compressed_pages),
        jnp.asarray(window_pages),
        jnp.full((workload.batch,), sequence, jnp.int32),
        jnp.asarray(distribution),
        jnp.asarray(logical_index_q, jnp.bfloat16),
        index_weights,
        attention_q,
        new_kv,
        sink,
        jnp.asarray(main_state),
        jnp.asarray(index_state),
        main_nope,
        main_rope,
        index_cache,
        window_cache,
    ]


class _StatefulCall:
    def __init__(self, function, operands, update):
        self.function = function
        self.operands = operands
        self.update = update
        self.index = 0

    def __call__(self):
        values = self.operands[self.index]
        result = self.function(*values)
        self.update(values, result)
        self.index = (self.index + 1) % len(self.operands)
        return result


def _update_operands(values, result):
    values[20], values[21] = result[2], result[3]
    values[22], values[23], values[24], values[25] = result[4:8]


def _build(workload: Workload, sequence: int):
    operands = [
        _operands(workload, sequence, BENCHMARK_SEED + ring)
        for ring in range(ROTATING_BUFFERS)
    ]
    return _StatefulCall(
        build_csa_step(
            workload.query_lengths,
            query_start_slots=tuple(
                int(value)
                for value in (sequence - np.asarray(workload.query_lengths))
                % CSA_COMPRESSION_RATIO
            ),
            uniform_prefill=workload.pattern == "prefill",
        ),
        operands,
        _update_operands,
    )


def _report(workload: Workload, sequence: int, run_profile: bool):
    call = _build(workload, sequence)
    timing = measure(call)
    print(
        f"{workload.pattern} B={workload.batch} T={workload.tokens} S={sequence}: "
        f"mean={timing.mean_ms:.4f} ms "
        f"range={timing.minimum_ms:.4f}..{timing.maximum_ms:.4f} ms"
    )
    if run_profile:
        usage = profile(call)
        print(
            f"  device={usage.device_active_ms:.4f} ms "
            f"MXU={usage.mxu_busy_pct:.1f}% VPU={usage.vector_issue_pct:.1f}% "
            f"HBM={usage.hbm_gb_s:.1f} GB/s ({usage.hbm_pct:.1f}%) "
            f"traffic={usage.hbm_bytes_per_call / 1e6:.3f} MB "
            f"top_ops={usage.top_ops}"
        )
    return Benchmark(
        workload.pattern,
        workload.batch,
        workload.tokens,
        sequence,
        timing.mean_ms,
        timing.minimum_ms,
        timing.maximum_ms,
    )


def _print_summary(results: list[Benchmark]) -> None:
    print("\nAll cells are mean latency in ms:")
    for pattern in dict.fromkeys(result.pattern for result in results):
        pattern_results = [result for result in results if result.pattern == pattern]
        sequences = tuple(dict.fromkeys(result.sequence for result in pattern_results))
        print(f"{pattern}: S=" + " ".join(map(str, sequences)))
        for batch in dict.fromkeys(result.batch for result in pattern_results):
            row = [result for result in pattern_results if result.batch == batch]
            print(
                f"  B={batch}: " + " ".join(f"{result.mean_ms:.4f}" for result in row)
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", default="decode,prefill,ragged")
    parser.add_argument("--batches", default="1,4,8,16,32")
    parser.add_argument("--sequences", default="512,1024,2048,4096,8192")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError("benchmark requires a physical TPU")
    print(f"device={jax.devices()[0]} JAX={jax.__version__}")
    results = []
    for pattern in args.patterns.split(","):
        for batch in map(int, args.batches.split(",")):
            workload = _workload(pattern, batch)
            for sequence in map(int, args.sequences.split(",")):
                sequence_alignment = CSA_DEFAULT_PAGE_SIZE * CSA_COMPRESSION_RATIO
                if sequence < workload.maximum_query or sequence % sequence_alignment:
                    raise ValueError(
                        "sequence must cover q and align to a compressed page"
                    )
                results.append(
                    _report(
                        workload,
                        sequence,
                        args.profile,
                    )
                )
                jax.clear_caches()
    _print_summary(results)


if __name__ == "__main__":
    main()
