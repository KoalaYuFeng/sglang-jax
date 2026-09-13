"""PR boundaries: no legacy import shims or test-only B1 code in serving."""

import ast
import hashlib
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "srt"
REPO = ROOT.parents[2]
REMOVED = (
    "attention",
    "collectives",
    "compressor",
    "csa",
    "dense",
    "fp8",
    "hca",
    "mhc",
    "moe",
    "moe_gmm",
    "numerics",
    "projections",
)
CANONICAL = (
    "attention",
    "collectives",
    "compressor",
    "csa",
    "hca",
    "mhc",
    "linear",
    "moe",
    "numerics",
)


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


@pytest.mark.parametrize("name", CANONICAL)
def test_canonical_model_modules_import(name):
    module = importlib.import_module(f"sgl_jax.srt.layers.deepseek_v4.{name}")
    assert module.__name__.endswith("." + name)


def test_kernel_implementations_do_not_import_model_adaptation():
    paths = [
        *ROOT.glob("kernels/low_bit/*.py"),
        ROOT / "kernels/deepseek_v4/normalization.py",
        ROOT / "kernels/deepseek_v4/projection_kernels.py",
    ]
    forbidden = (
        "sgl_jax.srt.layers",
        "sgl_jax.srt.models",
        "sgl_jax.srt.model_loader",
        "sgl_jax.srt.mem_cache",
        "sgl_jax.srt.managers",
        "sgl_jax.test",
    )
    for path in paths:
        assert not any(name.startswith(forbidden) for name in imports(path)), path


def test_all_repository_consumers_use_canonical_paths():
    obsolete = {f"sgl_jax.srt.kernels.deepseek_v4.{name}" for name in REMOVED}
    obsolete.update(
        {
            "sgl_jax.srt.layers.deepseek_v4.dense",
            "sgl_jax.srt.layers.deepseek_v4.projections",
            "sgl_jax.srt.layers.deepseek_v4.moe_gmm",
            "sgl_jax.srt.kernels.low_bit.gmm",
            "sgl_jax.srt.kernels.low_bit.fp4_tuning",
            "sgl_jax.srt.layers.attention.deepseek_v4_backend",
            "sgl_jax.srt.mem_cache.deepseek_v4_pool",
        }
    )
    for folder in ("python", "scripts", "test"):
        for path in (REPO / folder).rglob("*.py"):
            assert not obsolete.intersection(imports(path)), path


def test_removed_shims_and_b1_runtime_files_are_absent():
    for name in REMOVED:
        assert not (ROOT / f"kernels/deepseek_v4/{name}.py").exists()
    for name in ("dense", "projections", "moe_gmm"):
        assert not (ROOT / f"layers/deepseek_v4/{name}.py").exists()
    assert not (ROOT / "layers/attention/deepseek_v4_backend.py").exists()
    assert not (ROOT / "mem_cache/deepseek_v4_pool.py").exists()


def test_serving_does_not_depend_on_legacy_test_support():
    for path in ROOT.rglob("*.py"):
        # Existing unrelated runtime modules may depend on test utilities.
        # Only enforce the V4 legacy fixture boundary here.
        assert "sgl_jax.test.deepseek_v4_legacy" not in set(imports(path)), path


def test_checkpoint_packing_does_not_depend_on_device_or_serving_code():
    assert set(imports(ROOT / "model_loader/deepseek_v4_packing.py")) == {"numpy"}


def test_linear_does_not_reexport_checkpoint_packing_or_fusion():
    linear = importlib.import_module("sgl_jax.srt.layers.deepseek_v4.linear")
    for name in (
        "MERGED_PROJECTIONS",
        "pack_merged_weights",
        "unpack_merged_weights",
        "inverse_rope_fp8_wo_a",
    ):
        assert not hasattr(linear, name)


def test_moe_merge_keeps_explicit_legacy_default():
    moe = importlib.import_module("sgl_jax.srt.layers.deepseek_v4.moe")
    assert moe.moe.__kwdefaults__["backend"] == "legacy"
    for name in ("pack_routes", "gmm_fp4_experts", "grouped_fp4_experts"):
        assert callable(getattr(moe, name))


def test_hca_public_entries_are_the_existing_implementations():
    attention = importlib.import_module("sgl_jax.srt.kernels.hca.attention")
    compressor = importlib.import_module("sgl_jax.srt.kernels.hca.compressor")
    assert attention.streaming_attention_pallas is attention._streaming_attention
    assert compressor.hca_emit_selected_pallas is compressor._hca_emit_selected_pallas


def load_fingerprint_function(path, name, **namespace):
    """Load only trusted filesystem-hashing code, without serving imports."""
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace.update(__file__=str(path), Path=Path, hashlib=hashlib)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


def test_profile_and_execution_fingerprints_agree_and_cover_new_files(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    profile = importlib.import_module("analyze_deepseek_v4_native_profile")
    reference = load_fingerprint_function(
        ROOT / "model_executor/deepseek_v4_reference.py", "source_fingerprint"
    )
    provenance = importlib.import_module("deepseek_v4_source")
    assert provenance.reference_fingerprint() == reference()
    assert profile.fingerprint is provenance.framework_fingerprint
    assert "deepseek_v4_source.framework_fingerprint" in set(
        imports(REPO / "scripts/run_deepseek_v4_framework.py")
    )
    read_bytes = Path.read_bytes
    seen = set()

    def record(path):
        seen.add(path.resolve())
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record)
    profile.fingerprint()
    required = {
        *(p.resolve() for p in ROOT.glob("layers/deepseek_v4/*.py")),
        ROOT / "model_loader/deepseek_v4_packing.py",
        ROOT / "kernels/low_bit/fp8.py",
        ROOT / "kernels/low_bit/fp4.py",
        ROOT / "kernels/deepseek_v4/projection_kernels.py",
        ROOT / "configs/deepseek_v4_execution.py",
        REPO / "scripts/deepseek_v4_source.py",
    }
    assert required <= seen
    assert any(name == "compress" for _, _, name in profile.NATIVE_RANGES["compressor"])


@pytest.mark.parametrize(
    "source,expected",
    [
        ("/repo/kernels/low_bit/fp8.py:100", "fp8_gmm_metadata_padding_and_glue"),
        ("/repo/kernels/deepseek_v4/fp8.py:100", "fp8_gmm_metadata_padding_and_glue"),
        (
            "/repo/kernels/deepseek_v4/projection_kernels.py:50",
            "fp8_projection_layout_and_glue",
        ),
        ("/repo/layers/deepseek_v4/linear.py:20", "fp8_projection_layout_and_glue"),
    ],
)
def test_profile_recognizes_moved_projection_sources(monkeypatch, source, expected):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    profile = importlib.import_module("analyze_deepseek_v4_fp4_moe")
    row = {
        "source_info": source,
        "hlo_op_name": "copy.1",
        "tf_op_name": "",
        "category": "copy",
    }
    assert profile.stage(row, isolated=False) == expected


@pytest.mark.parametrize(
    "function,native_stage,detailed_stage",
    [
        (
            "pack_routes",
            "routed_fp4_experts_including_online_dequant",
            "moe_route_pack_and_gather",
        ),
        (
            "gmm_fp4_experts",
            "routed_fp4_experts_including_online_dequant",
            "moe_other_glue_including_swiglu_and_combine",
        ),
    ],
)
def test_merged_moe_profile_uses_function_ranges(
    monkeypatch, function, native_stage, detailed_stage
):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    profile = importlib.import_module("analyze_deepseek_v4_fp4_moe")
    ranges = profile.native.NATIVE_RANGES["moe"]
    line = lambda name: next(start for start, _, fn in ranges if fn == name)
    source = "\n".join(
        f"/repo/layers/deepseek_v4/moe.py:{line(name)}"
        for name in ("moe", "gmm_fp4_experts", function)
    )
    event = {"args": {"source_stack": source, "hlo_category": "copy"}}
    assert profile.native.native_stage(event, ()) == native_stage
    row = {
        "source_info": source,
        "hlo_op_name": "copy.1",
        "tf_op_name": "",
        "category": "copy",
    }
    assert profile.stage(row, isolated=False) == detailed_stage


def test_linear_norm_is_not_misclassified_as_fp8_projection(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    profile = importlib.import_module("analyze_deepseek_v4_fp4_moe")
    start = next(
        start
        for start, _, name in profile.native.NATIVE_RANGES["linear"]
        if name == "norm"
    )
    row = {
        "source_info": f"/repo/layers/deepseek_v4/linear.py:{start}",
        "hlo_op_name": "copy.1",
        "tf_op_name": "",
        "category": "copy",
    }
    assert profile.stage(row, isolated=False) == "normalization"
