"""CSA tensor-parallel contract for shared-KV attention."""

from __future__ import annotations

from sgl_jax.srt.kernels.csa.tune import CSA_ATTENTION_HEADS, TPU_V6E


def validate_csa_mesh(mesh) -> int:
    """Return TP size; request/cache data parallelism is not implemented yet.

    Attention queries and sinks are head-sharded. The compressor, Lightning
    Indexer (whose score reduces over all index heads), recurrent state and
    head-independent packed KV caches are replicated over the tensor axis.
    """
    if mesh is None:
        raise ValueError("CSA requires a mesh")
    if mesh.size == 1:
        return 1
    if "tensor" not in mesh.axis_names or any(
        size != 1 for name, size in mesh.shape.items() if name != "tensor"
    ):
        raise ValueError("CSA supports tensor parallelism only; all non-tensor axes must be 1")
    tp_size = mesh.shape["tensor"]
    if CSA_ATTENTION_HEADS % tp_size or (CSA_ATTENTION_HEADS // tp_size) % TPU_V6E.sublanes:
        raise ValueError("CSA tensor size must divide 64 heads into multiples of 8 local heads")
    return tp_size
