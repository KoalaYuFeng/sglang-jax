"""Read the official V4 checkpoint without dtype coercion or dequantization.

Uses safetensors headers and read-only mmap slices so an expert/output shard
can be loaded without first allocating a full layer on the host or device.
Unsupported dtypes and malformed shapes fail closed instead of defaulting to
FP32. This is a format reader, not a full serving-model registration.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np

_DTYPES = {
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "F8_E4M3": np.dtype("u1"),
    "F8_E8M0": np.dtype("u1"),
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F32": np.dtype("<f4"),
    "I32": np.dtype("<i4"),
    "I64": np.dtype("<i8"),
}


@dataclass(frozen=True)
class HostLinearWeight:
    data: np.ndarray
    scales: np.ndarray | None
    weight_format: str
    logical_shape: tuple[int, int]

    @property
    def nbytes(self):
        return self.data.nbytes + (0 if self.scales is None else self.scales.nbytes)


class DeepSeekV4Checkpoint:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.config = json.loads((self.directory / "config.json").read_text())
        if self.config.get("model_type") != "deepseek_v4":
            raise ValueError("expected an official deepseek_v4 checkpoint")
        index = json.loads((self.directory / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers = {}

    def _entry(self, name):
        filename = self.weight_map[name]
        path = (self.directory / filename).resolve()
        if not path.is_relative_to(self.directory) or path.suffix != ".safetensors":
            raise ValueError(f"checkpoint path escapes model directory: {filename}")
        if path not in self._headers:
            with path.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"truncated safetensors header: {filename}")
                length = struct.unpack("<Q", prefix)[0]
                if not 2 <= length <= 32 * 1024 * 1024:
                    raise ValueError(f"invalid safetensors header length: {length}")
                raw = stream.read(length)
                if len(raw) != length:
                    raise ValueError(f"truncated safetensors header: {filename}")
                self._headers[path] = (json.loads(raw), 8 + length, path.stat().st_size)
        header, offset, size = self._headers[path]
        entry = header[name]
        dtype = _DTYPES.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported checkpoint dtype {entry['dtype']} for {name}")
        shape = tuple(entry["shape"])
        if any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError(f"invalid tensor shape for {name}: {shape}")
        begin, end = entry["data_offsets"]
        if (
            type(begin) is not int
            or type(end) is not int
            or begin < 0
            or end - begin != math.prod(shape) * dtype.itemsize
            or offset + end > size
        ):
            raise ValueError(f"invalid tensor data offsets for {name}")
        return path, entry, offset + begin, dtype

    def tensor_info(self, name):
        _, entry, _, _ = self._entry(name)
        return dict(entry)

    def read_tensor(self, name, selection=None):
        path, entry, offset, dtype = self._entry(name)
        mapped = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=tuple(entry["shape"]))
        # The explicit copy detaches the returned shard from the file mapping.
        result = np.array(mapped if selection is None else mapped[selection], copy=True)
        del mapped
        return result

    def load_linear(self, prefix, *, shard_index=0, shard_count=1):
        """Load an output-channel shard; preserve packed bytes and compact scales."""
        name = prefix + ".weight"
        info = self.tensor_info(name)
        shape = tuple(info["shape"])
        if len(shape) != 2 or shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("invalid linear shape or shard selection")
        n, stored_k = shape
        if n % shard_count:
            raise ValueError("output channels must divide evenly among shards")
        start, stop = n * shard_index // shard_count, n * (shard_index + 1) // shard_count
        dtype = info["dtype"]
        if dtype == "I8":
            if self.config.get("expert_dtype") != "fp4" or ".ffn.experts." not in name:
                raise ValueError(
                    "I8 is only accepted as an explicitly declared FP4 expert container"
                )
            fmt, k = "fp4", 2 * stored_k
            expected_scale_shape = (n, k // 32)
            scale_selection = (slice(start, stop), slice(None))
            if k % 32:
                raise ValueError("FP4 expert K must be divisible by 32")
        elif dtype == "F8_E4M3":
            fmt, k = "fp8", stored_k
            expected_scale_shape = ((n + 127) // 128, (k + 127) // 128)
            if shard_count > 1 and (start % 128 or stop % 128):
                raise ValueError("FP8 output shards must align to 128-channel scale blocks")
            scale_selection = (slice(start // 128, (stop + 127) // 128), slice(None))
        elif dtype == "BF16":
            data = self.read_tensor(name, (slice(start, stop), slice(None)))
            return HostLinearWeight(data, None, "bf16", (stop - start, stored_k))
        else:
            raise ValueError(f"not a supported low-bit/BF16 linear weight: {dtype}")
        scale_name = prefix + ".scale"
        scale_info = self.tensor_info(scale_name)
        if scale_info["dtype"] != "F8_E8M0" or tuple(scale_info["shape"]) != expected_scale_shape:
            raise ValueError(f"invalid {fmt} checkpoint scale metadata for {prefix}")
        data = self.read_tensor(name, (slice(start, stop), slice(None))).view(np.uint8)
        scales = self.read_tensor(scale_name, scale_selection)
        if np.any(scales == 255):
            raise ValueError(f"NaN E8M0 scale in {scale_name}")
        return HostLinearWeight(data, scales, fmt, (stop - start, k))
