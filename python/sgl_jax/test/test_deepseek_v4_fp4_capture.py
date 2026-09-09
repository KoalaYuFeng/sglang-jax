"""Byte-preserving fixture preparation, without JAX/NumPy dependencies."""

import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from prepare_deepseek_v4_fp4_capture import prepare  # noqa: E402


class CaptureSubsetTest(unittest.TestCase):
    def test_subset_preserves_member_bytes_and_schema_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "original"
            schema = b'{"fixture": "schema copied, not numerically re-encoded"}\n'
            prefix.with_suffix(".json").write_bytes(schema)
            members = {
                name + ".npy": bytes(range(256)) * (index + 1)
                for index, name in enumerate(("ffn.norm", "expert_ids", "routing_weights"))
            }
            with zipfile.ZipFile(prefix.with_suffix(".npz"), "x") as original:
                for name, raw in {**members, "unused_layer_state.npy": b"not needed"}.items():
                    original.writestr(name, raw)
            source_hash = hashlib.sha256(prefix.with_suffix(".npz").read_bytes()).hexdigest()
            output = root / "subset"
            manifest = prepare(prefix, output)
            self.assertEqual(manifest["original_npz_sha256"], source_hash)
            self.assertEqual((output / "capture.json").read_bytes(), schema)
            with zipfile.ZipFile(output / "capture.npz") as subset:
                self.assertEqual(set(subset.namelist()), set(members))
                for name, raw in members.items():
                    self.assertEqual(subset.read(name), raw)
                    self.assertEqual(
                        manifest["members"][name]["sha256"], hashlib.sha256(raw).hexdigest()
                    )
            with self.assertRaises(FileExistsError):
                prepare(prefix, output)
            self.assertEqual(
                hashlib.sha256(prefix.with_suffix(".npz").read_bytes()).hexdigest(), source_hash
            )


if __name__ == "__main__":
    unittest.main()
