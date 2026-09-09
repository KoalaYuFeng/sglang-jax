"""Copy only the three original NPZ members used by isolated MoE tests.

Uses zipfile, not numerical decoding/re-encoding. Every .npy member is copied
byte-for-byte and hashed against the original capture. The full schema remains
unchanged. The manifest explicitly identifies this as a subset, not a full
layer-state capture.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def prepare(prefix, output):
    output.mkdir(parents=True, exist_ok=False)
    digest = lambda value: hashlib.sha256(value).hexdigest()
    manifest = {"scope": __doc__, "original_capture": str(prefix), "members": {}}
    with prefix.with_suffix(".npz").open("rb") as stream:
        manifest["original_npz_sha256"] = hashlib.file_digest(
            stream, "sha256"
        ).hexdigest()
    with (
        zipfile.ZipFile(prefix.with_suffix(".npz")) as source,
        zipfile.ZipFile(
            output / "capture.npz", "x", compression=zipfile.ZIP_DEFLATED
        ) as target,
    ):
        for key in ("ffn.norm", "expert_ids", "routing_weights"):
            member = key + ".npy"
            raw = source.read(member)
            target.writestr(member, raw)
            manifest["members"][member] = {"sha256": digest(raw), "bytes": len(raw)}
    shutil.copyfile(prefix.with_suffix(".json"), output / "capture.json")
    manifest["schema_sha256"] = digest((output / "capture.json").read_bytes())
    with zipfile.ZipFile(output / "capture.npz") as target:
        for member, expected in manifest["members"].items():
            assert digest(target.read(member)) == expected["sha256"]
    manifest["subset_npz_sha256"] = digest((output / "capture.npz").read_bytes())
    (output / "capture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.capture_prefix, args.output), indent=2))
