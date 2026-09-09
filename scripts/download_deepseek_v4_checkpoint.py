"""Download and verify an unmodified official checkpoint on a durable data disk.

Run only after the low-bit correctness gate. Downloads the *entire* pinned
repository, not just safetensors; verifies SHA256 for LFS files and Git blob
SHA1 for ordinary files. Hugging Face resumes incomplete downloads. The
manifest is written outside the official snapshot directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "deepseek-ai/DeepSeek-V4-Flash"
REVISION = "60d8d70770c6776ff598c94bb586a859a38244f1"


def write_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.data_root.resolve()
    if not root.is_mount() or root == Path("/"):
        raise SystemExit(
            "--data-root must be a separately mounted data disk, not the boot filesystem"
        )
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be in [1,16]")
    destination = root / "models" / "deepseek-ai--DeepSeek-V4-Flash" / REVISION
    destination.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError("resolved revision does not match the pinned checkpoint")
    siblings = info.siblings
    total_size = sum(item.size for item in siblings)
    stat = os.statvfs(root)
    available = stat.f_bavail * stat.f_frsize
    already_present = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if available + already_present < total_size + 10 * 1024**3:
        raise RuntimeError("not enough space for the checkpoint and a 10 GiB safety margin")
    manifest_path = destination.parent / f"{REVISION}.manifest.json"
    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "destination": str(destination),
        "file_count": len(siblings),
        "total_bytes": total_size,
        "complete": False,
        "verified_files": [],
    }
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "phase": "download",
                **{
                    k: manifest[k] for k in ("revision", "file_count", "total_bytes", "destination")
                },
            }
        ),
        flush=True,
    )
    started = time.monotonic()
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=destination,
        max_workers=args.workers,
    )
    print(
        json.dumps({"phase": "verify", "download_seconds": time.monotonic() - started}),
        flush=True,
    )

    def verify(item):
        path = (destination / item.rfilename).resolve()
        if not path.is_relative_to(destination) or not path.is_file():
            raise RuntimeError(f"invalid or missing snapshot file: {item.rfilename}")
        size = path.stat().st_size
        if size != item.size:
            raise RuntimeError(f"wrong size: {item.rfilename}: {size} != {item.size}")
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                sha256.update(chunk)
                if item.lfs is None:
                    git_blob.update(chunk)
        expected = item.lfs.sha256 if item.lfs is not None else item.blob_id
        actual = sha256.hexdigest() if item.lfs is not None else git_blob.hexdigest()
        if actual != expected:
            raise RuntimeError(f"published digest mismatch: {item.rfilename}")
        record = {
            "name": item.rfilename,
            "size": size,
            "sha256": sha256.hexdigest(),
            "published_digest": expected,
        }
        print(json.dumps({"verified": item.rfilename, "bytes": size}), flush=True)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        for record in pool.map(verify, siblings):
            manifest["verified_files"].append(record)
            write_manifest(manifest_path, manifest)
    index = json.loads((destination / "model.safetensors.index.json").read_text())
    required = set(index["weight_map"].values())
    verified = {item["name"] for item in manifest["verified_files"]}
    if not required.issubset(verified):
        raise RuntimeError("checkpoint index references unverified shards")
    manifest["complete"] = True
    manifest["safetensors_shards"] = len(required)
    manifest["total_seconds"] = time.monotonic() - started
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "phase": "complete",
                "manifest": str(manifest_path),
                "shards": len(required),
                "total_seconds": manifest["total_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
