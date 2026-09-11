"""Download the pinned original GSM8K test split into a new private directory."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATA_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
URL = f"https://raw.githubusercontent.com/openai/grade-school-math/{REVISION}/grade_school_math/data/test.jsonl"


def main(out):
    out.mkdir(mode=0o700, exist_ok=False)
    with urllib.request.urlopen(URL, timeout=60) as response:
        raw = response.read()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256
    # Read physical JSONL lines, not Unicode-aware str.splitlines().
    rows = [json.loads(line) for line in raw.split(b"\n") if line.strip()]
    assert len(rows) == 1319
    (out / "test.jsonl").write_bytes(raw)
    source = {
        "repository": "https://github.com/openai/grade-school-math",
        "revision": REVISION,
        "url": URL,
        "sha256": DATA_SHA256,
        "split": "original test",
    }
    (out / "dataset-source.json").write_text(json.dumps(source, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "sha256": DATA_SHA256}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
