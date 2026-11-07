#!/usr/bin/env python3
"""CODING_SPEC_08_FINAL_PACKAGE.md 2.2/2.3: fetch the 15 EMA checkpoint files
from the GitHub Release asset bundle into weights/, verifying each against
weights/MANIFEST.json's sha256 before accepting it. This is the only
supported way to populate weights/ -- they are gitignored and never
committed (CLAUDE.md: never commit checkpoints)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_k import io as skio

DEFAULT_RELEASE = "https://github.com/nicky5851/solar-k/releases/download/v1.0-final"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-url", default=DEFAULT_RELEASE,
                     help="base URL whose <file>/<basename> gives each asset "
                          "(default: this repo's v1.0-final GitHub Release)")
    ap.add_argument("--weights-dir", type=Path, default=None, help="default: weights/")
    ap.add_argument("--force", action="store_true", help="re-download even if a verified file already exists")
    args = ap.parse_args()

    repo = skio.repo_root()
    weights_dir = args.weights_dir or (repo / "weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((repo / "weights" / "MANIFEST.json").read_text())

    for entry in manifest:
        dst = repo / entry["file"]
        if dst.exists() and not args.force and sha256_of(dst) == entry["sha256"]:
            print(f"ok (cached): {entry['file']}")
            continue
        url = f"{args.release_url}/{dst.name}"
        print(f"downloading {url} -> {dst}")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dst.write_bytes(resp.content)
        got = sha256_of(dst)
        if got != entry["sha256"]:
            dst.unlink()
            raise SystemExit(f"checksum mismatch for {entry['file']}: expected {entry['sha256']}, got {got}")
        print(f"ok (verified): {entry['file']} ({entry['bytes']} bytes)")

    print(f"all {len(manifest)} checkpoints present and verified in {weights_dir}")


if __name__ == "__main__":
    main()
