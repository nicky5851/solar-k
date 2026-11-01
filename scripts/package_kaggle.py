#!/usr/bin/env python3
"""Zip src/, scripts/, configs/, requirements.txt into outputs/kaggle_pkg/
for upload as a private Kaggle dataset (CODING_SPEC_02_BASELINE.md, Stage C).

Never put credentials in code -- the Kaggle upload and training run are
done manually by Nicky; see reports/02_baseline_report.md for the steps.

Usage: python scripts/package_kaggle.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_k import io as skio

REPO_ROOT = skio.repo_root()
INCLUDE = ["src", "scripts", "configs", "requirements.txt", "data/splits/folds_v1.csv"]
EXCLUDE_SUFFIXES = {".pyc"}
EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".DS_Store"}


def iter_files():
    for rel in INCLUDE:
        path = REPO_ROOT / rel
        if path.is_file():
            yield path, path.relative_to(REPO_ROOT)
            continue
        for p in sorted(path.rglob("*")):
            if p.is_dir():
                continue
            if p.suffix in EXCLUDE_SUFFIXES or any(part in EXCLUDE_DIR_NAMES for part in p.parts):
                continue
            yield p, p.relative_to(REPO_ROOT)


def main() -> None:
    out_dir = REPO_ROOT / "outputs" / "kaggle_pkg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip = out_dir / "solar_k_package.zip"
    n = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, rel in iter_files():
            zf.write(path, arcname=str(rel))
            n += 1
    print(f"Wrote {n} files to {out_zip}")
    print("Upload this zip as a new private Kaggle dataset, then attach it to notebooks/kaggle_train.ipynb.")


if __name__ == "__main__":
    main()
