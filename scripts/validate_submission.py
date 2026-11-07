#!/usr/bin/env python3
"""Validate a submission CSV against the format in CLAUDE.md.

Checks: exact columns, unique filament_id, id -> image mapping against the
test folder, every RLE decodes to 2048x2048, no empty masks, no overlapping
masks within an image, and a clean reload. Reports test images with zero
rows (that's allowed by spec, but worth surfacing).

Usage: python scripts/validate_submission.py path/to/submission.csv [--height 2048] [--width 2048]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import io as skio
from solar_k import masks as skmasks

REQUIRED_COLUMNS = ["filament_id", "segmentation_rle"]


def validate(csv_path: Path, height: int = 2048, width: int = 2048, image_dir: Path | None = None) -> list[str]:
    errors: list[str] = []

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as e:  # noqa: BLE001 -- user-facing validator must not crash on any malformed input
        return [f"could not read CSV: {e}"]

    if list(df.columns) != REQUIRED_COLUMNS:
        errors.append(f"columns are {list(df.columns)}, expected exactly {REQUIRED_COLUMNS}")
        return errors  # nothing else is safe to check

    dup_ids = df["filament_id"][df["filament_id"].duplicated()].unique()
    if len(dup_ids) > 0:
        errors.append(f"{len(dup_ids)} duplicate filament_id(s), e.g. {list(dup_ids[:5])}")

    image_dir = image_dir or skio.test_images_dir()
    test_stems = {p.stem for p in skio.iter_image_paths(image_dir)}

    row_stems = df["filament_id"].str.rsplit("_", n=1).str[0]
    bad_mapping = sorted(set(row_stems) - test_stems)
    if bad_mapping:
        errors.append(f"{len(bad_mapping)} filament_id(s) don't map to a test image, e.g. {bad_mapping[:5]}")

    # One accumulator array at a time, not one per image, so this stays well
    # under memory on an 8 GB Mac even for the full test set.
    df = df.assign(_stem=row_stems)
    overlap_stems: list[str] = []
    for stem, group in df.groupby("_stem"):
        if stem not in test_stems:
            continue
        acc = np.zeros((height, width), dtype=np.uint16)
        for counts in group["segmentation_rle"]:
            try:
                mask = skmasks.decode_rle(str(counts), height, width)
            except Exception as e:  # noqa: BLE001 -- user-facing validator must not crash on any malformed input
                errors.append(f"filament_id for image {stem}: RLE failed to decode ({e})")
                continue
            if mask.shape != (height, width):
                errors.append(f"filament_id for image {stem}: decoded shape {mask.shape} != ({height}, {width})")
                continue
            if mask.sum() == 0:
                errors.append(f"filament_id for image {stem}: decoded mask is empty")
                continue
            acc += mask.astype(np.uint16)
        if (acc > 1).any():
            overlap_stems.append(stem)
        del acc

    if overlap_stems:
        errors.append(f"{len(overlap_stems)} image(s) have overlapping masks, e.g. {sorted(overlap_stems)[:5]}")

    zero_row_images = sorted(test_stems - set(row_stems))
    if zero_row_images:
        print(f"Note: {len(zero_row_images)} test image(s) have zero rows (allowed): {zero_row_images[:5]}...")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--image-dir", type=Path, default=None)
    args = parser.parse_args()

    errors = validate(args.csv_path, args.height, args.width, args.image_dir)
    if errors:
        print("Submission validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"{args.csv_path} is a valid submission.")


if __name__ == "__main__":
    main()
