#!/usr/bin/env python3
"""Build `data/splits/folds_v1.csv`: one row per physical train image.

Usage: python scripts/make_splits.py [--n-folds 5] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import ids as skids
from solar_k import io as skio
from solar_k import splits as sksplits


def build_image_table() -> pd.DataFrame:
    """One row per physical train image: file_stem, site, date, n_annotators."""
    ann = skio.load_coco_annotations()

    per_file_annotators: dict[str, set[str]] = {}
    for img in ann["images"]:
        annotator, stem = skids.parse_image_id(img["id"])
        per_file_annotators.setdefault(stem, set()).add(annotator)

    rows = []
    for stem, annotators in per_file_annotators.items():
        parsed = skids.parse_stem(stem)
        if parsed is None:
            continue
        rows.append(
            {
                "file_stem": stem,
                "site": parsed.site,
                "date": parsed.timestamp.strftime("%Y-%m-%d"),
                "n_annotators": len(annotators),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=skio.data_root() / "splits" / "folds_v1.csv"
    )
    args = parser.parse_args()

    images = build_image_table()
    folded = sksplits.assign_folds(images, n_folds=args.n_folds, seed=args.seed)

    expected_stems = {p.stem for p in skio.iter_train_images()}
    issues = sksplits.validate_folds(folded, expected_stems=expected_stems)
    if issues:
        print("Split validation FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    folded[["file_stem", "fold", "site", "date", "n_annotators"]].sort_values("file_stem").to_csv(
        args.out, index=False
    )
    print(f"Wrote {len(folded)} rows to {args.out}")
    print(sksplits.fold_report(folded))


if __name__ == "__main__":
    main()
