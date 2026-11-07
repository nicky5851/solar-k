#!/usr/bin/env python3
"""CODING_SPEC_06_RECALL_AND_CONNECTIVITY.md E024: one command from saved
probability maps -> submission CSV -> validate_submission.py.

Fits the final post-processing chain (cfg1, shape gate, scorer, regrow) on
all 707 OOF images of the given members (no CV split -- this is the final
build, not an evaluation), then applies it to the mean of every member's
test probability maps and writes+validates outputs/submissions/<experiment>.csv.

Does not submit anything. The actual logic lives in
solar_k.cv_pipeline.write_submission, shared with scripts/run_p5.py's
--write-submission flag.

Usage:
  python scripts/build_final.py --experiment E024_primary --members E007tta4s E015tta4s
  python scripts/build_final.py --experiment E024_primary --members E007 E015 --weight-grid 1.0,1.0 1.5,1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import cv_pipeline as cvp
from solar_k import ensemble as skens
from solar_k import io as skio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, help="output name; writes outputs/submissions/<experiment>.csv")
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--weight-grid", nargs="*", default=None, help="e.g. 1.0,1.0 1.5,1.0 (comma-separated per-member weights)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--postproc-out", type=Path, default=None,
                     help="CODING_SPEC_08 2.1/2.3: also write the frozen post-processing chain "
                          "(cfg1, shape gate, scorer, regrow, members/folds/weights) to this JSON path, "
                          "for scripts/predict.py to reload without refitting.")
    args = ap.parse_args()

    weight_grid = None
    if args.weight_grid:
        weight_grid = [tuple(float(x) for x in w.split(",")) for w in args.weight_grid]
        for w in weight_grid:
            if len(w) != len(args.members):
                raise SystemExit(f"weight vector {w} has {len(w)} entries for {len(args.members)} members")

    import pandas as pd

    folds_csv = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    fold_by_stem = dict(zip(folds_csv["file_stem"], folds_csv["fold"].astype(int)))
    stems = skens.available_oof_stems(args.members, fold_by_stem)
    if not stems:
        raise SystemExit("no OOF images available for every member")

    result = cvp.write_submission(args.experiment, args.members, stems, args.workers, weight_grid, postproc_out=args.postproc_out)
    print(f"{args.experiment}: {result['n_instances']} instances -> {result['path']}")


if __name__ == "__main__":
    main()
