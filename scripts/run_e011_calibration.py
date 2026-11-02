#!/usr/bin/env python3
"""E011 calibration diagnostic (CODING_SPEC_05_ENSEMBLE.md section 3, report
only -- nothing is tuned on it).

Compares the scorer's input features (component prob_max / prob_mean / area /
elongation) between single-member OOF maps (E004, E007) and their two-member
mean, on the same E011 images (model folds 0-1), at one fixed cfg1: the cfg1
P5 chose most often across E011_mean's CV folds (ties -> first in grid
order). Only components that pass min_area are compared (smaller ones never
reach the scorer's keep decision). Two-sample KS statistic per feature.

Also writes the E011 gate table (E004 vs E007 vs mean, per CV fold and per
model fold) from the three run_p5.py outputs.

Usage: python scripts/run_e011_calibration.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import cv_pipeline as cvp
from solar_k import ensemble as skens
from solar_k import io as skio
from solar_k import postprocess as skpp

INTERIM = skio.repo_root() / "data" / "interim"
FEATURES = ["prob_max", "prob_mean", "area", "elongation"]
VARIANTS = {"E011_E004": ["E004"], "E011_E007": ["E007"], "E011_mean": ["E004", "E007"]}


def main() -> None:
    summaries = {k: json.loads((INTERIM / f"{k}_p5_summary.json").read_text()) for k in VARIANTS}
    mean_cfgs = [cvp.cfg_key(skpp.PostprocessConfig.from_dict(f["cfg1"])) for f in summaries["E011_mean"]["per_fold"]]
    counts = collections.Counter(mean_cfgs)
    grid_keys = [cvp.cfg_key(c) for c in cvp.CFG1_GRID]
    top = max(counts.values())
    key = next(k for k in grid_keys if counts.get(k, 0) == top)
    cfg = cvp.CFG1_GRID[grid_keys.index(key)]
    print(f"fixed cfg1 for the diagnostic: {key} (chosen in {top}/{len(mean_cfgs)} E011_mean CV folds)")

    fold_by_stem = skens.model_fold_by_stem()
    stems = skens.available_oof_stems(["E004", "E007"], fold_by_stem, [0, 1])
    feats = {}
    for name, members in VARIANTS.items():
        runner = cvp.Runner(members, 3)
        try:
            tables = cvp.component_tables(runner, stems, cfg)
        finally:
            runner.close()
        df = pd.concat([tables[s].features_df for s in stems], ignore_index=True)
        feats[name] = df[df["area"] >= cvp.MIN_AREA]
        print(f"{name}: {len(feats[name])} components >= min_area")

    ks = {}
    for single in ("E011_E004", "E011_E007"):
        ks[single + "_vs_mean"] = {}
        for f in FEATURES:
            a, b = feats[single][f].dropna(), feats["E011_mean"][f].dropna()
            r = ks_2samp(a, b)
            ks[single + "_vs_mean"][f] = {"ks": round(float(r.statistic), 4), "p": float(r.pvalue),
                                           "median_single": round(float(a.median()), 4), "median_mean": round(float(b.median()), 4)}
    n_comp = {k: int(len(v)) for k, v in feats.items()}

    gate = {}
    for k, s in summaries.items():
        gate[k] = {"cv_pq_mean": s["cv_pq_mean"], "cv_pq_std": s["cv_pq_std"], "pooled_pq": s["pooled_pq"], "pooled_sq": s["pooled_sq"],
                   "pooled_rq": s["pooled_rq"], "pooled_tp": s["pooled_tp"], "pooled_fp": s["pooled_fp"], "pooled_fn": s["pooled_fn"],
                   "per_model_fold_pq": s.get("per_model_fold_pq"), "per_cv_fold_pq": [f["pq"] for f in s["per_fold"]],
                   "per_cv_fold_cfg1": [cvp.cfg_key(skpp.PostprocessConfig.from_dict(f["cfg1"])) for f in s["per_fold"]],
                   "vs_E004": s.get("vs_reference")}
    out = {"cfg1_for_diagnostic": key, "n_images": len(stems), "n_components": n_comp, "ks": ks, "gate_table": gate,
           "max_ks_prob_features": max(ks[v][f]["ks"] for v in ks for f in ("prob_max", "prob_mean"))}
    (INTERIM / "E011_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
