#!/usr/bin/env python3
"""Per-image held-out statistics for E009 (CODING_SPEC_05_ENSEMBLE.md E010:
the paired E010-vs-E009 comparison needs an image-level bootstrap, and
E009 only saved per-fold numbers).

Re-runs exactly scripts/run_e009.py's CV loop (same E003 cfg1/shape gate,
same per-fold scorer refit, E005's per-fold c, E008's regrow), streaming one
image at a time, and writes data/interim/E009_p5_per_image.csv in
run_p5.py's per-image format. Checks every fold's PQ against
E009_cv_results.csv and fails loudly if it does not reproduce.

Usage: python scripts/run_e009_per_image.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import cv_pipeline as cvp
from solar_k import io as skio
from solar_k import metric as skmetric
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc

from run_e002_eval import build_gt_for_stem, load_annotation_index, load_prob_png
from run_e003 import build_component_cache
from run_e009 import regrow_pred

INTERIM = skio.repo_root() / "data" / "interim"


def main() -> None:
    e003 = json.loads((INTERIM / "E003_eval_summary.json").read_text())
    cfg1 = skpp.PostprocessConfig.from_dict(e003["best_cfg1"])
    me, ml = e003["shape_gate"]["min_elongation"], e003["shape_gate"]["min_length"]
    ccfg = cvp.components_cfg(cfg1)
    e005_cv = pd.read_csv(INTERIM / "E005_cv_results.csv")
    e009_cv = pd.read_csv(INTERIM / "E009_cv_results.csv")
    e008 = json.loads((INTERIM / "E008_eval_summary.json").read_text())
    t_mask, d = e008["regrow_best"]["t_mask"], e008["regrow_best"]["d"]

    folds_df = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    images_by_file, anns_by_image_id = load_annotation_index()
    train_paths = {p.stem: p for p in skio.iter_train_images()}
    day_by_stem = dict(zip(folds_df["file_stem"], folds_df["date"]))
    fold_by_stem = dict(zip(folds_df["file_stem"], folds_df["fold"].astype(int)))
    png_by_stem = {}
    for _, row in folds_df.iterrows():
        p = skio.output_root() / f"E004-f{row['fold']}" / "oof" / f"{row['file_stem']}.png"
        if p.exists():
            png_by_stem[row["file_stem"]] = p
    stems = sorted(png_by_stem)

    cache = {}
    for i, s in enumerate(stems):
        processed, disk_info = skpre.load_or_build(s, train_paths[s])
        gt = build_gt_for_stem(s, images_by_file, anns_by_image_id)
        cache[s] = build_component_cache(s, load_prob_png(png_by_stem[s]), processed, disk_info, ccfg, gt)
        del processed, gt
        if (i + 1) % 100 == 0:
            print(f"  cache {i + 1}/{len(stems)}", flush=True)
    comps = pd.concat([cache[s].features_df.assign(stem=s) for s in stems], ignore_index=True)

    rows = []
    for i, (train, held) in enumerate(cvp.cv_splits(stems, day_by_stem)):
        scorer = sksc.fit_scorer(comps[comps["stem"].isin(train)], seed=0)
        c = float(e005_cv.loc[e005_cv["cv_fold"] == i, "c"].iloc[0])
        fold_rows = []
        for s in held:
            df = cache[s].features_df
            keep = (cvp.gate_keep(df, me, ml) & (sksc.keep_score(scorer, df) > c)) if len(df) else np.zeros(0, dtype=bool)
            probs = load_prob_png(png_by_stem[s])
            pred = regrow_pred(probs, cache[s], keep, t_mask, d)
            gt = build_gt_for_stem(s, images_by_file, anns_by_image_id)
            ms = []
            for g in gt.values():
                if g.shape[0] == 0 or pred.shape[0] == 0:
                    ms.append(skmetric.match_scores(np.zeros((0, 0)), np.zeros((0, 0)), g.shape[0], pred.shape[0]))
                else:
                    iou, dice = skmetric.pairwise_iou_dice(g, pred)
                    ms.append(skmetric.match_scores(iou, dice, g.shape[0], pred.shape[0]))
            m = skmetric.aggregate_matches(ms)
            fold_rows.append({"stem": s, "cv_fold": i, "model_fold": fold_by_stem[s], "day": day_by_stem[s],
                              "sum_iou": sum(m.tp_ious), "tp": m.tp, "fp": m.fp, "fn": m.fn})
            del probs, pred, gt
        f = pd.DataFrame(fold_rows)
        pq = round(cvp.pooled_pq_from_images(f), 4)
        ref = float(e009_cv.loc[e009_cv["cv_fold"] == i, "pq"].iloc[0])
        print(f"CV fold {i}: PQ {pq} (E009_cv_results.csv: {ref})", flush=True)
        if abs(pq - ref) > 1e-4:
            raise SystemExit(f"fold {i} does not reproduce E009 ({pq} vs {ref})")
        rows.extend(fold_rows)
    pd.DataFrame(rows).to_csv(INTERIM / "E009_p5_per_image.csv", index=False)
    print(f"wrote {INTERIM / 'E009_p5_per_image.csv'}")


if __name__ == "__main__":
    main()
