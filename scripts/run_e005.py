#!/usr/bin/env python3
"""E005: the 5-fold cross-validated PQ baseline every CODING_SPEC_04_MODEL_SIDE.md
experiment must beat (its section 1's "operational goal").

E003's threshold grid + closing radius + min_area + shape gate (best_cfg1,
shape_gate in data/interim/E003_eval_summary.json) are kept fixed -- they
were already chosen on a leakage-safe tune/report split and re-deriving them
via a ~370-config grid search nested inside 5 more CV folds would cost
hours of CPU for a result expected to land in the same place (E003's step1/
step3 report-half numbers were stable). What *is* refit here, properly
cross-validated this time, is the piece CODING_SPEC_04 flagged as the likely
source of E003's local-vs-public gap: the learned scorer and its keep
cutoff `c`. 5 day-grouped CV folds (GroupKFold over each image's capture
date, so no two images from the same day split across train/held-out,
mirroring CODING_SPEC_03's tune/report day-split): fit the scorer + pick c
on 4/5, score PQ on the held-out 1/5, for every fold. Reports mean +/- std
PQ/SQ/RQ across the 5 (what the spec calls for) and the pooled aggregate
across all held-out predictions (tp_ious/fp/fn summed, not averaged -- the
more statistically grounded single number, same aggregation PQResult uses
everywhere else in this repo).

Uses whichever of outputs/E004-f{0..4}/{oof,test_probs} are present locally
-- fewer than 5 folds gives a provisional number; re-run once all 5 exist.

Final step: refits the scorer on the full combined dataset, ensembles the
available folds' test-set probability maps (plain mean of probabilities),
applies cfg1+shape_gate+scorer, writes outputs/submissions/E005.csv (not
submitted to Kaggle -- CODING_SPEC_04 section 6 requires sign-off first).

Usage: python scripts/run_e005.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_submission as vs
from run_e002_eval import build_gt_for_stem, load_annotation_index, load_prob_png
from run_e003 import (
    ComponentCache,
    build_component_cache,
    summary_row,
)

from solar_k import components as skcomp
from solar_k import evaluate as skeval
from solar_k import io as skio
from solar_k import masks as skmasks
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc

REPO_ROOT = skio.repo_root()
H = W = 2048
C_GRID = [round(x, 3) for x in np.arange(0.0, 0.525, 0.025)]
N_CV_FOLDS = 5


def _gt_to_sparse(gt_by_annotator: dict) -> dict:
    """CLAUDE.md requires streaming on this 8GB Mac: a dense (n, H, W)
    instance-mask stack per annotator, held for every one of ~565 images at
    once, is tens of GB. Filament coverage is a tiny fraction of 2048x2048,
    so converting to (n, H*W) sparse CSR up front and reconstructing dense
    only transiently per-stem (see `_gt_to_dense`) keeps the persistent
    cache under ~20MB total."""
    out = {}
    for a, masks in gt_by_annotator.items():
        n = masks.shape[0]
        out[a] = sp.csr_matrix(masks.reshape(n, -1)) if n else sp.csr_matrix((0, H * W), dtype=np.uint8)
    return out


def _gt_to_dense(gt_sparse: dict) -> dict:
    out = {}
    for a, csr in gt_sparse.items():
        n = csr.shape[0]
        out[a] = np.asarray(csr.todense()).reshape(n, H, W).astype(np.uint8) if n else np.zeros((0, H, W), dtype=np.uint8)
    return out


def load_available_fold_dirs(kind: str) -> dict[int, Path]:
    dirs = {}
    for fold in range(5):
        d = skio.output_root() / f"E004-f{fold}" / kind
        if d.exists():
            dirs[fold] = d
    return dirs


def main() -> None:
    t_start = time.time()
    summary_path = REPO_ROOT / "data" / "interim" / "E003_eval_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Need {summary_path} -- run scripts/run_e003.py first.")
    e003_summary = json.loads(summary_path.read_text())
    cfg1 = skpp.PostprocessConfig.from_dict(e003_summary["best_cfg1"])
    min_elong = e003_summary["shape_gate"]["min_elongation"]
    min_len = e003_summary["shape_gate"]["min_length"]
    cfg_components = skpp.PostprocessConfig(
        mode=cfg1.mode, t=cfg1.t, t_low=cfg1.t_low, t_high=cfg1.t_high,
        closing_radius=cfg1.closing_radius, min_area=1, max_instances=5000,
    )

    def step1_shape_keep(df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.zeros(0, dtype=bool)
        return (
            (df["area"] >= cfg1.min_area)
            & (df["elongation"].fillna(0) >= min_elong)
            & (df["length"].fillna(0) >= min_len)
        ).to_numpy()

    oof_dirs = load_available_fold_dirs("oof")
    test_dirs = load_available_fold_dirs("test_probs")
    available_folds = sorted(oof_dirs)
    print(f"Folds available locally: {available_folds}")
    if not available_folds:
        raise SystemExit("No outputs/E004-fN/oof found locally -- sync at least one fold first.")

    images_by_file, anns_by_image_id = load_annotation_index()
    folds_df = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    train_paths = {p.stem: p for p in skio.iter_train_images()}
    test_paths = {p.stem: p for p in skio.iter_test_images()}

    instance_stats = pd.read_csv(REPO_ROOT / "data" / "interim" / "instance_stats.csv")
    quartile_edges = instance_stats["area"].quantile([0.25, 0.5, 0.75]).tolist()

    gt_sparse_by_stem, disk_info_by_stem, site_by_stem, day_by_stem = {}, {}, {}, {}
    cache_by_stem: dict[str, ComponentCache] = {}
    for fold in available_folds:
        fold_rows = folds_df[folds_df["fold"] == fold]
        for _, row in fold_rows.iterrows():
            stem = row["file_stem"]
            png = oof_dirs[fold] / f"{stem}.png"
            if not png.exists():
                continue
            probs = load_prob_png(png)
            processed, disk_info = skpre.load_or_build(stem, train_paths[stem])
            disk_info_by_stem[stem] = disk_info
            site_by_stem[stem] = row["site"]
            day_by_stem[stem] = row["date"]
            gt_dense = build_gt_for_stem(stem, images_by_file, anns_by_image_id)
            gt_sparse_by_stem[stem] = _gt_to_sparse(gt_dense)
            cache_by_stem[stem] = build_component_cache(stem, probs, processed, disk_info, cfg_components, gt_dense)
            del processed, gt_dense, probs

    all_stems = sorted(cache_by_stem)
    print(f"{len(all_stems)} images with local OOF across {len(available_folds)} fold(s)")

    components_df = pd.concat(
        [cache_by_stem[s].features_df.assign(stem=s) for s in all_stems], ignore_index=True
    )
    components_df.to_csv(REPO_ROOT / "data" / "interim" / "E005_component_table.csv", index=False)

    # --- 5-fold day-grouped CV: refit scorer + cutoff c, score held-out ----
    days = [day_by_stem[s] for s in all_stems]
    n_splits = min(N_CV_FOLDS, len(set(days)))
    gkf = GroupKFold(n_splits=n_splits)
    stems_arr = np.array(all_stems)

    def evaluate_scorer_on(stems, scorer, c, compute_pixel_metrics=False) -> skeval.EvaluationSummary:
        results = []
        for stem in stems:
            cache = cache_by_stem[stem]
            df = cache.features_df
            if len(df) == 0:
                keep = np.zeros(0, dtype=bool)
            else:
                keep = step1_shape_keep(df) & (sksc.keep_score(scorer, df) > c)
            pred_sparse = cache.sparse[keep] if len(df) else cache.sparse
            gt_dense = _gt_to_dense(gt_sparse_by_stem[stem])
            for annotator, gt_masks in gt_dense.items():
                r = skeval.evaluate_annotator_image(
                    stem, annotator, pred_sparse, gt_masks,
                    disk_info=disk_info_by_stem.get(stem), site=site_by_stem.get(stem),
                    compute_pixel_metrics=compute_pixel_metrics,
                )
                results.append(r)
            del gt_dense
        return skeval.aggregate_evaluation(results, quartile_edges=quartile_edges, noisy_stems=set())

    cv_rows = []
    pooled_matches_tp_ious, pooled_fp, pooled_fn = [], 0, 0
    for i, (train_idx, held_idx) in enumerate(gkf.split(stems_arr, groups=days)):
        train_stems, held_stems = stems_arr[train_idx].tolist(), stems_arr[held_idx].tolist()
        train_df = components_df[components_df["stem"].isin(train_stems)]
        scorer_i = sksc.fit_scorer(train_df, seed=0)

        best_c_i, best_pq_i = 0.0, -1.0
        for c in C_GRID:
            s = evaluate_scorer_on(train_stems, scorer_i, c)
            if s.overall.pq > best_pq_i:
                best_pq_i, best_c_i = s.overall.pq, c

        held_summary = evaluate_scorer_on(held_stems, scorer_i, best_c_i, compute_pixel_metrics=True)
        o = held_summary.overall
        cv_rows.append({"cv_fold": i, "n_train": len(train_stems), "n_held": len(held_stems), "c": best_c_i, **summary_row(held_summary)})
        pooled_matches_tp_ious.extend(o.tp_ious)
        pooled_fp += o.fp
        pooled_fn += o.fn
        print(f"CV fold {i}: held={len(held_stems)} images, c={best_c_i}, PQ={o.pq:.4f} SQ={o.sq:.4f} RQ={o.rq:.4f}")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(REPO_ROOT / "data" / "interim" / "E005_cv_results.csv", index=False)
    pooled_denom = len(pooled_matches_tp_ious) + 0.5 * pooled_fp + 0.5 * pooled_fn
    pooled_pq = sum(pooled_matches_tp_ious) / pooled_denom if pooled_denom > 0 else 0.0
    pooled_sq = sum(pooled_matches_tp_ious) / len(pooled_matches_tp_ious) if pooled_matches_tp_ious else 0.0
    pooled_rq = len(pooled_matches_tp_ious) / pooled_denom if pooled_denom > 0 else 0.0

    summary = {
        "available_folds": available_folds,
        "n_images": len(all_stems),
        "n_cv_folds": n_splits,
        "cv_pq_mean": round(float(cv_df["pq"].mean()), 4),
        "cv_pq_std": round(float(cv_df["pq"].std()), 4),
        "cv_sq_mean": round(float(cv_df["sq"].mean()), 4),
        "cv_sq_std": round(float(cv_df["sq"].std()), 4),
        "cv_rq_mean": round(float(cv_df["rq"].mean()), 4),
        "cv_rq_std": round(float(cv_df["rq"].std()), 4),
        "pooled_pq": round(pooled_pq, 4),
        "pooled_sq": round(pooled_sq, 4),
        "pooled_rq": round(pooled_rq, 4),
        "pooled_tp": len(pooled_matches_tp_ious),
        "pooled_fp": pooled_fp,
        "pooled_fn": pooled_fn,
    }
    print(f"\nE005 5-fold-CV PQ (mean +/- std): {summary['cv_pq_mean']:.4f} +/- {summary['cv_pq_std']:.4f}")
    print(f"E005 pooled PQ: {summary['pooled_pq']:.4f} (SQ={summary['pooled_sq']:.4f} RQ={summary['pooled_rq']:.4f})")
    if len(available_folds) < 5:
        print(f"NOTE: only {len(available_folds)}/5 folds available locally -- this is provisional, re-run once all 5 are synced.")

    # --- Final production pipeline: refit on everything, ensemble test probs
    scorer_full = sksc.fit_scorer(components_df, seed=0)
    best_c_full, best_pq_full = 0.0, -1.0
    for c in C_GRID:
        s = evaluate_scorer_on(all_stems, scorer_full, c)
        if s.overall.pq > best_pq_full:
            best_pq_full, best_c_full = s.overall.pq, c
    print(f"Final scorer (fit on all {len(all_stems)} images): c={best_c_full}")

    rows = []
    missing_test = []
    for path in skio.iter_test_images():
        stem = path.stem
        probs_list = []
        for fold in available_folds:
            prob_path = test_dirs[fold] / f"{stem}.png"
            if prob_path.exists():
                probs_list.append(load_prob_png(prob_path))
        if not probs_list:
            missing_test.append(stem)
            continue
        probs = np.mean(probs_list, axis=0)
        processed, disk_info = skpre.load_or_build(stem, test_paths[stem])
        features, labels = skcomp.extract_components(stem, probs, processed, disk_info, cfg_components)
        df = skcomp.to_dataframe(features)
        if len(df) == 0:
            del probs, processed
            continue
        keep = step1_shape_keep(df) & (sksc.keep_score(scorer_full, df) > best_c_full)
        kept_labels = [f.label for f, k in zip(features, keep) if k]
        for lid in kept_labels:
            mask = (labels == lid).astype(np.uint8)
            if mask.sum() == 0:
                continue
            counts = skmasks.encode_rle(mask)
            rows.append({"filament_id": f"{stem}_{lid}", "segmentation_rle": counts})
        del probs, processed, labels
    if missing_test:
        print(f"WARNING: {len(missing_test)} test images had no probability map from any available fold (e.g. {missing_test[:3]})")

    submission_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_path = REPO_ROOT / "outputs" / "submissions" / "E005.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_path, index=False)
    errors = vs.validate(out_path, 2048, 2048, image_dir=skio.test_images_dir())
    if errors:
        print(f"{out_path} FAILED validate_submission.py:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"Wrote + validated {out_path} ({len(submission_df)} instances)")

    summary["final_scorer_c"] = best_c_full
    summary["ensemble_folds_used_for_test"] = available_folds
    (REPO_ROOT / "data" / "interim" / "E005_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
