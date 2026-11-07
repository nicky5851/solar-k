#!/usr/bin/env python3
"""E009: combine the Spec 04 winners (CODING_SPEC_04_MODEL_SIDE.md section 3).

Both model-side candidates were rejected by the pre-registered decision
rule (see data/interim/E006_decision.json, E007_decision.json and
reports/04_model_side_report.md) -- hard-negative mining and the stronger
encoder each failed to clear >=0.02 PQ on both compared folds without SQ
falling. Per the spec ("if nothing passed, skip the training"), E009 needs
no new model training: it is E005's 5-fold cross-validated pipeline
(cfg1 + shape gate + per-fold-refit scorer) with E008's two-stage mask
re-growing (t_mask=0.6, d=8, data/interim/E008_eval_summary.json) layered
on top of each fold's kept instances before scoring.

To avoid re-running E005's expensive C_GRID search (the dominant cost of
that ~5.8h run), this reuses the *already-chosen* scorer cutoff `c` per CV
fold from data/interim/E005_cv_results.csv -- the scorer is refit
identically (same seed, same deterministic GroupKFold split over the same
stems/days, so bit-identical to E005's), so no new grid search is needed,
only one extra evaluation pass per fold (with regrow added) instead of 19.

Usage: python scripts/run_e009.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_submission as vs
from run_e002_eval import build_gt_for_stem, load_annotation_index, load_prob_png
from run_e003 import ComponentCache, build_component_cache, summary_row
from run_e005 import _gt_to_dense, _gt_to_sparse, load_available_fold_dirs

from solar_k import components as skcomp
from solar_k import evaluate as skeval
from solar_k import io as skio
from solar_k import masks as skmasks
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc

REPO_ROOT = skio.repo_root()
H = W = 2048
N_CV_FOLDS = 5


def regrow_pred(probs: np.ndarray, cache: ComponentCache, keep: np.ndarray, t_mask: float, d: int) -> np.ndarray:
    if not keep.any():
        return np.zeros((0, H, W), dtype=np.uint8)
    seed_rows = cache.sparse[keep]
    seed_masks = [np.asarray(seed_rows[i].todense()).reshape(H, W) for i in range(seed_rows.shape[0])]
    regrown = skpp.regrow_instances(probs, seed_masks, t_mask, d)
    return np.stack(regrown).astype(np.uint8) if regrown else np.zeros((0, H, W), dtype=np.uint8)


def main() -> None:
    t_start = time.time()
    e003_path = REPO_ROOT / "data" / "interim" / "E003_eval_summary.json"
    e005_cv_path = REPO_ROOT / "data" / "interim" / "E005_cv_results.csv"
    e005_summary_path = REPO_ROOT / "data" / "interim" / "E005_eval_summary.json"
    e008_path = REPO_ROOT / "data" / "interim" / "E008_eval_summary.json"
    for p in (e003_path, e005_cv_path, e005_summary_path, e008_path):
        if not p.exists():
            raise SystemExit(f"Need {p} -- run the corresponding script first.")

    e003_summary = json.loads(e003_path.read_text())
    cfg1 = skpp.PostprocessConfig.from_dict(e003_summary["best_cfg1"])
    min_elong = e003_summary["shape_gate"]["min_elongation"]
    min_len = e003_summary["shape_gate"]["min_length"]
    cfg_components = skpp.PostprocessConfig(
        mode=cfg1.mode, t=cfg1.t, t_low=cfg1.t_low, t_high=cfg1.t_high,
        closing_radius=cfg1.closing_radius, min_area=1, max_instances=5000,
    )
    e005_cv_df = pd.read_csv(e005_cv_path)
    e005_summary = json.loads(e005_summary_path.read_text())
    e008_summary = json.loads(e008_path.read_text())
    t_mask = e008_summary["regrow_best"]["t_mask"]
    d = e008_summary["regrow_best"]["d"]
    print(f"Re-growing config from E008: t_mask={t_mask}, d={d}")

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

    images_by_file, anns_by_image_id = load_annotation_index()
    folds_df = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    train_paths = {p.stem: p for p in skio.iter_train_images()}
    test_paths = {p.stem: p for p in skio.iter_test_images()}

    instance_stats = pd.read_csv(REPO_ROOT / "data" / "interim" / "instance_stats.csv")
    quartile_edges = instance_stats["area"].quantile([0.25, 0.5, 0.75]).tolist()

    gt_sparse_by_stem, disk_info_by_stem, site_by_stem, day_by_stem = {}, {}, {}, {}
    cache_by_stem: dict[str, ComponentCache] = {}
    probs_path_by_stem: dict[str, Path] = {}
    for fold in available_folds:
        fold_rows = folds_df[folds_df["fold"] == fold]
        for _, row in fold_rows.iterrows():
            stem = row["file_stem"]
            png = oof_dirs[fold] / f"{stem}.png"
            if not png.exists():
                continue
            probs = load_prob_png(png)
            probs_path_by_stem[stem] = png
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

    days = [day_by_stem[s] for s in all_stems]
    n_splits = min(N_CV_FOLDS, len(set(days)))
    gkf = GroupKFold(n_splits=n_splits)
    stems_arr = np.array(all_stems)

    def evaluate_fold(stems, scorer, c, with_regrow: bool, compute_pixel_metrics=False) -> skeval.EvaluationSummary:
        results = []
        for stem in stems:
            cache = cache_by_stem[stem]
            df = cache.features_df
            if len(df) == 0:
                keep = np.zeros(0, dtype=bool)
            else:
                keep = step1_shape_keep(df) & (sksc.keep_score(scorer, df) > c)
            if with_regrow:
                probs = load_prob_png(probs_path_by_stem[stem])
                pred = regrow_pred(probs, cache, keep, t_mask, d)
                del probs
            else:
                pred = cache.sparse[keep] if len(df) else cache.sparse
            gt_dense = _gt_to_dense(gt_sparse_by_stem[stem])
            for annotator, gt_masks in gt_dense.items():
                r = skeval.evaluate_annotator_image(
                    stem, annotator, pred, gt_masks,
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
        c_i = float(e005_cv_df.loc[e005_cv_df["cv_fold"] == i, "c"].iloc[0])

        held_summary = evaluate_fold(held_stems, scorer_i, c_i, with_regrow=True, compute_pixel_metrics=True)
        o = held_summary.overall
        cv_rows.append({"cv_fold": i, "n_train": len(train_stems), "n_held": len(held_stems), "c": c_i, **summary_row(held_summary)})
        pooled_matches_tp_ious.extend(o.tp_ious)
        pooled_fp += o.fp
        pooled_fn += o.fn
        print(f"CV fold {i}: held={len(held_stems)} images, c={c_i}, PQ={o.pq:.4f} SQ={o.sq:.4f} RQ={o.rq:.4f}", flush=True)

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(REPO_ROOT / "data" / "interim" / "E009_cv_results.csv", index=False)
    pooled_denom = len(pooled_matches_tp_ious) + 0.5 * pooled_fp + 0.5 * pooled_fn
    pooled_pq = sum(pooled_matches_tp_ious) / pooled_denom if pooled_denom > 0 else 0.0
    pooled_sq = sum(pooled_matches_tp_ious) / len(pooled_matches_tp_ious) if pooled_matches_tp_ious else 0.0
    pooled_rq = len(pooled_matches_tp_ious) / pooled_denom if pooled_denom > 0 else 0.0

    summary = {
        "note": "E009 = E005 (5-fold CV scorer, cfg1+shape_gate fixed from E003) + E008 (regrow t_mask/d fixed from E008). No new training -- E006/E007 were both rejected.",
        "available_folds": available_folds,
        "n_images": len(all_stems),
        "n_cv_folds": n_splits,
        "regrow_t_mask": t_mask,
        "regrow_d": d,
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
        "e005_cv_pq_mean": e005_summary["cv_pq_mean"],
        "e005_cv_pq_std": e005_summary["cv_pq_std"],
    }
    print(f"\nE009 5-fold-CV PQ (mean +/- std): {summary['cv_pq_mean']:.4f} +/- {summary['cv_pq_std']:.4f}")
    print(f"E009 pooled PQ: {summary['pooled_pq']:.4f} (SQ={summary['pooled_sq']:.4f} RQ={summary['pooled_rq']:.4f})")
    print(f"vs E005 (no regrow): PQ {e005_summary['cv_pq_mean']:.4f} +/- {e005_summary['cv_pq_std']:.4f} -- delta {summary['cv_pq_mean'] - e005_summary['cv_pq_mean']:+.4f}")

    # --- Final production pipeline: refit scorer on everything (reuse E005's
    # already-chosen c for the same reason as the CV loop above), ensemble
    # test probs, apply cfg1+shape_gate+scorer+regrow, write submission.
    scorer_full = sksc.fit_scorer(components_df, seed=0)
    c_full = float(e005_summary["final_scorer_c"])
    print(f"Final scorer (fit on all {len(all_stems)} images): c={c_full} (reused from E005)")

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
        sparse = skcomp.components_to_sparse(labels, [f.label for f in features])
        cache = ComponentCache(features_df=df, sparse=sparse)
        keep = step1_shape_keep(df) & (sksc.keep_score(scorer_full, df) > c_full)
        pred = regrow_pred(probs, cache, keep, t_mask, d)
        for inst_idx, mask in enumerate(pred):
            if mask.sum() == 0:
                continue
            counts = skmasks.encode_rle(mask.astype(np.uint8))
            rows.append({"filament_id": f"{stem}_{inst_idx}", "segmentation_rle": counts})
        del probs, processed, labels, sparse, cache
    if missing_test:
        print(f"WARNING: {len(missing_test)} test images had no probability map from any available fold (e.g. {missing_test[:3]})")

    submission_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_path = REPO_ROOT / "outputs" / "submissions" / "E009.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out_path, index=False)
    errors = vs.validate(out_path, 2048, 2048, image_dir=skio.test_images_dir())
    if errors:
        print(f"{out_path} FAILED validate_submission.py:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"Wrote + validated {out_path} ({len(submission_df)} instances)")

    summary["final_scorer_c"] = c_full
    summary["ensemble_folds_used_for_test"] = available_folds
    (REPO_ROOT / "data" / "interim" / "E009_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
