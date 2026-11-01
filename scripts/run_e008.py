#!/usr/bin/env python3
"""E008 (CODING_SPEC_04_MODEL_SIDE.md): mask-extent threshold vs detection
threshold, and an SQ ceiling estimate. Three parts:

1. SQ ceiling: on multi-annotator images, for each annotator A, build a
   "consensus of the other annotators" target (soft_consensus_target over
   just the others, thresholded at 0.5, connected components -- no model
   involved), and score A's own instances against it the same way PQ scores
   predictions against GT. The mean matched IoU (= aggregate_matches(...).sq)
   is how much SQ headroom even a perfect model could have, since annotators
   themselves disagree on exact boundaries.

2. Area-ratio / boundary-offset diagnosis: on every fold's OOF (whichever of
   outputs/E004-f{0..4}/oof exist locally -- re-run this once more folds are
   synced), apply E003's chosen postprocessing (best_cfg1 + shape gate, no
   scorer -- the scorer's job is recognition, not extent) and match kept
   instances to GT at IoU > 0.5. For matched (TP) pairs, record area_ratio
   (pred/gt) and a signed boundary offset (median value of the GT's signed
   distance transform sampled at the predicted mask's boundary pixels --
   positive means the predicted boundary sits outside the GT, i.e. the mask
   extends past the true filament; negative means it's inside, i.e. thin).

3. Two-stage re-growing grid search: using those same TP seed instances (now
   from the *scorer-gated* step-3 pipeline, not just shape-gated) and
   postprocess.regrow_instances, grid-search t_mask x d on `tune`, report on
   `report` -- CODING_SPEC_03_FP_AND_FOLDS.md's leakage-safe split, reused
   so this isn't tuned on data it's scored against.

Writes data/interim/E008_area_boundary.csv (per-TP-pair rows), data/interim/
E008_regrow_grid.csv, and data/interim/E008_eval_summary.json (SQ ceiling,
report-half PQ/SQ/RQ before and after re-growing, chosen t_mask/d).

Usage: python scripts/run_e008.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import components as skcomp
from solar_k import evaluate as skeval
from solar_k import ids as skids
from solar_k import io as skio
from solar_k import metric as skmetric
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc
from solar_k import targets as sktgt
from solar_k import tuning as sktune

from run_e002_eval import build_gt_for_stem, load_annotation_index, load_prob_png

H = W = 2048
REPO_ROOT = skio.repo_root()
T_MASK_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
D_GRID = [2, 4, 8]


def sq_ceiling(images_by_file, anns_by_image_id) -> dict:
    """Part 1: annotator-vs-consensus-of-others SQ, over every multi-annotator
    training image (no model, no probability maps involved)."""
    matches = []
    n_images = 0
    for file_name, entries in images_by_file.items():
        if len(entries) < 2:
            continue
        n_images += 1
        anns_by_annotator = {}
        for e in entries:
            annotator, _ = skids.parse_image_id(e["id"])
            anns_by_annotator[annotator] = anns_by_image_id.get(e["id"], [])

        for a in anns_by_annotator:
            own_masks = sktgt.annotator_instance_masks(anns_by_annotator[a], H, W)
            own = np.stack(own_masks) if own_masks else np.zeros((0, H, W), dtype=np.uint8)

            others = {k: v for k, v in anns_by_annotator.items() if k != a}
            consensus = sktgt.soft_consensus_target(others, H, W)
            consensus_bin = (consensus >= 0.5).astype(np.uint8)
            consensus_instances = skpp.extract_instances(consensus_bin, min_area=1)
            cons = np.stack(consensus_instances) if consensus_instances else np.zeros((0, H, W), dtype=np.uint8)

            matches.append(skmetric.score_image(cons, own))

    result = skmetric.aggregate_matches(matches)
    print(f"SQ ceiling: {n_images} multi-annotator images, {len(matches)} annotator-vs-others comparisons, "
          f"TP={result.tp} FP={result.fp} FN={result.fn}, SQ_ceiling={result.sq:.4f}")
    return {"n_multi_annotator_images": n_images, "n_comparisons": len(matches),
            "tp": result.tp, "fp": result.fp, "fn": result.fn, "sq_ceiling": result.sq}


def signed_boundary_offset(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Median value of GT's signed distance transform, sampled at pred's
    boundary pixels. Positive: pred boundary sits outside GT (mask too
    large there). Negative: pred boundary sits inside GT (mask too thin)."""
    gt_bool = gt_mask.astype(bool)
    dist_out = distance_transform_edt(~gt_bool)  # 0 inside GT, >0 outside
    dist_in = distance_transform_edt(gt_bool)  # >0 inside GT, 0 outside
    signed = np.where(gt_bool, -dist_in, dist_out)

    pred_u8 = pred_mask.astype(np.uint8)
    import cv2
    eroded = cv2.erode(pred_u8, np.ones((3, 3), np.uint8))
    boundary = (pred_u8 - eroded).astype(bool)
    if not boundary.any():
        boundary = pred_u8.astype(bool)
    return float(np.median(signed[boundary]))


def main() -> None:
    t_start = time.time()
    images_by_file, anns_by_image_id = load_annotation_index()

    ceiling = sq_ceiling(images_by_file, anns_by_image_id)

    summary_path = REPO_ROOT / "data" / "interim" / "E003_eval_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Need {summary_path} -- run scripts/run_e003.py first.")
    e003_summary = json.loads(summary_path.read_text())
    cfg1 = skpp.PostprocessConfig.from_dict(e003_summary["best_cfg1"])
    min_elong = e003_summary["shape_gate"]["min_elongation"]
    min_len = e003_summary["shape_gate"]["min_length"]
    best_c = e003_summary["scorer_c"]
    cfg_components = skpp.PostprocessConfig(
        mode=cfg1.mode, t=cfg1.t, t_low=cfg1.t_low, t_high=cfg1.t_high,
        closing_radius=cfg1.closing_radius, min_area=1, max_instances=5000,
    )

    comp_table = pd.read_csv(REPO_ROOT / "data" / "interim" / "E003_component_table.csv")
    tune_table = comp_table[comp_table["split"] == "tune"]
    scorer = sksc.fit_scorer(tune_table, seed=0)

    folds_df = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    train_paths = {p.stem: p for p in skio.iter_train_images()}

    gt_by_stem, disk_info_by_stem, day_by_stem, site_by_stem = {}, {}, {}, {}
    probs_by_stem: dict[str, np.ndarray] = {}
    available_folds = []
    for fold in range(5):
        oof_dir = skio.output_root() / f"E004-f{fold}" / "oof"
        if not oof_dir.exists():
            print(f"fold {fold}: no local OOF yet -- skipping (re-run once outputs/E004-f{fold}/oof is synced)")
            continue
        available_folds.append(fold)
        fold_rows = folds_df[folds_df["fold"] == fold]
        for _, row in fold_rows.iterrows():
            stem = row["file_stem"]
            png = oof_dir / f"{stem}.png"
            if not png.exists():
                continue
            probs_by_stem[stem] = load_prob_png(png)
            _, disk_info = skpre.load_or_build(stem, train_paths[stem])
            disk_info_by_stem[stem] = disk_info
            site_by_stem[stem] = row["site"]
            day_by_stem[stem] = row["date"]
            gt_by_stem[stem] = build_gt_for_stem(stem, images_by_file, anns_by_image_id)
    print(f"Folds available locally: {available_folds} ({len(probs_by_stem)} images total)")

    tune_stems, report_stems = sktune.split_tune_report(day_by_stem, seed=0)
    tune_stems = sorted(s for s in tune_stems if s in probs_by_stem)
    report_stems = sorted(s for s in report_stems if s in probs_by_stem)
    all_stems = tune_stems + report_stems
    print(f"Tune: {len(tune_stems)} images, Report: {len(report_stems)} images")

    # --- Part 2: area ratio / boundary offset on TP pairs, step-3 pipeline --
    ar_rows = []
    seeds_by_stem: dict[str, list[np.ndarray]] = {}
    for stem in all_stems:
        probs = probs_by_stem[stem]
        processed, _ = skpre.load_or_build(stem, train_paths[stem])
        features, labels = skcomp.extract_components(stem, probs, processed, disk_info_by_stem[stem], cfg_components)
        skcomp.label_components(features, labels, gt_by_stem[stem])
        df = skcomp.to_dataframe(features)
        if df.empty:
            seeds_by_stem[stem] = []
            del labels
            continue
        shape_keep = (df["area"] >= cfg1.min_area) & (df["elongation"].fillna(0) >= min_elong) & (df["length"].fillna(0) >= min_len)
        score_keep = sksc.keep_score(scorer, df) > best_c
        keep = (shape_keep & score_keep).to_numpy()
        kept_df = df.loc[keep].reset_index(drop=True)
        kept_labels = df.loc[keep, "label"].tolist()
        seeds_by_stem[stem] = [(labels == lid).astype(np.uint8) for lid in kept_labels]
        del labels

        for i, mask in enumerate(seeds_by_stem[stem]):
            best_iou, best_gt_mask = 0.0, None
            for annotator, gt_masks in gt_by_stem[stem].items():
                if gt_masks.shape[0] == 0:
                    continue
                iou, _ = skmetric.pairwise_iou_dice(gt_masks, mask[None, :, :])
                j = int(np.argmax(iou[:, 0]))
                if iou[j, 0] > best_iou:
                    best_iou, best_gt_mask = float(iou[j, 0]), gt_masks[j]
            if best_iou > skmetric.IOU_THRESHOLD and best_gt_mask is not None:
                area_ratio = float(mask.sum()) / max(1.0, float(best_gt_mask.sum()))
                boundary_offset = signed_boundary_offset(mask, best_gt_mask)
                ar_rows.append({"stem": stem, "area_ratio": area_ratio, "boundary_offset_px": boundary_offset, "iou": best_iou})

    ar_df = pd.DataFrame(ar_rows)
    ar_path = REPO_ROOT / "data" / "interim" / "E008_area_boundary.csv"
    ar_df.to_csv(ar_path, index=False)
    if len(ar_df):
        print(f"Area ratio: median={ar_df['area_ratio'].median():.3f} IQR=[{ar_df['area_ratio'].quantile(.25):.3f}, {ar_df['area_ratio'].quantile(.75):.3f}]")
        print(f"Boundary offset (px): median={ar_df['boundary_offset_px'].median():.2f} IQR=[{ar_df['boundary_offset_px'].quantile(.25):.2f}, {ar_df['boundary_offset_px'].quantile(.75):.2f}]")
        print(f"(negative median => masks systematically thin, supporting the re-growing hypothesis)")
    else:
        print("No TP pairs found for area/boundary diagnosis.")

    # --- Part 3: two-stage re-growing grid search ---------------------------
    quartile_edges_path = REPO_ROOT / "data" / "interim" / "instance_stats.csv"
    quartile_edges = pd.read_csv(quartile_edges_path)["area"].quantile([0.25, 0.5, 0.75]).tolist() if quartile_edges_path.exists() else None

    def evaluate_regrow(stems, t_mask: float, d: int, compute_pixel_metrics: bool = False) -> skeval.EvaluationSummary:
        results = []
        for stem in stems:
            seeds = seeds_by_stem.get(stem, [])
            grown = skpp.regrow_instances(probs_by_stem[stem], seeds, t_mask=t_mask, d=d) if seeds else []
            pred_sparse = _stack_to_sparse(grown) if grown else sp.csr_matrix((0, H * W), dtype=np.uint8)
            for annotator, gt_masks in gt_by_stem[stem].items():
                r = skeval.evaluate_annotator_image(
                    stem, annotator, pred_sparse, gt_masks,
                    disk_info=disk_info_by_stem.get(stem), site=site_by_stem.get(stem),
                    compute_pixel_metrics=compute_pixel_metrics,
                )
                results.append(r)
        return skeval.aggregate_evaluation(results, quartile_edges=quartile_edges, noisy_stems=set())

    t0 = time.time()
    grid_rows = []
    best_t_mask, best_d, best_pq = cfg1.t, 0, -1.0
    for t_mask in T_MASK_GRID:
        if t_mask >= cfg1.t:
            continue  # re-growing only makes sense below the detection threshold
        for d in D_GRID:
            s = evaluate_regrow(tune_stems, t_mask, d)
            grid_rows.append({"t_mask": t_mask, "d": d, "pq": s.overall.pq, "sq": s.overall.sq, "rq": s.overall.rq})
            if s.overall.pq > best_pq:
                best_pq, best_t_mask, best_d = s.overall.pq, t_mask, d
    pd.DataFrame(grid_rows).to_csv(REPO_ROOT / "data" / "interim" / "E008_regrow_grid.csv", index=False)
    print(f"Re-grow grid search done in {time.time() - t0:.1f}s. Best on tune: t_mask={best_t_mask}, d={best_d}, PQ={best_pq:.4f}")

    baseline_report = evaluate_regrow(report_stems, cfg1.t, 0, compute_pixel_metrics=True)  # d=0 => no growth, same as step-4 seeds
    grown_report = evaluate_regrow(report_stems, best_t_mask, best_d, compute_pixel_metrics=True)

    def row(s):
        o = s.overall
        return {"pq": round(o.pq, 4), "sq": round(o.sq, 4), "rq": round(o.rq, 4), "tp": o.tp, "fp": o.fp, "fn": o.fn}

    summary = {
        "sq_ceiling": ceiling,
        "available_folds": available_folds,
        "area_ratio_median": float(ar_df["area_ratio"].median()) if len(ar_df) else None,
        "boundary_offset_median_px": float(ar_df["boundary_offset_px"].median()) if len(ar_df) else None,
        "regrow_best": {"t_mask": best_t_mask, "d": best_d},
        "report_half_before_regrow": row(baseline_report),
        "report_half_after_regrow": row(grown_report),
    }
    (REPO_ROOT / "data" / "interim" / "E008_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Done in {(time.time() - t_start) / 60:.1f} min")


def _stack_to_sparse(masks: list[np.ndarray]):
    flat = np.stack([m.reshape(-1) for m in masks])
    return sp.csr_matrix(flat)


if __name__ == "__main__":
    main()
