#!/usr/bin/env python3
"""Build the hard-negative pool for E006 (CODING_SPEC_04_MODEL_SIDE.md).

From the 5-fold OOF probability maps (each image predicted only by the model
that excluded it from training), extracts components with E003's chosen
postprocessing + shape gate, scores P(match) with a scorer refit on E003's
`tune` split (the same supervision E003's reported numbers used -- nothing
new is fit here), classifies FPs with the near_miss -> annotator_disagreement
-> fragment -> isolated taxonomy (run_e003.py's fp_class_counts, adapted to
return one class per component instead of an aggregate count), and keeps the
`isolated` ones as hard negatives. On single-annotator images, only keeps
FPs with P(match) < 0.2 (CODING_SPEC_03's caveat: an "isolated" FP there
might just be a real filament the lone annotator skipped).

Writes data/interim/hard_negatives_v1.csv: stem, fold, bbox, centroid, area,
class, p_match, single_annotator -- one row per hard-negative component.
scripts/train.py's sampler_mix + hard_negatives_csv config options filter
this to a fold's own training stems before use (CODING_SPEC_04's requirement
that each fold's hard negatives come only from that fold's training images).

Re-run this once more folds' OOF exists (it skips folds whose
outputs/E004-fN/oof/ isn't there yet and says so).

Usage: python scripts/build_hard_negatives.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import components as skcomp
from solar_k import io as skio
from solar_k import metric as skmetric
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc

from run_e002_eval import build_gt_for_stem, load_annotation_index, load_prob_png

H = W = 2048
REPO_ROOT = skio.repo_root()
E003_SUMMARY_PATH = REPO_ROOT / "data" / "interim" / "E003_eval_summary.json"
COMPONENT_TABLE_PATH = REPO_ROOT / "data" / "interim" / "E003_component_table.csv"
OUT_CSV = REPO_ROOT / "data" / "interim" / "hard_negatives_v1.csv"

IOU_T = skmetric.IOU_THRESHOLD
WEAK_IOU = 0.1
NEAR_MISS = (0.3, 0.5)
SINGLE_ANNOTATOR_P_MATCH_CEILING = 0.2


def classify_fp_components(sparse, gt_by_annotator: dict) -> list[str | None]:
    """One class per component row (None if matched by >=1 annotator).

    Uses the image's first annotator (fixed order) as the classification
    reference when there's more than one -- this pool only needs a single
    representative class per component, not run_e003.py's full per-annotator
    FP-count breakdown.
    """
    annotators = list(gt_by_annotator.keys())
    n_pred = sparse.shape[0]
    iou_by_ann = {}
    for a in annotators:
        gt_masks = gt_by_annotator[a]
        if gt_masks.shape[0] == 0 or n_pred == 0:
            iou_by_ann[a] = np.zeros((gt_masks.shape[0], n_pred))
        else:
            iou_by_ann[a], _ = skmetric.pairwise_iou_dice(gt_masks, sparse)

    any_hit = np.zeros(n_pred, dtype=bool)
    for a in annotators:
        if iou_by_ann[a].size:
            any_hit |= (iou_by_ann[a] > IOU_T).any(axis=0)

    ref = annotators[0]
    iou_ref = iou_by_ann[ref]
    n_gt_ref = iou_ref.shape[0]
    tp_gt_ref = set(np.where((iou_ref > IOU_T).any(axis=1))[0]) if iou_ref.size else set()

    classes: list[str | None] = [None] * n_pred
    for j in range(n_pred):
        if any_hit[j]:
            continue
        best_same = iou_ref[:, j].max() if n_gt_ref else 0.0
        if NEAR_MISS[0] <= best_same < NEAR_MISS[1]:
            classes[j] = "near_miss"
        elif any(iou_by_ann[b].shape[0] and iou_by_ann[b][:, j].max() > IOU_T for b in annotators if b != ref):
            classes[j] = "annotator_disagreement"
        elif n_gt_ref and tp_gt_ref and (iou_ref[list(tp_gt_ref), j] > WEAK_IOU).any():
            classes[j] = "fragment"
        else:
            classes[j] = "isolated"
    return classes


def main() -> None:
    t_start = time.time()
    if not E003_SUMMARY_PATH.exists() or not COMPONENT_TABLE_PATH.exists():
        raise SystemExit(f"Need {E003_SUMMARY_PATH} and {COMPONENT_TABLE_PATH} -- run scripts/run_e003.py first.")

    summary = json.loads(E003_SUMMARY_PATH.read_text())
    cfg1 = skpp.PostprocessConfig.from_dict(summary["best_cfg1"])
    cfg_components = skpp.PostprocessConfig(
        mode=cfg1.mode, t=cfg1.t, t_low=cfg1.t_low, t_high=cfg1.t_high,
        closing_radius=cfg1.closing_radius, min_area=1, max_instances=5000,
    )
    min_elong = summary["shape_gate"]["min_elongation"]
    min_len = summary["shape_gate"]["min_length"]

    comp_table = pd.read_csv(COMPONENT_TABLE_PATH)
    tune_df = comp_table[comp_table["split"] == "tune"]
    scorer = sksc.fit_scorer(tune_df, seed=0)
    print(f"Refit scorer on {len(tune_df)} tune-split components (same as E003 step 4)")

    images_by_file, anns_by_image_id = load_annotation_index()
    folds_df = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    train_paths = {p.stem: p for p in skio.iter_train_images()}

    rows: list[dict] = []
    for fold in range(5):
        oof_dir = skio.output_root() / f"E004-f{fold}" / "oof"
        if not oof_dir.exists():
            print(f"fold {fold}: no local OOF yet ({oof_dir}) -- skipping, re-run once it's synced")
            continue
        fold_stems = folds_df.loc[folds_df["fold"] == fold, "file_stem"].tolist()
        n_processed = 0
        for stem in fold_stems:
            png = oof_dir / f"{stem}.png"
            if not png.exists():
                continue
            probs = load_prob_png(png)
            processed, disk_info = skpre.load_or_build(stem, train_paths[stem])
            gt_by_annotator = build_gt_for_stem(stem, images_by_file, anns_by_image_id)

            features, labels = skcomp.extract_components(stem, probs, processed, disk_info, cfg_components)
            skcomp.label_components(features, labels, gt_by_annotator)
            df = skcomp.to_dataframe(features)
            if df.empty:
                del labels
                continue
            shape_keep = (
                (df["area"] >= cfg1.min_area)
                & (df["elongation"].fillna(0) >= min_elong)
                & (df["length"].fillna(0) >= min_len)
            )
            df_k = df.loc[shape_keep].reset_index(drop=True)
            if df_k.empty:
                del labels
                continue
            sparse_k = skcomp.components_to_sparse(labels, df_k["label"].tolist())
            del labels

            p_match = sksc.predict_match_prob(scorer, df_k)
            classes = classify_fp_components(sparse_k, gt_by_annotator)
            single_annotator = len(gt_by_annotator) == 1

            for i in range(len(df_k)):
                if classes[i] != "isolated":
                    continue
                pm = float(p_match[i])
                if single_annotator and pm >= SINGLE_ANNOTATOR_P_MATCH_CEILING:
                    continue
                row = sparse_k.getrow(i)
                cols = row.indices
                if len(cols) == 0:
                    continue
                ys, xs = np.divmod(cols, W)
                rows.append({
                    "stem": stem, "fold": fold,
                    "bbox_y0": int(ys.min()), "bbox_x0": int(xs.min()),
                    "bbox_y1": int(ys.max()), "bbox_x1": int(xs.max()),
                    "centroid_y": int(round(ys.mean())), "centroid_x": int(round(xs.mean())),
                    "area": int(len(cols)), "class": classes[i], "p_match": round(pm, 4),
                    "single_annotator": single_annotator,
                })
            n_processed += 1
        print(f"fold {fold}: {n_processed} images processed")

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} hard-negative components to {OUT_CSV} in {time.time() - t_start:.1f}s")
    if len(out_df):
        print(out_df.groupby("fold").size().to_dict())
        print(f"single-annotator rows: {int(out_df['single_annotator'].sum())}/{len(out_df)}")


if __name__ == "__main__":
    main()
