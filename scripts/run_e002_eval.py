#!/usr/bin/env python3
"""Stage C (Spec 02): evaluate E002 against the Kaggle-saved OOF/test probs.

scripts/train.py (run on Kaggle) already trained the U-Net and saved
per-image flip-TTA probability maps as compressed uint8 PNGs for fold-0 val
(OOF, outputs/E002/oof/) and the 180 test images (outputs/E002/test_probs/),
with no postprocessing/tuning performed there by design. This script runs
the same tune/report threshold-grid protocol as run_e001.py against those
cached OOF probs (no model inference here, just loading PNGs), then applies
the winning config to the cached test probs to produce the submission CSV.

Usage: python scripts/run_e002_eval.py
"""
from __future__ import annotations

import collections
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_submission as vs

from solar_k import evaluate as skeval
from solar_k import ids as skids
from solar_k import io as skio
from solar_k import masks as skmasks
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import targets as sktgt
from solar_k import tuning as sktune

H = W = 2048
REPO_ROOT = skio.repo_root()
OOF_DIR = skio.output_root() / "E002" / "oof"
TEST_PROBS_DIR = skio.output_root() / "E002" / "test_probs"


def load_annotation_index():
    ann = skio.load_coco_annotations()
    images_by_file = collections.defaultdict(list)
    for img in ann["images"]:
        images_by_file[img["file_name"]].append(img)
    anns_by_image_id = collections.defaultdict(list)
    for a in ann["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)
    return images_by_file, anns_by_image_id


def fold0_val_stems() -> pd.DataFrame:
    folds = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    return folds[folds["fold"] == 0].copy()


def build_gt_for_stem(stem: str, images_by_file: dict, anns_by_image_id: dict) -> dict[str, np.ndarray]:
    entries = images_by_file.get(f"{stem}.jpeg", [])
    out = {}
    for entry in entries:
        annotator, _ = skids.parse_image_id(entry["id"])
        anns = anns_by_image_id.get(entry["id"], [])
        masks = sktgt.annotator_instance_masks(anns, H, W)
        out[annotator] = np.stack(masks) if masks else np.zeros((0, H, W), dtype=np.uint8)
    return out


def load_prob_png(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return arr.astype(np.float32) / 255.0


def main() -> None:
    t_start = time.time()
    images_by_file, anns_by_image_id = load_annotation_index()
    val_df = fold0_val_stems()
    print(f"Fold-0 val: {len(val_df)} images")

    train_paths_by_stem = {p.stem: p for p in skio.iter_train_images()}

    gt_by_stem, disk_info_by_stem, site_by_stem, day_by_stem = {}, {}, {}, {}
    missing_oof = []
    for _, row in val_df.iterrows():
        stem = row["file_stem"]
        if not (OOF_DIR / f"{stem}.png").exists():
            missing_oof.append(stem)
            continue
        _, disk_info = skpre.load_or_build(stem, train_paths_by_stem[stem])
        disk_info_by_stem[stem] = disk_info
        site_by_stem[stem] = row["site"]
        day_by_stem[stem] = row["date"]
        gt_by_stem[stem] = build_gt_for_stem(stem, images_by_file, anns_by_image_id)
    if missing_oof:
        print(f"Warning: {len(missing_oof)} fold-0 val stems have no cached OOF prob map, skipped: {missing_oof[:5]}")

    tune_stems, report_stems = sktune.split_tune_report(day_by_stem, seed=0)
    print(f"Tune: {len(tune_stems)} images, Report: {len(report_stems)} images")

    instance_stats = pd.read_csv(REPO_ROOT / "data" / "interim" / "instance_stats.csv")
    eda_p1_area = float(instance_stats["area"].quantile(0.01))
    quartile_edges = instance_stats["area"].quantile([0.25, 0.5, 0.75]).tolist()
    configs = sktune.full_grid(eda_p1_area)
    print(f"Grid size: {len(configs)} configs (EDA p1 area = {eda_p1_area:.0f})")

    def compute_probs(stem: str) -> np.ndarray:
        return load_prob_png(OOF_DIR / f"{stem}.png")

    t0 = time.time()
    best_cfg, grid_df = sktune.grid_search(
        tune_stems, compute_probs, gt_by_stem, disk_info_by_stem, site_by_stem, configs, progress=True
    )
    print(f"Grid search done in {(time.time() - t0) / 60:.1f} min. Best on tune: PQ={grid_df['pq'].max():.4f} cfg={best_cfg}")
    grid_df.to_csv(REPO_ROOT / "data" / "interim" / "E002_grid_search.csv", index=False)

    report_results = []
    noisy_stems: set[str] = set()
    if (REPO_ROOT / "data" / "interim" / "image_stats.csv").exists():
        image_stats = pd.read_csv(REPO_ROOT / "data" / "interim" / "image_stats.csv")
        noise_p90 = image_stats["noise"].quantile(0.9)
        noisy_stems = set(image_stats.loc[image_stats["noise"] >= noise_p90, "file_stem"])

    for stem in report_stems:
        if stem not in gt_by_stem:
            continue
        probs = compute_probs(stem)
        pred_sparse = skpp.postprocess_sparse(probs, best_cfg)
        for annotator, gt_masks in gt_by_stem[stem].items():
            r = skeval.evaluate_annotator_image(
                stem, annotator, pred_sparse, gt_masks,
                disk_info=disk_info_by_stem.get(stem), site=site_by_stem.get(stem),
            )
            report_results.append(r)
        del probs

    summary = skeval.aggregate_evaluation(report_results, quartile_edges=quartile_edges, noisy_stems=noisy_stems)
    print(f"Report-half PQ={summary.overall.pq:.4f} SQ={summary.overall.sq:.4f} RQ={summary.overall.rq:.4f} "
          f"(TP={summary.overall.tp} FP={summary.overall.fp} FN={summary.overall.fn})")

    # Apply the tuned config to the cached TEST probs -> submission CSV.
    test_paths = list(skio.iter_test_images())
    rows = []
    missing_test = []
    for path in test_paths:
        stem = path.stem
        prob_path = TEST_PROBS_DIR / f"{stem}.png"
        if not prob_path.exists():
            missing_test.append(stem)
            continue
        probs = load_prob_png(prob_path)
        instances = skpp.postprocess(probs, best_cfg)
        for k, m in enumerate(instances):
            if m.sum() == 0:
                continue
            counts = skmasks.encode_rle(m)
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": counts})
        del probs, instances
    if missing_test:
        print(f"Warning: {len(missing_test)} test stems have no cached prob map, skipped: {missing_test[:5]}")

    sub_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_csv = REPO_ROOT / "outputs" / "submissions" / "E002.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(sub_df)} rows for {len(test_paths) - len(missing_test)} test images to {out_csv}")

    errors = vs.validate(out_csv, H, W, image_dir=skio.test_images_dir())
    if errors:
        print("E002.csv FAILED validate_submission.py:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("E002.csv validates OK.")

    exp_row = {
        "id": "E002",
        "date": date.today().isoformat(),
        "spec": "CODING_SPEC_02_BASELINE",
        "hypothesis": "semantic U-Net (resnet34/imagenet) + connected-components postprocess beats the classical baseline",
        "config_path": "configs/E002.yaml",
        "split_fold": "fold0 (tune/report split within)",
        "seed": 42,
        "compute": "Kaggle (T4 GPU, 6000 iters, resnet34/imagenet) + Mac (local threshold tuning)",
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
        "val_pq_mean": round(summary.overall.pq, 4),
        "val_pq_std": "",
        "sq": round(summary.overall.sq, 4),
        "rq": round(summary.overall.rq, 4),
        "secondary_metrics": json.dumps({"loss_decomposition": summary.loss_decomposition}),
        "decision": "",
        "notes": f"tuned cfg: {best_cfg}",
        "kaggle_public_score": "",
    }
    exp_csv = REPO_ROOT / "experiments.csv"
    write_header = not exp_csv.exists()
    with open(exp_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(exp_row)
    print(f"Appended E002 row to {exp_csv}")

    eval_summary_path = REPO_ROOT / "data" / "interim" / "E002_eval_summary.json"
    eval_summary_path.write_text(
        json.dumps(
            {
                "best_config": best_cfg.__dict__,
                "tune_pq": float(grid_df["pq"].max()),
                "report_overall": summary.overall.__dict__,
                "per_site_pq": summary.per_site_pq,
                "per_r_bin_pq": summary.per_r_bin_pq,
                "per_size_quartile_pq": summary.per_size_quartile,
                "loss_decomposition": summary.loss_decomposition,
                "noisy_slice_pq": summary.noisy_slice_pq,
                "non_noisy_slice_pq": summary.non_noisy_slice_pq,
                "n_test_images": len(test_paths),
                "n_submission_rows": len(sub_df),
            },
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
        )
    )
    print(f"Wrote eval summary to {eval_summary_path}")
    print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
