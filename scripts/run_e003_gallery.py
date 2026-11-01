#!/usr/bin/env python3
"""Render reports/figures/E003_gallery.png: the report-half images with the
most remaining FPs under E003's final pipeline (CODING_SPEC_03_FP_AND_FOLDS.md's
"a gallery of the remaining worst FPs"), GT boundary vs kept predicted
instances, after re-deriving per-image PQ/FP counts from the already-chosen
step1/step3/step4 parameters in data/interim/E003_eval_summary.json.

Refits the scorer on `tune` only (not tune+report) so this gallery matches
the actual report-half numbers in reports/03_fp_report.md, not the
production (tune+report-refit) scorer that scripts/run_e003.py used for the
test-set submission.

Usage: python scripts/run_e003_gallery.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import components as skcomp
from solar_k import evaluate as skeval
from solar_k import io as skio
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc
from solar_k import tuning as sktune

from run_e002_eval import build_gt_for_stem, fold0_val_stems, load_annotation_index, load_prob_png
from run_e003 import build_component_cache

REPO_ROOT = skio.repo_root()
OOF_DIR = skio.output_root() / "E002" / "oof"
FIG_DIR = REPO_ROOT / "reports" / "figures"
H = W = 2048


def main() -> None:
    t_start = time.time()
    summary = json.loads((REPO_ROOT / "data" / "interim" / "E003_eval_summary.json").read_text())
    cfg1 = skpp.PostprocessConfig.from_dict(summary["best_cfg1"])
    cfg_components = skpp.PostprocessConfig(
        mode=cfg1.mode, t=cfg1.t, t_low=cfg1.t_low, t_high=cfg1.t_high,
        closing_radius=cfg1.closing_radius, min_area=1, max_instances=5000,
    )
    min_elong, min_len = summary["shape_gate"]["min_elongation"], summary["shape_gate"]["min_length"]
    c = summary["scorer_c"]

    images_by_file, anns_by_image_id = load_annotation_index()
    val_df = fold0_val_stems()
    train_paths_by_stem = {p.stem: p for p in skio.iter_train_images()}

    gt_by_stem, disk_info_by_stem, day_by_stem = {}, {}, {}
    for _, row in val_df.iterrows():
        stem = row["file_stem"]
        if not (OOF_DIR / f"{stem}.png").exists():
            continue
        _, disk_info = skpre.load_or_build(stem, train_paths_by_stem[stem])
        disk_info_by_stem[stem] = disk_info
        day_by_stem[stem] = row["date"]
        gt_by_stem[stem] = build_gt_for_stem(stem, images_by_file, anns_by_image_id)

    tune_stems, report_stems = sktune.split_tune_report(day_by_stem, seed=0)
    tune_stems = sorted(s for s in tune_stems if s in gt_by_stem)
    report_stems = sorted(s for s in report_stems if s in gt_by_stem)

    def compute_probs(stem: str) -> np.ndarray:
        return load_prob_png(OOF_DIR / f"{stem}.png")

    print("Rebuilding component cache for tune (to refit the scorer) + report (for the gallery) ...")
    cache_by_stem = {}
    for stem in tune_stems + report_stems:
        processed, _ = skpre.load_or_build(stem, train_paths_by_stem[stem])
        cache_by_stem[stem] = build_component_cache(stem, compute_probs(stem), processed, disk_info_by_stem[stem], cfg_components, gt_by_stem[stem])
        del processed
    print(f"  done in {time.time() - t_start:.1f}s")

    tune_df = pd.concat([cache_by_stem[s].features_df.assign(stem=s) for s in tune_stems], ignore_index=True)
    scorer = sksc.fit_scorer(tune_df, seed=0)

    def keep_mask(df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.zeros(0, dtype=bool)
        base = (df["area"] >= cfg1.min_area).to_numpy() & (df["elongation"].fillna(0) >= min_elong).to_numpy() & (df["length"].fillna(0) >= min_len).to_numpy()
        score = sksc.keep_score(scorer, df)
        return base & (score > c)

    per_stem_pq, per_stem_fp = {}, {}
    for stem in report_stems:
        cache = cache_by_stem[stem]
        df = cache.features_df
        keep = keep_mask(df)
        pred_sparse = cache.sparse[keep] if len(df) else cache.sparse
        matches = []
        fp_count = 0
        for annotator, gt_masks in gt_by_stem[stem].items():
            r = skeval.evaluate_annotator_image(stem, annotator, pred_sparse, gt_masks, disk_info=disk_info_by_stem.get(stem))
            matches.append(r.match)
            fp_count += r.match.fp
        from solar_k import metric as skmetric

        overall = skmetric.aggregate_matches(matches)
        per_stem_pq[stem] = overall.pq
        per_stem_fp[stem] = fp_count

    worst8 = sorted(report_stems, key=lambda s: (-per_stem_fp[s], per_stem_pq[s]))[:8]
    print(f"Worst 8 by remaining FP count: {[(s, per_stem_fp[s], round(per_stem_pq[s], 3)) for s in worst8]}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()
    cmap = plt.get_cmap("tab20")
    for ax, stem in zip(axes, worst8):
        img = skio.load_grayscale(train_paths_by_stem[stem])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        gt_union = np.zeros((H, W), dtype=np.uint8)
        for a_masks in gt_by_stem[stem].values():
            for m in a_masks:
                np.maximum(gt_union, m, out=gt_union)
        contours, _ = cv2.findContours(gt_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c_ in contours:
            pts = c_.reshape(-1, 2)
            ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=0.8)

        cache = cache_by_stem[stem]
        df = cache.features_df
        keep = keep_mask(df)
        pred_sparse = cache.sparse[keep] if len(df) else cache.sparse
        for k in range(pred_sparse.shape[0]):
            row = pred_sparse.getrow(k)
            cols = row.indices
            ys, xs = np.divmod(cols, W)
            ax.scatter(xs[::11], ys[::11], s=2, color=cmap(k % 20))
        ax.set_title(f"{stem}\nPQ={per_stem_pq[stem]:.3f} remaining FP={per_stem_fp[stem]}", fontsize=8)
        ax.axis("off")
    fig.suptitle("E003 report-half: 8 images with the most remaining FPs after the full pipeline. Black=GT boundary, colour=kept predicted instance.")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "E003_gallery.png", dpi=110)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'E003_gallery.png'}")
    print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
