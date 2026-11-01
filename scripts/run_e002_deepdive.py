#!/usr/bin/env python3
"""FP/FN deep-dive for E002 on the report half (CODING_SPEC_02_BASELINE.md's
H1 verdict needs more than counts -- this classifies every FP/FN instance
into a cause and gives each cause's effect on PQ).

Reuses the exact tune/report split and tuned PostprocessConfig that
scripts/run_e002_eval.py already produced (data/interim/E002_eval_summary.json),
so nothing here re-tunes or peeks at the report half for threshold choice.

FP classes (checked in this order, first match wins):
  - near_miss: best IoU with this annotator's own GT is in [0.3, 0.5) -- the
    right filament, IoU just under the strict panoptic threshold.
  - annotator_disagreement: IoU > 0.5 with a *different* annotator's GT on
    the same physical image -- the model agrees with someone, just not the
    annotator metric.py happens to be scoring against.
  - fragment: no match, but weakly overlaps (IoU > 0.1) a GT instance of
    this annotator that *is* already matched (TP) by a different
    prediction -- an extra fragment carved off an instance the model found.
  - isolated: none of the above -- a spurious detection with no GT nearby.

FN classes:
  - missed_only_this_annotator: unmatched for this annotator, but the same
    region (weak IoU > 0.1) was matched (TP) for at least one *other*
    annotator of the same image -- an annotator-identity/boundary issue,
    not a true detection miss.
  - missed_by_all: unmatched for every annotator who drew anything there
    (including single-annotator images, trivially).

Usage: python scripts/run_e002_deepdive.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import evaluate as skeval
from solar_k import io as skio
from solar_k import metric as skmetric
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import tuning as sktune

from run_e002_eval import build_gt_for_stem, fold0_val_stems, load_annotation_index, load_prob_png

H = W = 2048
REPO_ROOT = skio.repo_root()
OOF_DIR = skio.output_root() / "E002" / "oof"
IOU_T = skmetric.IOU_THRESHOLD
WEAK_IOU = 0.1
NEAR_MISS = (0.3, 0.5)
FIG_DIR = REPO_ROOT / "reports" / "figures"

# The full val_dice-vs-iteration series parsed from the Kaggle kernel output
# log (kaggle kernels output potcharapanchumrak/solar-k-e002-train), training
# run completed 2026-11-01. That raw log lives outside the repo (it is
# training/Kaggle-run output, not source); these numbers are the complete,
# unedited series -- every checkpoint eval train.py ran, in order.
TRAIN_CURVE_FALLBACK = [
    (500, 0.1733, 0.1733), (1000, 0.0551, 0.1733), (1500, 0.0774, 0.1733),
    (2000, 0.1822, 0.1822), (2500, 0.1352, 0.1822), (3000, 0.0725, 0.1822),
    (3500, 0.1326, 0.1822), (4000, 0.0946, 0.1822), (4500, 0.0881, 0.1822),
    (5000, 0.0998, 0.1822), (5500, 0.0644, 0.1822), (6000, 0.0813, 0.1822),
]


def parse_train_curve(log_path: Path | None) -> list[tuple[int, float, float]]:
    if log_path is None or not log_path.exists():
        return TRAIN_CURVE_FALLBACK
    text = log_path.read_text()
    pat = re.compile(r"iter (\d+)/\d+ loss=[\d.]+ val_dice=([\d.]+) best=([\d.]+)")
    seen, rows = set(), []
    for m in pat.finditer(text):
        it = int(m.group(1))
        if it in seen:
            continue
        seen.add(it)
        rows.append((it, float(m.group(2)), float(m.group(3))))
    rows.sort()
    return rows or TRAIN_CURVE_FALLBACK


def bbox_pad(mask: np.ndarray, pad: int = 2) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (max(0, ys.min() - pad), min(H - 1, ys.max() + pad), max(0, xs.min() - pad), min(W - 1, xs.max() + pad))


def instance_length(mask: np.ndarray) -> float:
    box = bbox_pad(mask)
    if box is None:
        return float("nan")
    y0, y1, x0, x1 = box
    skel = skeletonize(mask[y0 : y1 + 1, x0 : x1 + 1].astype(bool))
    return float(skel.sum())


def local_contrast(mask: np.ndarray, processed: np.ndarray) -> float:
    """mean(ring around the mask) - mean(mask) on the preprocessed image.
    Positive = the instance is darker than its immediate surroundings, the
    expected sign for a dark absorption feature; our own instance-level
    proxy, not the EDA's whole-image std/mean contrast."""
    import cv2

    box = bbox_pad(mask, pad=6)
    if box is None:
        return float("nan")
    y0, y1, x0, x1 = box
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
    crop_img = processed[y0 : y1 + 1, x0 : x1 + 1]
    ring = cv2.dilate(crop_mask, np.ones((5, 5), np.uint8)) & (1 - crop_mask)
    if not ring.any() or not crop_mask.any():
        return float("nan")
    return float(crop_img[ring.astype(bool)].mean() - crop_img[crop_mask.astype(bool)].mean())


def r_over_r(mask: np.ndarray, cx: float, cy: float, radius: float) -> float:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0 or radius <= 0:
        return float("nan")
    return float(np.sqrt((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2) / radius)


def main() -> None:
    t_start = time.time()
    summary = json.loads((REPO_ROOT / "data" / "interim" / "E002_eval_summary.json").read_text())
    best_cfg = skpp.PostprocessConfig.from_dict(summary["best_config"])
    print(f"Using tuned config: {best_cfg}")

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

    _, report_stems = sktune.split_tune_report(day_by_stem, seed=0)
    report_stems = sorted(s for s in report_stems if s in gt_by_stem)
    print(f"Report half: {len(report_stems)} images")

    fp_rows, fn_rows, uniqueness_violations = [], [], 0
    stem_matches: dict[str, list] = collections.defaultdict(list)

    for si, stem in enumerate(report_stems):
        probs = load_prob_png(OOF_DIR / f"{stem}.png")
        processed, disk_info = skpre.load_or_build(stem, train_paths_by_stem[stem])
        cx, cy, radius = disk_info
        pred_masks_list = skpp.postprocess(probs, best_cfg)
        pred_stack = np.stack(pred_masks_list) if pred_masks_list else np.zeros((0, H, W), dtype=np.uint8)

        gt = gt_by_stem[stem]
        annotators = list(gt.keys())
        iou_by_ann: dict[str, np.ndarray] = {}
        hit_by_ann: dict[str, np.ndarray] = {}
        tp_gt_by_ann: dict[str, set] = {}

        for a in annotators:
            gt_masks = gt[a]
            if gt_masks.shape[0] == 0 or pred_stack.shape[0] == 0:
                iou = np.zeros((gt_masks.shape[0], pred_stack.shape[0]))
            else:
                iou, _ = skmetric.pairwise_iou_dice(gt_masks, pred_stack)
            hit = iou > IOU_T
            if ((hit.sum(axis=1) > 1).any()) or ((hit.sum(axis=0) > 1).any()):
                uniqueness_violations += 1
            iou_by_ann[a], hit_by_ann[a] = iou, hit
            tp_gt_by_ann[a] = set(np.where(hit.any(axis=1))[0]) if hit.size else set()

            match = skmetric.match_scores(iou, np.zeros_like(iou), gt_masks.shape[0], pred_stack.shape[0])
            stem_matches[stem].append(match)

        # Cross-annotator GT-GT IoU, for the FN missed-only-this-annotator check.
        cross_iou: dict[tuple[str, str], np.ndarray] = {}
        for a in annotators:
            for b in annotators:
                if a == b:
                    continue
                ga, gb = gt[a], gt[b]
                if ga.shape[0] == 0 or gb.shape[0] == 0:
                    cross_iou[(a, b)] = np.zeros((ga.shape[0], gb.shape[0]))
                else:
                    cross_iou[(a, b)], _ = skmetric.pairwise_iou_dice(ga, gb)

        for a in annotators:
            gt_masks = gt[a]
            iou, hit = iou_by_ann[a], hit_by_ann[a]
            n_gt, n_pred = iou.shape
            fp_idx = np.where(~hit.any(axis=0))[0] if n_pred else np.array([], dtype=int)
            fn_idx = np.where(~hit.any(axis=1))[0] if n_gt else np.array([], dtype=int)
            tp_gt_idx = tp_gt_by_ann[a]

            for j in fp_idx:
                pred_mask = pred_stack[j]
                best_same = iou[:, j].max() if n_gt else 0.0
                if NEAR_MISS[0] <= best_same < NEAR_MISS[1]:
                    cls = "near_miss"
                elif any(
                    iou_by_ann[b].shape[0] and iou_by_ann[b][:, j].max() > IOU_T for b in annotators if b != a
                ):
                    cls = "annotator_disagreement"
                elif n_gt and tp_gt_idx and (iou[list(tp_gt_idx), j] > WEAK_IOU).any():
                    cls = "fragment"
                else:
                    cls = "isolated"
                fp_rows.append(
                    {
                        "stem": stem, "annotator": a, "class": cls,
                        "area": int(pred_mask.sum()),
                        "mean_prob": float(probs[pred_mask.astype(bool)].mean()),
                        "max_prob": float(probs[pred_mask.astype(bool)].max()),
                        "length": instance_length(pred_mask),
                        "contrast": local_contrast(pred_mask, processed),
                        "r_over_R": r_over_r(pred_mask, cx, cy, radius),
                    }
                )

            for i in fn_idx:
                gt_mask = gt_masks[i]
                missed_by_all = True
                for b in annotators:
                    if b == a:
                        continue
                    row = cross_iou[(a, b)][i]
                    overlapping_b = np.where(row > WEAK_IOU)[0]
                    if len(overlapping_b) and any(k in tp_gt_by_ann[b] for k in overlapping_b):
                        missed_by_all = False
                        break
                cls = "missed_only_this_annotator" if not missed_by_all else "missed_by_all"
                fn_rows.append(
                    {
                        "stem": stem, "annotator": a, "class": cls,
                        "area": int(gt_mask.sum()),
                        "mean_prob": float(probs[gt_mask.astype(bool)].mean()),
                        "max_prob": float(probs[gt_mask.astype(bool)].max()),
                        "length": instance_length(gt_mask),
                        "contrast": local_contrast(gt_mask, processed),
                        "r_over_R": r_over_r(gt_mask, cx, cy, radius),
                    }
                )
        del probs, processed, pred_stack, pred_masks_list
        if (si + 1) % 10 == 0 or si + 1 == len(report_stems):
            print(f"  deepdive: stem {si + 1}/{len(report_stems)} ({stem}) done, elapsed {time.time() - t_start:.1f}s", flush=True)

    fp_df = pd.DataFrame(fp_rows)
    fn_df = pd.DataFrame(fn_rows)
    fp_df.to_csv(REPO_ROOT / "data" / "interim" / "E002_fp_deepdive.csv", index=False)
    fn_df.to_csv(REPO_ROOT / "data" / "interim" / "E002_fn_deepdive.csv", index=False)
    print(f"Wrote {len(fp_df)} FP rows and {len(fn_df)} FN rows.")
    print(f"IoU>0.5 uniqueness-violation count (should be 0): {uniqueness_violations}")

    overall = skmetric.aggregate_matches([m for ms in stem_matches.values() for m in ms])
    tp_ious_all = list(overall.tp_ious)

    def pq_without(fp_drop: int, fn_drop: int) -> float:
        tp = len(tp_ious_all)
        fp = overall.fp - fp_drop
        fn = overall.fn - fn_drop
        denom = tp + 0.5 * fp + 0.5 * fn
        return sum(tp_ious_all) / denom if denom > 0 else 0.0

    actual_pq = overall.pq
    fp_class_impact = {}
    for cls, n in fp_df["class"].value_counts().items():
        fp_class_impact[cls] = {"count": int(n), "pq_gain_if_eliminated": round(pq_without(int(n), 0) - actual_pq, 4)}
    fn_class_impact = {}
    for cls, n in fn_df["class"].value_counts().items():
        fn_class_impact[cls] = {"count": int(n), "pq_gain_if_eliminated": round(pq_without(0, int(n)) - actual_pq, 4)}

    # Per-stem PQ, for the report gallery's worst-8/best-4 ranking.
    per_stem_pq = {}
    for stem, matches in stem_matches.items():
        r = skmetric.aggregate_matches(matches)
        per_stem_pq[stem] = {"pq": r.pq, "tp": r.tp, "fp": r.fp, "fn": r.fn}
    ranked = sorted(per_stem_pq.items(), key=lambda kv: kv[1]["pq"])
    worst8 = [s for s, _ in ranked[:8]]
    best4 = [s for s, _ in ranked[-4:]]

    deepdive_summary = {
        "actual_report_pq": actual_pq,
        "fp_class_impact": fp_class_impact,
        "fn_class_impact": fn_class_impact,
        "fp_area_stats_by_class": {c: fp_df[fp_df["class"] == c]["area"].describe().to_dict() for c in fp_df["class"].unique()},
        "fp_prob_stats_by_class": {c: fp_df[fp_df["class"] == c][["mean_prob", "max_prob"]].describe().to_dict() for c in fp_df["class"].unique()},
        "fn_area_stats_by_class": {c: fn_df[fn_df["class"] == c]["area"].describe().to_dict() for c in fn_df["class"].unique()},
        "fn_prob_stats_by_class": {c: fn_df[fn_df["class"] == c][["mean_prob", "max_prob"]].describe().to_dict() for c in fn_df["class"].unique()},
        "uniqueness_violations": uniqueness_violations,
        "per_stem_pq": per_stem_pq,
        "gallery_worst8": worst8,
        "gallery_best4": best4,
    }
    (REPO_ROOT / "data" / "interim" / "E002_deepdive_summary.json").write_text(
        json.dumps(deepdive_summary, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    )
    print("Wrote data/interim/E002_deepdive_summary.json")

    # --- Figures ---------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    curve = parse_train_curve(None)  # no on-disk Kaggle log this run -> fallback series (see module docstring)
    iters, dices, bests = zip(*curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, dices, "o-", label="val_dice (this checkpoint)")
    ax.plot(iters, bests, "--", label="best so far")
    ax.axvline(iters[int(np.argmax(dices))], color="gray", linestyle=":", label="best checkpoint")
    ax.set_xlabel("iteration")
    ax.set_ylabel("pixel Dice @ t=0.5 (fold-0 val)")
    ax.set_title("E002: val Dice vs iteration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "E002_val_dice_vs_iter.png", dpi=120)
    plt.close(fig)

    grid_df = pd.read_csv(REPO_ROOT / "data" / "interim" / "E002_grid_search.csv")
    sub = grid_df[
        (grid_df["mode"] == "threshold")
        & (grid_df["closing_radius"] == best_cfg.closing_radius)
        & (grid_df["min_area"] == best_cfg.min_area)
    ].sort_values("t")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sub["t"], sub["pq"], "o-")
    ax.axvline(best_cfg.t, color="red", linestyle="--", label=f"chosen t={best_cfg.t}")
    ax.set_xlabel("threshold t")
    ax.set_ylabel("PQ (tune half)")
    ax.set_title(f"E002: PQ vs threshold (closing_radius={best_cfg.closing_radius}, min_area={best_cfg.min_area})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "E002_pq_vs_threshold.png", dpi=120)
    plt.close(fig)

    # Gallery: 8 worst + 4 best report-half images, GT boundary (black) vs
    # predicted instances (tab20, one colour each), titled with per-image PQ
    # and dominant failure class among that stem's FP/FN rows.
    import cv2

    def dominant_class(stem: str) -> str:
        sub_fp = fp_df[fp_df["stem"] == stem]["class"]
        sub_fn = fn_df[fn_df["stem"] == stem]["class"]
        counts = pd.concat([sub_fp, sub_fn]).value_counts()
        return counts.index[0] if len(counts) else "none"

    gallery_stems = worst8 + best4
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    axes = axes.ravel()
    cmap = plt.get_cmap("tab20")
    for ax, stem in zip(axes, gallery_stems):
        img = skio.load_grayscale(train_paths_by_stem[stem])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        gt_union = np.zeros((H, W), dtype=np.uint8)
        for a_masks in gt_by_stem[stem].values():
            for m in a_masks:
                np.maximum(gt_union, m, out=gt_union)
        contours, _ = cv2.findContours(gt_union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = c.reshape(-1, 2)
            ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=0.8)
        probs = load_prob_png(OOF_DIR / f"{stem}.png")
        for k, m in enumerate(skpp.postprocess(probs, best_cfg)):
            ys, xs = np.nonzero(m)
            ax.scatter(xs[::29], ys[::29], s=1.5, color=cmap(k % 20))
        del probs
        pq = per_stem_pq[stem]["pq"]
        ax.set_title(f"{stem}\nPQ={pq:.3f} dominant={dominant_class(stem)}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(gallery_stems):]:
        ax.axis("off")
    fig.suptitle("E002 report-half gallery: 8 worst (top 2 rows) + 4 best (bottom row). Black=GT boundary, colour=predicted instance.")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "E002_gallery.png", dpi=110)
    plt.close(fig)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
