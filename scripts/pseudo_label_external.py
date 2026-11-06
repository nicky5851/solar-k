#!/usr/bin/env python3
"""CODING_SPEC_07 E028: generate per-fold pseudo-labels for the external
unlabelled GONG images.

Leakage safety: for model fold k, pseudo-labels are produced using *only*
that fold's own E007-f{k} and E015-f{k} checkpoints (averaged, same TTA
recipe as E024: 4-flip views + scales 1.0/1.25). Those two models never saw
fold k's held-out validation images during training, so nothing about this
process can leak fold k's own OOF into training signal -- and since every
external image already excludes any date within +/-3 days of *any* real
train/test image (scripts/fetch_external_gong.py), there's no cross-fold
leakage via near-duplicate external frames either.

Confidence handling: per pixel, confidence = max(p, 1-p) (distance from the
decision boundary, rescaled to [0.5, 1] -> this script stores it as-is and
callers derive the ignore-mask). An external image is dropped entirely if
its mean confidence (over the disk region) falls in the bottom 20% of the
whole candidate pool's mean-confidence distribution, *computed separately
per fold* (so the drop threshold itself can't leak information between
folds).

Outputs (gitignored, like all of data/): for each kept stem,
  data/external/pseudo/E028-f{k}/<stem>.png       -- uint8 soft target (p*255)
and a small per-fold manifest of which stems were kept/dropped + their mean
confidence, for inspection:
  data/external/pseudo/E028-f{k}/_kept_stems.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import io as skio
from solar_k import preprocess as skpre
from solar_k import unet as sku

TTA_VIEWS = 4
TTA_SCALES = (1.0, 1.25)
DROP_BOTTOM_PCT = 20.0


def load_fold_members(fold: int, device: torch.device, tile_size: int, overlap: int):
    """E007-f{k} and E015-f{k}, EMA weights if present -- exactly how
    scripts/infer.py loads a checkpoint for inference."""
    prob_fns = []
    for member in ("E007", "E015"):
        ckpt_path = skio.output_root() / f"{member}-f{fold}" / "checkpoints" / "best.pt"
        encoder = "efficientnet-b4" if member == "E007" else "tu-regnety_040"
        model = sku.build_model(encoder_name=encoder, encoder_weights=None).to(device)
        ckpt = sku.load_checkpoint(ckpt_path, model, map_location=device, restore_rng=False)
        if ckpt.get("ema") is not None:
            ema = sku.EMA(model, 0.999)
            ema.load_state_dict(ckpt["ema"])
            ema.copy_to(model)
        model.eval()
        prob_fns.append(sku.make_prob_fn(model, device, tta_flips=True, tile_size=tile_size, overlap=overlap,
                                          tta_views=TTA_VIEWS, tta_scales=TTA_SCALES))
    return prob_fns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--manifest", type=Path, default=skio.repo_root() / "data" / "external" / "gong_unlabelled_manifest.csv")
    ap.add_argument("--images-dir", type=Path, default=skio.data_root() / "external" / "images")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = args.out_dir or (skio.data_root() / "external" / "pseudo" / f"E028-f{args.fold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open() as f:
        stems = [row["stem"] for row in csv.DictReader(f)]
    print(f"fold {args.fold}: {len(stems)} candidate external images")

    device = sku.pick_device(args.device)
    print(f"Device: {device}")
    prob_fns = load_fold_members(args.fold, device, args.tile_size, args.overlap)

    mean_conf_by_stem: dict[str, float] = {}
    t0 = time.time()
    with torch.no_grad():
        for i, stem in enumerate(stems):
            img_path = args.images_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue
            processed, disk_info = skpre.load_or_build(stem, img_path)
            disk_mask, _ = skpre.disk_mask_and_r(processed.shape, *disk_info)
            probs = np.mean([fn(processed, disk_mask) for fn in prob_fns], axis=0)
            confidence = np.maximum(probs, 1.0 - probs)
            mean_conf = float(confidence[disk_mask].mean()) if disk_mask.any() else 0.0
            mean_conf_by_stem[stem] = mean_conf
            np.save(out_dir / f"{stem}__probs.npy", (probs * 255).astype(np.uint8))
            del processed, probs, confidence
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(stems)} scored ({(time.time() - t0) / 60:.1f} min)", flush=True)

    if not mean_conf_by_stem:
        print("No external images found -- nothing to pseudo-label.")
        return

    confidences = np.array(list(mean_conf_by_stem.values()))
    drop_threshold = float(np.percentile(confidences, DROP_BOTTOM_PCT))
    print(f"fold {args.fold}: mean-confidence drop threshold (bottom {DROP_BOTTOM_PCT}%) = {drop_threshold:.4f}")

    kept, dropped = [], []
    for stem, conf in mean_conf_by_stem.items():
        probs_path = out_dir / f"{stem}__probs.npy"
        if conf < drop_threshold:
            dropped.append((stem, conf))
            probs_path.unlink(missing_ok=True)
            continue
        probs_u8 = np.load(probs_path)
        cv2.imwrite(str(out_dir / f"{stem}.png"), probs_u8)
        probs_path.unlink(missing_ok=True)
        kept.append((stem, conf))

    with (out_dir / "_kept_stems.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "mean_confidence", "kept"])
        for stem, conf in kept:
            w.writerow([stem, f"{conf:.4f}", True])
        for stem, conf in dropped:
            w.writerow([stem, f"{conf:.4f}", False])

    print(f"fold {args.fold}: kept {len(kept)}/{len(mean_conf_by_stem)} "
          f"({(time.time() - t0) / 60:.1f} min total) -> {out_dir}")


if __name__ == "__main__":
    main()
