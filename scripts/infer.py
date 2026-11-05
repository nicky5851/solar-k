#!/usr/bin/env python3
"""Regenerate OOF and test probability maps from an already-trained checkpoint
(CODING_SPEC_06_RECALL_AND_CONNECTIVITY.md E023: flip+scale TTA re-inference).

Unlike ``scripts/train.py`` (which trains then infers once, in-process),
this script only infers, from a saved ``best.pt``, using its EMA shadow
weights -- the same weights ``train.py`` itself uses for OOF/test maps. Run
it only on RunPod or Kaggle (GPU), never the Mac: full-resolution flip+scale
TTA over hundreds of images is exactly the CPU-heavy, GPU-absent workload
CLAUDE.md's 8 GB Mac rule exists to keep off the laptop.

Usage:
  python scripts/infer.py --config configs/E007-f0.yaml \
      --checkpoint outputs/E007-f0/checkpoints/best.pt \
      --out-experiment E007tta4-f0 --tta-views 4 --tta-scales 1.0
  python scripts/infer.py --config configs/E007-f0.yaml \
      --checkpoint outputs/E007-f0/checkpoints/best.pt \
      --out-experiment E007tta4s-f0 --tta-views 4 --tta-scales 1.0,1.25
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import io as skio
from solar_k import unet as sku

# scripts/ isn't a package; import sibling module by path manipulation (same
# directory as this file, already prepended to sys.path by the shell's cwd
# or explicit invocation -- add it explicitly to be safe either way).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import fold_stems  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="Training config (for fold, encoder_name, infer_tile_size/overlap).")
    ap.add_argument("--checkpoint", type=Path, required=True, help="Path to a best.pt (or any checkpoint with a saved EMA shadow).")
    ap.add_argument("--out-experiment", required=True, help="Output experiment id; maps land under outputs/<out-experiment>/{oof,test_probs}.")
    ap.add_argument("--tta-views", type=int, default=4, choices=(3, 4))
    ap.add_argument("--tta-scales", default="1.0", help="Comma-separated scales, e.g. '1.0' or '1.0,1.25'. Must all be >= 1.0.")
    args = ap.parse_args()

    tta_scales = tuple(float(s) for s in args.tta_scales.split(","))

    cfg = yaml.safe_load(args.config.read_text())
    device = sku.pick_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    model = sku.build_model(encoder_name=cfg.get("encoder_name", "resnet34"), encoder_weights=None).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("ema") is not None:
        ema = sku.EMA(model, decay=cfg.get("ema_decay", 0.999))
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)
        print("Using EMA weights (from the checkpoint's saved shadow) for inference.")
    else:
        print("Checkpoint has no saved EMA shadow; using its raw model weights.")
    model.eval()

    all_val_stems = fold_stems(cfg["fold"], exclude=False)
    train_paths = {p.stem: p for p in skio.iter_train_images()}
    test_paths = {p.stem: p for p in skio.iter_test_images()}
    tile_size, overlap = cfg.get("infer_tile_size", 1024), cfg.get("infer_overlap", 128)

    # run_inference_and_save's tta_views param drives make_prob_fn; patch in
    # tta_scales via a thin wrapper since the shared helper predates E023.
    def run(items, out_dir):
        prob_fn = sku.make_prob_fn(model, device, tta_flips=True, tile_size=tile_size, overlap=overlap, tta_views=args.tta_views, tta_scales=tta_scales)
        model.eval()
        from solar_k import preprocess as skpre

        with torch.no_grad():
            for stem, path in items:
                processed, disk_info = skpre.load_or_build(stem, path)
                disk_mask, _ = skpre.disk_mask_and_r(processed.shape, *disk_info)
                probs = prob_fn(processed, disk_mask)
                from train import save_probs_png

                save_probs_png(probs, out_dir / f"{stem}.png")
                del processed, probs

    oof_dir = skio.output_root() / args.out_experiment / "oof"
    t0 = time.time()
    run([(s, train_paths[s]) for s in all_val_stems], oof_dir)
    print(f"Saved {len(all_val_stems)} OOF prob maps to {oof_dir} in {(time.time() - t0) / 60:.1f} min")

    test_dir = skio.output_root() / args.out_experiment / "test_probs"
    t0 = time.time()
    run(list(test_paths.items()), test_dir)
    print(f"Saved {len(test_paths)} test prob maps to {test_dir} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
