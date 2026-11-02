#!/usr/bin/env python3
"""Regenerate OOF + test probability maps from an existing checkpoint with a
different TTA view count (CODING_SPEC_05_ENSEMBLE.md E014), using the
checkpoint's EMA weights exactly as scripts/train.py does at the end of a
run. Writes outputs/<experiment_id>-tta<N>/{oof,test_probs}/<stem>.png.

Usage:
  python scripts/infer.py --config configs/E004-f0.yaml --checkpoint outputs/E004-f0/checkpoints/best.pt --tta_views 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import io as skio
from solar_k import unet as sku

from train import fold_stems, run_inference_and_save


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tta_views", type=int, choices=(3, 4), default=4)
    ap.add_argument("--skip_test", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = sku.pick_device(cfg.get("device", "auto"))
    model = sku.build_model(encoder_name=cfg.get("encoder_name", "resnet34"), encoder_weights=None).to(device)
    ckpt = sku.load_checkpoint(args.checkpoint, model, map_location=device, restore_rng=False)
    if cfg.get("ema_decay") and ckpt.get("ema") is not None:
        ema = sku.EMA(model, cfg["ema_decay"])
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(model)
        print("Using EMA weights from the checkpoint.")
    tile, overlap = cfg.get("infer_tile_size", 1024), cfg.get("infer_overlap", 128)
    out_root = skio.output_root() / f"{cfg['experiment_id']}-tta{args.tta_views}"
    train_paths = {p.stem: p for p in skio.iter_train_images()}
    val = fold_stems(cfg["fold"], exclude=False)
    t0 = time.time()
    run_inference_and_save(model, device, [(s, train_paths[s]) for s in val], out_root / "oof", tile, overlap, args.tta_views)
    print(f"{len(val)} OOF maps -> {out_root / 'oof'} in {(time.time() - t0) / 60:.1f} min")
    if not args.skip_test:
        t0 = time.time()
        items = [(p.stem, p) for p in skio.iter_test_images()]
        run_inference_and_save(model, device, items, out_root / "test_probs", tile, overlap, args.tta_views)
        print(f"{len(items)} test maps -> {out_root / 'test_probs'} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    with torch.no_grad():
        main()
