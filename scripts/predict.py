#!/usr/bin/env python3
"""Run a model (classical or trained U-Net) over a folder of images ->
instances -> CSV -> validate_submission.py.

Usage:
  python scripts/predict.py --config configs/E001.yaml \
      --images-dir data/raw/test/test_images --out outputs/submissions/E001.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import classical as skcl
from solar_k import io as skio
from solar_k import postprocess as skpp
from solar_k import predict as skpred

import validate_submission as vs


def build_prob_fn(model_cfg: dict):
    mtype = model_cfg["type"]
    if mtype == "classical":
        kernel_radius = model_cfg.get("kernel_radius", 9)
        return lambda processed, disk_mask: skcl.dark_feature_map(processed, disk_mask, kernel_radius=kernel_radius)
    if mtype == "unet":
        from solar_k import unet as sku

        model, device = sku.load_model(
            model_cfg["checkpoint"], encoder_name=model_cfg.get("encoder_name", "resnet34"), device=model_cfg.get("device", "auto")
        )
        return sku.make_prob_fn(model, device, tta_flips=model_cfg.get("tta_flips", True))
    raise ValueError(f"unknown model type: {mtype!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, default=None, help="default: data/raw/test/test_images")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-cache", action="store_true", help="don't read/write the preprocessed-image cache")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    prob_fn = build_prob_fn(cfg["model"])
    pp_cfg = skpp.PostprocessConfig.from_dict(cfg["postprocess"])

    images_dir = args.images_dir or skio.test_images_dir()
    paths = list(skio.iter_image_paths(images_dir))

    df, timings = skpred.run_over_paths(prob_fn, paths, pp_cfg, use_cache=not args.no_cache)
    skpred.write_submission(df, args.out)
    mean_t = sum(timings.values()) / max(len(timings), 1)
    print(f"Wrote {len(df)} rows for {len(paths)} images to {args.out}; mean inference {mean_t:.3f}s/image")

    errors = vs.validate(args.out, 2048, 2048, image_dir=images_dir)
    if errors:
        print("Submission validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("Submission validates OK.")


if __name__ == "__main__":
    main()
