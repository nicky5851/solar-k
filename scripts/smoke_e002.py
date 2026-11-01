#!/usr/bin/env python3
"""Mac smoke test, infer+evaluate half: run the just-trained E002_smoke
checkpoint over 4 fold-0 val images and score it against GT
(CODING_SPEC_02_BASELINE.md: "train -> resume -> infer -> evaluate on 4 val
images before anything goes to Kaggle"). Not a real experiment -- the model
only saw 50 iterations, so a near-zero PQ is expected; this only proves the
checkpoint -> inference -> evaluation plumbing works.

Usage: python scripts/smoke_e002.py
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import evaluate as skeval
from solar_k import ids as skids
from solar_k import io as skio
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import targets as sktgt
from solar_k import unet as sku

REPO_ROOT = skio.repo_root()


def main() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "E002_smoke.yaml").read_text())
    ckpt_path = skio.output_root() / cfg["experiment_id"] / "checkpoints" / "best.pt"
    model, device = sku.load_model(ckpt_path, encoder_name=cfg.get("encoder_name", "resnet34"), device=cfg.get("device", "auto"))
    print(f"Loaded {ckpt_path} on {device}")
    prob_fn = sku.make_prob_fn(model, device, tta_flips=True, tile_size=512, overlap=64)

    ann = skio.load_coco_annotations()
    images_by_file = collections.defaultdict(list)
    for img in ann["images"]:
        images_by_file[img["file_name"]].append(img)
    anns_by_image_id = collections.defaultdict(list)
    for a in ann["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)

    folds = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    val_stems = folds.loc[folds["fold"] == cfg["fold"], "file_stem"].tolist()[: cfg.get("n_val_images", 4)]
    train_paths = {p.stem: p for p in skio.iter_train_images()}
    pp_cfg = skpp.PostprocessConfig.from_dict(cfg["postprocess"])

    results = []
    for stem in val_stems:
        processed, disk_info = skpre.load_or_build(stem, train_paths[stem])
        disk_mask, _ = skpre.disk_mask_and_r(processed.shape, *disk_info)
        probs = prob_fn(processed, disk_mask)
        pred_sparse = skpp.postprocess_sparse(probs, pp_cfg)
        print(f"  {stem}: n_pred={pred_sparse.shape[0]}")

        entries = images_by_file.get(f"{stem}.jpeg", [])
        for e in entries:
            annotator, _ = skids.parse_image_id(e["id"])
            anns = anns_by_image_id.get(e["id"], [])
            masks = sktgt.annotator_instance_masks(anns, *processed.shape)
            gt_masks = np.stack(masks) if masks else np.zeros((0,) + processed.shape, dtype=np.uint8)
            r = skeval.evaluate_annotator_image(stem, annotator, pred_sparse, gt_masks, disk_info=disk_info)
            results.append(r)
        del probs

    summary = skeval.aggregate_evaluation(results, quartile_edges=[1, 2, 3])
    print(f"Smoke-test PQ={summary.overall.pq:.4f} SQ={summary.overall.sq:.4f} RQ={summary.overall.rq:.4f} "
          f"(TP={summary.overall.tp} FP={summary.overall.fp} FN={summary.overall.fn})")
    print("Smoke test passed: checkpoint -> inference -> evaluation ran end to end.")


if __name__ == "__main__":
    main()
