#!/usr/bin/env python3
"""Train the E002 semantic U-Net (CODING_SPEC_02_BASELINE.md, Stage C).

Fixed-iteration training (not epoch-based): AdamW + cosine schedule with
warmup, BCE + soft Dice on the soft-consensus target, checkpointed every
`checkpoint_every` iterations with full resume state (model, optimizer,
scheduler, RNG, iteration count). Validates on fold-0 val by pixel Dice at
t=0.5 on downsampled full images (cheap) and keeps the best checkpoint.

In the same run, once training stops (iteration limit or --max_hours),
saves full-resolution flip-TTA probability maps for fold-0 val (OOF) and
the test set as compressed uint8 PNGs -- no postprocessing/tuning here,
that happens locally against these cached probs.

Paths: checkpoints/logs/probs go under `solar_k.io.output_root()`
(`SOLAR_K_OUTPUT`, default `<repo_root>/outputs`) and the preprocessing
cache under `solar_k.io.cache_root()` (`SOLAR_K_CACHE`, default
`<data_root>/processed`) -- on Kaggle, set `SOLAR_K_OUTPUT=/kaggle/working/outputs`
and `SOLAR_K_CACHE=/kaggle/temp/...` so results land directly in the
persisted working directory and the cache sits on fast local disk, with no
copy-at-the-end step needed.

Usage:
  python scripts/train.py --config configs/E002.yaml
  python scripts/train.py --config configs/E002.yaml --resume            # this experiment's own last.pt
  python scripts/train.py --config configs/E002.yaml --resume /path/to/last.pt  # e.g. a prior Kaggle version's output
  python scripts/train.py --config configs/E002.yaml --max_hours 8.5     # save + exit cleanly at the wall-clock limit
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import dataset as skds
from solar_k import io as skio
from solar_k import preprocess as skpre
from solar_k import sampler as sksamp
from solar_k import unet as sku


def load_annotation_index():
    ann = skio.load_coco_annotations()
    images_by_file = collections.defaultdict(list)
    for img in ann["images"]:
        images_by_file[img["file_name"]].append(img)
    anns_by_image_id = collections.defaultdict(list)
    for a in ann["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)
    return images_by_file, anns_by_image_id


def fold_stems(fold: int, exclude: bool = False) -> list[str]:
    import pandas as pd

    folds = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    mask = folds["fold"] != fold if exclude else folds["fold"] == fold
    return folds.loc[mask, "file_stem"].tolist()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def pixel_dice_at_threshold(prob_map: np.ndarray, target: np.ndarray, t: float = 0.5) -> float:
    pred = prob_map >= t
    gt = target >= 0.5
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def _load_processed_and_target(stem: str, train_paths: dict, images_by_file: dict, anns_by_image_id: dict) -> tuple[np.ndarray, np.ndarray]:
    from solar_k import ids as skids
    from solar_k import targets as sktgt

    processed, _ = skpre.load_or_build(stem, train_paths[stem])

    entries = images_by_file.get(f"{stem}.jpeg", [])
    anns_by_annotator = {}
    for e in entries:
        annotator, _ = skids.parse_image_id(e["id"])
        anns_by_annotator[annotator] = anns_by_image_id.get(e["id"], [])
    target = sktgt.soft_consensus_target(anns_by_annotator, processed.shape[0], processed.shape[1])
    return processed, target


def run_pq_validation(model, device, stems, train_paths, images_by_file, anns_by_image_id, pp_cfg, tile_size: int, overlap: int) -> float:
    """CODING_SPEC_03_FP_AND_FOLDS.md E004: checkpoint selection by instance
    PQ instead of pixel Dice -- full-resolution flip-TTA inference + the
    fixed postprocess config (E002's tuned settings by default) on a small
    fixed val subset, scored the same way the report scripts do."""
    from solar_k import evaluate as skeval
    from solar_k import ids as skids
    from solar_k import metric as skmetric
    from solar_k import postprocess as skpp
    from solar_k import targets as sktgt

    model.eval()
    prob_fn = sku.make_prob_fn(model, device, tta_flips=True, tile_size=tile_size, overlap=overlap)
    matches = []
    with torch.no_grad():
        for stem in stems:
            processed, disk_info = skpre.load_or_build(stem, train_paths[stem])
            disk_mask, _ = skpre.disk_mask_and_r(processed.shape, *disk_info)
            probs = prob_fn(processed, disk_mask)
            pred_sparse = skpp.postprocess_sparse(probs, pp_cfg)
            for e in images_by_file.get(f"{stem}.jpeg", []):
                annotator, _ = skids.parse_image_id(e["id"])
                inst_masks = sktgt.annotator_instance_masks(anns_by_image_id.get(e["id"], []), *processed.shape)
                gt_masks = np.stack(inst_masks) if inst_masks else np.zeros((0, *processed.shape), dtype=np.uint8)
                r = skeval.evaluate_annotator_image(stem, annotator, pred_sparse, gt_masks, disk_info=disk_info)
                matches.append(r.match)
            del probs, pred_sparse
    model.train()
    return skmetric.aggregate_matches(matches).pq if matches else 0.0


def downsampled_validate(model, device, val_stems, train_paths, images_by_file, anns_by_image_id, size: int = 512) -> float:
    model.eval()
    dices = []
    with torch.no_grad():
        for stem in val_stems:
            processed, target = _load_processed_and_target(stem, train_paths, images_by_file, anns_by_image_id)
            small_img = cv2.resize(processed, (size, size), interpolation=cv2.INTER_AREA)
            small_tgt = cv2.resize(target, (size, size), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(small_img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            probs = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            dices.append(pixel_dice_at_threshold(probs, small_tgt))
    model.train()
    return float(np.mean(dices)) if dices else 0.0


def build_optimizer_scheduler(model, lr, weight_decay, warmup_steps, total_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def _worker_init_fn(_worker_id: int) -> None:
    # Forked DataLoader workers inherit an identical copy of the dataset's
    # random.Random state; without reseeding, every worker draws the exact
    # same sequence of crops. DataLoader already derives a unique `seed`
    # per worker, so reuse that.
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(info.seed % (2**32))


def save_probs_png(probs: np.ndarray, path: Path) -> None:
    """Compressed uint8 probability map -- PNG is lossless and naturally
    compressed, and stays directly viewable for sanity-checking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_u8 = np.clip(probs * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), arr_u8)


def run_inference_and_save(model, device, items: list[tuple[str, Path]], out_dir: Path, tile_size: int, overlap: int, tta_views: int = 3) -> None:
    """``items`` is a list of (stem, raw_image_path). No postprocessing or
    tuning here by design -- that runs locally against these saved probs
    (CODING_SPEC_02_BASELINE.md's Kaggle path). ``tta_views``: see
    ``unet.make_prob_fn`` (3 = every map saved before CODING_SPEC_05)."""
    prob_fn = sku.make_prob_fn(model, device, tta_flips=True, tile_size=tile_size, overlap=overlap, tta_views=tta_views)
    model.eval()
    with torch.no_grad():
        for stem, path in items:
            processed, disk_info = skpre.load_or_build(stem, path)
            disk_mask, _ = skpre.disk_mask_and_r(processed.shape, *disk_info)
            probs = prob_fn(processed, disk_mask)
            save_probs_png(probs, out_dir / f"{stem}.png")
            del processed, probs
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--resume", nargs="?", const="__default__", default=None,
        help="Resume from this experiment's own last.pt (bare flag) or from an explicit checkpoint path "
             "(e.g. a previous Kaggle notebook version's output attached under /kaggle/input).",
    )
    ap.add_argument("--max_hours", type=float, default=None, help="Save a checkpoint and exit cleanly once this wall-clock budget is used up.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    device = sku.pick_device(cfg.get("device", "auto"))
    print(f"Device: {device}")

    images_by_file, anns_by_image_id = load_annotation_index()
    train_stems = fold_stems(cfg["fold"], exclude=True)
    all_val_stems = fold_stems(cfg["fold"], exclude=False)
    val_stems_for_training = all_val_stems[: cfg.get("n_val_images", 4)]

    train_paths = {p.stem: p for p in skio.iter_train_images()}
    test_paths = {p.stem: p for p in skio.iter_test_images()}

    if cfg.get("prebuild_cache", True):
        stems_with_paths = [(s, train_paths[s]) for s in set(train_stems) | set(all_val_stems)]
        stems_with_paths += list(test_paths.items())
        t0 = time.time()
        n_built = skpre.build_cache(stems_with_paths, workers=cfg.get("cache_workers", 1))
        print(f"Preprocessing cache: built {n_built}/{len(stems_with_paths)} new entries in {time.time() - t0:.1f}s")

    num_workers = cfg.get("num_workers", 0)

    # CODING_SPEC_04_MODEL_SIDE.md E006: hard-negative-centred crops, mixed in
    # via an explicit sampler_mix config. Absent -> sampler.DEFAULT_MIX, which
    # reproduces E002/E004's fixed 2/3 positive, 1/3 random split exactly.
    sampler_mix = cfg.get("sampler_mix")
    crop_sampler = None
    if sampler_mix:
        hard_negatives_csv = cfg.get("hard_negatives_csv")
        hard_negatives_by_stem = {}
        if hard_negatives_csv:
            hard_negatives_by_stem = sksamp.load_hard_negatives(Path(hard_negatives_csv), fold_stems=set(train_stems))
            print(f"Loaded hard negatives for {len(hard_negatives_by_stem)}/{len(train_stems)} training stems")
        crop_sampler = sksamp.CropCenterSampler(mix=sampler_mix, hard_negatives_by_stem=hard_negatives_by_stem)

    train_ds = skds.SolarCropDataset(
        train_stems, images_by_file, anns_by_image_id,
        crop_size=cfg.get("crop_size", 512), length=cfg.get("dataset_length"), seed=cfg.get("seed", 42),
        sampler=crop_sampler,
    )
    loader = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True, num_workers=num_workers,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    model = sku.build_model(encoder_name=cfg.get("encoder_name", "resnet34")).to(device)
    total_iters = cfg.get("total_iterations", 6000)
    optimizer, scheduler = build_optimizer_scheduler(
        model, cfg.get("lr", 3e-4), cfg.get("weight_decay", 1e-4), cfg.get("warmup_steps", 300), total_iters
    )

    # CODING_SPEC_03_FP_AND_FOLDS.md E004: EMA of weights, used for both the
    # periodic validation metric and what gets saved to best.pt/last.pt.
    # Opt-in via config (absent -> old E002 behaviour: validate/save the raw
    # model), so configs/E002.yaml stays reproducible unchanged.
    ema_decay = cfg.get("ema_decay")
    ema = sku.EMA(model, ema_decay) if ema_decay else None
    eval_model = sku.build_model(encoder_name=cfg.get("encoder_name", "resnet34"), encoder_weights=None).to(device) if ema is not None else None

    checkpoint_select_by = cfg.get("checkpoint_select_by", "pixel_dice")
    pq_val_stems, pq_postprocess_cfg = None, None
    if checkpoint_select_by == "pq":
        from solar_k import postprocess as skpp

        pq_val_stems = all_val_stems[: cfg.get("n_pq_val_images", 40)]
        pp = cfg.get("checkpoint_postprocess", {})
        pq_postprocess_cfg = skpp.PostprocessConfig(
            mode=pp.get("mode", "threshold"), t=pp.get("t", 0.7),
            closing_radius=pp.get("closing_radius", 2), min_area=pp.get("min_area", 418),
        )

    ckpt_dir = skio.output_root() / cfg["experiment_id"] / "checkpoints"
    last_ckpt_path = ckpt_dir / "last.pt"
    best_ckpt_path = ckpt_dir / "best.pt"

    start_iter = 0
    best_metric = -1.0
    if args.resume is not None:
        resume_path = last_ckpt_path if args.resume == "__default__" else Path(args.resume)
        if resume_path.exists():
            ckpt = sku.load_checkpoint(resume_path, model, optimizer, scheduler, map_location=device, ema=ema)
            start_iter = ckpt["iteration"]
            best_metric = ckpt.get("best_metric", -1.0)
            print(f"Resumed from {resume_path} at iteration {start_iter}")
        else:
            print(f"--resume path {resume_path} does not exist; starting fresh")

    checkpoint_every = cfg.get("checkpoint_every", 500)
    log_every = cfg.get("log_every", 100)
    use_amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    data_iter = iter(loader)
    run_start = time.time()
    log_t, log_iter = run_start, start_iter
    it = start_iter
    stopped_early = False
    while it < total_iters:
        try:
            imgs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, targets = next(data_iter)
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
            loss = F.binary_cross_entropy_with_logits(logits, targets) + soft_dice_loss(logits, targets)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        it += 1
        if ema is not None:
            ema.update(model)

        if it % log_every == 0:
            now = time.time()
            its = (it - log_iter) / max(now - log_t, 1e-6)
            eta_min = (total_iters - it) / its / 60 if its > 0 else float("nan")
            print(f"iter {it}/{total_iters} loss={loss.item():.4f} {its:.2f} it/s ETA {eta_min:.1f} min", flush=True)
            log_t, log_iter = now, it

        if it % checkpoint_every == 0 or it == total_iters:
            if ema is not None:
                ema.copy_to(eval_model)
                eval_target = eval_model
            else:
                eval_target = model
            if checkpoint_select_by == "pq":
                val_metric = run_pq_validation(
                    eval_target, device, pq_val_stems, train_paths, images_by_file, anns_by_image_id,
                    pq_postprocess_cfg, cfg.get("infer_tile_size", 1024), cfg.get("infer_overlap", 128),
                )
                metric_name = "val_pq"
            else:
                val_metric = downsampled_validate(eval_target, device, val_stems_for_training, train_paths, images_by_file, anns_by_image_id)
                metric_name = "val_dice"
            is_best = val_metric > best_metric
            best_metric = max(best_metric, val_metric)
            sku.save_checkpoint(last_ckpt_path, model, optimizer, scheduler, it, best_metric, ema=ema)
            if is_best:
                sku.save_checkpoint(best_ckpt_path, model, optimizer, scheduler, it, best_metric, ema=ema)
            elapsed = time.time() - run_start
            print(f"iter {it}/{total_iters} loss={loss.item():.4f} {metric_name}={val_metric:.4f} "
                  f"best={best_metric:.4f} ({elapsed:.1f}s elapsed)")

        if args.max_hours is not None and (time.time() - run_start) / 3600 >= args.max_hours:
            sku.save_checkpoint(last_ckpt_path, model, optimizer, scheduler, it, best_metric, ema=ema)
            print(f"--max_hours {args.max_hours} reached at iter {it}/{total_iters}; checkpoint saved, stopping cleanly.")
            stopped_early = True
            break

    print(f"Training {'stopped early' if stopped_early else 'done'}. Best val dice={best_metric:.4f}. Checkpoints in {ckpt_dir}")

    if cfg.get("save_oof_and_test_probs", True):
        best_path = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
        best_ckpt = sku.load_checkpoint(best_path, model, map_location=device, restore_rng=False)
        if ema is not None and best_ckpt.get("ema") is not None:
            ema_for_infer = sku.EMA(model, ema_decay)
            ema_for_infer.load_state_dict(best_ckpt["ema"])
            ema_for_infer.copy_to(model)
            print("Using EMA weights (from the checkpoint's saved shadow) for OOF/test inference.")
        tile_size, overlap = cfg.get("infer_tile_size", 1024), cfg.get("infer_overlap", 128)

        oof_dir = skio.output_root() / cfg["experiment_id"] / "oof"
        t0 = time.time()
        run_inference_and_save(model, device, [(s, train_paths[s]) for s in all_val_stems], oof_dir, tile_size, overlap, cfg.get("tta_views", 3))
        print(f"Saved {len(all_val_stems)} OOF prob maps to {oof_dir} in {(time.time() - t0) / 60:.1f} min")

        test_dir = skio.output_root() / cfg["experiment_id"] / "test_probs"
        t0 = time.time()
        run_inference_and_save(model, device, list(test_paths.items()), test_dir, tile_size, overlap, cfg.get("tta_views", 3))
        print(f"Saved {len(test_paths)} test prob maps to {test_dir} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
