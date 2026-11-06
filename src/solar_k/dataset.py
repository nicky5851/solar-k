"""Training dataset: random crops from the disk area with a soft-consensus
target (CODING_SPEC_02_BASELINE.md, Stage C). Two thirds of crops are
centred on a random GT filament pixel, one third random; flips, 90-degree
rotations, and mild brightness/contrast/gamma augment the image (never the
target). No resizing.

Samples are computed on demand, not pre-cached in memory: training mostly
runs on Kaggle GPUs with plenty of RAM, and the Mac smoke test only needs a
handful of iterations, so an elaborate target cache isn't worth the
complexity or the memory it would cost on an 8 GB dev machine.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import ids as skids
from . import io as skio
from . import preprocess as skpre
from . import sampler as sksamp
from . import targets as sktgt

CROP_SIZE = 512
CENTERED_FRACTION = 2 / 3  # kept for reference; see sampler.DEFAULT_MIX


class SolarCropDataset(Dataset):
    def __init__(
        self,
        stems: list[str],
        images_by_file: dict,
        anns_by_image_id: dict,
        crop_size: int = CROP_SIZE,
        length: int | None = None,
        seed: int = 42,
        augment: bool = True,
        sampler: sksamp.CropCenterSampler | None = None,
        skeleton_recall: bool = False,
        train_scale: float = 1.0,
    ):
        if not stems:
            raise ValueError("SolarCropDataset needs at least one stem")
        if skeleton_recall and train_scale != 1.0:
            raise NotImplementedError("skeleton_recall + train_scale != 1.0 is not supported")
        self.stems = stems
        self.images_by_file = images_by_file
        self.anns_by_image_id = anns_by_image_id
        self.crop_size = crop_size
        self.length = length or len(stems) * 20
        self.augment = augment
        self.rng = random.Random(seed)
        self.sampler = sampler or sksamp.CropCenterSampler()
        self.skeleton_recall = skeleton_recall
        self.train_scale = train_scale
        self._train_paths = {p.stem: p for p in skio.iter_train_images()}

    def __len__(self) -> int:
        return self.length

    def _load_image_and_target(self, stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        processed, _ = skpre.load_or_build(stem, self._train_paths[stem], scale=self.train_scale)

        entries = self.images_by_file.get(f"{stem}.jpeg", [])
        anns_by_annotator = {}
        for e in entries:
            annotator, _ = skids.parse_image_id(e["id"])
            anns_by_annotator[annotator] = self.anns_by_image_id.get(e["id"], [])
        if self.train_scale == 1.0:
            target = sktgt.soft_consensus_target(anns_by_annotator, processed.shape[0], processed.shape[1])
        else:
            # Annotation polygon coordinates are in the image's native
            # (scale=1.0) pixel space -- rasterise there, then upsample the
            # resulting soft target to match the upsampled image, exactly
            # as CODING_SPEC_07_RESOLUTION_AND_SELF_TRAINING.md E027 asks
            # ("bilinear for the image, bilinear then the same soft target").
            native_processed, _ = skpre.load_or_build(stem, self._train_paths[stem], scale=1.0)
            native_target = sktgt.soft_consensus_target(anns_by_annotator, native_processed.shape[0], native_processed.shape[1])
            target = cv2.resize(native_target, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_LINEAR)
        skeleton = sktgt.load_or_build_skeleton(stem, target) if self.skeleton_recall else None
        return processed, target, skeleton

    def _crop(self, arr: np.ndarray, cy: int, cx: int) -> np.ndarray:
        h, w = arr.shape
        half = self.crop_size // 2
        y0 = min(max(cy - half, 0), max(h - self.crop_size, 0))
        x0 = min(max(cx - half, 0), max(w - self.crop_size, 0))
        return arr[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]

    def _augment(self, img: np.ndarray, tgt: np.ndarray, skel: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.rng.random() < 0.5:
            img, tgt = img[:, ::-1].copy(), tgt[:, ::-1].copy()
            if skel is not None:
                skel = skel[:, ::-1].copy()
        if self.rng.random() < 0.5:
            img, tgt = img[::-1, :].copy(), tgt[::-1, :].copy()
            if skel is not None:
                skel = skel[::-1, :].copy()
        k = self.rng.randrange(4)
        if k:
            img, tgt = np.rot90(img, k).copy(), np.rot90(tgt, k).copy()
            if skel is not None:
                skel = np.rot90(skel, k).copy()
        if self.rng.random() < 0.8:
            brightness = self.rng.uniform(-0.05, 0.05)
            contrast = self.rng.uniform(0.9, 1.1)
            gamma = self.rng.uniform(0.9, 1.1)
            img = np.clip((np.clip(img, 0, 1) - 0.5) * contrast + 0.5 + brightness, 0, 1)
            img = np.clip(img, 1e-6, 1.0) ** gamma
        return img, tgt, skel

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        stem = self.stems[self.rng.randrange(len(self.stems))]
        processed, target, skeleton = self._load_image_and_target(stem)
        h, w = processed.shape
        cy, cx = self.sampler.pick_center(stem, target, h, w, self.rng)
        img_crop = self._crop(processed, cy, cx)
        tgt_crop = self._crop(target, cy, cx)
        skel_crop = self._crop(skeleton, cy, cx) if skeleton is not None else None
        if img_crop.shape != (self.crop_size, self.crop_size):
            pad_h = self.crop_size - img_crop.shape[0]
            pad_w = self.crop_size - img_crop.shape[1]
            img_crop = np.pad(img_crop, ((0, pad_h), (0, pad_w)))
            tgt_crop = np.pad(tgt_crop, ((0, pad_h), (0, pad_w)))
            if skel_crop is not None:
                skel_crop = np.pad(skel_crop, ((0, pad_h), (0, pad_w)))
        if self.augment:
            img_crop, tgt_crop, skel_crop = self._augment(img_crop, tgt_crop, skel_crop)
        img_t = torch.from_numpy(np.ascontiguousarray(img_crop, dtype=np.float32)).unsqueeze(0)
        tgt_t = torch.from_numpy(np.ascontiguousarray(tgt_crop, dtype=np.float32)).unsqueeze(0)
        if skel_crop is None:
            return img_t, tgt_t
        skel_t = torch.from_numpy(np.ascontiguousarray(skel_crop, dtype=np.float32)).unsqueeze(0)
        return img_t, tgt_t, skel_t


# CODING_SPEC_07 E028: confidence band ignored in the pseudo-label loss.
# Pixels here are too ambiguous to trust as either filament or background.
PSEUDO_IGNORE_BAND = (0.3, 0.7)


def pseudo_weight_mask(target: np.ndarray, band: tuple[float, float] = PSEUDO_IGNORE_BAND) -> np.ndarray:
    """1 outside ``band`` (confident), 0 inside it (ignored). Pure function,
    kept separate from ExternalPseudoCropDataset for direct unit testing."""
    lo, hi = band
    return ((target <= lo) | (target >= hi)).astype(np.float32)


class ExternalPseudoCropDataset(Dataset):
    """Random crops from externally-sourced, unlabelled GONG images, paired
    with a fold-specific pseudo-target produced by
    scripts/pseudo_label_external.py (CODING_SPEC_07 E028). Same crop/
    augment logic as SolarCropDataset, but every sample also carries a
    per-pixel weight mask (0 inside PSEUDO_IGNORE_BAND, 1 elsewhere) so the
    loss ignores low-confidence pixels instead of trusting them as ground
    truth. Never mixed with GT labels -- this dataset doesn't know the
    difference between "ambiguous" and "wrong", only "confident enough to
    use"."""

    def __init__(
        self,
        stems: list[str],
        images_dir,
        pseudo_dir,
        crop_size: int = CROP_SIZE,
        length: int | None = None,
        seed: int = 42,
        augment: bool = True,
        sampler: sksamp.CropCenterSampler | None = None,
    ):
        if not stems:
            raise ValueError("ExternalPseudoCropDataset needs at least one stem")
        self.stems = stems
        self.images_dir = images_dir
        self.pseudo_dir = pseudo_dir
        self.crop_size = crop_size
        self.length = length or len(stems) * 20
        self.augment = augment
        self.rng = random.Random(seed)
        self.sampler = sampler or sksamp.CropCenterSampler()

    def __len__(self) -> int:
        return self.length

    def _load_image_and_target(self, stem: str) -> tuple[np.ndarray, np.ndarray]:
        img_path = self.images_dir / f"{stem}.jpg"
        processed, _ = skpre.load_or_build(stem, img_path)
        pseudo_u8 = cv2.imread(str(self.pseudo_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        target = pseudo_u8.astype(np.float32) / 255.0
        return processed, target

    def _crop(self, arr: np.ndarray, cy: int, cx: int) -> np.ndarray:
        h, w = arr.shape
        half = self.crop_size // 2
        y0 = min(max(cy - half, 0), max(h - self.crop_size, 0))
        x0 = min(max(cx - half, 0), max(w - self.crop_size, 0))
        return arr[y0 : y0 + self.crop_size, x0 : x0 + self.crop_size]

    def _augment(self, img: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            img, tgt = img[:, ::-1].copy(), tgt[:, ::-1].copy()
        if self.rng.random() < 0.5:
            img, tgt = img[::-1, :].copy(), tgt[::-1, :].copy()
        k = self.rng.randrange(4)
        if k:
            img, tgt = np.rot90(img, k).copy(), np.rot90(tgt, k).copy()
        if self.rng.random() < 0.8:
            brightness = self.rng.uniform(-0.05, 0.05)
            contrast = self.rng.uniform(0.9, 1.1)
            gamma = self.rng.uniform(0.9, 1.1)
            img = np.clip((np.clip(img, 0, 1) - 0.5) * contrast + 0.5 + brightness, 0, 1)
            img = np.clip(img, 1e-6, 1.0) ** gamma
        return img, tgt

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stems[self.rng.randrange(len(self.stems))]
        processed, target = self._load_image_and_target(stem)
        h, w = processed.shape
        cy, cx = self.sampler.pick_center(stem, target, h, w, self.rng)
        img_crop = self._crop(processed, cy, cx)
        tgt_crop = self._crop(target, cy, cx)
        if img_crop.shape != (self.crop_size, self.crop_size):
            pad_h = self.crop_size - img_crop.shape[0]
            pad_w = self.crop_size - img_crop.shape[1]
            img_crop = np.pad(img_crop, ((0, pad_h), (0, pad_w)))
            tgt_crop = np.pad(tgt_crop, ((0, pad_h), (0, pad_w)), constant_values=0.5)  # padding stays ignored
        if self.augment:
            img_crop, tgt_crop = self._augment(img_crop, tgt_crop)
        weight_crop = pseudo_weight_mask(tgt_crop)
        img_t = torch.from_numpy(np.ascontiguousarray(img_crop, dtype=np.float32)).unsqueeze(0)
        tgt_t = torch.from_numpy(np.ascontiguousarray(tgt_crop, dtype=np.float32)).unsqueeze(0)
        weight_t = torch.from_numpy(np.ascontiguousarray(weight_crop, dtype=np.float32)).unsqueeze(0)
        return img_t, tgt_t, weight_t


class MixedCropDataset(Dataset):
    """50/50 (configurable) mix of a labelled SolarCropDataset and an
    ExternalPseudoCropDataset (CODING_SPEC_07 E028). Always yields
    (img, target, weight) triples: labelled crops get an all-ones weight
    (every pixel is real ground truth, nothing to ignore), pseudo crops get
    ExternalPseudoCropDataset's confidence-derived weight. ``real_dataset``
    must not itself be a skeleton-recall dataset (E028 doesn't combine with
    skeleton-recall; raises NotImplementedError if it returns 3-tuples).
    ``real_fraction`` in [0, 1]; 0 or 1 degenerate to pure pseudo/real (handy
    for testing, not how E028 actually trains)."""

    def __init__(self, real_dataset: SolarCropDataset, pseudo_dataset: ExternalPseudoCropDataset, real_fraction: float = 0.5, seed: int = 42):
        if not 0.0 <= real_fraction <= 1.0:
            raise ValueError(f"real_fraction must be in [0, 1], got {real_fraction}")
        self.real_dataset = real_dataset
        self.pseudo_dataset = pseudo_dataset
        self.real_fraction = real_fraction
        self.rng = random.Random(seed)
        self.length = max(len(real_dataset), len(pseudo_dataset))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.rng.random() < self.real_fraction:
            sample = self.real_dataset[self.rng.randrange(len(self.real_dataset))]
            if len(sample) != 2:
                raise NotImplementedError("MixedCropDataset does not support a skeleton-recall real_dataset")
            img_t, tgt_t = sample
            weight_t = torch.ones_like(tgt_t)
        else:
            img_t, tgt_t, weight_t = self.pseudo_dataset[self.rng.randrange(len(self.pseudo_dataset))]
        return img_t, tgt_t, weight_t
