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
    ):
        if not stems:
            raise ValueError("SolarCropDataset needs at least one stem")
        self.stems = stems
        self.images_by_file = images_by_file
        self.anns_by_image_id = anns_by_image_id
        self.crop_size = crop_size
        self.length = length or len(stems) * 20
        self.augment = augment
        self.rng = random.Random(seed)
        self.sampler = sampler or sksamp.CropCenterSampler()
        self._train_paths = {p.stem: p for p in skio.iter_train_images()}

    def __len__(self) -> int:
        return self.length

    def _load_image_and_target(self, stem: str) -> tuple[np.ndarray, np.ndarray]:
        processed, _ = skpre.load_or_build(stem, self._train_paths[stem])

        entries = self.images_by_file.get(f"{stem}.jpeg", [])
        anns_by_annotator = {}
        for e in entries:
            annotator, _ = skids.parse_image_id(e["id"])
            anns_by_annotator[annotator] = self.anns_by_image_id.get(e["id"], [])
        target = sktgt.soft_consensus_target(anns_by_annotator, processed.shape[0], processed.shape[1])
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
            tgt_crop = np.pad(tgt_crop, ((0, pad_h), (0, pad_w)))
        if self.augment:
            img_crop, tgt_crop = self._augment(img_crop, tgt_crop)
        img_t = torch.from_numpy(np.ascontiguousarray(img_crop, dtype=np.float32)).unsqueeze(0)
        tgt_t = torch.from_numpy(np.ascontiguousarray(tgt_crop, dtype=np.float32)).unsqueeze(0)
        return img_t, tgt_t
