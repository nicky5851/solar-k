"""Pluggable crop-center sampler for SolarCropDataset (CODING_SPEC_04_MODEL_SIDE.md).

Three center-picking modes, mixed by configurable probability:
- "positive": a random GT-filament pixel (the old, only, behaviour)
- "random": a uniform-random pixel on the full image
- "hard_negative": a pixel near a cataloged false-positive component for this
  stem (see scripts/build_hard_negatives.py), jittered +/- HARD_NEGATIVE_JITTER_PX

DEFAULT_MIX reproduces E002/E004's fixed 2/3 positive, 1/3 random split, so a
dataset built without an explicit sampler is unaffected by this refactor.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MIX = {"positive": 2 / 3, "random": 1 / 3}
HARD_NEGATIVE_JITTER_PX = 128


class CropCenterSampler:
    def __init__(
        self,
        mix: dict[str, float] | None = None,
        hard_negatives_by_stem: dict[str, list[tuple[int, int]]] | None = None,
    ):
        mix = dict(mix or DEFAULT_MIX)
        total = sum(mix.values())
        if total <= 0:
            raise ValueError("sampler mix must have positive total weight")
        if any(m not in ("positive", "random", "hard_negative") for m in mix):
            raise ValueError(f"unknown sampler mode(s) in {list(mix)}")
        self.modes = list(mix.keys())
        self.weights = [v / total for v in mix.values()]
        self.hard_negatives_by_stem = hard_negatives_by_stem or {}

    def _pick_mode(self, rng: random.Random) -> str:
        r = rng.random()
        cum = 0.0
        for mode, w in zip(self.modes, self.weights):
            cum += w
            if r < cum:
                return mode
        return self.modes[-1]

    def pick_center(self, stem: str, target: np.ndarray, h: int, w: int, rng: random.Random) -> tuple[int, int]:
        mode = self._pick_mode(rng)
        if mode == "hard_negative":
            centers = self.hard_negatives_by_stem.get(stem)
            if centers:
                cy, cx = centers[rng.randrange(len(centers))]
                jy = rng.randint(-HARD_NEGATIVE_JITTER_PX, HARD_NEGATIVE_JITTER_PX)
                jx = rng.randint(-HARD_NEGATIVE_JITTER_PX, HARD_NEGATIVE_JITTER_PX)
                return int(np.clip(cy + jy, 0, h - 1)), int(np.clip(cx + jx, 0, w - 1))
            mode = "positive"  # no catalogued hard negatives for this stem -- fall back
        if mode == "positive":
            return self._positive_center(target, h, w, rng)
        return rng.randrange(h), rng.randrange(w)

    @staticmethod
    def _positive_center(target: np.ndarray, h: int, w: int, rng: random.Random) -> tuple[int, int]:
        ys, xs = np.nonzero(target > 0)
        if len(ys):
            idx = rng.randrange(len(ys))
            return int(ys[idx]), int(xs[idx])
        return rng.randrange(h), rng.randrange(w)


def load_hard_negatives(csv_path: Path, fold_stems: set[str] | None = None) -> dict[str, list[tuple[int, int]]]:
    """Load scripts/build_hard_negatives.py's output into {stem: [(cy, cx), ...]}.

    Restricting to `fold_stems` (a fold's training stems) is the caller's
    responsibility -- CODING_SPEC_04_MODEL_SIDE.md requires each fold's hard
    negatives come only from that fold's own training images.
    """
    df = pd.read_csv(csv_path)
    if fold_stems is not None:
        df = df[df["stem"].isin(fold_stems)]
    out: dict[str, list[tuple[int, int]]] = {}
    for stem, group in df.groupby("stem"):
        out[stem] = list(zip(group["centroid_y"].astype(int), group["centroid_x"].astype(int)))
    return out
