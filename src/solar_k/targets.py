"""Per-physical-image training targets built from all of that image's annotators.

Strategy (CODING_SPEC_02_BASELINE.md, "Decisions taken from the EDA"):
**soft consensus** -- each pixel's value is the fraction of that image's
annotators who marked it; single-annotator images give a binary target.
Union is rejected: it inflates masks, which lowers IoU against each
individual annotator. With BCE, soft consensus has the same expected
gradient as sampling a random annotator each epoch, but less noise.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import io as skio
from . import masks as skmasks

TARGET_STRATEGY = "soft_consensus"
SKELETON_VERSION = "skeleton_v1"  # CODING_SPEC_06 E020


def annotator_instance_masks(anns: list[dict], height: int, width: int) -> list[np.ndarray]:
    """One mask per instance for a single annotator -- kept available for
    evaluation (scoring is always against individual annotators' instances,
    never against the soft-consensus target)."""
    return [skmasks.polygon_to_mask(a["segmentation"], height, width) for a in anns]


def annotator_semantic_mask(anns: list[dict], height: int, width: int) -> np.ndarray:
    """Union of one annotator's own instances into a single binary semantic mask."""
    out = np.zeros((height, width), dtype=np.uint8)
    for a in anns:
        m = skmasks.polygon_to_mask(a["segmentation"], height, width)
        np.maximum(out, m, out=out)
    return out


def soft_consensus_target(anns_by_annotator: dict[str, list[dict]], height: int, width: int) -> np.ndarray:
    """Per-pixel fraction of annotators who marked that pixel, in [0, 1].
    ``anns_by_annotator`` maps annotator code -> that annotator's annotation
    records for one physical image."""
    n = len(anns_by_annotator)
    if n == 0:
        return np.zeros((height, width), dtype=np.float32)
    acc = np.zeros((height, width), dtype=np.float32)
    for anns in anns_by_annotator.values():
        acc += annotator_semantic_mask(anns, height, width).astype(np.float32)
    return acc / n


# ---------------------------------------------- skeleton cache (E020, GPU-only)

def skeleton_cache_dir() -> Path:
    return skio.cache_root() / SKELETON_VERSION


def skeleton_path(stem: str) -> Path:
    return skeleton_cache_dir() / f"{stem}.png"


def build_skeleton(target: np.ndarray) -> np.ndarray:
    """1-px-dilated skeleton of the binarised (t=0.5) soft-consensus target
    (Kirchhoff et al., ECCV 2024's skeleton-recall loss). ``skimage.morphology
    .skeletonize`` is topology-preserving, so a connected filament stays a
    connected 1-px line before the dilation widens it back to a forgiving
    1-px tolerance band."""
    from skimage.morphology import dilation, skeletonize

    skel = skeletonize(target >= 0.5)
    return dilation(skel, footprint=np.ones((3, 3), dtype=bool)).astype(np.uint8)


def load_or_build_skeleton(stem: str, target: np.ndarray) -> np.ndarray:
    """Cache-aware: computed and cached once per stem (CODING_SPEC_06 E020
    says to build this on the GPU machine, never the Mac -- CLAUDE.md's
    8 GB rule -- by only ever calling this from training/inference code,
    never from any Mac-run CPU evaluation path)."""
    path = skeleton_path(stem)
    if path.exists():
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            return (arr > 0).astype(np.uint8)
    skel = build_skeleton(target)
    skeleton_cache_dir().mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), skel * 255)
    return skel
