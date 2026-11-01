"""Per-physical-image training targets built from all of that image's annotators.

Strategy (CODING_SPEC_02_BASELINE.md, "Decisions taken from the EDA"):
**soft consensus** -- each pixel's value is the fraction of that image's
annotators who marked it; single-annotator images give a binary target.
Union is rejected: it inflates masks, which lowers IoU against each
individual annotator. With BCE, soft consensus has the same expected
gradient as sampling a random annotator each epoch, but less noise.
"""
from __future__ import annotations

import numpy as np

from . import masks as skmasks

TARGET_STRATEGY = "soft_consensus"


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
