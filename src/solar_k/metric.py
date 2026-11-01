"""Panoptic Quality scorer, reproducing the organisers' self-evaluation notebook
(``data/external/self-evaluation-notebook.ipynb``).

The notebook's rule (read from the source, not inferred):
  - ``get_overlap_matrices``: dense IoU and Dice between every GT layer and
    every predicted layer of one image (``(n_gt, n_pred)`` matrices).
  - ``get_pq_score``: a GT/pred pair is a true positive (TP) iff
    ``IoU > 0.5`` (strict, not >=). If an image has no GT, every prediction
    on it is a false positive (FP); if it has no predictions, every GT is a
    false negative (FN); if both are empty the image contributes nothing.
    Otherwise FP/FN are "no row/column of the hit matrix is set"-counts.
  - Aggregation is a single sum over *all* images, not a per-image mean:
    ``PQ = sum(IoU of TP pairs) / (TP + 0.5*FP + 0.5*FN)``, 0.0 if the
    denominator is 0.
  - Matching is done per *annotator-image* (``get_overlap_df`` groups GT by
    ``"<annotator>-<image_id>"``), so with several GT annotators each one is
    scored separately against the same prediction set and then summed in.

This module reimplements that logic with numpy instead of torch (no benefit
to a GPU-shaped dependency for images processed one at a time on an 8 GB
Mac), and adds SQ/RQ/TP/FP/FN/per-match IoU-Dice, which the notebook does not
return but which the spec asks for. SQ/RQ follow the standard decomposition
PQ = SQ * RQ, RQ = TP / (TP + 0.5FP + 0.5FN), SQ = mean IoU of TP pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLD = 0.5


def _flatten(masks: np.ndarray) -> np.ndarray:
    n = masks.shape[0]
    return masks.reshape(n, -1).astype(np.float64)


def pairwise_iou_dice(gt_masks: np.ndarray, pred_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n_gt, n_pred)`` IoU and Dice matrices. Mirrors ``get_overlap_matrices``."""
    gt_flat = _flatten(gt_masks)
    pred_flat = _flatten(pred_masks)

    intersection = gt_flat @ pred_flat.T
    gt_areas = gt_flat.sum(axis=1, keepdims=True)
    pred_areas = pred_flat.sum(axis=1, keepdims=True).T

    union = gt_areas + pred_areas - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    dice_den = gt_areas + pred_areas
    dice = np.divide(2 * intersection, dice_den, out=np.zeros_like(intersection), where=dice_den > 0)

    return iou, dice


@dataclass
class ImageMatch:
    """Per-image (or per-annotator-image) match outcome."""

    tp_ious: list[float] = field(default_factory=list)
    tp_dices: list[float] = field(default_factory=list)
    fp: int = 0
    fn: int = 0


def match_scores(
    iou: np.ndarray, dice: np.ndarray, n_gt: int, n_pred: int, iou_threshold: float = IOU_THRESHOLD
) -> ImageMatch:
    """Match two score matrices, mirroring ``get_pq_score``'s per-image branch."""
    if n_gt == 0:
        return ImageMatch(fp=n_pred, fn=0)
    if n_pred == 0:
        return ImageMatch(fp=0, fn=n_gt)

    hit = iou > iou_threshold
    tp_ious = iou[hit].tolist()
    tp_dices = dice[hit].tolist()
    fp = int((hit.sum(axis=0) == 0).sum())
    fn = int((hit.sum(axis=1) == 0).sum())
    return ImageMatch(tp_ious=tp_ious, tp_dices=tp_dices, fp=fp, fn=fn)


def score_image(
    gt_masks: np.ndarray, pred_masks: np.ndarray, iou_threshold: float = IOU_THRESHOLD
) -> ImageMatch:
    """Score one image's GT mask stack ``(n_gt, H, W)`` against predictions ``(n_pred, H, W)``."""
    n_gt, n_pred = gt_masks.shape[0], pred_masks.shape[0]
    if n_gt == 0 or n_pred == 0:
        return match_scores(np.zeros((0, 0)), np.zeros((0, 0)), n_gt, n_pred, iou_threshold)
    iou, dice = pairwise_iou_dice(gt_masks, pred_masks)
    return match_scores(iou, dice, n_gt, n_pred, iou_threshold)


@dataclass
class PQResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    tp_ious: list[float] = field(default_factory=list)
    tp_dices: list[float] = field(default_factory=list)


def aggregate_matches(matches: list[ImageMatch]) -> PQResult:
    """Sum per-image matches into one global PQ/SQ/RQ, matching the notebook's
    global-sum aggregation (not a per-image mean of PQ)."""
    tp_ious: list[float] = []
    tp_dices: list[float] = []
    fp = 0
    fn = 0
    for m in matches:
        tp_ious.extend(m.tp_ious)
        tp_dices.extend(m.tp_dices)
        fp += m.fp
        fn += m.fn

    tp = len(tp_ious)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum(tp_ious) / denom if denom > 0 else 0.0
    rq = tp / denom if denom > 0 else 0.0
    sq = sum(tp_ious) / tp if tp > 0 else 0.0
    return PQResult(pq=pq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn, tp_ious=tp_ious, tp_dices=tp_dices)


def score_dataset(image_pairs) -> PQResult:
    """Stream ``(gt_masks, pred_masks)`` pairs, one image at a time, and aggregate.

    ``image_pairs`` is any iterable (ideally a generator) so the caller never
    has to hold more than one image's mask stacks in memory at once.
    """
    matches = [score_image(gt, pred) for gt, pred in image_pairs]
    return aggregate_matches(matches)
