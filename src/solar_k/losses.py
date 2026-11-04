"""Training losses (CODING_SPEC_06_RECALL_AND_CONNECTIVITY.md E020).

BCE + soft Dice on the soft-consensus target is the baseline (moved here
verbatim from ``scripts/train.py``, which now imports it). The skeleton-recall
term (Kirchhoff et al., ECCV 2024) adds a recall-only penalty on a
precomputed, 1-px-dilated skeleton of the binarised target (``targets.
build_skeleton``): it only rewards covering skeleton pixels, so it pushes
the model to keep thin filaments connected without any extra penalty for
being generous around them (that's soft Dice's job already).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def skeleton_recall_loss(logits: torch.Tensor, skeleton: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - recall of the predicted probability over skeleton pixels only.
    0 for a prediction that is 1 everywhere the skeleton is 1 (perfect
    recall); never penalises predictions outside the skeleton."""
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    covered = (probs * skeleton).sum(dims)
    total = skeleton.sum(dims)
    recall = (covered + eps) / (total + eps)
    return 1.0 - recall.mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, skeleton: torch.Tensor | None = None, skeleton_lambda: float = 1.0) -> torch.Tensor:
    """BCE + soft Dice, plus ``skeleton_lambda`` * skeleton-recall when a
    skeleton mask is given (absent -> exactly the Spec 02-05 loss)."""
    loss = F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)
    if skeleton is not None:
        loss = loss + skeleton_lambda * skeleton_recall_loss(logits, skeleton)
    return loss
