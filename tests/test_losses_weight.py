"""CODING_SPEC_07 E028: the optional per-pixel ``weight`` on
segmentation_loss/soft_dice_loss/weighted_bce_loss must (a) reduce to
exactly the pre-E028 behaviour when absent, and (b) genuinely exclude
0-weighted pixels rather than just shrinking their contribution."""
from __future__ import annotations

import torch

from solar_k import losses as sklo


def test_weight_none_matches_pre_e028_segmentation_loss():
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    target = torch.randint(0, 2, (2, 1, 8, 8)).float()
    without_weight = sklo.segmentation_loss(logits, target)
    with_full_weight = sklo.segmentation_loss(logits, target, weight=torch.ones_like(target))
    assert torch.allclose(without_weight, with_full_weight, atol=1e-6)


def test_weighted_bce_ignores_zero_weighted_pixels():
    # A batch where half the pixels are wildly wrong but 0-weighted, and the
    # other half are perfectly predicted and weighted 1: the weighted loss
    # must come out near zero, driven only by the confident half.
    logits = torch.cat([torch.full((1, 1, 4, 8), -20.0), torch.full((1, 1, 4, 8), 20.0)], dim=2)
    target = torch.cat([torch.zeros(1, 1, 4, 8), torch.ones(1, 1, 4, 8)], dim=2)
    # flip the "wrong" half's target so it would be a large loss if not masked out
    wrong_target = target.clone()
    wrong_target[:, :, :4, :] = 1.0
    weight = torch.cat([torch.zeros(1, 1, 4, 8), torch.ones(1, 1, 4, 8)], dim=2)
    loss = sklo.weighted_bce_loss(logits, wrong_target, weight)
    assert loss.item() < 1e-3


def test_weighted_bce_normalises_by_weight_sum_not_pixel_count():
    # Same confident region, but padded with many more 0-weighted pixels.
    # The loss value must stay the same either way (normalised by weight
    # sum), not shrink as more zero-weighted padding is added.
    logits_small = torch.full((1, 1, 2, 2), 20.0)
    target_small = torch.ones(1, 1, 2, 2)
    weight_small = torch.ones(1, 1, 2, 2)
    small_loss = sklo.weighted_bce_loss(logits_small, target_small, weight_small)

    logits_padded = torch.cat([logits_small, torch.full((1, 1, 2, 98), -999.0)], dim=3)
    target_padded = torch.cat([target_small, torch.ones(1, 1, 2, 98)], dim=3)
    weight_padded = torch.cat([weight_small, torch.zeros(1, 1, 2, 98)], dim=3)
    padded_loss = sklo.weighted_bce_loss(logits_padded, target_padded, weight_padded)
    assert torch.allclose(small_loss, padded_loss, atol=1e-5)


def test_soft_dice_loss_weight_excludes_pixels_from_both_sums():
    # A fully-wrong 0-weighted region must not affect the dice score at all.
    probs_logits = torch.full((1, 1, 4, 4), 20.0)  # predicts ~1 everywhere
    target = torch.zeros(1, 1, 4, 4)  # true label 0 everywhere -> would be a bad dice
    target[:, :, :2, :] = 1.0  # except this region, where it's a perfect match
    weight = torch.zeros(1, 1, 4, 4)
    weight[:, :, :2, :] = 1.0  # only the perfect-match region counts
    loss = sklo.soft_dice_loss(probs_logits, target, weight)
    assert loss.item() < 1e-3
