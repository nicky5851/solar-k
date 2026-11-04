"""CODING_SPEC_06_RECALL_AND_CONNECTIVITY.md E020: the skeleton-recall loss
is 0 for a perfect prediction, and skeletonizing a straight line keeps it
1-px connected."""
from __future__ import annotations

import numpy as np
import torch

from solar_k import losses as sklo
from solar_k import targets as sktgt


def test_skeleton_recall_loss_is_zero_for_perfect_prediction():
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 4:12, 7] = 1.0
    skeleton = (target > 0).float()
    logits = (target * 2 - 1) * 20.0  # +20 where skeleton=1, -20 elsewhere
    assert sklo.skeleton_recall_loss(logits, skeleton).item() < 1e-6
    assert sklo.segmentation_loss(logits, target, skeleton).item() < 1e-3


def test_skeleton_recall_loss_penalises_missed_skeleton_pixels():
    skeleton = torch.zeros(1, 1, 8, 8)
    skeleton[0, 0, 3, :] = 1.0
    missed_logits = torch.full((1, 1, 8, 8), -20.0)
    hit_logits = torch.full((1, 1, 8, 8), 20.0)
    assert sklo.skeleton_recall_loss(missed_logits, skeleton).item() > 0.99
    assert sklo.skeleton_recall_loss(hit_logits, skeleton).item() < 1e-6


def test_build_skeleton_keeps_a_straight_line_connected():
    target = np.zeros((40, 40), dtype=np.float32)
    target[20, 5:35] = 1.0  # a straight 1-px filament
    skel = sktgt.build_skeleton(target)
    ys, xs = np.nonzero(skel)
    assert len(xs) > 0
    # every skeleton pixel has at least one 8-connected skeleton neighbour,
    # i.e. the line never breaks into isolated points.
    pts = set(zip(ys.tolist(), xs.tolist()))
    for y, x in pts:
        neighbours = {(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)}
        assert neighbours & pts, f"isolated skeleton pixel at {(y, x)}"


def test_load_or_build_skeleton_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLAR_K_CACHE", str(tmp_path / "cache"))
    target = np.zeros((30, 30), dtype=np.float32)
    target[10, 2:28] = 1.0
    first = sktgt.load_or_build_skeleton("stemA", target)
    assert sktgt.skeleton_path("stemA").exists()
    second = sktgt.load_or_build_skeleton("stemA", np.zeros_like(target))  # cache hit ignores the new target
    np.testing.assert_array_equal(first, second)
