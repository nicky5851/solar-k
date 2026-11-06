"""CODING_SPEC_07 E028: pseudo-label weight masking and the real/pseudo
dataset mix."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from solar_k import dataset as skds


def test_pseudo_weight_mask_ignores_middle_band_only():
    target = np.array([0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0], dtype=np.float32)
    weight = skds.pseudo_weight_mask(target)
    # band is (0.3, 0.7) exclusive -> 0.3 and 0.7 themselves are boundary
    # (<=0.3 or >=0.7), so confident; only strictly inside is ignored.
    expected = np.array([1, 1, 1, 0, 1, 1, 1], dtype=np.float32)
    np.testing.assert_array_equal(weight, expected)


def test_pseudo_weight_mask_all_confident_when_bimodal():
    target = np.zeros((10, 10), dtype=np.float32)
    target[5:, :] = 1.0  # sharp 0/1 split, no ambiguous pixels
    weight = skds.pseudo_weight_mask(target)
    assert weight.sum() == target.size


def test_pseudo_weight_mask_all_ignored_when_everything_ambiguous():
    target = np.full((10, 10), 0.5, dtype=np.float32)
    weight = skds.pseudo_weight_mask(target)
    assert weight.sum() == 0


class _FakeTwoTupleDataset:
    """Stand-in for SolarCropDataset: always returns (img, target)."""

    def __init__(self, n=5):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.full((1, 4, 4), float(idx))
        tgt = torch.zeros(1, 4, 4)
        return img, tgt


class _FakeThreeTupleDataset:
    """Stand-in for SolarCropDataset with skeleton_recall=True."""

    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return torch.zeros(1, 4, 4), torch.zeros(1, 4, 4), torch.zeros(1, 4, 4)


class _FakePseudoDataset:
    def __init__(self, n=5):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = torch.full((1, 4, 4), -1.0)
        tgt = torch.full((1, 4, 4), 0.9)
        weight = torch.full((1, 4, 4), 1.0)
        return img, tgt, weight


def test_mixed_dataset_real_fraction_controls_the_split():
    rng_check = random.Random(123)
    n = 4000
    real_count = sum(1 for _ in range(n) if rng_check.random() < 0.5)
    # sanity: sampling with p=0.5 over many draws lands near 50%
    assert 0.4 * n < real_count < 0.6 * n


def test_mixed_dataset_real_samples_get_all_ones_weight():
    mixed = skds.MixedCropDataset(_FakeTwoTupleDataset(), _FakePseudoDataset(), real_fraction=1.0, seed=0)
    _, _, weight = mixed[0]
    assert torch.all(weight == 1.0)


def test_mixed_dataset_pseudo_samples_pass_through_their_own_weight():
    mixed = skds.MixedCropDataset(_FakeTwoTupleDataset(), _FakePseudoDataset(), real_fraction=0.0, seed=0)
    img, tgt, weight = mixed[0]
    assert torch.all(img == -1.0)
    assert torch.all(tgt == 0.9)
    assert torch.all(weight == 1.0)


def test_mixed_dataset_rejects_skeleton_recall_real_dataset():
    mixed = skds.MixedCropDataset(_FakeThreeTupleDataset(), _FakePseudoDataset(), real_fraction=1.0, seed=0)
    with pytest.raises(NotImplementedError):
        mixed[0]


def test_mixed_dataset_rejects_invalid_real_fraction():
    with pytest.raises(ValueError):
        skds.MixedCropDataset(_FakeTwoTupleDataset(), _FakePseudoDataset(), real_fraction=-0.1)
    with pytest.raises(ValueError):
        skds.MixedCropDataset(_FakeTwoTupleDataset(), _FakePseudoDataset(), real_fraction=1.1)
