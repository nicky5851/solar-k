import random

import numpy as np
import pandas as pd
import pytest

from solar_k import sampler as sksamp


def _target_with_blob(h=64, w=64):
    t = np.zeros((h, w), dtype=np.float32)
    t[10:15, 10:15] = 1.0
    return t


def test_positive_only_mix_always_lands_on_target():
    s = sksamp.CropCenterSampler(mix={"positive": 1.0})
    t = _target_with_blob()
    rng = random.Random(0)
    for _ in range(50):
        cy, cx = s.pick_center("stemA", t, 64, 64, rng)
        assert 10 <= cy < 15 and 10 <= cx < 15


def test_random_only_mix_covers_full_image():
    s = sksamp.CropCenterSampler(mix={"random": 1.0})
    t = _target_with_blob()
    rng = random.Random(0)
    centers = [s.pick_center("stemA", t, 64, 64, rng) for _ in range(200)]
    ys = [c[0] for c in centers]
    assert min(ys) < 10 and max(ys) >= 15  # spread beyond the blob


def test_hard_negative_mode_uses_catalogued_center_with_jitter():
    s = sksamp.CropCenterSampler(mix={"hard_negative": 1.0}, hard_negatives_by_stem={"stemA": [(500, 500)]})
    t = _target_with_blob(1024, 1024)
    rng = random.Random(0)
    for _ in range(50):
        cy, cx = s.pick_center("stemA", t, 1024, 1024, rng)
        assert abs(cy - 500) <= sksamp.HARD_NEGATIVE_JITTER_PX
        assert abs(cx - 500) <= sksamp.HARD_NEGATIVE_JITTER_PX


def test_hard_negative_mode_falls_back_to_positive_when_stem_uncatalogued():
    s = sksamp.CropCenterSampler(mix={"hard_negative": 1.0}, hard_negatives_by_stem={"other_stem": [(5, 5)]})
    t = _target_with_blob()
    rng = random.Random(0)
    cy, cx = s.pick_center("stemA", t, 64, 64, rng)
    assert 10 <= cy < 15 and 10 <= cx < 15


def test_default_mix_matches_old_two_thirds_one_third_split():
    assert sksamp.DEFAULT_MIX == {"positive": pytest.approx(2 / 3), "random": pytest.approx(1 / 3)}


def test_rejects_unknown_mode():
    with pytest.raises(ValueError):
        sksamp.CropCenterSampler(mix={"bogus": 1.0})


def test_rejects_zero_weight_mix():
    with pytest.raises(ValueError):
        sksamp.CropCenterSampler(mix={"positive": 0.0, "random": 0.0})


def test_load_hard_negatives_groups_by_stem_and_filters_to_fold(tmp_path):
    df = pd.DataFrame({
        "stem": ["a", "a", "b"],
        "centroid_y": [10, 20, 30],
        "centroid_x": [11, 21, 31],
        "class": ["isolated", "isolated", "isolated"],
    })
    csv_path = tmp_path / "hn.csv"
    df.to_csv(csv_path, index=False)

    out_all = sksamp.load_hard_negatives(csv_path)
    assert out_all["a"] == [(10, 11), (20, 21)]
    assert out_all["b"] == [(30, 31)]

    out_filtered = sksamp.load_hard_negatives(csv_path, fold_stems={"a"})
    assert list(out_filtered.keys()) == ["a"]
