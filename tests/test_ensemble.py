"""CODING_SPEC_05_ENSEMBLE.md section 6(b): the ensemble's OOF mapping never
gives an image a model that trained on it; plus 6(d): tta_views=3
reproduces the existing flip-TTA output exactly."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from solar_k import ensemble as skens
from solar_k import unet as sku


@pytest.fixture
def fake_outputs(tmp_path, monkeypatch):
    np.random.RandomState(0)
    stems = [f"s{i:02d}" for i in range(20)]
    folds = pd.DataFrame({"file_stem": stems, "fold": [i % 5 for i in range(20)]})
    (tmp_path / "data" / "splits").mkdir(parents=True)
    folds.to_csv(tmp_path / "data" / "splits" / "folds_v1.csv", index=False)
    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("SOLAR_K_OUTPUT", str(tmp_path / "outputs"))
    # Every model fold writes a map for EVERY stem (as if it had leaked), with
    # pixel value = 10*member + fold, so the test can see which model was used.
    for mi, member in enumerate(["EA", "EB"]):
        for k in range(5):
            for kind in ("oof", "test_probs"):
                d = tmp_path / "outputs" / f"{member}-f{k}" / kind
                d.mkdir(parents=True)
                for s in stems:
                    cv2.imwrite(str(d / f"{s}.png"), np.full((4, 4), 10 * mi + k, dtype=np.uint8))
    return stems, dict(zip(folds["file_stem"], folds["fold"]))


def test_oof_uses_only_the_held_out_fold_model(fake_outputs):
    stems, fold = fake_outputs
    fold_by_stem = skens.model_fold_by_stem()
    assert fold_by_stem == fold
    for s in stems:
        srcs = skens.oof_sources(["EA", "EB"], s, fold_by_stem)
        # a model trained on fold j saw every stem whose fold != j
        for p in srcs:
            model_fold = int(p.parent.parent.name.split("-f")[1])
            assert model_fold == fold[s], f"{s} (fold {fold[s]}) averaged a fold-{model_fold} model that trained on it"
        mean = skens.load_oof_mean(["EA", "EB"], s, fold_by_stem)
        expected = ((0 + fold[s]) + (10 + fold[s])) / 2 / 255.0
        np.testing.assert_allclose(mean, expected, rtol=0, atol=1e-7)


def test_test_mean_uses_all_members_and_folds(fake_outputs):
    stems, _ = fake_outputs
    srcs = skens.test_sources(["EA", "EB"], stems[0])
    assert len(srcs) == 10
    mean = skens.load_test_mean(["EA", "EB"], stems[0])
    expected = np.mean([10 * m + k for m in range(2) for k in range(5)]) / 255.0
    np.testing.assert_allclose(mean, expected, atol=1e-6)


def test_weighted_oof_mean_matches_manual_weighted_average(fake_outputs):
    # CODING_SPEC_06 E018: member weights for load_oof_mean.
    stems, fold = fake_outputs
    fold_by_stem = skens.model_fold_by_stem()
    s = stems[5]
    k = fold[s]
    unweighted = skens.load_oof_mean(["EA", "EB"], s, fold_by_stem)
    equal = skens.load_oof_mean(["EA", "EB"], s, fold_by_stem, weights=[1.0, 1.0])
    np.testing.assert_allclose(unweighted, equal, rtol=0, atol=1e-7)
    weighted = skens.load_oof_mean(["EA", "EB"], s, fold_by_stem, weights=[0.3, 0.7])
    a_val, b_val = (0 + k) / 255.0, (10 + k) / 255.0
    expected = 0.3 * a_val + 0.7 * b_val
    np.testing.assert_allclose(weighted, expected, rtol=0, atol=1e-6)


def test_weighted_test_mean_expands_member_weight_over_its_folds(fake_outputs):
    stems, _ = fake_outputs
    unweighted = skens.load_test_mean(["EA", "EB"], stems[0])
    weighted = skens.load_test_mean(["EA", "EB"], stems[0], weights=[1.0, 1.0])
    np.testing.assert_allclose(unweighted, weighted, rtol=0, atol=1e-6)
    only_eb = skens.load_test_mean(["EA", "EB"], stems[0], weights=[0.0, 1.0])
    expected = np.mean([10 + k for k in range(5)]) / 255.0
    np.testing.assert_allclose(only_eb, expected, atol=1e-6)


def test_single_member_mean_is_the_member_map(fake_outputs):
    stems, fold = fake_outputs
    m = skens.load_oof_mean(["EB"], stems[3], skens.model_fold_by_stem())
    ref = skens.load_prob_png(skens.member_dir("EB", fold[stems[3]]) / "oof" / f"{stems[3]}.png")
    assert np.array_equal(m, ref)


class _Asym(torch.nn.Module):
    """Orientation-sensitive toy model, so every flip view differs."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(1, 1, 5, padding=2)

    def forward(self, x):
        h, w = x.shape[-2:]
        ramp = torch.linspace(0, 1, w).view(1, 1, 1, w) + torch.linspace(0, 2, h).view(1, 1, h, 1)
        return self.conv(x) + ramp


def _old_prob_fn(model, device):
    # verbatim copy of make_prob_fn's 3-view body before tta_views existed
    def infer(arr):
        return sku._infer_once(model, device, arr)

    def prob_fn(processed, disk_mask):
        probs = infer(processed)
        probs_h = infer(processed[:, ::-1])[:, ::-1]
        probs_v = infer(processed[::-1, :])[::-1, :]
        probs = (probs + probs_h + probs_v) / 3.0
        return probs * disk_mask.astype(np.float32)

    return prob_fn


def test_tta_views_3_reproduces_existing_output():
    model = _Asym().eval()
    dev = torch.device("cpu")
    x = np.random.RandomState(0).rand(32, 48).astype(np.float32)
    disk = np.ones_like(x, dtype=bool)
    ref = _old_prob_fn(model, dev)(x, disk)
    assert np.array_equal(sku.make_prob_fn(model, dev)(x, disk), ref)
    assert np.array_equal(sku.make_prob_fn(model, dev, tta_views=3)(x, disk), ref)
    four = sku.make_prob_fn(model, dev, tta_views=4)(x, disk)
    assert four.shape == ref.shape and not np.array_equal(four, ref)


def test_tta_scales_default_reproduces_existing_output():
    model = _Asym().eval()
    dev = torch.device("cpu")
    x = np.random.RandomState(0).rand(32, 48).astype(np.float32)
    disk = np.ones_like(x, dtype=bool)
    ref = _old_prob_fn(model, dev)(x, disk)
    assert np.array_equal(sku.make_prob_fn(model, dev, tta_scales=(1.0,))(x, disk), ref)


def test_tta_scales_with_upscale_changes_output_same_shape():
    model = _Asym().eval()
    dev = torch.device("cpu")
    x = np.random.RandomState(0).rand(32, 48).astype(np.float32)
    disk = np.ones_like(x, dtype=bool)
    ref = sku.make_prob_fn(model, dev)(x, disk)
    scaled = sku.make_prob_fn(model, dev, tta_scales=(1.0, 1.25))(x, disk)
    assert scaled.shape == ref.shape
    assert not np.array_equal(scaled, ref)


def test_tta_scales_rejects_downscaling():
    model = _Asym().eval()
    dev = torch.device("cpu")
    with pytest.raises(ValueError):
        sku.make_prob_fn(model, dev, tta_scales=(0.8,))
