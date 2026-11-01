from __future__ import annotations

import numpy as np

from solar_k import components as skcomp
from solar_k import postprocess as skpp


def _toy_scene(h: int = 64, w: int = 64):
    """One elongated, high-probability component (rows 10-10, cols 5-40,
    a straight filament-like line) plus a bright patch elsewhere so the
    'bright pixels nearby' context feature has something to find."""
    probs = np.zeros((h, w), dtype=np.float32)
    probs[10, 5:40] = 0.9
    processed = np.full((h, w), 0.3, dtype=np.float32)
    processed[10, 5:40] = 0.1  # the filament is a dark absorption feature
    processed[40:50, 40:50] = 0.95  # a bright (plage-like) patch nearby-ish
    disk_info = (w / 2, h / 2, w / 2 - 1)
    return probs, processed, disk_info


def test_extract_components_basic_shape_and_prob_features():
    probs, processed, disk_info = _toy_scene()
    cfg = skpp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=0, min_area=1)
    features, labels = skcomp.extract_components("toy", probs, processed, disk_info, cfg)

    assert len(features) == 1
    f = features[0]
    assert f.area == 35
    assert f.elongation > 5  # a 35px straight line is highly elongated
    assert f.prob_mean > 0.8
    assert f.prob_max > 0.8
    assert f.contrast > 0  # darker than its surrounding ring
    assert 0.0 <= f.r_over_R <= 2.0
    assert labels.shape == probs.shape
    assert (labels == f.label).sum() == f.area


def test_extract_components_empty_probs_gives_no_components():
    probs = np.zeros((64, 64), dtype=np.float32)
    processed = np.full((64, 64), 0.3, dtype=np.float32)
    cfg = skpp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=0, min_area=1)
    features, labels = skcomp.extract_components("toy", probs, processed, (32, 32, 31), cfg)
    assert features == []
    assert (labels == 0).all()


def test_label_components_tp_and_fp():
    probs, processed, disk_info = _toy_scene()
    cfg = skpp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=0, min_area=1)
    features, labels = skcomp.extract_components("toy", probs, processed, disk_info, cfg)

    h, w = probs.shape
    gt_match = np.zeros((h, w), dtype=np.uint8)
    gt_match[10, 5:40] = 1  # same pixels as the component -> IoU 1.0
    gt_by_annotator = {"A": np.stack([gt_match])}
    skcomp.label_components(features, labels, gt_by_annotator)
    assert features[0].is_tp is True
    assert features[0].best_iou_by_annotator["A"] == 1.0

    gt_elsewhere = np.zeros((h, w), dtype=np.uint8)
    gt_elsewhere[0, 0] = 1
    gt_by_annotator_no_match = {"A": np.stack([gt_elsewhere])}
    features2, labels2 = skcomp.extract_components("toy", probs, processed, disk_info, cfg)
    skcomp.label_components(features2, labels2, gt_by_annotator_no_match)
    assert features2[0].is_tp is False
    assert features2[0].best_iou_by_annotator["A"] == 0.0


def test_components_to_sparse_matches_dense_masks():
    probs, processed, disk_info = _toy_scene()
    cfg = skpp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=0, min_area=1)
    features, labels = skcomp.extract_components("toy", probs, processed, disk_info, cfg)
    ids = [f.label for f in features]
    sparse = skcomp.components_to_sparse(labels, ids)
    assert sparse.shape == (len(features), probs.size)
    for i, f in enumerate(features):
        dense_row = np.asarray(sparse[i].todense()).reshape(labels.shape)
        assert (dense_row == (labels == f.label)).all()

    empty = skcomp.components_to_sparse(labels, [])
    assert empty.shape == (0, probs.size)


def test_to_dataframe_has_expected_columns():
    probs, processed, disk_info = _toy_scene()
    cfg = skpp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=0, min_area=1)
    features, _ = skcomp.extract_components("toy", probs, processed, disk_info, cfg)
    df = skcomp.to_dataframe(features)
    assert len(df) == 1
    for col in skcomp.ALL_FEATURE_COLS:
        assert col in df.columns
    assert "is_tp" in df.columns and "best_iou" in df.columns
