import numpy as np
import pytest

from solar_k import evaluate as ev


def _box(h, w, box):
    m = np.zeros((h, w), dtype=np.uint8)
    r0, r1, c0, c1 = box
    m[r0 : r1 + 1, c0 : c1 + 1] = 1
    return m


def test_pixel_iou_dice_identical_masks():
    m = _box(10, 10, (2, 5, 2, 5))
    iou, dice = ev.pixel_iou_dice(m, m)
    assert iou == pytest.approx(1.0)
    assert dice == pytest.approx(1.0)


def test_pixel_iou_dice_disjoint_masks():
    a = _box(10, 10, (0, 1, 0, 1))
    b = _box(10, 10, (5, 6, 5, 6))
    iou, dice = ev.pixel_iou_dice(a, b)
    assert iou == 0.0
    assert dice == 0.0


def test_pixel_iou_dice_both_empty_is_perfect():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    iou, dice = ev.pixel_iou_dice(a, b)
    assert iou == 1.0 and dice == 1.0


def test_r_bin_thresholds():
    assert ev.r_bin(0.1) == "centre"
    assert ev.r_bin(0.5) == "middle"
    assert ev.r_bin(0.9) == "limb"
    assert ev.r_bin(float("nan")) == "unknown"


def test_size_bucket():
    edges = [100, 200, 300]
    assert ev.size_bucket(50, edges) == "q1_smallest"
    assert ev.size_bucket(150, edges) == "q2"
    assert ev.size_bucket(250, edges) == "q3"
    assert ev.size_bucket(350, edges) == "q4_largest"


def test_loss_decomposition_pure_detection_miss():
    # one GT, no predictions at all anywhere near it
    iou = np.zeros((1, 1))
    counts = ev.pq_loss_decomposition(iou, n_gt=1, n_pred=1)
    assert counts["a"] == {"fn": 1, "fp": 1}
    for c in "bcd":
        assert counts[c] == {"fn": 0, "fp": 0}


def test_loss_decomposition_near_miss():
    iou = np.array([[0.4]])
    counts = ev.pq_loss_decomposition(iou, n_gt=1, n_pred=1)
    assert counts["c"] == {"fn": 1, "fp": 1}
    assert counts["a"] == {"fn": 0, "fp": 0}


def test_loss_decomposition_fragmentation():
    # one GT weakly overlapped by two predictions, none crossing the TP threshold
    iou = np.array([[0.2, 0.15]])
    counts = ev.pq_loss_decomposition(iou, n_gt=1, n_pred=2)
    assert counts["b"] == {"fn": 1, "fp": 2}
    for c in "acd":
        assert counts[c] == {"fn": 0, "fp": 0}


def test_loss_decomposition_merging():
    # two GT weakly overlapped by one prediction, none crossing the TP threshold
    iou = np.array([[0.2], [0.15]])
    counts = ev.pq_loss_decomposition(iou, n_gt=2, n_pred=1)
    assert counts["d"] == {"fn": 2, "fp": 1}
    for c in "abc":
        assert counts[c] == {"fn": 0, "fp": 0}


def test_loss_decomposition_true_positive_contributes_nothing():
    iou = np.array([[0.8]])
    counts = ev.pq_loss_decomposition(iou, n_gt=1, n_pred=1)
    for c in "abcd":
        assert counts[c] == {"fn": 0, "fp": 0}


def test_loss_decomposition_no_gt_all_fp_detection():
    counts = ev.pq_loss_decomposition(np.zeros((0, 3)), n_gt=0, n_pred=3)
    assert counts["a"] == {"fn": 0, "fp": 3}


def test_loss_decomposition_no_pred_all_fn_detection():
    counts = ev.pq_loss_decomposition(np.zeros((3, 0)), n_gt=3, n_pred=0)
    assert counts["a"] == {"fn": 3, "fp": 0}


def test_evaluate_annotator_image_and_aggregate_end_to_end():
    h = w = 10
    gt = np.stack([_box(h, w, (0, 1, 0, 1)), _box(h, w, (5, 6, 5, 6))])
    pred = np.stack([_box(h, w, (0, 1, 0, 1))])  # matches gt[0] exactly, misses gt[1]

    result = ev.evaluate_annotator_image(
        "img1", "ann1", pred, gt, disk_info=(5.0, 5.0, 10.0), site="A"
    )
    assert result.match.tp_ious == [pytest.approx(1.0)]
    assert result.loss_counts["a"]["fn"] == 1  # the missed gt[1]

    summary = ev.aggregate_evaluation([result], quartile_edges=[1, 2, 3])
    assert summary.overall.tp == 1
    assert summary.overall.fn == 1
    assert summary.loss_decomposition["a"]["fn"] == 1


def test_evaluate_annotator_image_sparse_pred_matches_dense():
    import scipy.sparse as sp

    h = w = 10
    gt = np.stack([_box(h, w, (0, 1, 0, 1)), _box(h, w, (5, 6, 5, 6))])
    pred_dense = np.stack([_box(h, w, (0, 1, 0, 1)), _box(h, w, (8, 8, 8, 8))])  # 1 match, 1 lone FP
    pred_sparse = sp.csr_matrix(pred_dense.reshape(2, -1))

    r_dense = ev.evaluate_annotator_image("img1", "ann1", pred_dense, gt, disk_info=(5.0, 5.0, 10.0), site="A")
    r_sparse = ev.evaluate_annotator_image("img1", "ann1", pred_sparse, gt, disk_info=(5.0, 5.0, 10.0), site="A")

    assert r_dense.match.tp_ious == r_sparse.match.tp_ious
    assert r_dense.match.fp == r_sparse.match.fp == 1
    assert r_dense.fp_areas == r_sparse.fp_areas
    assert r_dense.fp_r_over_r == pytest.approx(r_sparse.fp_r_over_r)
    assert r_dense.loss_counts == r_sparse.loss_counts
