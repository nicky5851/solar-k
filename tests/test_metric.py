import numpy as np
import pytest
import scipy.sparse as sp

from solar_k import metric


def _mask(h, w, box):
    """box = (r0, r1, c0, c1), inclusive."""
    m = np.zeros((h, w), dtype=np.uint8)
    r0, r1, c0, c1 = box
    m[r0 : r1 + 1, c0 : c1 + 1] = 1
    return m


def _stack(masks):
    return np.stack(masks, axis=0) if masks else np.zeros((0, 6, 6), dtype=np.uint8)


def test_gt_vs_itself_is_pq_one():
    gt = _stack([_mask(6, 6, (0, 1, 0, 1)), _mask(6, 6, (3, 5, 3, 5))])
    result = metric.score_dataset([(gt, gt)])
    assert result.pq == pytest.approx(1.0)
    assert result.sq == pytest.approx(1.0)
    assert result.rq == pytest.approx(1.0)
    assert result.tp == 2
    assert result.fp == 0
    assert result.fn == 0


def test_empty_prediction_is_pq_zero():
    gt = _stack([_mask(6, 6, (0, 1, 0, 1))])
    pred = _stack([])
    result = metric.score_dataset([(gt, pred)])
    assert result.pq == 0.0
    assert result.tp == 0
    assert result.fn == 1
    assert result.fp == 0


def test_both_empty_contributes_nothing():
    gt = np.zeros((0, 6, 6), dtype=np.uint8)
    pred = np.zeros((0, 6, 6), dtype=np.uint8)
    result = metric.score_dataset([(gt, pred)])
    assert result.pq == 0.0
    assert result.tp == 0 and result.fp == 0 and result.fn == 0


def test_no_gt_all_predictions_are_false_positives():
    gt = np.zeros((0, 6, 6), dtype=np.uint8)
    pred = _stack([_mask(6, 6, (0, 1, 0, 1)), _mask(6, 6, (3, 5, 3, 5))])
    result = metric.score_dataset([(gt, pred)])
    assert result.fp == 2
    assert result.tp == 0 and result.fn == 0
    assert result.pq == 0.0


def test_hand_built_two_instance_case_known_answer():
    # instance 1: exact match, IoU = Dice = 1
    gt1 = _mask(6, 6, (0, 1, 0, 1))
    pred1 = _mask(6, 6, (0, 1, 0, 1))
    # instance 2: gt area 9 (3x3), pred area 6 (3x2), intersection 6 (3x2 overlap)
    # IoU = 6 / (9 + 6 - 6) = 6/9 = 2/3; Dice = 12/15 = 0.8
    gt2 = _mask(6, 6, (3, 5, 3, 5))
    pred2 = _mask(6, 6, (3, 5, 3, 4))

    gt = _stack([gt1, gt2])
    pred = _stack([pred1, pred2])

    iou, dice = metric.pairwise_iou_dice(gt, pred)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[1, 1] == pytest.approx(2 / 3)
    assert dice[1, 1] == pytest.approx(0.8)

    result = metric.score_dataset([(gt, pred)])
    assert result.tp == 2
    assert result.fp == 0 and result.fn == 0
    expected_pq = (1.0 + 2 / 3) / 2
    assert result.pq == pytest.approx(expected_pq)


def test_merged_instance_case():
    # two separate GT instances; one prediction spans both plus a gap between
    # them, so IoU with each GT stays below 0.5 -> neither matches.
    gt_a = _mask(6, 6, (0, 1, 0, 1))
    gt_b = _mask(6, 6, (0, 1, 3, 4))
    pred_merged = _mask(6, 6, (0, 1, 0, 4))

    gt = _stack([gt_a, gt_b])
    pred = _stack([pred_merged])

    result = metric.score_dataset([(gt, pred)])
    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 2
    assert result.pq == 0.0


def test_fragmented_instance_case():
    # one GT instance split into two non-overlapping predictions; each
    # prediction's IoU with the GT stays below 0.5.
    gt_full = _mask(6, 6, (0, 1, 0, 4))
    pred_a = _mask(6, 6, (0, 1, 0, 1))
    pred_b = _mask(6, 6, (0, 1, 3, 4))

    gt = _stack([gt_full])
    pred = _stack([pred_a, pred_b])

    result = metric.score_dataset([(gt, pred)])
    assert result.tp == 0
    assert result.fn == 1
    assert result.fp == 2
    assert result.pq == 0.0


def test_aggregation_sums_across_images_not_mean_of_pq():
    # image 1: perfect match (contributes TP=1, IoU=1)
    gt1 = _stack([_mask(6, 6, (0, 1, 0, 1))])
    pred1 = gt1
    # image 2: no predictions (contributes FN=1)
    gt2 = _stack([_mask(6, 6, (0, 1, 0, 1))])
    pred2 = _stack([])

    result = metric.score_dataset([(gt1, pred1), (gt2, pred2)])
    # global: TP=1, FP=0, FN=1 -> PQ = 1 / (1 + 0.5) = 2/3, NOT (1 + 0)/2 = 0.5
    assert result.pq == pytest.approx(2 / 3)


def test_pairwise_iou_dice_sparse_pred_matches_dense():
    gt = _stack([_mask(6, 6, (0, 1, 0, 1)), _mask(6, 6, (3, 5, 3, 5))])
    pred_dense = _stack([_mask(6, 6, (0, 1, 0, 1)), _mask(6, 6, (3, 4, 3, 4))])  # partial overlap on 2nd

    iou_dense, dice_dense = metric.pairwise_iou_dice(gt, pred_dense)

    pred_sparse = sp.csr_matrix(pred_dense.reshape(2, -1))
    iou_sparse, dice_sparse = metric.pairwise_iou_dice(gt, pred_sparse)

    assert np.allclose(iou_dense, iou_sparse)
    assert np.allclose(dice_dense, dice_sparse)


def test_pairwise_iou_dice_sparse_zero_pred_rows():
    gt = _stack([_mask(6, 6, (0, 1, 0, 1))])
    pred_sparse = sp.csr_matrix((0, 36), dtype=np.uint8)
    iou, dice = metric.pairwise_iou_dice(gt, pred_sparse)
    assert iou.shape == (1, 0)
    assert dice.shape == (1, 0)
