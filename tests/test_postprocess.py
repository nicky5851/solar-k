import numpy as np
import pytest

from solar_k import postprocess as pp


def test_threshold_known_toy_case():
    probs = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    cfg = pp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=0, min_area=1)
    instances = pp.postprocess(probs, cfg)
    # one 2x2 blob (area 4) and one isolated pixel (area 1)
    areas = sorted(int(m.sum()) for m in instances)
    assert areas == [1, 4]


def test_min_area_drops_small_components():
    probs = np.array(
        [
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.9, 0.1, 0.1],
            [0.1, 0.9, 0.9, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    cfg = pp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=0, min_area=2)
    instances = pp.postprocess(probs, cfg)
    assert len(instances) == 1
    assert int(instances[0].sum()) == 4


def test_instances_are_exclusive_and_non_empty():
    rng = np.random.RandomState(0)
    probs = rng.rand(50, 50).astype(np.float32)
    cfg = pp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=1, min_area=3)
    instances = pp.postprocess(probs, cfg)
    assert len(instances) > 0
    for m in instances:
        assert m.sum() > 0
    if len(instances) > 1:
        stacked = np.stack(instances)
        overlap = stacked.sum(axis=0)
        assert overlap.max() <= 1  # no two instances share a pixel


def test_hysteresis_grows_seeds_into_low_threshold_region():
    probs = np.zeros((10, 10), dtype=np.float32)
    probs[4, 4] = 0.9  # seed above t_high
    probs[4, 5] = 0.4  # only above t_low, connected to the seed
    probs[0, 0] = 0.4  # above t_low but isolated -- should NOT be kept

    cfg = pp.PostprocessConfig(mode="hysteresis", t_low=0.3, t_high=0.7, closing_radius=0, min_area=1)
    instances = pp.postprocess(probs, cfg)
    assert len(instances) == 1
    assert int(instances[0].sum()) == 2
    assert instances[0][4, 4] == 1 and instances[0][4, 5] == 1
    assert instances[0][0, 0] == 0


def test_hysteresis_rejects_t_low_greater_than_t_high():
    probs = np.zeros((5, 5), dtype=np.float32)
    cfg = pp.PostprocessConfig(mode="hysteresis", t_low=0.8, t_high=0.3)
    with pytest.raises(ValueError):
        pp.probs_to_binary(probs, cfg)


def test_morphological_closing_fills_small_gap():
    # two 4px-tall blobs with a 1px gap between them -- a cross-shaped
    # radius-1 kernel needs vertical neighbours on both sides of the
    # bridged pixel too, so the blobs must be taller than 2px for the gap
    # to actually close under erosion.
    binary = np.zeros((10, 10), dtype=np.uint8)
    binary[3:7, 2:4] = 1
    binary[3:7, 5:7] = 1  # a 1px gap at column 4
    closed = pp.morphological_close(binary, radius=1)
    assert closed[4, 4] == 1 and closed[5, 4] == 1


def test_empty_probs_give_no_instances():
    probs = np.zeros((20, 20), dtype=np.float32)
    cfg = pp.PostprocessConfig(mode="threshold", t=0.5, min_area=1)
    instances = pp.postprocess(probs, cfg)
    assert instances == []


def test_postprocess_sparse_matches_dense():
    rng = np.random.RandomState(0)
    probs = rng.rand(50, 50).astype(np.float32)
    cfg = pp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=1, min_area=3)
    dense = pp.postprocess(probs, cfg)
    sparse = pp.postprocess_sparse(probs, cfg)

    assert sparse.shape == (len(dense), 50 * 50)
    dense_areas = sorted(int(m.sum()) for m in dense)
    sparse_areas = sorted(int(sparse.getrow(i).sum()) for i in range(sparse.shape[0]))
    assert dense_areas == sparse_areas

    dense_stack = np.stack(dense) if dense else np.zeros((0, 50, 50), dtype=np.uint8)
    dense_union = dense_stack.reshape(len(dense), -1).sum(axis=0) if dense else np.zeros(50 * 50)
    sparse_union = np.asarray(sparse.sum(axis=0)).ravel()
    assert np.array_equal(dense_union, sparse_union)


def test_postprocess_sparse_empty_gives_zero_row_matrix():
    probs = np.zeros((20, 20), dtype=np.float32)
    cfg = pp.PostprocessConfig(mode="threshold", t=0.5, min_area=1)
    sparse = pp.postprocess_sparse(probs, cfg)
    assert sparse.shape == (0, 400)


def test_extract_instances_sparse_caps_to_largest_by_area():
    # three isolated single pixels far apart, plus one 2x2 blob
    binary = np.zeros((20, 20), dtype=np.uint8)
    binary[0, 0] = 1
    binary[0, 10] = 1
    binary[10, 0] = 1
    binary[15:17, 15:17] = 1  # area 4, the largest
    sparse = pp.extract_instances_sparse(binary, min_area=1, max_instances=1)
    assert sparse.shape[0] == 1
    assert int(sparse.sum()) == 4  # kept the 2x2 blob, not a lone pixel


def test_regrow_instances_empty_seeds_returns_empty_list():
    probs = np.zeros((20, 20), dtype=np.float32)
    assert pp.regrow_instances(probs, [], t_mask=0.3, d=4) == []


def test_regrow_instances_grows_single_seed_toward_lower_threshold():
    probs = np.zeros((20, 20), dtype=np.float32)
    probs[10, 10] = 0.9  # a single confident seed pixel
    probs[10, 11] = 0.4  # neighbour just above t_mask, below t_det
    probs[10, 9] = 0.1  # neighbour below t_mask -- should not be grown into
    seed = np.zeros((20, 20), dtype=np.uint8)
    seed[10, 10] = 1

    grown = pp.regrow_instances(probs, [seed], t_mask=0.3, d=2)
    assert len(grown) == 1
    assert grown[0][10, 10] == 1
    assert grown[0][10, 11] == 1  # grown in: above t_mask, within d
    assert grown[0][10, 9] == 0  # not grown in: below t_mask


def test_regrow_instances_respects_geodesic_distance_cap():
    probs = np.full((40, 40), 0.9, dtype=np.float32)  # whole image above t_mask
    seed = np.zeros((40, 40), dtype=np.uint8)
    seed[20, 20] = 1

    grown = pp.regrow_instances(probs, [seed], t_mask=0.3, d=3)
    ys, xs = np.nonzero(grown[0])
    # every grown pixel stays within the seed's d-pixel dilated footprint
    assert np.all(np.abs(ys - 20) <= 3) and np.all(np.abs(xs - 20) <= 3)
    # and it did grow beyond the single seed pixel
    assert len(ys) > 1


def test_regrow_instances_does_not_merge_adjacent_instances():
    probs = np.full((20, 20), 0.9, dtype=np.float32)  # everything eligible to grow
    seed_a = np.zeros((20, 20), dtype=np.uint8)
    seed_a[5, 5] = 1
    seed_b = np.zeros((20, 20), dtype=np.uint8)
    seed_b[5, 10] = 1  # close enough that a naive independent dilation by d=4 would overlap

    grown = pp.regrow_instances(probs, [seed_a, seed_b], t_mask=0.3, d=4)
    assert len(grown) == 2
    overlap = grown[0].astype(bool) & grown[1].astype(bool)
    assert not overlap.any()  # exclusive masks: no pixel claimed by both
    assert grown[0][5, 5] == 1 and grown[1][5, 10] == 1  # each still contains its own seed
