import numpy as np
import pytest

from solar_k import masks


def test_rle_round_trip_pixel_exact():
    rng = np.random.RandomState(0)
    mask = (rng.rand(32, 40) > 0.7).astype(np.uint8)
    counts = masks.encode_rle(mask)
    assert isinstance(counts, str)
    decoded = masks.decode_rle(counts, 32, 40)
    assert decoded.shape == (32, 40)
    assert np.array_equal(decoded, mask)


def test_rle_round_trip_empty_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    counts = masks.encode_rle(mask)
    decoded = masks.decode_rle(counts, 10, 10)
    assert np.array_equal(decoded, mask)
    assert masks.mask_area(decoded) == 0


def test_rle_round_trip_full_mask():
    mask = np.ones((10, 10), dtype=np.uint8)
    counts = masks.encode_rle(mask)
    decoded = masks.decode_rle(counts, 10, 10)
    assert np.array_equal(decoded, mask)
    assert masks.mask_area(decoded) == 100


def test_polygon_to_mask_simple_square():
    # a 4x4 square, (x, y) pairs flattened as COCO expects
    polygon = [[0, 0, 4, 0, 4, 4, 0, 4]]
    mask = masks.polygon_to_mask(polygon, height=10, width=10)
    assert mask.shape == (10, 10)
    assert masks.mask_area(mask) == pytest.approx(16, abs=2)
    assert mask[1, 1] == 1
    assert mask[8, 8] == 0


def test_polygon_to_mask_multi_part_unions():
    square_a = [0, 0, 2, 0, 2, 2, 0, 2]
    square_b = [5, 5, 7, 5, 7, 7, 5, 7]
    mask = masks.polygon_to_mask([square_a, square_b], height=10, width=10)
    assert mask[1, 1] == 1
    assert mask[6, 6] == 1
    assert mask[3, 3] == 0


def test_polygon_to_mask_empty_segmentation():
    mask = masks.polygon_to_mask([], height=5, width=5)
    assert mask.shape == (5, 5)
    assert masks.mask_area(mask) == 0
