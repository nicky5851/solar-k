"""Polygon -> mask rasterisation and RLE encode/decode.

Submission RLE is the pycocotools compressed ``counts`` string from a
Fortran-ordered uint8 mask, with no ``size`` field and no JSON wrapper
(CLAUDE.md). Decoding therefore needs the height/width supplied out of band.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from pycocotools import mask as mask_util


def polygon_to_mask(
    segmentation: Sequence[Sequence[float]], height: int, width: int
) -> np.ndarray:
    """Rasterise a COCO polygon segmentation (possibly multi-part) to a uint8 mask.

    Multiple parts are unioned into one instance mask, matching how a single
    filament can be stored as several disjoint polygon rings.
    """
    if len(segmentation) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    rles = mask_util.frPyObjects(list(segmentation), height, width)
    merged = mask_util.merge(rles)
    return mask_util.decode(merged).astype(np.uint8)


def encode_rle(mask: np.ndarray) -> str:
    """Encode a uint8 {0,1} mask to the bare pycocotools ``counts`` string."""
    fortran = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_util.encode(fortran)
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def decode_rle(counts: str, height: int, width: int) -> np.ndarray:
    """Decode a bare ``counts`` string (no size field) back to a uint8 mask."""
    rle = {"size": [height, width], "counts": counts.encode("ascii")}
    return mask_util.decode(rle).astype(np.uint8)


def mask_area(mask: np.ndarray) -> int:
    return int(mask.sum())
