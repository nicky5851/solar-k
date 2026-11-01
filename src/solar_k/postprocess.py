"""Probability map -> exclusive instance masks.

probs -> binary (single threshold or hysteresis) -> optional morphological
closing -> connected components (8-connectivity) -> drop components smaller
than min_area -> exclusive instance masks. All parameters come from config
(CODING_SPEC_02_BASELINE.md, Stage A).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import scipy.sparse as sp
from skimage.filters import apply_hysteresis_threshold
from skimage.segmentation import watershed


@dataclass
class PostprocessConfig:
    mode: str = "threshold"  # "threshold" or "hysteresis"
    t: float = 0.5
    t_low: float = 0.3
    t_high: float = 0.6
    closing_radius: int = 0
    min_area: int = 1
    # Safety cap, not a tuning knob: no real GT image has anywhere near this
    # many filaments (EDA's busiest image had 26). A bad (t, min_area) combo
    # during the grid search can leave thousands of noise-speckle components
    # above min_area, and stacking that many full-resolution instance masks
    # for IoU/Dice scoring would blow past 8 GB. Keep the largest-area ones;
    # a config this noisy is already scored as heavily FP-penalized either way.
    max_instances: int = 2000

    @classmethod
    def from_dict(cls, d: dict) -> "PostprocessConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def probs_to_binary(probs: np.ndarray, cfg: PostprocessConfig) -> np.ndarray:
    if cfg.mode == "hysteresis":
        if cfg.t_low > cfg.t_high:
            raise ValueError(f"t_low ({cfg.t_low}) must be <= t_high ({cfg.t_high})")
        return apply_hysteresis_threshold(probs, cfg.t_low, cfg.t_high).astype(np.uint8)
    if cfg.mode == "threshold":
        return (probs >= cfg.t).astype(np.uint8)
    raise ValueError(f"unknown postprocess mode: {cfg.mode!r}")


def morphological_close(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def extract_instances(binary: np.ndarray, min_area: int, connectivity: int = 8, max_instances: int | None = None) -> list[np.ndarray]:
    """Connected components -> exclusive instance masks, dropping small ones."""
    cc_connectivity = 8 if connectivity == 8 else 4
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=cc_connectivity)
    kept = [(label, stats[label, cv2.CC_STAT_AREA]) for label in range(1, n) if stats[label, cv2.CC_STAT_AREA] >= min_area]
    if max_instances is not None and len(kept) > max_instances:
        kept.sort(key=lambda t: t[1], reverse=True)
        kept = kept[:max_instances]
    return [(labels == label).astype(np.uint8) for label, _ in kept]


def postprocess(probs: np.ndarray, cfg: PostprocessConfig) -> list[np.ndarray]:
    binary = probs_to_binary(probs, cfg)
    closed = morphological_close(binary, cfg.closing_radius)
    return extract_instances(closed, cfg.min_area, max_instances=cfg.max_instances)


def extract_instances_sparse(binary: np.ndarray, min_area: int, connectivity: int = 8, max_instances: int | None = None) -> sp.csr_matrix:
    """Like ``extract_instances`` but returns a ``(n_kept, H*W)`` boolean CSR
    matrix instead of a list of dense ``(H, W)`` masks.

    Built directly from the label image in one vectorised pass, so memory
    scales with the number of foreground pixels, not with
    ``n_instances * H * W``. Needed for scoring during the tuning grid
    search: an untuned (low threshold, small min_area) cell can leave
    thousands of noise-speckle components, and stacking that many
    full-resolution dense masks (as ``extract_instances`` does, meant for
    realistic instance counts at inference time) would blow past 8 GB.
    """
    cc_connectivity = 8 if connectivity == 8 else 4
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=cc_connectivity)
    h, w = labels.shape
    if n <= 1:
        return sp.csr_matrix((0, h * w), dtype=np.uint8)

    areas_all = stats[1:n, cv2.CC_STAT_AREA]
    label_ids_all = np.arange(1, n)
    keep = areas_all >= min_area
    kept_ids, kept_areas = label_ids_all[keep], areas_all[keep]
    if max_instances is not None and len(kept_ids) > max_instances:
        order = np.argsort(kept_areas)[::-1][:max_instances]
        kept_ids = kept_ids[order]
    if len(kept_ids) == 0:
        return sp.csr_matrix((0, h * w), dtype=np.uint8)

    row_lookup = np.full(n, -1, dtype=np.int64)
    row_lookup[kept_ids] = np.arange(len(kept_ids))

    flat_labels = labels.reshape(-1)
    fg_idx = np.flatnonzero(flat_labels)
    rows = row_lookup[flat_labels[fg_idx]]
    valid = rows >= 0
    rows, cols = rows[valid], fg_idx[valid]
    data = np.ones(len(rows), dtype=np.uint8)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(kept_ids), h * w))


def postprocess_sparse(probs: np.ndarray, cfg: PostprocessConfig) -> sp.csr_matrix:
    binary = probs_to_binary(probs, cfg)
    closed = morphological_close(binary, cfg.closing_radius)
    return extract_instances_sparse(closed, cfg.min_area, max_instances=cfg.max_instances)


def regrow_instances(probs: np.ndarray, seed_masks: list[np.ndarray], t_mask: float, d: int) -> list[np.ndarray]:
    """Two-stage mask extent fix (CODING_SPEC_04_MODEL_SIDE.md E008).

    `seed_masks` are the final kept instances from the normal detect+scorer
    pipeline (one mask decides *whether* an instance exists); this re-grows
    each one's *extent* out to a lower threshold `t_mask`, capped to within a
    `d`-pixel dilation of its seed. A marker-controlled watershed (markers =
    seed instance ids, elevation = -probs, flood region = each seed's
    d-pixel-dilated footprint intersected with probs >= t_mask) assigns every
    candidate pixel to at most one instance, so regrown masks never merge or
    overlap -- unlike dilating each seed independently.
    """
    if not seed_masks:
        return []
    h, w = probs.shape
    n = len(seed_masks)
    markers = np.zeros((h, w), dtype=np.int32)
    for i, m in enumerate(seed_masks, start=1):
        markers[m.astype(bool)] = i

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)) if d > 0 else None
    allowed = np.zeros((h, w), dtype=bool)
    for m in seed_masks:
        m_u8 = m.astype(np.uint8)
        dil = cv2.dilate(m_u8, kernel).astype(bool) if d > 0 else m_u8.astype(bool)
        allowed |= dil

    region = allowed & ((probs >= t_mask) | (markers > 0))
    labels_grown = watershed(-probs, markers=markers, mask=region)
    return [(labels_grown == i).astype(np.uint8) for i in range(1, n + 1)]
