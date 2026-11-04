"""Per-predicted-component feature table (CODING_SPEC_03_FP_AND_FOLDS.md,
E003 step 2): one row per connected component of ``postprocess.py``'s
binarize -> morphological-close -> connected-components pipeline, built
*before* the min_area/max_instances filter is applied, so the shape gate
and learned scorer (E003 steps 3-4) have small/weak components to learn
to drop, not just the ones already large enough to survive.

Feature groups (CODING_SPEC_03_FP_AND_FOLDS.md):
  - shape: area, skeleton length, elongation, mean/max width, solidity,
    orientation
  - probability: mean, max, p90 under the mask
  - context: mean intensity inside the component, mean intensity in a
    15 px ring around it, their contrast, local texture (std) in a 64 px
    window, and the fraction of bright pixels nearby (a plage indicator)
  - position: r/R
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from . import metric as skmetric
from . import postprocess as skpp
from . import preprocess as skpre

RING_RADIUS = 15
TEXTURE_WINDOW = 64
BRIGHT_PERCENTILE = 90.0


@dataclass
class ComponentFeatures:
    stem: str
    label: int
    area: int
    length: float
    elongation: float
    mean_width: float
    max_width: float
    solidity: float
    orientation: float
    prob_mean: float
    prob_max: float
    prob_p90: float
    local_intensity: float
    ring_intensity: float
    contrast: float
    texture_std: float
    frac_bright_nearby: float
    r_over_R: float
    is_tp: bool = False
    best_iou_by_annotator: dict[str, float] = field(default_factory=dict)
    # CODING_SPEC_06 E019: cross-member agreement features, nan unless
    # extract_components is given member_probs.
    agree_mean: float = float("nan")
    agree_min: float = float("nan")
    agree_max: float = float("nan")
    agree_std: float = float("nan")
    agree_frac_majority: float = float("nan")
    boundary_ring_prob: float = float("nan")


def _bbox(mask: np.ndarray, pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    h, w = shape
    return max(0, ys.min() - pad), min(h - 1, ys.max() + pad), max(0, xs.min() - pad), min(w - 1, xs.max() + pad)


def _shape_features(mask_crop: np.ndarray) -> dict:
    area = int(mask_crop.sum())
    skel = skeletonize(mask_crop.astype(bool))
    skel_len = float(skel.sum())
    elongation = skel_len / np.sqrt(area) if area > 0 else float("nan")
    if skel_len > 0:
        dt = distance_transform_edt(mask_crop)
        widths = 2 * dt[skel]
        mean_width, max_width = float(widths.mean()), float(widths.max())
    else:
        mean_width = max_width = float("nan")

    contours, _ = cv2.findContours(mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = orientation = float("nan")
    if contours:
        c = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area > 0:
            solidity = float(area / hull_area)
        if len(c) >= 5:
            orientation = _orientation(c)
    return {
        "area": area, "length": skel_len, "elongation": elongation,
        "mean_width": mean_width, "max_width": max_width, "solidity": solidity, "orientation": orientation,
    }


def _orientation(contour: np.ndarray) -> float:
    """Ellipse-fit angle in degrees, [0, 180). ``cv2.fitEllipse`` on a
    degenerate contour (all points collinear, e.g. a 1-px-wide straight
    component) returns values that depend on uninitialised memory -- the
    same component gave 22.5 or 0.0 depending on what ran earlier in the
    process, which made the learned scorer and its cutoff c
    non-deterministic (CODING_SPEC_05_ENSEMBLE.md section 2.1). For those,
    use the second-moment angle of the contour points instead (deterministic,
    same [0, 180) range); well-conditioned fits are unchanged."""
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    if min(w, h) >= 1.0:
        (_, _), (_, _), angle = cv2.fitEllipse(contour)
        return float(angle)
    pts = contour.reshape(-1, 2).astype(np.float64)
    d = pts - pts.mean(axis=0)
    mu20, mu02, mu11 = (d[:, 0] ** 2).mean(), (d[:, 1] ** 2).mean(), (d[:, 0] * d[:, 1]).mean()
    return float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)) % 180.0)


def _prob_features(mask: np.ndarray, probs: np.ndarray) -> dict:
    vals = probs[mask.astype(bool)]
    if len(vals) == 0:
        return {"prob_mean": float("nan"), "prob_max": float("nan"), "prob_p90": float("nan")}
    return {"prob_mean": float(vals.mean()), "prob_max": float(vals.max()), "prob_p90": float(np.percentile(vals, 90))}


def _context_features(mask: np.ndarray, processed: np.ndarray, bright_threshold: float) -> dict:
    h, w = mask.shape
    box = _bbox(mask, pad=RING_RADIUS + 2, shape=(h, w))
    nan_out = {
        "local_intensity": float("nan"), "ring_intensity": float("nan"), "contrast": float("nan"),
        "texture_std": float("nan"), "frac_bright_nearby": float("nan"),
    }
    if box is None:
        return nan_out
    y0, y1, x0, x1 = box
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
    crop_img = processed[y0 : y1 + 1, x0 : x1 + 1]
    ring = cv2.dilate(crop_mask, np.ones((2 * RING_RADIUS + 1, 2 * RING_RADIUS + 1), np.uint8)) & (1 - crop_mask)
    if not crop_mask.any() or not ring.any():
        return nan_out
    local_intensity = float(crop_img[crop_mask.astype(bool)].mean())
    ring_intensity = float(crop_img[ring.astype(bool)].mean())

    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    half = TEXTURE_WINDOW // 2
    ty0, ty1 = int(max(0, cy - half)), int(min(h, cy + half))
    tx0, tx1 = int(max(0, cx - half)), int(min(w, cx + half))
    window = processed[ty0:ty1, tx0:tx1]
    texture_std = float(window.std()) if window.size else float("nan")
    frac_bright = float((window > bright_threshold).mean()) if window.size else float("nan")
    return {
        "local_intensity": local_intensity, "ring_intensity": ring_intensity,
        # positive = component darker than its surroundings, the expected
        # sign for a dark absorption feature (same convention as the Spec 02
        # deep-dive's instance-level contrast proxy).
        "contrast": ring_intensity - local_intensity,
        "texture_std": texture_std, "frac_bright_nearby": frac_bright,
    }


def _agreement_features(mask: np.ndarray, member_probs: dict[str, np.ndarray] | None, avg_probs: np.ndarray) -> dict:
    """CODING_SPEC_06 E019 scorer v2: per-member agreement on this component,
    plus a probability-map (not image-intensity) boundary-contrast feature.
    ``member_probs`` comes only from the members that exist for this OOF
    image (fold-matched) -- see CODING_SPEC_06's agreement-feature note."""
    nan_out = {"agree_mean": float("nan"), "agree_min": float("nan"), "agree_max": float("nan"),
               "agree_std": float("nan"), "agree_frac_majority": float("nan"), "boundary_ring_prob": float("nan")}
    if not mask.any():
        return nan_out
    ring = cv2.dilate(mask, np.ones((3, 3), np.uint8)) & (1 - mask)
    boundary_ring_prob = float(avg_probs[ring.astype(bool)].mean()) if ring.any() else float("nan")
    if not member_probs:
        return {**nan_out, "boundary_ring_prob": boundary_ring_prob}
    mask_bool = mask.astype(bool)
    member_means, majority_flags = [], []
    for probs in member_probs.values():
        vals = probs[mask_bool]
        member_means.append(float(vals.mean()))
        majority_flags.append(float((vals > 0.5).mean() >= 0.5))
    arr = np.array(member_means)
    return {
        "agree_mean": float(arr.mean()), "agree_min": float(arr.min()), "agree_max": float(arr.max()),
        "agree_std": float(arr.std()) if len(arr) > 1 else 0.0,
        "agree_frac_majority": float(np.mean(majority_flags)),
        "boundary_ring_prob": boundary_ring_prob,
    }


def _r_over_r(mask: np.ndarray, cx: float, cy: float, radius: float) -> float:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0 or radius <= 0:
        return float("nan")
    return float(np.sqrt((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2) / radius)


def extract_components(
    stem: str,
    probs: np.ndarray,
    processed: np.ndarray,
    disk_info: tuple[float, float, float],
    cfg: skpp.PostprocessConfig,
    member_probs: dict[str, np.ndarray] | None = None,
) -> tuple[list[ComponentFeatures], np.ndarray]:
    """Binarize -> close -> connected components (``cfg``'s mode/t/closing_radius;
    ``cfg.min_area``/``cfg.max_instances`` are ignored here on purpose -- the
    shape gate and scorer need the small/weak components too) -> one
    ``ComponentFeatures`` row per component. Returns ``(features, labels)``;
    ``labels == f.label`` reconstructs a given component's mask without
    storing every mask densely."""
    cx, cy, radius = disk_info
    binary = skpp.probs_to_binary(probs, cfg)
    closed = skpp.morphological_close(binary, cfg.closing_radius)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
    h, w = labels.shape

    disk_mask, _ = skpre.disk_mask_and_r((h, w), cx, cy, radius)
    bright_threshold = float(np.percentile(processed[disk_mask], BRIGHT_PERCENTILE)) if disk_mask.any() else float("nan")

    out = []
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] < 1:
            continue
        mask = (labels == label).astype(np.uint8)
        box = _bbox(mask, pad=2, shape=(h, w))
        if box is None:
            continue
        y0, y1, x0, x1 = box
        shape_feats = _shape_features(mask[y0 : y1 + 1, x0 : x1 + 1])
        prob_feats = _prob_features(mask, probs)
        ctx_feats = _context_features(mask, processed, bright_threshold)
        agree_feats = _agreement_features(mask, member_probs, probs)
        out.append(
            ComponentFeatures(
                stem=stem, label=label,
                r_over_R=_r_over_r(mask, cx, cy, radius),
                **shape_feats, **prob_feats, **ctx_feats, **agree_feats,
            )
        )
    return out, labels


def label_components(
    features: list[ComponentFeatures],
    labels_img: np.ndarray,
    gt_by_annotator: dict[str, np.ndarray],
    iou_threshold: float = skmetric.IOU_THRESHOLD,
) -> None:
    """Mutate each ``ComponentFeatures`` in place: ``is_tp`` (matched, IoU >
    threshold, for at least one annotator) and ``best_iou_by_annotator``."""
    if not features:
        return
    pred_stack = np.stack([(labels_img == f.label).astype(np.uint8) for f in features])
    for annotator, gt_masks in gt_by_annotator.items():
        if gt_masks.shape[0] == 0:
            for f in features:
                f.best_iou_by_annotator[annotator] = 0.0
            continue
        iou, _ = skmetric.pairwise_iou_dice(gt_masks, pred_stack)
        best_per_pred = iou.max(axis=0)
        for f, b in zip(features, best_per_pred):
            f.best_iou_by_annotator[annotator] = float(b)
    for f in features:
        f.is_tp = any(v > iou_threshold for v in f.best_iou_by_annotator.values())


def components_to_sparse(labels: np.ndarray, label_ids: list[int]) -> sp.csr_matrix:
    """Vectorised dense-labels -> ``(n, H*W)`` boolean CSR, one row per id in
    ``label_ids`` (order preserved, e.g. matching a feature table's row
    order) -- the same technique as ``postprocess.extract_instances_sparse``,
    so cost scales with foreground pixel count, not ``n_labels * H * W``.
    Lets a caller build the full per-image component stack once and reuse
    it across many candidate keep-filters via cheap boolean row selection
    (``sparse[mask]``), instead of re-running connected components per
    filter choice."""
    h, w = labels.shape
    if not label_ids:
        return sp.csr_matrix((0, h * w), dtype=np.uint8)
    n = int(labels.max()) + 1
    row_lookup = np.full(n, -1, dtype=np.int64)
    row_lookup[label_ids] = np.arange(len(label_ids))
    flat = labels.reshape(-1)
    fg_idx = np.flatnonzero(flat)
    rows = row_lookup[flat[fg_idx]]
    valid = rows >= 0
    rows, cols = rows[valid], fg_idx[valid]
    data = np.ones(len(rows), dtype=np.uint8)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(label_ids), h * w))


SHAPE_FEATURE_COLS =["area", "length", "elongation", "mean_width", "max_width", "solidity", "orientation"]
PROB_FEATURE_COLS = ["prob_mean", "prob_max", "prob_p90"]
CONTEXT_FEATURE_COLS = ["local_intensity", "ring_intensity", "contrast", "texture_std", "frac_bright_nearby"]
POSITION_FEATURE_COLS = ["r_over_R"]
ALL_FEATURE_COLS = SHAPE_FEATURE_COLS + PROB_FEATURE_COLS + CONTEXT_FEATURE_COLS + POSITION_FEATURE_COLS
# CODING_SPEC_06 E019 scorer v2: agreement features, nan unless member_probs
# was supplied to extract_components.
AGREEMENT_FEATURE_COLS = ["agree_mean", "agree_min", "agree_max", "agree_std", "agree_frac_majority", "boundary_ring_prob"]
ALL_FEATURE_COLS_V2 = ALL_FEATURE_COLS + AGREEMENT_FEATURE_COLS


def to_dataframe(features: list[ComponentFeatures]) -> pd.DataFrame:
    rows = []
    for f in features:
        d = asdict(f)
        best_iou_by_annotator = d.pop("best_iou_by_annotator")
        d["best_iou"] = max(best_iou_by_annotator.values()) if best_iou_by_annotator else 0.0
        rows.append(d)
    cols = ["stem", "label", *ALL_FEATURE_COLS_V2, "is_tp", "best_iou"]
    return pd.DataFrame(rows, columns=cols)
