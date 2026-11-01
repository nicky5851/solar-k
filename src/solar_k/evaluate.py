"""Evaluation: official PQ/SQ/RQ plus the diagnostics CODING_SPEC_02_BASELINE.md
asks for (pixel Dice/IoU, size-bucket FP/FN, fragmentation/merge counts,
near-misses, PQ by size/position/site, and a PQ-loss decomposition into
(a) detection (b) fragmentation (c) near-miss IoU (d) merging).

Atomic unit throughout is the (annotator, image) pair, the same unit the
official notebook (and metric.py) aggregates over: a prediction is scored
against every annotator of an image separately, never against one
arbitrarily-chosen annotator or a union.

The (a)-(d) decomposition explains the **RQ loss** (the FP/FN side of
PQ = SQ * RQ) -- it assigns every FP and FN metric.py counts to one failure
mode. It does not explain the **SQ loss** (imperfect IoU among TPs); that is
reported separately, consistent with the inter-annotator SQ reference.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import metric as skmetric

WEAK_IOU = 0.1
NEAR_MISS_RANGE = (0.3, 0.5)
FAILURE_MODES = {"a": "detection", "b": "fragmentation", "c": "near_miss", "d": "merging"}


def pixel_iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    pred_b, gt_b = pred_mask.astype(bool), gt_mask.astype(bool)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    iou = (inter / union) if union > 0 else 1.0
    denom = int(pred_b.sum()) + int(gt_b.sum())
    dice = (2 * inter / denom) if denom > 0 else 1.0
    return float(iou), float(dice)


def semantic_union(instances: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for m in instances:
        np.maximum(out, m, out=out)
    return out


def _semantic_union_any(masks, shape: tuple[int, int]) -> np.ndarray:
    """``semantic_union`` that also accepts a sparse ``(n, H*W)`` CSR stack
    (as produced by ``postprocess.extract_instances_sparse``) without ever
    densifying the individual rows."""
    if sp.issparse(masks):
        if masks.shape[0] == 0:
            return np.zeros(shape, dtype=np.uint8)
        summed = np.asarray(masks.sum(axis=0)).reshape(shape)
        return (summed > 0).astype(np.uint8)
    return semantic_union(list(masks), shape)


def _union_find_components(n_gt: int, n_pred: int, edges: list[tuple[int, int]]):
    parent = list(range(n_gt + n_pred))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for gi, pj in edges:
        union(gi, n_gt + pj)

    groups: dict[int, tuple[list[int], list[int]]] = collections.defaultdict(lambda: ([], []))
    for gi in range(n_gt):
        groups[find(gi)][0].append(gi)
    for pj in range(n_pred):
        groups[find(n_gt + pj)][1].append(pj)
    return list(groups.values())


def classify_component(gt_idxs: list[int], pred_idxs: list[int], iou_submatrix: np.ndarray) -> tuple[str, int, int]:
    """One connected component of the weak-overlap graph -> (failure code, fn count, fp count)."""
    ng, npd = len(gt_idxs), len(pred_idxs)
    if ng == 0:
        return "a", 0, npd
    if npd == 0:
        return "a", ng, 0
    if ng == 1 and npd == 1:
        iou_val = iou_submatrix[gt_idxs[0], pred_idxs[0]]
        cat = "c" if NEAR_MISS_RANGE[0] <= iou_val < NEAR_MISS_RANGE[1] else "a"
        return cat, 1, 1
    if ng == 1 and npd > 1:
        return "b", 1, npd
    if npd == 1 and ng > 1:
        return "d", ng, 1
    # ambiguous many-to-many: lean on whichever side has more nodes
    cat = "b" if npd > ng else ("d" if ng > npd else "a")
    return cat, ng, npd


def pq_loss_decomposition(
    iou_matrix: np.ndarray, n_gt: int, n_pred: int, iou_threshold: float = skmetric.IOU_THRESHOLD, weak_iou: float = WEAK_IOU
) -> dict[str, dict[str, int]]:
    """Assign every FP/FN metric.py would count (iou_matrix > iou_threshold hit
    test) to exactly one of a/b/c/d, via connected components of a weak
    (iou > weak_iou) overlap graph restricted to the unmatched instances."""
    counts = {c: {"fn": 0, "fp": 0} for c in "abcd"}
    if n_gt == 0:
        counts["a"]["fp"] += n_pred
        return counts
    if n_pred == 0:
        counts["a"]["fn"] += n_gt
        return counts

    hit = iou_matrix > iou_threshold
    unmatched_gt = np.where(hit.sum(axis=1) == 0)[0]
    unmatched_pred = np.where(hit.sum(axis=0) == 0)[0]
    if len(unmatched_gt) == 0 and len(unmatched_pred) == 0:
        return counts

    sub_iou = iou_matrix[np.ix_(unmatched_gt, unmatched_pred)]
    weak = sub_iou > weak_iou
    edges = [(i, j) for i in range(len(unmatched_gt)) for j in range(len(unmatched_pred)) if weak[i, j]]
    for gt_local, pred_local in _union_find_components(len(unmatched_gt), len(unmatched_pred), edges):
        cat, fn, fp = classify_component(gt_local, pred_local, sub_iou)
        counts[cat]["fn"] += fn
        counts[cat]["fp"] += fp
    return counts


@dataclass
class AnnotatorImageResult:
    stem: str
    annotator: str
    match: skmetric.ImageMatch
    loss_counts: dict[str, dict[str, int]]
    fn_areas: list[int] = field(default_factory=list)
    fp_areas: list[int] = field(default_factory=list)
    tp_areas: list[int] = field(default_factory=list)
    fn_r_over_r: list[float] = field(default_factory=list)
    fp_r_over_r: list[float] = field(default_factory=list)
    tp_r_over_r: list[float] = field(default_factory=list)
    site: str | None = None
    pixel_iou: float | None = None
    pixel_dice: float | None = None


def _r_over_r_from_coords(ys: np.ndarray, xs: np.ndarray, cx: float, cy: float, radius: float) -> float:
    if len(ys) == 0 or radius <= 0:
        return float("nan")
    return float(np.sqrt((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2) / radius)


def _instance_r_over_r(mask: np.ndarray, cx: float, cy: float, radius: float) -> float:
    ys, xs = np.nonzero(mask)
    return _r_over_r_from_coords(ys, xs, cx, cy, radius)


def _pred_instance_area_and_coords(pred_masks, j: int, shape: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray]:
    """Works whether ``pred_masks`` is a dense ``(n_pred, H, W)`` stack or a
    sparse ``(n_pred, H*W)`` matrix (see ``postprocess.extract_instances_sparse``),
    without ever densifying a row."""
    if sp.issparse(pred_masks):
        row = pred_masks.getrow(j)
        cols = row.indices
        h, w = shape
        ys, xs = np.divmod(cols, w)
        return int(row.sum()), ys, xs
    mask = pred_masks[j]
    ys, xs = np.nonzero(mask)
    return int(mask.sum()), ys, xs


def evaluate_annotator_image(
    stem: str,
    annotator: str,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    disk_info: tuple[float, float, float] | None = None,
    site: str | None = None,
    iou_threshold: float = skmetric.IOU_THRESHOLD,
    weak_iou: float = WEAK_IOU,
    compute_pixel_metrics: bool = False,
) -> AnnotatorImageResult:
    n_gt, n_pred = gt_masks.shape[0], pred_masks.shape[0]
    if n_gt == 0 or n_pred == 0:
        iou_matrix = np.zeros((n_gt, n_pred))
        dice_matrix = np.zeros((n_gt, n_pred))
    else:
        iou_matrix, dice_matrix = skmetric.pairwise_iou_dice(gt_masks, pred_masks)

    match = skmetric.match_scores(iou_matrix, dice_matrix, n_gt, n_pred, iou_threshold)
    loss = pq_loss_decomposition(iou_matrix, n_gt, n_pred, iou_threshold, weak_iou)

    hit = iou_matrix > iou_threshold if n_gt and n_pred else np.zeros((n_gt, n_pred), dtype=bool)
    fn_idxs = np.where(hit.sum(axis=1) == 0)[0] if n_gt else np.array([], dtype=int)
    fp_idxs = np.where(hit.sum(axis=0) == 0)[0] if n_pred else np.array([], dtype=int)
    # np.argwhere on a boolean array walks row-major (C order), same as the
    # boolean fancy-indexing `iou[hit]` match.py uses to build tp_ious -- so
    # this stays index-aligned with match.tp_ious without re-deriving it.
    tp_pairs = np.argwhere(hit)

    shape = gt_masks.shape[1:]
    fn_areas = [int(gt_masks[i].sum()) for i in fn_idxs]
    tp_areas = [int(gt_masks[i].sum()) for i, _ in tp_pairs]
    fp_instances = [_pred_instance_area_and_coords(pred_masks, j, shape) for j in fp_idxs]
    fp_areas = [area for area, _, _ in fp_instances]

    fn_r, fp_r, tp_r = [], [], []
    if disk_info is not None:
        cx, cy, radius = disk_info
        fn_r = [_instance_r_over_r(gt_masks[i], cx, cy, radius) for i in fn_idxs]
        fp_r = [_r_over_r_from_coords(ys, xs, cx, cy, radius) for _, ys, xs in fp_instances]
        tp_r = [_instance_r_over_r(gt_masks[i], cx, cy, radius) for i, _ in tp_pairs]

    pixel_iou = pixel_dice = None
    if compute_pixel_metrics:
        pred_semantic = _semantic_union_any(pred_masks, shape)
        gt_semantic = _semantic_union_any(gt_masks, shape)
        pixel_iou, pixel_dice = pixel_iou_dice(pred_semantic, gt_semantic)

    return AnnotatorImageResult(
        stem=stem,
        annotator=annotator,
        match=match,
        loss_counts=loss,
        fn_areas=fn_areas,
        fp_areas=fp_areas,
        tp_areas=tp_areas,
        fn_r_over_r=fn_r,
        fp_r_over_r=fp_r,
        tp_r_over_r=tp_r,
        site=site,
        pixel_iou=pixel_iou,
        pixel_dice=pixel_dice,
    )


def r_bin(r_over_r: float) -> str:
    if np.isnan(r_over_r):
        return "unknown"
    if r_over_r < 1 / 3:
        return "centre"
    if r_over_r < 2 / 3:
        return "middle"
    return "limb"


def size_bucket(area: int, quartile_edges: list[float]) -> str:
    """quartile_edges = [q25, q50, q75] from the EDA's train instance areas."""
    q25, q50, q75 = quartile_edges
    if area < q25:
        return "q1_smallest"
    if area < q50:
        return "q2"
    if area < q75:
        return "q3"
    return "q4_largest"


@dataclass
class EvaluationSummary:
    overall: skmetric.PQResult
    per_site_pq: dict
    per_r_bin_pq: dict
    per_size_quartile: dict
    loss_decomposition: dict[str, dict[str, int]]
    noisy_slice_pq: float | None
    non_noisy_slice_pq: float | None
    mean_inference_seconds: float | None
    details_df: pd.DataFrame
    pixel_iou_mean: float | None = None
    pixel_dice_mean: float | None = None


def _pq_from_tp_fp_fn(tp_ious: list[float], fp: int, fn: int) -> dict:
    tp = len(tp_ious)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum(tp_ious) / denom if denom > 0 else 0.0
    rq = tp / denom if denom > 0 else 0.0
    sq = sum(tp_ious) / tp if tp > 0 else 0.0
    return {"pq": pq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn}


def aggregate_evaluation(
    results: list[AnnotatorImageResult],
    quartile_edges: list[float],
    noisy_stems: set[str] | None = None,
    inference_seconds_by_stem: dict[str, float] | None = None,
) -> EvaluationSummary:
    noisy_stems = noisy_stems or set()
    inference_seconds_by_stem = inference_seconds_by_stem or {}

    overall = skmetric.aggregate_matches([r.match for r in results])

    rows = []
    loss_totals = {c: {"fn": 0, "fp": 0} for c in "abcd"}
    for r in results:
        for c in "abcd":
            loss_totals[c]["fn"] += r.loss_counts[c]["fn"]
            loss_totals[c]["fp"] += r.loss_counts[c]["fp"]
        tp_r_list = r.tp_r_over_r or [float("nan")] * len(r.match.tp_ious)
        tp_area_list = r.tp_areas or [None] * len(r.match.tp_ious)
        for iou, rr, area in zip(r.match.tp_ious, tp_r_list, tp_area_list):
            rows.append(
                {
                    "stem": r.stem,
                    "site": r.site,
                    "kind": "tp",
                    "iou": iou,
                    "r_bin": r_bin(rr),
                    "size": size_bucket(area, quartile_edges) if area is not None else None,
                    "noisy": r.stem in noisy_stems,
                }
            )
        for area, rr in zip(r.fn_areas, r.fn_r_over_r or [float("nan")] * len(r.fn_areas)):
            rows.append({"stem": r.stem, "site": r.site, "kind": "fn", "iou": 0.0, "r_bin": r_bin(rr), "size": size_bucket(area, quartile_edges), "noisy": r.stem in noisy_stems})
        for area, rr in zip(r.fp_areas, r.fp_r_over_r or [float("nan")] * len(r.fp_areas)):
            rows.append({"stem": r.stem, "site": r.site, "kind": "fp", "iou": 0.0, "r_bin": r_bin(rr), "size": size_bucket(area, quartile_edges), "noisy": r.stem in noisy_stems})
    df = pd.DataFrame(rows)

    def pq_for_subset(mask: pd.Series) -> float:
        sub = df[mask]
        tp_ious = sub.loc[sub["kind"] == "tp", "iou"].tolist()
        fp = int((sub["kind"] == "fp").sum())
        fn = int((sub["kind"] == "fn").sum())
        return _pq_from_tp_fp_fn(tp_ious, fp, fn)["pq"]

    per_site_pq = {site: pq_for_subset(df["site"] == site) for site in df["site"].dropna().unique()} if len(df) else {}
    per_r_bin_pq = {b: pq_for_subset(df["r_bin"] == b) for b in ["centre", "middle", "limb"]} if len(df) else {}
    per_size_quartile = (
        {q: pq_for_subset(df["size"] == q) for q in ["q1_smallest", "q2", "q3", "q4_largest"]} if len(df) else {}
    )

    noisy_pq = pq_for_subset(df["noisy"]) if len(df) and df["noisy"].any() else None
    non_noisy_pq = pq_for_subset(~df["noisy"]) if len(df) and (~df["noisy"]).any() else None

    inference_times = list(inference_seconds_by_stem.values())
    mean_inf = float(np.mean(inference_times)) if inference_times else None

    pixel_ious = [r.pixel_iou for r in results if r.pixel_iou is not None]
    pixel_dices = [r.pixel_dice for r in results if r.pixel_dice is not None]

    return EvaluationSummary(
        overall=overall,
        per_site_pq=per_site_pq,
        per_r_bin_pq=per_r_bin_pq,
        per_size_quartile=per_size_quartile,
        loss_decomposition=loss_totals,
        noisy_slice_pq=noisy_pq,
        non_noisy_slice_pq=non_noisy_pq,
        mean_inference_seconds=mean_inf,
        details_df=df,
        pixel_iou_mean=float(np.mean(pixel_ious)) if pixel_ious else None,
        pixel_dice_mean=float(np.mean(pixel_dices)) if pixel_dices else None,
    )
