"""Protocol P5 (CODING_SPEC_05_ENSEMBLE.md section 2.1): nested, per-CV-fold
refit of the whole post-processing chain, over day-grouped 5-fold CV of the
OOF images.

Inside each CV fold, everything is fit on the 4/5 train portion and scored
on the held-out 1/5:
  1. cfg1 (threshold/hysteresis + closing, min_area 418) from CFG1_GRID,
  2. shape gate (elongation x length, same grid as run_e007_eval.py),
  3. scorer fit_scorer(seed=0) + cutoff c from C_GRID,
  4. regrow (t_mask, d) from REGROW_GRID, tuned on a seeded day-grouped
     subsample of <= REGROW_MAX_IMAGES train-portion images.

Cost control (spec 2.1): IoU > 0.5 matches are unique for disjoint
predictions, so a fixed cfg1's per-image component table (per component and
per annotator: the GT instances it hits and their IoUs, plus each
annotator's GT count) is a sufficient statistic for PQ of *any* keep-subset
(shape gate, scorer cutoff). Steps 1-3 therefore become sums over cached
per-image statistics; only regrow and the final held-out pass rasterise
masks again. ``pq_from_tables`` is tested to equal ``aggregate_evaluation``.

Determinism (spec 2.1/6): stems are always processed in sorted order, the
scorer is fit on rows concatenated in sorted-stem order, ties in every grid
search go to the first grid entry, worker results are reassembled by stem
(never by completion order), and every random choice takes an explicit seed.

Memory (CLAUDE.md): every worker streams one image at a time; only sparse
GT/statistics persist across images.
"""
from __future__ import annotations

import collections
import hashlib
import json
import multiprocessing as mp
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold

from . import components as skcomp
from . import ensemble as skens
from . import evaluate as skeval
from . import ids as skids
from . import io as skio
from . import metric as skmetric
from . import postprocess as skpp
from . import preprocess as skpre
from . import scorer as sksc
from . import targets as sktgt

H = W = 2048
MIN_AREA = 418
N_CV_FOLDS = 5

# CODING_SPEC_06 E018: widened from Spec 05's grid (every P5 winner sat at a
# grid edge -- hysteresis t_high at its max, t_low at its min, regrow d at
# its min). t in {0.45, 0.5} is added to the threshold grid; the hysteresis
# and regrow grids are replaced by their widened ranges below (both still
# contain Spec 05's edge winners: hyst (0.5, 0.975), regrow d=8).
_T_GRID = [0.45, 0.5, 0.55, 0.575, 0.6, 0.625, 0.65, 0.7, 0.8, 0.925]
_HYST_GRID = [(lo, hi) for lo in (0.35, 0.4, 0.45, 0.5) for hi in (0.975, 0.985, 0.99)]
_R_GRID = [1, 2, 3]
CFG1_GRID: list[skpp.PostprocessConfig] = (
    [skpp.PostprocessConfig(mode="threshold", t=t, closing_radius=r, min_area=MIN_AREA) for t in _T_GRID for r in _R_GRID]
    + [skpp.PostprocessConfig(mode="hysteresis", t_low=lo, t_high=hi, closing_radius=r, min_area=MIN_AREA) for lo, hi in _HYST_GRID for r in _R_GRID]
)
ELONG_GRID = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LEN_GRID = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
C_GRID = [round(x, 3) for x in np.arange(0.0, 0.525, 0.025)]
REGROW_GRID = [(t_mask, d) for t_mask in (0.4, 0.5, 0.6) for d in (4, 6, 8)]
REGROW_MAX_IMAGES = 150
SEED = 0


# ---------------------------------------------------------------- helpers

def cfg_key(cfg: skpp.PostprocessConfig) -> str:
    if cfg.mode == "threshold":
        return f"thr{cfg.t}_r{cfg.closing_radius}"
    return f"hys{cfg.t_low}-{cfg.t_high}_r{cfg.closing_radius}"


def components_cfg(cfg: skpp.PostprocessConfig) -> skpp.PostprocessConfig:
    """Same binarisation/closing, but keep every component (the gate and the
    scorer need the small/weak ones too) -- as run_e003/run_e005 do."""
    return skpp.PostprocessConfig(
        mode=cfg.mode, t=cfg.t, t_low=cfg.t_low, t_high=cfg.t_high,
        closing_radius=cfg.closing_radius, min_area=1, max_instances=5000,
    )


def pq_of(sum_iou: float, tp: int, fp: int, fn: int) -> float:
    denom = tp + 0.5 * fp + 0.5 * fn
    return sum_iou / denom if denom > 0 else 0.0


def load_annotation_index() -> tuple[dict, dict]:
    ann = skio.load_coco_annotations()
    images_by_file = collections.defaultdict(list)
    for img in ann["images"]:
        images_by_file[img["file_name"]].append(img)
    anns_by_image_id = collections.defaultdict(list)
    for a in ann["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)
    return dict(images_by_file), dict(anns_by_image_id)


def gt_for_stem(stem: str, images_by_file: dict, anns_by_image_id: dict) -> dict[str, np.ndarray]:
    """Same as scripts/run_e002_eval.build_gt_for_stem: annotator -> dense
    (n, H, W) uint8 instance stack, in annotation-file order."""
    out = {}
    for entry in images_by_file.get(f"{stem}.jpeg", []):
        annotator, _ = skids.parse_image_id(entry["id"])
        masks = sktgt.annotator_instance_masks(anns_by_image_id.get(entry["id"], []), H, W)
        out[annotator] = np.stack(masks) if masks else np.zeros((0, H, W), dtype=np.uint8)
    return out


def cv_splits(stems: list[str], day_by_stem: dict[str, str], n_splits: int = N_CV_FOLDS) -> list[tuple[list[str], list[str]]]:
    """GroupKFold over capture date on sorted stems -- with all 707 OOF
    stems this reproduces E005_cv_results.csv's folds exactly."""
    stems = sorted(stems)
    days = [day_by_stem[s] for s in stems]
    arr = np.array(stems)
    gkf = GroupKFold(n_splits=min(n_splits, len(set(days))))
    return [(arr[tr].tolist(), arr[ho].tolist()) for tr, ho in gkf.split(arr, groups=days)]


def day_grouped_subsample(stems: list[str], day_by_stem: dict[str, str], max_images: int = REGROW_MAX_IMAGES, seed: int = SEED) -> list[str]:
    """Whole days in seeded random order, added while the total stays <= max_images."""
    by_day = collections.defaultdict(list)
    for s in sorted(stems):
        by_day[day_by_stem[s]].append(s)
    days = sorted(by_day)
    order = np.random.RandomState(seed).permutation(len(days))
    out: list[str] = []
    for i in order:
        group = by_day[days[i]]
        if len(out) + len(group) <= max_images:
            out.extend(group)
    return sorted(out)


# ------------------------------------------------- per-image computations

def component_labels(probs: np.ndarray, cfg: skpp.PostprocessConfig) -> np.ndarray:
    """The label image extract_components builds (same binarise/close/CC call)."""
    closed = skpp.morphological_close(skpp.probs_to_binary(probs, cfg), cfg.closing_radius)
    _, labels, _, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
    return labels


def match_stats(gt_by_annotator: dict[str, np.ndarray], pred) -> tuple[float, int, int, int]:
    """Summed (sum TP IoU, TP, FP, FN) over annotators -- metric.match_scores' rule."""
    s_iou, tp, fp, fn = 0.0, 0, 0, 0
    n_pred = pred.shape[0]
    for gt in gt_by_annotator.values():
        n_gt = gt.shape[0]
        if n_gt == 0 or n_pred == 0:
            m = skmetric.match_scores(np.zeros((0, 0)), np.zeros((0, 0)), n_gt, n_pred)
        else:
            iou, dice = skmetric.pairwise_iou_dice(gt, pred)
            m = skmetric.match_scores(iou, dice, n_gt, n_pred)
        s_iou += float(sum(m.tp_ious))
        tp += len(m.tp_ious)
        fp += m.fp
        fn += m.fn
    return s_iou, tp, fp, fn


def cfg1_stats_for_image(probs: np.ndarray, gt_by_annotator: dict, configs: list[skpp.PostprocessConfig] = CFG1_GRID) -> np.ndarray:
    """(n_configs, 4): sum TP IoU, TP, FP, FN per config -- what
    tuning.grid_search sums, per image."""
    out = np.zeros((len(configs), 4), dtype=np.float64)
    for i, cfg in enumerate(configs):
        pred = skpp.postprocess_sparse(probs, cfg)
        out[i] = match_stats(gt_by_annotator, pred)
    return out


@dataclass
class ComponentTable:
    """One image's sufficient statistics at one cfg1."""
    stem: str
    features_df: pd.DataFrame  # one row per component, extract_components order; is_tp/best_iou set
    n_gt: dict[str, int]  # annotator -> GT count (annotators in annotation-file order)
    hits: np.ndarray  # (n_hits, 3) int: component row, annotator index (into n_gt order), gt index
    hit_iou: np.ndarray  # (n_hits,) float64


def component_table(stem: str, probs: np.ndarray, processed: np.ndarray, disk_info, cfg: skpp.PostprocessConfig, gt_by_annotator: dict, iou_threshold: float = skmetric.IOU_THRESHOLD) -> ComponentTable:
    """extract_components + label_components, with IoUs from sparse stacks
    (same arithmetic as label_components, without a dense (n, H, W) pred
    stack) and every IoU > threshold hit recorded."""
    ccfg = components_cfg(cfg)
    features, labels = skcomp.extract_components(stem, probs, processed, disk_info, ccfg)
    sparse = skcomp.components_to_sparse(labels, [f.label for f in features])
    del labels
    hits, hit_iou = [], []
    n_gt = {}
    for a_idx, (annotator, gt) in enumerate(gt_by_annotator.items()):
        n_gt[annotator] = int(gt.shape[0])
        if gt.shape[0] == 0 or not features:
            for f in features:
                f.best_iou_by_annotator[annotator] = 0.0
            continue
        iou, _ = skmetric.pairwise_iou_dice(gt, sparse)
        best = iou.max(axis=0)
        for f, b in zip(features, best):
            f.best_iou_by_annotator[annotator] = float(b)
        gi, pj = np.nonzero(iou > iou_threshold)
        for g, p in zip(gi.tolist(), pj.tolist()):
            hits.append((p, a_idx, g))
            hit_iou.append(float(iou[g, p]))
    for f in features:
        f.is_tp = any(v > iou_threshold for v in f.best_iou_by_annotator.values())
    df = skcomp.to_dataframe(features)
    hits_arr = np.array(hits, dtype=np.int64).reshape(-1, 3)
    order = np.lexsort((hits_arr[:, 2], hits_arr[:, 1], hits_arr[:, 0])) if len(hits_arr) else np.zeros(0, dtype=int)
    return ComponentTable(stem=stem, features_df=df, n_gt=n_gt, hits=hits_arr[order], hit_iou=np.array(hit_iou, dtype=np.float64)[order])


# ---------------------------------------------- PQ from cached statistics

class StatsIndex:
    """Flattened component tables for a fixed list of stems, so PQ of any
    keep vector over any stem subset is a few numpy reductions."""

    def __init__(self, tables: list[ComponentTable]):
        self.stems = [t.stem for t in tables]
        self.stem_pos = {s: i for i, s in enumerate(self.stems)}
        comp_stem, pair_stem, pair_ngt = [], [], []
        hit_comp, hit_pair, hit_gt, hit_iou = [], [], [], []
        self.comp_offset = []
        n_comp = n_pair = n_gt_global = 0
        for si, t in enumerate(tables):
            nc = len(t.features_df)
            self.comp_offset.append(n_comp)
            comp_stem.extend([si] * nc)
            gt_base = {}
            for a_idx, (_, ng) in enumerate(t.n_gt.items()):
                pair_stem.append(si)
                pair_ngt.append(ng)
                gt_base[a_idx] = (n_pair + a_idx, n_gt_global)
                n_gt_global += ng
            for (p, a_idx, g), iou in zip(t.hits.tolist(), t.hit_iou.tolist()):
                pair_id, gbase = gt_base[a_idx]
                hit_comp.append(n_comp + p)
                hit_pair.append(pair_id)
                hit_gt.append(gbase + g)
                hit_iou.append(iou)
            n_comp += nc
            n_pair += len(t.n_gt)
        self.n_comp = n_comp
        self.comp_stem = np.array(comp_stem, dtype=np.int64)
        self.pair_stem = np.array(pair_stem, dtype=np.int64)
        self.pair_ngt = np.array(pair_ngt, dtype=np.int64)
        self.hit_comp = np.array(hit_comp, dtype=np.int64)
        self.hit_gt = np.array(hit_gt, dtype=np.int64)
        self.hit_iou = np.array(hit_iou, dtype=np.float64)
        cp = np.unique(np.stack([self.hit_comp, np.array(hit_pair, dtype=np.int64)], axis=1), axis=0) if hit_comp else np.zeros((0, 2), dtype=np.int64)
        self.cp_comp = cp[:, 0]
        self.features_df = pd.concat([t.features_df for t in tables], ignore_index=True) if tables else pd.DataFrame()

    def stem_mask(self, stems) -> np.ndarray:
        m = np.zeros(len(self.stems), dtype=bool)
        m[[self.stem_pos[s] for s in stems]] = True
        return m

    def stats(self, keep: np.ndarray, stem_mask: np.ndarray) -> tuple[float, int, int, int]:
        kept = keep & stem_mask[self.comp_stem]
        h = kept[self.hit_comp]
        tp = int(h.sum())
        s_iou = float(self.hit_iou[h].sum())
        n_kept = np.bincount(self.comp_stem[kept], minlength=len(self.stems))
        pair_in = stem_mask[self.pair_stem]
        fp = int(n_kept[self.pair_stem[pair_in]].sum()) - int(kept[self.cp_comp].sum())
        fn = int(self.pair_ngt[pair_in].sum()) - len(np.unique(self.hit_gt[h]))
        return s_iou, tp, fp, fn

    def pq(self, keep: np.ndarray, stem_mask: np.ndarray) -> float:
        return pq_of(*self.stats(keep, stem_mask))


def gate_keep(df: pd.DataFrame, min_elong: float, min_len: float) -> np.ndarray:
    if len(df) == 0:
        return np.zeros(0, dtype=bool)
    return ((df["area"] >= MIN_AREA) & (df["elongation"].fillna(0) >= min_elong) & (df["length"].fillna(0) >= min_len)).to_numpy()


# ------------------------------------------------- multiprocessing workers

_W: dict = {}


def _worker_init(members: list[str], output_root: str | None) -> None:
    import os

    if output_root:
        os.environ["SOLAR_K_OUTPUT"] = output_root
    cv2.setNumThreads(1)
    _W["members"] = members
    _W["fold_by_stem"] = skens.model_fold_by_stem()
    _W["index"] = load_annotation_index()
    _W["train_paths"] = {p.stem: p for p in skio.iter_train_images()}


def _probs(stem: str, weights: tuple[float, ...] | None = None) -> np.ndarray:
    return skens.load_oof_mean(_W["members"], stem, _W["fold_by_stem"], list(weights) if weights else None)


def _gt(stem: str) -> dict:
    return gt_for_stem(stem, *_W["index"])


def _sparse_gt(gt: dict) -> dict:
    """Converted once per image, not once per config (identical IoUs)."""
    return {a: sp.csr_matrix(g.reshape(g.shape[0], -1)) for a, g in gt.items()}


def _job_cfg1(args):
    stem, weights = args
    probs = _probs(stem, weights)
    out = cfg1_stats_for_image(probs, _sparse_gt(_gt(stem)))
    return stem, out


def _job_table(args):
    stem, cfg_dict, weights = args
    cfg = skpp.PostprocessConfig.from_dict(cfg_dict)
    probs = _probs(stem, weights)
    processed, disk_info = skpre.load_or_build(stem, _W["train_paths"][stem])
    t = component_table(stem, probs, processed, disk_info, cfg, _gt(stem))
    return stem, t


def _kept_masks(probs: np.ndarray, cfg: skpp.PostprocessConfig, kept_labels: list[int]) -> list[np.ndarray]:
    if not kept_labels:
        return []
    labels = component_labels(probs, components_cfg(cfg))
    return [(labels == lid).astype(np.uint8) for lid in kept_labels]


def _stack(masks: list[np.ndarray]) -> np.ndarray:
    return np.stack(masks).astype(np.uint8) if masks else np.zeros((0, H, W), dtype=np.uint8)


def _job_regrow(args):
    stem, cfg_dict, kept_labels, grid, weights = args
    cfg = skpp.PostprocessConfig.from_dict(cfg_dict)
    probs = _probs(stem, weights)
    seeds = _kept_masks(probs, cfg, kept_labels)
    gt = _sparse_gt(_gt(stem))
    out = np.zeros((len(grid), 4), dtype=np.float64)
    for i, (t_mask, d) in enumerate(grid):
        grown = skpp.regrow_instances(probs, seeds, t_mask, d) if seeds else []
        out[i] = match_stats(gt, _stack(grown))
    return stem, out


def _job_final(args):
    stem, cfg_dict, kept_labels, t_mask, d, disk_info, site, weights = args
    cfg = skpp.PostprocessConfig.from_dict(cfg_dict)
    probs = _probs(stem, weights)
    seeds = _kept_masks(probs, cfg, kept_labels)
    pred = _stack(skpp.regrow_instances(probs, seeds, t_mask, d) if seeds else [])
    results = [
        skeval.evaluate_annotator_image(stem, a, pred, g, disk_info=disk_info, site=site, compute_pixel_metrics=True)
        for a, g in _gt(stem).items()
    ]
    return stem, results


class Runner:
    """Ordered map over stems, in-process (workers=0) or in a spawn pool."""

    def __init__(self, members: list[str], workers: int):
        self.members, self.workers = members, workers
        self.pool = None
        if workers > 0:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(workers, initializer=_worker_init, initargs=(members, str(skio.output_root())))
        else:
            _worker_init(members, None)

    def map(self, fn, items, log=None, label=""):
        it = self.pool.imap(fn, items, chunksize=1) if self.pool else map(fn, items)
        out = {}
        n = len(items)
        for i, (stem, res) in enumerate(it):
            out[stem] = res
            if log and ((i + 1) % 50 == 0 or i + 1 == n):
                log(f"  {label}: {i + 1}/{n}")
        return out

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


# ------------------------------------------------------------- the protocol

@dataclass
class P5Result:
    per_fold: list[dict] = field(default_factory=list)
    per_image: pd.DataFrame | None = None
    summary: dict = field(default_factory=dict)


CACHE_TAG = ""  # set e.g. "rerun" to force a cold, separate cache (determinism checks)


def weight_tag(weights: tuple[float, ...] | None) -> str:
    if not weights:
        return "equal"
    return "w" + "-".join(f"{w:.3f}" for w in weights)


def _cache_dir(members: list[str], weights: tuple[float, ...] | None = None) -> Path:
    name = "+".join(members)
    if weights:
        name += f"_{weight_tag(weights)}"
    d = skio.output_root() / "p5_cache" / (name + (f"__{CACHE_TAG}" if CACHE_TAG else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _grid_hash() -> str:
    return hashlib.md5(json.dumps([c.__dict__ for c in CFG1_GRID], sort_keys=True).encode()).hexdigest()[:8]


def cfg1_stats(runner: Runner, stems: list[str], weights: tuple[float, ...] | None = None, log=print) -> dict[str, np.ndarray]:
    path = _cache_dir(runner.members, weights) / f"cfg1_stats_{_grid_hash()}.pkl"
    cached = pickle.loads(path.read_bytes()) if path.exists() else {}
    todo = [s for s in stems if s not in cached]
    if todo:
        log(f"cfg1 sweep ({weight_tag(weights)}): {len(todo)} images x {len(CFG1_GRID)} configs")
        cached.update(runner.map(_job_cfg1, [(s, weights) for s in todo], log, "cfg1"))
        path.write_bytes(pickle.dumps(cached))
    return {s: cached[s] for s in stems}


def component_tables(runner: Runner, stems: list[str], cfg: skpp.PostprocessConfig, weights: tuple[float, ...] | None = None, log=print) -> dict[str, ComponentTable]:
    path = _cache_dir(runner.members, weights) / f"tables_{cfg_key(cfg)}.pkl"
    cached = pickle.loads(path.read_bytes()) if path.exists() else {}
    todo = [s for s in stems if s not in cached]
    if todo:
        log(f"component tables @ {cfg_key(cfg)} ({weight_tag(weights)}): {len(todo)} images")
        cached.update(runner.map(_job_table, [(s, cfg.__dict__, weights) for s in todo], log, "tables"))
        path.write_bytes(pickle.dumps(cached))
    return {s: cached[s] for s in stems}


def choose_cfg1(stats: dict[str, np.ndarray], stems: list[str]) -> tuple[skpp.PostprocessConfig, float]:
    total = np.zeros((len(CFG1_GRID), 4))
    for s in sorted(stems):
        total += stats[s]
    pqs = [pq_of(*row) for row in total]
    i = int(np.argmax(pqs))  # first max = first grid entry on ties
    return CFG1_GRID[i], float(pqs[i])


def choose_weights_and_cfg1(runner: Runner, stems: list[str], train: list[str], weight_grid: list[tuple[float, ...]], log=print) -> tuple[tuple[float, ...] | None, skpp.PostprocessConfig, float]:
    """CODING_SPEC_06 E018: member weights chosen inside the train portion
    only, jointly with cfg1 -- for each candidate weight vector, the cfg1
    sweep is run (and cached) on that weight's probability maps, and the
    (weight, cfg1) pair with the best train-portion PQ wins. Ties go to the
    first weight vector in ``weight_grid``, then the first cfg1 (grid order)."""
    best = None
    for w in weight_grid:
        stats = cfg1_stats(runner, stems, w, log)
        cfg1, pq = choose_cfg1(stats, train)
        if best is None or pq > best[2]:
            best = (w, cfg1, pq)
    return best


def fit_keep_rule(index: StatsIndex, train_stems: list[str]) -> dict:
    """Gate -> scorer -> c, all on train_stems only."""
    tr_mask = index.stem_mask(train_stems)
    df = index.features_df
    best_gate, best_gate_pq = (0.0, 0.0), -1.0
    for me in ELONG_GRID:
        for ml in LEN_GRID:
            pq = index.pq(gate_keep(df, me, ml), tr_mask)
            if pq > best_gate_pq:
                best_gate_pq, best_gate = pq, (me, ml)
    gate = gate_keep(df, *best_gate)
    train_rows = tr_mask[index.comp_stem]
    scorer = sksc.fit_scorer(df[train_rows].reset_index(drop=True), seed=SEED)
    scores = sksc.keep_score(scorer, df) if len(df) else np.zeros(0)
    best_c, best_c_pq = 0.0, -1.0
    for c in C_GRID:
        pq = index.pq(gate & (scores > c), tr_mask)
        if pq > best_c_pq:
            best_c_pq, best_c = pq, c
    return {"shape_gate": list(best_gate), "gate_train_pq": best_gate_pq, "scorer_c": best_c, "c_train_pq": best_c_pq,
            "keep": gate & (scores > best_c), "scorer": scorer}


def kept_labels_by_stem(index: StatsIndex, keep: np.ndarray, stems: list[str]) -> dict[str, list[int]]:
    labels = index.features_df["label"].to_numpy()
    out = {}
    for s in stems:
        i = index.stem_pos[s]
        sel = (index.comp_stem == i) & keep
        out[s] = [int(x) for x in labels[sel]]
    return out


def choose_regrow(runner: Runner, cfg: skpp.PostprocessConfig, kept: dict[str, list[int]], stems: list[str], weights: tuple[float, ...] | None = None, log=print) -> tuple[tuple[float, int], float, list[float]]:
    res = runner.map(_job_regrow, [(s, cfg.__dict__, kept[s], REGROW_GRID, weights) for s in stems], log, "regrow")
    total = np.zeros((len(REGROW_GRID), 4))
    for s in sorted(stems):
        total += res[s]
    pqs = [pq_of(*row) for row in total]
    i = int(np.argmax(pqs))
    return REGROW_GRID[i], float(pqs[i]), [round(p, 6) for p in pqs]


def run_p5(
    members: list[str],
    stems: list[str],
    day_by_stem: dict[str, str],
    site_by_stem: dict[str, str],
    disk_info_by_stem: dict[str, tuple],
    quartile_edges: list[float],
    workers: int = 3,
    cv_folds: list[int] | None = None,
    weight_grid: list[tuple[float, ...]] | None = None,
    log=print,
) -> P5Result:
    stems = sorted(stems)
    fold_by_stem = skens.model_fold_by_stem()
    splits = cv_splits(stems, day_by_stem)
    runner = Runner(members, workers)
    try:
        stats = None if weight_grid else cfg1_stats(runner, stems, None, log)
        result = P5Result()
        image_rows = []
        for i, (train, held) in enumerate(splits):
            if cv_folds is not None and i not in cv_folds:
                continue
            if weight_grid:
                w, cfg1, cfg1_pq = choose_weights_and_cfg1(runner, stems, train, weight_grid, log)
                log(f"CV fold {i}: weights={weight_tag(w)} cfg1={cfg_key(cfg1)} (train PQ {cfg1_pq:.4f})")
            else:
                w = None
                cfg1, cfg1_pq = choose_cfg1(stats, train)
                log(f"CV fold {i}: cfg1={cfg_key(cfg1)} (train PQ {cfg1_pq:.4f})")
            tables = component_tables(runner, stems, cfg1, w, log)
            index = StatsIndex([tables[s] for s in stems])
            rule = fit_keep_rule(index, train)
            kept = kept_labels_by_stem(index, rule["keep"], stems)
            sub = day_grouped_subsample(train, day_by_stem)
            (t_mask, d), regrow_pq, regrow_pqs = choose_regrow(runner, cfg1, kept, sub, w, log)
            log(f"  gate={rule['shape_gate']} c={rule['scorer_c']} regrow=({t_mask},{d}) weights={weight_tag(w)} on {len(sub)} imgs")
            final = runner.map(_job_final, [(s, cfg1.__dict__, kept[s], t_mask, d, disk_info_by_stem[s], site_by_stem[s], w) for s in held], log, "held-out")
            results = [r for s in held for r in final[s]]
            summ = skeval.aggregate_evaluation(results, quartile_edges=quartile_edges, noisy_stems=set())
            o = summ.overall
            for s in held:
                m = skmetric.aggregate_matches([r.match for r in final[s]])
                image_rows.append({"stem": s, "cv_fold": i, "model_fold": fold_by_stem[s], "day": day_by_stem[s],
                                   "sum_iou": sum(m.tp_ious), "tp": m.tp, "fp": m.fp, "fn": m.fn})
            fold = {
                "cv_fold": i, "n_train": len(train), "n_held": len(held),
                "weights": list(w) if w else None,
                "cfg1": {k: v for k, v in cfg1.__dict__.items()}, "cfg1_train_pq": round(cfg1_pq, 6),
                "shape_gate": rule["shape_gate"], "scorer_c": rule["scorer_c"],
                "regrow_t_mask": t_mask, "regrow_d": d, "regrow_n_images": len(sub), "regrow_grid_pq": regrow_pqs,
                "pq": round(o.pq, 4), "sq": round(o.sq, 4), "rq": round(o.rq, 4), "tp": o.tp, "fp": o.fp, "fn": o.fn,
                "pixel_dice_mean": round(summ.pixel_dice_mean, 4) if summ.pixel_dice_mean is not None else None,
                "loss_decomposition": summ.loss_decomposition,
                "sum_tp_iou": float(sum(o.tp_ious)),
            }
            log(f"  held-out PQ={o.pq:.4f} SQ={o.sq:.4f} RQ={o.rq:.4f} TP={o.tp} FP={o.fp} FN={o.fn}")
            result.per_fold.append(fold)
        result.per_image = pd.DataFrame(image_rows)
        result.summary = summarise(result.per_fold)
        return result
    finally:
        runner.close()


def summarise(per_fold: list[dict]) -> dict:
    if not per_fold:
        return {}
    df = pd.DataFrame(per_fold)
    s_iou, tp, fp, fn = float(df["sum_tp_iou"].sum()), int(df["tp"].sum()), int(df["fp"].sum()), int(df["fn"].sum())
    denom = tp + 0.5 * fp + 0.5 * fn
    loss = {c: {"fn": 0, "fp": 0} for c in "abcd"}
    for f in per_fold:
        for c, v in f["loss_decomposition"].items():
            loss[c]["fn"] += v["fn"]
            loss[c]["fp"] += v["fp"]
    std = (lambda col: round(float(df[col].std()), 4) if len(df) > 1 else None)
    return {
        "n_cv_folds_run": len(df),
        "cv_pq_mean": round(float(df["pq"].mean()), 4), "cv_pq_std": std("pq"),
        "cv_sq_mean": round(float(df["sq"].mean()), 4), "cv_sq_std": std("sq"),
        "cv_rq_mean": round(float(df["rq"].mean()), 4), "cv_rq_std": std("rq"),
        "pooled_pq": round(s_iou / denom, 4) if denom else 0.0,
        "pooled_sq": round(s_iou / tp, 4) if tp else 0.0,
        "pooled_rq": round(tp / denom, 4) if denom else 0.0,
        "pooled_tp": tp, "pooled_fp": fp, "pooled_fn": fn,
        "loss_decomposition": loss,
    }


# ------------------------------------------------------- paired comparison

def pooled_pq_from_images(df: pd.DataFrame) -> float:
    return pq_of(df["sum_iou"].sum(), df["tp"].sum(), df["fp"].sum(), df["fn"].sum())


def pooled_sq_from_images(df: pd.DataFrame) -> float:
    tp = df["tp"].sum()
    return float(df["sum_iou"].sum() / tp) if tp else 0.0


def paired_comparison(cand: pd.DataFrame, ref: pd.DataFrame, group_col: str = "cv_fold", n_boot: int = 2000, seed: int = 0) -> dict:
    """Candidate vs reference on identical held-out images: per-group PQ
    deltas, pooled delta, and an image-level paired bootstrap 95% CI of the
    pooled delta (resampling stems with replacement, same draw for both)."""
    a = cand.sort_values("stem").reset_index(drop=True)
    b = ref.sort_values("stem").reset_index(drop=True)
    if a["stem"].tolist() != b["stem"].tolist() or (a["cv_fold"].to_numpy() != b["cv_fold"].to_numpy()).any():
        raise ValueError("paired comparison needs identical stems and CV folds")
    per_group = {}
    for g in sorted(a[group_col].unique()):
        ma, mb = a[group_col] == g, b[group_col] == g
        per_group[int(g)] = round(pooled_pq_from_images(a[ma]) - pooled_pq_from_images(b[mb]), 4)
    cols = ["sum_iou", "tp", "fp", "fn"]
    A, B = a[cols].to_numpy(dtype=np.float64), b[cols].to_numpy(dtype=np.float64)
    rng = np.random.RandomState(seed)
    n = len(a)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.randint(0, n, n)
        sa, sb = A[idx].sum(axis=0), B[idx].sum(axis=0)
        deltas[k] = pq_of(*sa) - pq_of(*sb)
    return {
        "per_group_delta_pq": per_group,
        "mean_group_delta_pq": round(float(np.mean(list(per_group.values()))), 4),
        "n_groups_positive": int(sum(v > 0 for v in per_group.values())),
        "pooled_delta_pq": round(pooled_pq_from_images(a) - pooled_pq_from_images(b), 4),
        "pooled_delta_sq": round(pooled_sq_from_images(a) - pooled_sq_from_images(b), 4),
        "bootstrap_ci95_pooled_delta_pq": [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)],
        "n_boot": n_boot, "boot_seed": seed,
    }
