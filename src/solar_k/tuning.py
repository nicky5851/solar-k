"""Tuning protocol shared by E001 and E002 (CODING_SPEC_02_BASELINE.md).

Split fold-0 val by **observation day** into ``tune``/``report`` halves with
a fixed seed. Grid-search postprocess params on ``tune`` by PQ; every other
metric is then reported on ``report`` only -- no threshold is ever chosen
on ``report``.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import evaluate as skeval
from . import metric as skmetric
from . import postprocess as skpp

SINGLE_T_GRID = [round(0.5 + 0.025 * i, 3) for i in range(19)]  # 0.50 .. 0.95, step 0.025 (CODING_SPEC_03)
HYSTERESIS_T_LOW = [0.5, 0.6, 0.7]
HYSTERESIS_T_HIGH = [0.7, 0.8, 0.9, 0.95]
R_GRID = [0, 1, 2, 3]


def min_area_grid(eda_p1_area: float) -> list[int]:
    return sorted({max(1, int(round(eda_p1_area * m))) for m in (0.5, 1.0, 2.0)})


def full_grid(eda_p1_area: float) -> list[skpp.PostprocessConfig]:
    configs = []
    for t, r, ma in itertools.product(SINGLE_T_GRID, R_GRID, min_area_grid(eda_p1_area)):
        configs.append(skpp.PostprocessConfig(mode="threshold", t=t, closing_radius=r, min_area=ma))
    for (t_low, t_high), r, ma in itertools.product(
        itertools.product(HYSTERESIS_T_LOW, HYSTERESIS_T_HIGH), R_GRID, min_area_grid(eda_p1_area)
    ):
        configs.append(
            skpp.PostprocessConfig(mode="hysteresis", t_low=t_low, t_high=t_high, closing_radius=r, min_area=ma)
        )
    return configs


def split_tune_report(day_by_stem: dict[str, str], seed: int = 0) -> tuple[set[str], set[str]]:
    """Group val stems by observation day, then split days ~50/50."""
    days = sorted(set(day_by_stem.values()))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(days))
    half = len(days) // 2
    tune_days = {days[i] for i in order[:half]}
    report_days = {days[i] for i in order[half:]}
    tune_stems = {s for s, d in day_by_stem.items() if d in tune_days}
    report_stems = {s for s, d in day_by_stem.items() if d in report_days}
    return tune_stems, report_stems


def grid_search(
    tune_stems: set[str],
    compute_probs,  # stem -> (probs, disk_mask) ; called once per stem, not once per config
    gt_by_stem: dict[str, dict[str, np.ndarray]],
    disk_info_by_stem: dict[str, tuple[float, float, float]],
    site_by_stem: dict[str, str],
    configs: list[skpp.PostprocessConfig],
    progress: bool = False,
) -> tuple[skpp.PostprocessConfig, pd.DataFrame]:
    """Evaluate every config on ``tune_stems`` by PQ.

    Loops images outer, configs inner, so ``compute_probs`` runs once per
    image and only one image's probability map is in memory at a time --
    holding every tune-image's full-resolution probs at once would be
    several GB on an 8 GB Mac for a fold-sized val set.
    """
    per_config_matches: list[list] = [[] for _ in configs]
    # sorted: a set's iteration order depends on PYTHONHASHSEED, which changes
    # the float summation order of tp_ious and can flip near-tied configs.
    for stem_i, stem in enumerate(sorted(tune_stems)):
        if stem not in gt_by_stem:
            continue
        if progress:
            import time

            t0 = time.time()
        probs = compute_probs(stem)
        for cfg_idx, cfg in enumerate(configs):
            pred_sparse = skpp.postprocess_sparse(probs, cfg)
            for annotator, gt_masks in gt_by_stem[stem].items():
                r = skeval.evaluate_annotator_image(
                    stem, annotator, pred_sparse, gt_masks,
                    disk_info=disk_info_by_stem.get(stem), site=site_by_stem.get(stem),
                )
                per_config_matches[cfg_idx].append(r.match)
        if progress:
            print(f"  grid_search: stem {stem_i + 1}/{len(tune_stems)} ({stem}) done in {time.time() - t0:.1f}s", flush=True)
        del probs

    rows = []
    best_cfg, best_pq = None, -1.0
    for cfg, matches in zip(configs, per_config_matches):
        overall = skmetric.aggregate_matches(matches) if matches else None
        pq = overall.pq if overall else 0.0
        rows.append({**cfg.__dict__, "pq": pq})
        if pq > best_pq:
            best_pq, best_cfg = pq, cfg
    return best_cfg, pd.DataFrame(rows)
