#!/usr/bin/env python3
"""Decision-rule comparison for CODING_SPEC_04_MODEL_SIDE.md section 2: is
a new recipe (E006 hard-negative mining, E007 stronger encoder, ...) worth
keeping, compared to E004 on the *same* fold(s)?

Two paths, computed for both the candidate and E004, on the same stems:
  - "raw": E002's fixed tuned postprocess config applied directly, no
    fitting -- isolates whether the new model's probability maps are
    better on their own, with zero new tuning.
  - "full": E003's step1 (threshold/closing/min_area grid) + step3 (shape
    gate) + step4 (learned scorer) pipeline, refit from scratch for each
    model on that fold's own tune stems (day-grouped split), reported on
    that fold's own report stems -- never tuned on data it's scored
    against, and never shared between the candidate and E004 (each gets
    its own independent refit, since a postprocessing config tuned for one
    model's probability calibration may not suit another's -- see E008's
    finding that E003's fold-0-tuned config under-detects on other folds).

Decision (pre-registered, CODING_SPEC_04 section 2): keep only if full PQ
improves by >= 0.02 on EVERY given fold, and SQ does not fall on any of them.

Usage:
  python scripts/run_e007_eval.py --experiment E006 --folds 0 1
  python scripts/run_e007_eval.py --experiment E007 --folds 0 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import components as skcomp
from solar_k import evaluate as skeval
from solar_k import io as skio
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc
from solar_k import tuning as sktune

from run_e002_eval import build_gt_for_stem, load_annotation_index, load_prob_png
from run_e003 import ComponentCache, build_component_cache, evaluate_filtered, summary_row

REPO_ROOT = skio.repo_root()
C_GRID = [round(x, 3) for x in np.arange(0.0, 0.525, 0.025)]


def load_fold_data(fold: int, images_by_file, anns_by_image_id, train_paths):
    folds_df = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    rows = folds_df[folds_df["fold"] == fold]
    gt_by_stem, disk_info_by_stem, site_by_stem, day_by_stem = {}, {}, {}, {}
    for _, row in rows.iterrows():
        stem = row["file_stem"]
        _, disk_info = skpre.load_or_build(stem, train_paths[stem])
        disk_info_by_stem[stem] = disk_info
        site_by_stem[stem] = row["site"]
        day_by_stem[stem] = row["date"]
        gt_by_stem[stem] = build_gt_for_stem(stem, images_by_file, anns_by_image_id)
    return gt_by_stem, disk_info_by_stem, site_by_stem, day_by_stem


def raw_path(oof_dir: Path, stems, gt_by_stem, disk_info_by_stem, site_by_stem, e002_cfg, quartile_edges) -> dict:
    results = []
    for stem in stems:
        png = oof_dir / f"{stem}.png"
        if not png.exists():
            continue
        probs = load_prob_png(png)
        pred_sparse = skpp.postprocess_sparse(probs, e002_cfg)
        for annotator, gt_masks in gt_by_stem[stem].items():
            results.append(skeval.evaluate_annotator_image(
                stem, annotator, pred_sparse, gt_masks,
                disk_info=disk_info_by_stem.get(stem), site=site_by_stem.get(stem),
            ))
        del probs
    s = skeval.aggregate_evaluation(results, quartile_edges=quartile_edges, noisy_stems=set())
    return summary_row(s)


def full_path(oof_dir: Path, tune_stems, report_stems, gt_by_stem, disk_info_by_stem, site_by_stem, eda_p1_area, quartile_edges) -> dict:
    """E003's step1+3+4, refit from scratch on this model's own OOF."""
    train_paths = {p.stem: p for p in skio.iter_train_images()}
    all_stems = tune_stems + report_stems

    def compute_probs(stem):
        return load_prob_png(oof_dir / f"{stem}.png")

    configs = sktune.full_grid(eda_p1_area)
    best_cfg1, _ = sktune.grid_search(set(tune_stems), compute_probs, gt_by_stem, disk_info_by_stem, site_by_stem, configs)

    cfg_components = skpp.PostprocessConfig(
        mode=best_cfg1.mode, t=best_cfg1.t, t_low=best_cfg1.t_low, t_high=best_cfg1.t_high,
        closing_radius=best_cfg1.closing_radius, min_area=1, max_instances=5000,
    )
    cache_by_stem: dict[str, ComponentCache] = {}
    for stem in all_stems:
        processed, _ = skpre.load_or_build(stem, train_paths[stem])
        cache_by_stem[stem] = build_component_cache(stem, compute_probs(stem), processed, disk_info_by_stem[stem], cfg_components, gt_by_stem[stem])
        del processed

    def area_keep(df):
        return (df["area"] >= best_cfg1.min_area).to_numpy() if len(df) else np.zeros(0, dtype=bool)

    elong_grid = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    len_grid = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]

    def shape_keep(me, ml):
        def fn(df):
            if len(df) == 0:
                return np.zeros(0, dtype=bool)
            return area_keep(df) & (df["elongation"].fillna(0) >= me).to_numpy() & (df["length"].fillna(0) >= ml).to_numpy()
        return fn

    best_shape, best_shape_pq = (0.0, 0.0), -1.0
    for me in elong_grid:
        for ml in len_grid:
            s = evaluate_filtered(tune_stems, cache_by_stem, gt_by_stem, disk_info_by_stem, site_by_stem, shape_keep(me, ml), quartile_edges, set())
            if s.overall.pq > best_shape_pq:
                best_shape_pq, best_shape = s.overall.pq, (me, ml)
    step3_keep = shape_keep(*best_shape)

    tune_df = pd.concat([cache_by_stem[s].features_df.assign(stem=s) for s in tune_stems], ignore_index=True)
    scorer = sksc.fit_scorer(tune_df, seed=0)
    scores_by_stem = {s: (sksc.keep_score(scorer, cache_by_stem[s].features_df) if len(cache_by_stem[s].features_df) else np.zeros(0)) for s in all_stems}

    def evaluate_scorer(stems, c, compute_pixel_metrics=False):
        results = []
        for stem in stems:
            cache = cache_by_stem[stem]
            df = cache.features_df
            keep = (step3_keep(df) & (scores_by_stem[stem] > c)) if len(df) else np.zeros(0, dtype=bool)
            pred_sparse = cache.sparse[keep] if len(df) else cache.sparse
            for annotator, gt_masks in gt_by_stem[stem].items():
                results.append(skeval.evaluate_annotator_image(
                    stem, annotator, pred_sparse, gt_masks,
                    disk_info=disk_info_by_stem.get(stem), site=site_by_stem.get(stem),
                    compute_pixel_metrics=compute_pixel_metrics,
                ))
        return skeval.aggregate_evaluation(results, quartile_edges=quartile_edges, noisy_stems=set())

    best_c, best_c_pq = 0.0, -1.0
    for c in C_GRID:
        s = evaluate_scorer(tune_stems, c)
        if s.overall.pq > best_c_pq:
            best_c_pq, best_c = s.overall.pq, c

    report_summary = evaluate_scorer(report_stems, best_c, compute_pixel_metrics=True)
    return {"cfg1": best_cfg1.__dict__, "shape_gate": best_shape, "scorer_c": best_c, **summary_row(report_summary)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="e.g. E006 or E007")
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    t_start = time.time()
    images_by_file, anns_by_image_id = load_annotation_index()
    train_paths = {p.stem: p for p in skio.iter_train_images()}

    e002_summary = json.loads((REPO_ROOT / "data" / "interim" / "E002_eval_summary.json").read_text())
    e002_cfg = skpp.PostprocessConfig.from_dict(e002_summary["best_config"])

    instance_stats = pd.read_csv(REPO_ROOT / "data" / "interim" / "instance_stats.csv")
    eda_p1_area = float(instance_stats["area"].quantile(0.01))
    quartile_edges = instance_stats["area"].quantile([0.25, 0.5, 0.75]).tolist()

    per_fold = {}
    for fold in args.folds:
        gt_by_stem, disk_info_by_stem, site_by_stem, day_by_stem = load_fold_data(fold, images_by_file, anns_by_image_id, train_paths)
        tune_stems, report_stems = sktune.split_tune_report(day_by_stem, seed=0)
        tune_stems, report_stems = sorted(tune_stems), sorted(report_stems)
        all_stems = tune_stems + report_stems

        candidate_dir = skio.output_root() / f"{args.experiment}-f{fold}" / "oof"
        baseline_dir = skio.output_root() / "E004-f{}".format(fold) / "oof"
        if not candidate_dir.exists():
            print(f"fold {fold}: {candidate_dir} missing, skipping")
            continue

        print(f"--- fold {fold}: {len(tune_stems)} tune, {len(report_stems)} report ---", flush=True)
        raw_candidate = raw_path(candidate_dir, all_stems, gt_by_stem, disk_info_by_stem, site_by_stem, e002_cfg, quartile_edges)
        raw_baseline = raw_path(baseline_dir, all_stems, gt_by_stem, disk_info_by_stem, site_by_stem, e002_cfg, quartile_edges)
        print(f"  raw: {args.experiment}={raw_candidate} E004={raw_baseline}", flush=True)

        full_candidate = full_path(candidate_dir, tune_stems, report_stems, gt_by_stem, disk_info_by_stem, site_by_stem, eda_p1_area, quartile_edges)
        full_baseline = full_path(baseline_dir, tune_stems, report_stems, gt_by_stem, disk_info_by_stem, site_by_stem, eda_p1_area, quartile_edges)
        print(f"  full: {args.experiment}={full_candidate} E004={full_baseline}", flush=True)

        pq_delta = full_candidate["pq"] - full_baseline["pq"]
        sq_delta = full_candidate["sq"] - full_baseline["sq"]
        per_fold[fold] = {
            "raw_candidate": raw_candidate, "raw_baseline": raw_baseline,
            "full_candidate": full_candidate, "full_baseline": full_baseline,
            "pq_delta": round(pq_delta, 4), "sq_delta": round(sq_delta, 4),
            "passes_fold": bool(pq_delta >= 0.02 and sq_delta >= 0.0),
        }
        print(f"  fold {fold}: PQ delta={pq_delta:+.4f} SQ delta={sq_delta:+.4f} passes={per_fold[fold]['passes_fold']}", flush=True)

    overall_pass = len(per_fold) > 0 and all(v["passes_fold"] for v in per_fold.values())
    summary = {"experiment": args.experiment, "folds": args.folds, "per_fold": per_fold, "decision": "keep" if overall_pass else "reject"}
    out_path = REPO_ROOT / "data" / "interim" / f"{args.experiment}_decision.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDecision for {args.experiment}: {summary['decision'].upper()}")
    print(json.dumps(summary, indent=2))
    print(f"Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
