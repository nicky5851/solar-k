#!/usr/bin/env python3
"""E003 (CODING_SPEC_03_FP_AND_FOLDS.md): post-processing only, fold-0 OOF
probs, no training. Five cumulative steps against the report-02 finding that
RQ (not SQ) is E002's gap, driven mainly by confident `isolated` FPs:

  1. Wider threshold grid (t 0.50..0.95, hysteresis widened to match).
  2. Component feature table (src/solar_k/components.py) over tune+report.
  3. Shape gate: tune min elongation / min length on top of step 1.
  4. Learned component scorer (src/solar_k/scorer.py): keep iff
     P(match) * E[IoU] > c, c tuned on tune, on top of steps 1+3.
  5. pixel_iou_dice wired into every report-half evaluation (report-02 gap).

Each step is evaluated on `report` only after its own parameter(s) are
chosen on `tune` -- nothing here is fit or tuned on data it's scored
against. Final pipeline (steps 1+3+4, refit on tune+report combined) is
applied to the cached E002 test probs -> outputs/submissions/E003.csv.

Usage: python scripts/run_e003.py
"""
from __future__ import annotations

import collections
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import components as skcomp
from solar_k import evaluate as skeval
from solar_k import io as skio
from solar_k import metric as skmetric
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc
from solar_k import tuning as sktune

import validate_submission as vs
from run_e002_eval import build_gt_for_stem, fold0_val_stems, load_annotation_index, load_prob_png

H = W = 2048
REPO_ROOT = skio.repo_root()
OOF_DIR = skio.output_root() / "E002" / "oof"
TEST_PROBS_DIR = skio.output_root() / "E002" / "test_probs"
IOU_T = skmetric.IOU_THRESHOLD
WEAK_IOU = 0.1
NEAR_MISS = (0.3, 0.5)

ELONGATION_GRID = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LENGTH_GRID = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
C_GRID = [round(x, 3) for x in np.arange(0.0, 0.525, 0.025)]


@dataclass
class ComponentCache:
    features_df: pd.DataFrame
    sparse: "object"  # scipy.sparse.csr_matrix, one row per features_df row, same order


def build_component_cache(stem, probs, processed, disk_info, cfg, gt_by_annotator) -> ComponentCache:
    features, labels = skcomp.extract_components(stem, probs, processed, disk_info, cfg)
    skcomp.label_components(features, labels, gt_by_annotator)
    df = skcomp.to_dataframe(features)
    sparse = skcomp.components_to_sparse(labels, [f.label for f in features])
    del labels
    return ComponentCache(features_df=df, sparse=sparse)


def evaluate_filtered(
    stems, cache_by_stem, gt_by_stem, disk_info_by_stem, site_by_stem, keep_mask_fn, quartile_edges, noisy_stems,
    compute_pixel_metrics=False,
) -> skeval.EvaluationSummary:
    results = []
    for stem in stems:
        cache = cache_by_stem[stem]
        df = cache.features_df
        keep = np.asarray(keep_mask_fn(df), dtype=bool) if len(df) else np.zeros(0, dtype=bool)
        pred_sparse = cache.sparse[keep] if len(df) else cache.sparse
        for annotator, gt_masks in gt_by_stem[stem].items():
            r = skeval.evaluate_annotator_image(
                stem, annotator, pred_sparse, gt_masks,
                disk_info=disk_info_by_stem.get(stem), site=site_by_stem.get(stem),
                compute_pixel_metrics=compute_pixel_metrics,
            )
            results.append(r)
    return skeval.aggregate_evaluation(results, quartile_edges=quartile_edges, noisy_stems=noisy_stems)


def summary_row(summary: skeval.EvaluationSummary) -> dict:
    o = summary.overall
    return {
        "pq": round(o.pq, 4), "sq": round(o.sq, 4), "rq": round(o.rq, 4),
        "tp": o.tp, "fp": o.fp, "fn": o.fn,
        "pixel_dice_mean": round(summary.pixel_dice_mean, 4) if summary.pixel_dice_mean is not None else None,
    }


def fp_class_counts(pred_sparse, gt_by_annotator: dict) -> collections.Counter:
    """Classify every FP (checked in this order, first match wins) --
    mirrors scripts/run_e002_deepdive.py, repeated here so E003 can report
    the FP mix before/after its own pipeline change."""
    annotators = list(gt_by_annotator.keys())
    iou_by_ann, hit_by_ann, tp_gt_by_ann = {}, {}, {}
    for a in annotators:
        gt_masks = gt_by_annotator[a]
        if gt_masks.shape[0] == 0 or pred_sparse.shape[0] == 0:
            iou = np.zeros((gt_masks.shape[0], pred_sparse.shape[0]))
        else:
            iou, _ = skmetric.pairwise_iou_dice(gt_masks, pred_sparse)
        hit = iou > IOU_T
        iou_by_ann[a], hit_by_ann[a] = iou, hit
        tp_gt_by_ann[a] = set(np.where(hit.any(axis=1))[0]) if hit.size else set()

    counts = collections.Counter()
    for a in annotators:
        iou, hit = iou_by_ann[a], hit_by_ann[a]
        n_gt, n_pred = iou.shape
        fp_idx = np.where(~hit.any(axis=0))[0] if n_pred else np.array([], dtype=int)
        tp_gt_idx = tp_gt_by_ann[a]
        for j in fp_idx:
            best_same = iou[:, j].max() if n_gt else 0.0
            if NEAR_MISS[0] <= best_same < NEAR_MISS[1]:
                counts["near_miss"] += 1
            elif any(iou_by_ann[b].shape[0] and iou_by_ann[b][:, j].max() > IOU_T for b in annotators if b != a):
                counts["annotator_disagreement"] += 1
            elif n_gt and tp_gt_idx and (iou[list(tp_gt_idx), j] > WEAK_IOU).any():
                counts["fragment"] += 1
            else:
                counts["isolated"] += 1
    return counts


def main() -> None:
    t_start = time.time()
    images_by_file, anns_by_image_id = load_annotation_index()
    val_df = fold0_val_stems()
    train_paths_by_stem = {p.stem: p for p in skio.iter_train_images()}
    test_paths_by_stem = {p.stem: p for p in skio.iter_test_images()}

    gt_by_stem, disk_info_by_stem, site_by_stem, day_by_stem = {}, {}, {}, {}
    for _, row in val_df.iterrows():
        stem = row["file_stem"]
        if not (OOF_DIR / f"{stem}.png").exists():
            continue
        _, disk_info = skpre.load_or_build(stem, train_paths_by_stem[stem])  # cache-aware; processed re-read (cheap) per stem below
        disk_info_by_stem[stem] = disk_info
        site_by_stem[stem] = row["site"]
        day_by_stem[stem] = row["date"]
        gt_by_stem[stem] = build_gt_for_stem(stem, images_by_file, anns_by_image_id)

    tune_stems, report_stems = sktune.split_tune_report(day_by_stem, seed=0)
    tune_stems = sorted(s for s in tune_stems if s in gt_by_stem)
    report_stems = sorted(s for s in report_stems if s in gt_by_stem)
    all_stems = tune_stems + report_stems
    print(f"Tune: {len(tune_stems)} images, Report: {len(report_stems)} images")

    instance_stats = pd.read_csv(REPO_ROOT / "data" / "interim" / "instance_stats.csv")
    eda_p1_area = float(instance_stats["area"].quantile(0.01))
    quartile_edges = instance_stats["area"].quantile([0.25, 0.5, 0.75]).tolist()
    noisy_stems: set[str] = set()
    if (REPO_ROOT / "data" / "interim" / "image_stats.csv").exists():
        image_stats = pd.read_csv(REPO_ROOT / "data" / "interim" / "image_stats.csv")
        noise_p90 = image_stats["noise"].quantile(0.9)
        noisy_stems = set(image_stats.loc[image_stats["noise"] >= noise_p90, "file_stem"])

    def compute_probs(stem: str) -> np.ndarray:
        return load_prob_png(OOF_DIR / f"{stem}.png")

    increments = {}

    # --- Step 1: wider threshold grid -------------------------------------
    configs = sktune.full_grid(eda_p1_area)
    print(f"Step 1: grid size {len(configs)} configs (t up to {max(sktune.SINGLE_T_GRID)})")
    t0 = time.time()
    best_cfg1, grid_df1 = sktune.grid_search(
        set(tune_stems), compute_probs, gt_by_stem, disk_info_by_stem, site_by_stem, configs, progress=True
    )
    print(f"Step 1 grid search done in {(time.time() - t0) / 60:.1f} min. Best on tune: PQ={grid_df1['pq'].max():.4f} cfg={best_cfg1}")
    grid_df1.to_csv(REPO_ROOT / "data" / "interim" / "E003_step1_grid_search.csv", index=False)

    def step1_keep(df: pd.DataFrame) -> np.ndarray:
        return np.ones(len(df), dtype=bool)

    # Component cache is built at step 1's winning (t, closing_radius) but
    # min_area=1 (unfiltered) -- so steps 1/3/4's area/shape/score filters
    # are all just boolean row-selects on one cached extraction per image.
    cfg_components = skpp.PostprocessConfig(
        mode=best_cfg1.mode, t=best_cfg1.t, t_low=best_cfg1.t_low, t_high=best_cfg1.t_high,
        closing_radius=best_cfg1.closing_radius, min_area=1, max_instances=5000,
    )
    cache_by_stem: dict[str, ComponentCache] = {}
    t0 = time.time()
    for i, stem in enumerate(all_stems):
        processed, _ = skpre.load_or_build(stem, train_paths_by_stem[stem])  # cache-aware, fast re-read
        cache_by_stem[stem] = build_component_cache(
            stem, compute_probs(stem), processed, disk_info_by_stem[stem], cfg_components, gt_by_stem[stem]
        )
        del processed
        if (i + 1) % 20 == 0 or i + 1 == len(all_stems):
            print(f"  component cache: {i + 1}/{len(all_stems)} images done, elapsed {time.time() - t0:.1f}s", flush=True)
    print(f"Step 2: component cache built in {(time.time() - t0) / 60:.1f} min")

    def area_keep(min_area: int):
        return lambda df: (df["area"] >= min_area).to_numpy() if len(df) else np.zeros(0, dtype=bool)

    step1_base_keep = area_keep(best_cfg1.min_area)
    step1_summary = evaluate_filtered(
        report_stems, cache_by_stem, gt_by_stem, disk_info_by_stem, site_by_stem, step1_base_keep,
        quartile_edges, noisy_stems, compute_pixel_metrics=True,
    )
    increments["step1_wider_grid"] = {"cfg": best_cfg1.__dict__, **summary_row(step1_summary)}
    print(f"Step 1 report-half: {increments['step1_wider_grid']}")

    all_tune_df = pd.concat([cache_by_stem[s].features_df.assign(stem=s, split="tune") for s in tune_stems], ignore_index=True)
    all_report_df = pd.concat([cache_by_stem[s].features_df.assign(stem=s, split="report") for s in report_stems], ignore_index=True)
    all_components_df = pd.concat([all_tune_df, all_report_df], ignore_index=True)
    all_components_df.to_csv(REPO_ROOT / "data" / "interim" / "E003_component_table.csv", index=False)
    print(f"Wrote {len(all_components_df)} component rows to data/interim/E003_component_table.csv")
    print(f"TP rate: tune={all_tune_df['is_tp'].mean():.3f} report={all_report_df['is_tp'].mean():.3f}")

    # --- Step 3: shape gate (min elongation, min length), on top of step 1 -
    def shape_keep(min_elong: float, min_len: float):
        base = step1_base_keep

        def fn(df: pd.DataFrame) -> np.ndarray:
            if len(df) == 0:
                return np.zeros(0, dtype=bool)
            return base(df) & (df["elongation"].fillna(0) >= min_elong).to_numpy() & (df["length"].fillna(0) >= min_len).to_numpy()
        return fn

    t0 = time.time()
    best_shape, best_shape_pq = (0.0, 0.0), -1.0
    shape_rows = []
    for me in ELONGATION_GRID:
        for ml in LENGTH_GRID:
            s = evaluate_filtered(tune_stems, cache_by_stem, gt_by_stem, disk_info_by_stem, site_by_stem, shape_keep(me, ml), quartile_edges, noisy_stems)
            shape_rows.append({"min_elongation": me, "min_length": ml, "pq": s.overall.pq})
            if s.overall.pq > best_shape_pq:
                best_shape_pq, best_shape = s.overall.pq, (me, ml)
    pd.DataFrame(shape_rows).to_csv(REPO_ROOT / "data" / "interim" / "E003_step3_shape_grid.csv", index=False)
    print(f"Step 3 shape-gate search done in {time.time() - t0:.1f}s. Best on tune: elongation>={best_shape[0]}, length>={best_shape[1]}, PQ={best_shape_pq:.4f}")

    step3_keep = shape_keep(*best_shape)
    step3_summary = evaluate_filtered(
        report_stems, cache_by_stem, gt_by_stem, disk_info_by_stem, site_by_stem, step3_keep,
        quartile_edges, noisy_stems, compute_pixel_metrics=True,
    )
    increments["step3_shape_gate"] = {"min_elongation": best_shape[0], "min_length": best_shape[1], **summary_row(step3_summary)}
    print(f"Step 3 report-half: {increments['step3_shape_gate']}")

    # --- Step 4: learned component scorer, on top of steps 1+3 ------------
    scorer_tune = sksc.fit_scorer(all_tune_df, seed=0)
    importances = sksc.feature_importances(scorer_tune, all_tune_df)
    importances.to_csv(REPO_ROOT / "data" / "interim" / "E003_feature_importances.csv", index=False)

    scores_by_stem: dict[str, np.ndarray] = {}
    for stem in all_stems:
        df = cache_by_stem[stem].features_df
        scores_by_stem[stem] = sksc.keep_score(scorer_tune, df) if len(df) else np.zeros(0)

    # The scorer's keep rule needs the per-stem predicted score array (built
    # once above, not a column recomputable from df alone), so this is a
    # stem-aware wrapper rather than the generic keep_mask_fn signature
    # steps 1/3 use through evaluate_filtered.
    def evaluate_scorer(stems, c: float, compute_pixel_metrics=False) -> skeval.EvaluationSummary:
        results = []
        for stem in stems:
            cache = cache_by_stem[stem]
            df = cache.features_df
            if len(df) == 0:
                keep = np.zeros(0, dtype=bool)
            else:
                keep = step3_keep(df) & (scores_by_stem[stem] > c)
            pred_sparse = cache.sparse[keep] if len(df) else cache.sparse
            for annotator, gt_masks in gt_by_stem[stem].items():
                r = skeval.evaluate_annotator_image(
                    stem, annotator, pred_sparse, gt_masks,
                    disk_info=disk_info_by_stem.get(stem), site=site_by_stem.get(stem),
                    compute_pixel_metrics=compute_pixel_metrics,
                )
                results.append(r)
        return skeval.aggregate_evaluation(results, quartile_edges=quartile_edges, noisy_stems=noisy_stems)

    t0 = time.time()
    c_rows = []
    best_c, best_c_pq = 0.0, -1.0
    for c in C_GRID:
        s = evaluate_scorer(tune_stems, c)
        c_rows.append({"c": c, "pq": s.overall.pq})
        if s.overall.pq > best_c_pq:
            best_c_pq, best_c = s.overall.pq, c
    pd.DataFrame(c_rows).to_csv(REPO_ROOT / "data" / "interim" / "E003_step4_c_grid.csv", index=False)
    print(f"Step 4 scorer-cutoff search done in {time.time() - t0:.1f}s. Best on tune: c={best_c}, PQ={best_c_pq:.4f}")

    step4_summary = evaluate_scorer(report_stems, best_c, compute_pixel_metrics=True)
    increments["step4_learned_scorer"] = {"c": best_c, **summary_row(step4_summary)}
    print(f"Step 4 report-half: {increments['step4_learned_scorer']}")

    context_cols = skcomp.CONTEXT_FEATURE_COLS
    shape_cols = skcomp.SHAPE_FEATURE_COLS
    imp_by_feat = importances.set_index("feature")["hgb_permutation_importance"]
    context_mean = float(imp_by_feat.reindex(context_cols).mean())
    shape_mean = float(imp_by_feat.reindex(shape_cols).mean())
    context_vs_shape = "context" if context_mean > shape_mean else "shape"
    print(f"Mean HGB permutation importance: context={context_mean:.4f} shape={shape_mean:.4f} -> {context_vs_shape} features win")

    # --- FP class shares before (E002's tuned cfg) / after (E003's final pipeline),
    # split by single- vs multi-annotator images, report half only ---------
    e002_summary = json.loads((REPO_ROOT / "data" / "interim" / "E002_eval_summary.json").read_text())
    e002_cfg = skpp.PostprocessConfig.from_dict(e002_summary["best_config"])

    def split_group(stems):
        single = [s for s in stems if len(gt_by_stem[s]) == 1]
        multi = [s for s in stems if len(gt_by_stem[s]) > 1]
        return {"single_annotator": single, "multi_annotator": multi}

    fp_shares = {"before_E002": {}, "after_E003": {}}
    groups = split_group(report_stems)
    for group_name, stems in groups.items():
        before_counts, after_counts = collections.Counter(), collections.Counter()
        for stem in stems:
            probs = compute_probs(stem)
            pred_e002 = skpp.postprocess_sparse(probs, e002_cfg)
            before_counts += fp_class_counts(pred_e002, gt_by_stem[stem])

            cache = cache_by_stem[stem]
            df = cache.features_df
            keep = (step3_keep(df) & (scores_by_stem[stem] > best_c)) if len(df) else np.zeros(0, dtype=bool)
            pred_e003 = cache.sparse[keep] if len(df) else cache.sparse
            after_counts += fp_class_counts(pred_e003, gt_by_stem[stem])
        fp_shares["before_E002"][group_name] = dict(before_counts)
        fp_shares["after_E003"][group_name] = dict(after_counts)
    print(f"FP class shares: {json.dumps(fp_shares, indent=2)}")

    # --- Final pipeline: refit scorer on tune+report, apply to test -------
    scorer_full = sksc.fit_scorer(all_components_df, seed=0)
    rows = []
    missing_test = []
    t0 = time.time()
    for path in skio.iter_test_images():
        stem = path.stem
        prob_path = TEST_PROBS_DIR / f"{stem}.png"
        if not prob_path.exists():
            missing_test.append(stem)
            continue
        probs = load_prob_png(prob_path)
        processed, disk_info = skpre.load_or_build(stem, test_paths_by_stem[stem])
        features, labels = skcomp.extract_components(stem, probs, processed, disk_info, cfg_components)
        df = skcomp.to_dataframe(features)
        if len(df) == 0:
            del probs, processed
            continue
        keep = step3_keep(df) & (sksc.keep_score(scorer_full, df) > best_c)
        kept_labels = [f.label for f, k in zip(features, keep) if k]
        for lid in kept_labels:
            mask = (labels == lid).astype(np.uint8)
            if mask.sum() == 0:
                continue
            from solar_k import masks as skmasks

            counts = skmasks.encode_rle(mask)
            rows.append({"filament_id": f"{stem}_{lid}", "segmentation_rle": counts})
        del probs, processed, labels
    if missing_test:
        print(f"Warning: {len(missing_test)} test stems have no cached prob map, skipped: {missing_test[:5]}")

    sub_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_csv = REPO_ROOT / "outputs" / "submissions" / "E003.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(sub_df)} rows to {out_csv} in {(time.time() - t0) / 60:.1f} min (not submitted -- wait for Nicky)")

    errors = vs.validate(out_csv, H, W, image_dir=skio.test_images_dir())
    if errors:
        print("E003.csv FAILED validate_submission.py:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("E003.csv validates OK.")

    exp_row = {
        "id": "E003",
        "date": date.today().isoformat(),
        "spec": "CODING_SPEC_03_FP_AND_FOLDS",
        "hypothesis": "post-processing alone (wider threshold grid, shape gate, learned component scorer) recovers most of E002's isolated-FP PQ loss without retraining",
        "config_path": "",
        "split_fold": "fold0 (tune/report split within)",
        "seed": 0,
        "compute": "Mac (CPU), post-processing + component scorer only, no training (fold-0 OOF probs from E002)",
        "runtime_minutes": round((time.time() - t_start) / 60, 1),
        "val_pq_mean": increments["step4_learned_scorer"]["pq"],
        "val_pq_std": "",
        "sq": increments["step4_learned_scorer"]["sq"],
        "rq": increments["step4_learned_scorer"]["rq"],
        "secondary_metrics": json.dumps({"increments": increments, "context_vs_shape_importance": context_vs_shape}),
        "decision": "",
        "notes": f"step1 cfg: {best_cfg1}; shape gate elongation>={best_shape[0]} length>={best_shape[1]}; scorer c={best_c}",
        "kaggle_public_score": "",
    }
    exp_csv = REPO_ROOT / "experiments.csv"
    with open(exp_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_row.keys()))
        writer.writerow(exp_row)
    print(f"Appended E003 row to {exp_csv}")

    eval_summary_path = REPO_ROOT / "data" / "interim" / "E003_eval_summary.json"
    eval_summary_path.write_text(
        json.dumps(
            {
                "increments": increments,
                "feature_importances": importances.to_dict(orient="records"),
                "context_vs_shape_importance": context_vs_shape,
                "fp_class_shares": fp_shares,
                "best_cfg1": best_cfg1.__dict__,
                "shape_gate": {"min_elongation": best_shape[0], "min_length": best_shape[1]},
                "scorer_c": best_c,
                "n_test_images": len(list(skio.iter_test_images())),
                "n_submission_rows": len(sub_df),
            },
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
        )
    )
    print(f"Wrote {eval_summary_path}")
    print(f"Total runtime: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
