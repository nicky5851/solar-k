#!/usr/bin/env python3
"""Protocol P5 (CODING_SPEC_05_ENSEMBLE.md section 2.1) for one variant: a
set of ensemble members, equal-weight averaged per image (src/solar_k/
ensemble.py), run through the nested per-CV-fold refit in
src/solar_k/cv_pipeline.py.

Writes data/interim/<experiment>_p5_summary.json, <experiment>_p5_cv.csv
and <experiment>_p5_per_image.csv (per held-out image TP-IoU sum / TP / FP /
FN, the input to paired comparisons). With --reference, adds the paired
deltas + bootstrap CI against that experiment's per-image file and applies
the pre-registered decision bar (section 2.2).

Usage:
  python scripts/run_p5.py --experiment E010 --members E004 --reference E009_like --bar pipeline
  python scripts/run_p5.py --experiment E011_mean --members E004 E007 --model-folds 0 1 --reference E011_E004 --bar early
  python scripts/run_p5.py --experiment E013 --members E004 E007 --reference E010 --bar ensemble --write-submission
  python scripts/run_p5.py --experiment E010_rerun --members E004 --cv-folds 0 --cache-tag rerun   # determinism check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import components as skcomp
from solar_k import cv_pipeline as cvp
from solar_k import ensemble as skens
from solar_k import io as skio
from solar_k import masks as skmasks
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc

REPO_ROOT = skio.repo_root()
INTERIM = REPO_ROOT / "data" / "interim"

BARS = {
    # CODING_SPEC_05 section 2.2
    "ensemble": {"min_mean_delta": 0.010, "min_folds_positive": 4, "sq_tolerance": 0.005, "ci_excludes_zero": True},
    "pipeline": {"min_mean_delta": 0.005, "min_folds_positive": 4, "sq_tolerance": 0.005, "ci_excludes_zero": True},
    "early": {"min_pooled_delta": 0.008},
}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def decide(bar: str, cmp_cv: dict, cmp_model: dict | None) -> dict:
    b = BARS[bar]
    if bar == "early":
        checks = {
            "pooled_delta_ge_0.008": bool(cmp_cv["pooled_delta_pq"] >= b["min_pooled_delta"]),
            "positive_on_every_model_fold": bool(all(v > 0 for v in cmp_model["per_group_delta_pq"].values())),
        }
    else:
        checks = {
            f"mean_delta_ge_{b['min_mean_delta']}": bool(cmp_cv["mean_group_delta_pq"] >= b["min_mean_delta"]),
            f"positive_on_ge_{b['min_folds_positive']}_folds": bool(cmp_cv["n_groups_positive"] >= b["min_folds_positive"]),
            "pooled_sq_drop_le_0.005": bool(cmp_cv["pooled_delta_sq"] >= -b["sq_tolerance"]),
            "bootstrap_ci_excludes_0": bool(cmp_cv["bootstrap_ci95_pooled_delta_pq"][0] > 0),
        }
    return {"bar": bar, "checks": checks, "passes": bool(all(checks.values()))}


def write_submission(experiment: str, members: list[str], stems: list[str], day_by_stem: dict, runner_workers: int, weight_grid: list[tuple[float, ...]] | None = None) -> dict:
    """Final post-processing fit on all OOF stems, applied to the mean of every
    member x fold test map (spec E010/E013)."""
    import validate_submission as vs

    runner = cvp.Runner(members, runner_workers)
    try:
        if weight_grid:
            weights, cfg1, cfg1_pq = cvp.choose_weights_and_cfg1(runner, stems, stems, weight_grid, log)
        else:
            weights = None
            stats = cvp.cfg1_stats(runner, stems, None, log)
            cfg1, cfg1_pq = cvp.choose_cfg1(stats, stems)
        tables = cvp.component_tables(runner, stems, cfg1, weights, log)
        index = cvp.StatsIndex([tables[s] for s in stems])
        rule = cvp.fit_keep_rule(index, stems)
        kept = cvp.kept_labels_by_stem(index, rule["keep"], stems)
        sub = cvp.day_grouped_subsample(stems, day_by_stem)
        (t_mask, d), _, _ = cvp.choose_regrow(runner, cfg1, kept, sub, weights, log)
    finally:
        runner.close()
    log(f"final fit on {len(stems)}: cfg1={cvp.cfg_key(cfg1)} gate={rule['shape_gate']} c={rule['scorer_c']} regrow=({t_mask},{d})")

    ccfg = cvp.components_cfg(cfg1)
    me, ml = rule["shape_gate"]
    rows = []
    test_paths = {p.stem: p for p in skio.iter_test_images()}
    for stem in sorted(test_paths):
        probs = skens.load_test_mean(members, stem, weights=list(weights) if weights else None)
        processed, disk_info = skpre.load_or_build(stem, test_paths[stem])
        features, labels = skcomp.extract_components(stem, probs, processed, disk_info, ccfg)
        df = skcomp.to_dataframe(features)
        if len(df):
            keep = cvp.gate_keep(df, me, ml) & (sksc.keep_score(rule["scorer"], df) > rule["scorer_c"])
            seeds = [(labels == f.label).astype(np.uint8) for f, k in zip(features, keep) if k]
            grown = skpp.regrow_instances(probs, seeds, t_mask, d) if seeds else []
            for k_i, mask in enumerate(grown):
                if mask.sum() == 0:
                    continue
                rows.append({"filament_id": f"{stem}_{k_i}", "segmentation_rle": skmasks.encode_rle(mask.astype(np.uint8))})
        del probs, processed, labels
    out = REPO_ROOT / "outputs" / "submissions" / f"{experiment}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(out, index=False)
    errors = vs.validate(out, 2048, 2048, image_dir=skio.test_images_dir())
    if errors:
        raise SystemExit(f"{out} FAILED validate_submission.py: {errors[:5]}")
    log(f"wrote + validated {out} ({len(rows)} instances)")
    return {"path": str(out.relative_to(REPO_ROOT)), "n_instances": len(rows), "cfg1": cfg1.__dict__,
            "shape_gate": rule["shape_gate"], "scorer_c": rule["scorer_c"], "regrow_t_mask": t_mask, "regrow_d": d,
            "test_members": members, "test_folds": list(range(skens.N_MODEL_FOLDS)),
            "member_weights": list(weights) if weights else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--model-folds", nargs="+", type=int, default=None, help="restrict to OOF images of these model folds (E011: 0 1)")
    ap.add_argument("--cv-folds", nargs="+", type=int, default=None, help="run only these CV folds (determinism checks)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reference", default=None, help="experiment id whose <id>_p5_per_image.csv to pair against")
    ap.add_argument("--bar", choices=sorted(BARS), default=None)
    ap.add_argument("--cache-tag", default="", help="separate cold cache (for the run-twice determinism check)")
    ap.add_argument("--write-submission", action="store_true")
    ap.add_argument("--weight-grid", nargs="+", default=None,
                     help="CODING_SPEC_06 E018: one or more comma-separated per-member weight vectors "
                          "(aligned with --members order), e.g. --weight-grid 0,0.5,0.5 0.15,0.425,0.425 0.3,0.35,0.35. "
                          "Chosen inside each CV fold's train portion jointly with cfg1.")
    args = ap.parse_args()

    weight_grid = None
    if args.weight_grid:
        weight_grid = [tuple(float(x) for x in w.split(",")) for w in args.weight_grid]
        for w in weight_grid:
            if len(w) != len(args.members):
                raise SystemExit(f"weight vector {w} has {len(w)} entries for {len(args.members)} members")

    t0 = time.time()
    cvp.CACHE_TAG = args.cache_tag
    folds_df = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    day_by_stem = dict(zip(folds_df["file_stem"], folds_df["date"]))
    site_by_stem = dict(zip(folds_df["file_stem"], folds_df["site"]))
    fold_by_stem = dict(zip(folds_df["file_stem"], folds_df["fold"].astype(int)))
    stems = skens.available_oof_stems(args.members, fold_by_stem, args.model_folds)
    if not stems:
        raise SystemExit("no OOF images available for every member")
    train_paths = {p.stem: p for p in skio.iter_train_images()}
    disk_info_by_stem = {s: skpre.load_or_build(s, train_paths[s])[1] for s in stems}
    instance_stats = pd.read_csv(INTERIM / "instance_stats.csv")
    quartile_edges = instance_stats["area"].quantile([0.25, 0.5, 0.75]).tolist()
    model_folds_present = sorted({fold_by_stem[s] for s in stems})
    log(f"{args.experiment}: members={args.members} images={len(stems)} model folds={model_folds_present}")

    res = cvp.run_p5(args.members, stems, day_by_stem, site_by_stem, disk_info_by_stem, quartile_edges,
                     workers=args.workers, cv_folds=args.cv_folds, weight_grid=weight_grid, log=log)

    per_image_path = INTERIM / f"{args.experiment}_p5_per_image.csv"
    res.per_image.to_csv(per_image_path, index=False)
    cv_rows = [{k: v for k, v in f.items() if k not in ("loss_decomposition", "regrow_grid_pq")} | {"cfg1": cvp.cfg_key(skpp.PostprocessConfig.from_dict(f["cfg1"]))} for f in res.per_fold]
    pd.DataFrame(cv_rows).to_csv(INTERIM / f"{args.experiment}_p5_cv.csv", index=False)

    summary = {
        "experiment": args.experiment, "protocol": "P5", "members": args.members,
        "model_folds": model_folds_present, "n_images": len(stems), "cv_folds_run": args.cv_folds or list(range(len(res.per_fold))),
        **res.summary, "per_fold": res.per_fold,
        "grids": {"cfg1": [cvp.cfg_key(c) for c in cvp.CFG1_GRID], "elongation": cvp.ELONG_GRID, "length": cvp.LEN_GRID,
                  "c": cvp.C_GRID, "regrow": cvp.REGROW_GRID, "regrow_max_images": cvp.REGROW_MAX_IMAGES, "seed": cvp.SEED,
                  "weight_grid": [list(w) for w in weight_grid] if weight_grid else None},
    }
    if len(model_folds_present) > 1 and res.per_image is not None and len(res.per_image):
        summary["per_model_fold_pq"] = {int(k): round(cvp.pooled_pq_from_images(g), 4) for k, g in res.per_image.groupby("model_fold")}

    if args.reference:
        ref = pd.read_csv(INTERIM / f"{args.reference}_p5_per_image.csv")
        cmp_cv = cvp.paired_comparison(res.per_image, ref, "cv_fold")
        cmp_model = cvp.paired_comparison(res.per_image, ref, "model_fold") if len(model_folds_present) > 1 else None
        summary["vs_reference"] = {"reference": args.reference, "by_cv_fold": cmp_cv, "by_model_fold": cmp_model}
        if args.bar:
            summary["vs_reference"]["decision"] = decide(args.bar, cmp_cv, cmp_model)
        log(f"vs {args.reference}: {json.dumps(summary['vs_reference'])}")

    if args.write_submission:
        summary["submission"] = write_submission(args.experiment, args.members, stems, day_by_stem, args.workers, weight_grid)

    summary["runtime_minutes"] = round((time.time() - t0) / 60, 1)
    out = INTERIM / f"{args.experiment}_p5_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"{args.experiment}: CV PQ {summary.get('cv_pq_mean')} +/- {summary.get('cv_pq_std')} pooled {summary.get('pooled_pq')} -> {out.relative_to(REPO_ROOT)} ({summary['runtime_minutes']} min)")


if __name__ == "__main__":
    main()
