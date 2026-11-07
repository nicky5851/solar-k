#!/usr/bin/env python3
"""CODING_SPEC_07 E026: min_area grid {150, 250} (418 already exists as
E023_4flip_scale) on the current best-known TTA maps (E007tta4s +
E015tta4s, the REF's own members). Reuses run_p5.py's main() logic but
monkey-patches cv_pipeline.MIN_AREA/CFG1_GRID per grid point, since MIN_AREA
is a module constant with no CLI flag.

Note (provisional vs spec text): Spec 07 says "run on the best E025 maps."
E025 (larger TTA scales) was still running when this was written, so this
runs against the current best-known maps (E023's TTA, REF's own members)
instead -- cheap to rerun against whatever E025 finds later if that changes
the leading member set.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime

import pandas as pd

sys.path.insert(0, "/workspace/Solar/src")
sys.path.insert(0, "/workspace/Solar/scripts")

from solar_k import cv_pipeline as cvp
from solar_k import ensemble as skens
from solar_k import io as skio
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre

REPO_ROOT = skio.repo_root()
INTERIM = REPO_ROOT / "data" / "interim"
MEMBERS = ["E007tta4s", "E015tta4s"]
REFERENCE = "E023_4flip_scale"
BAR = {"min_mean_delta": 0.005, "min_folds_positive": 4, "sq_tolerance": 0.005, "ci_excludes_zero": True}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def rebuild_cfg1_grid(min_area: int) -> list:
    return (
        [skpp.PostprocessConfig(mode="threshold", t=t, closing_radius=r, min_area=min_area) for t in cvp._T_GRID for r in cvp._R_GRID]
        + [skpp.PostprocessConfig(mode="hysteresis", t_low=lo, t_high=hi, closing_radius=r, min_area=min_area) for lo, hi in cvp._HYST_GRID for r in cvp._R_GRID]
    )


def decide(bar: dict, cmp_cv: dict) -> dict:
    checks = {
        f"mean_delta_ge_{bar['min_mean_delta']}": bool(cmp_cv["mean_group_delta_pq"] >= bar["min_mean_delta"]),
        f"positive_on_ge_{bar['min_folds_positive']}_folds": bool(cmp_cv["n_groups_positive"] >= bar["min_folds_positive"]),
        "pooled_sq_drop_le_0.005": bool(cmp_cv["pooled_delta_sq"] >= -bar["sq_tolerance"]),
        "bootstrap_ci_excludes_0": bool(cmp_cv["bootstrap_ci95_pooled_delta_pq"][0] > 0),
    }
    return {"bar": "pipeline", "checks": checks, "passes": bool(all(checks.values()))}


def run_one(min_area: int) -> dict:
    experiment = f"E026_ma{min_area}"
    t0 = time.time()
    cvp.MIN_AREA = min_area
    cvp.CFG1_GRID = rebuild_cfg1_grid(min_area)

    folds_df = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    day_by_stem = dict(zip(folds_df["file_stem"], folds_df["date"]))
    site_by_stem = dict(zip(folds_df["file_stem"], folds_df["site"]))
    fold_by_stem = dict(zip(folds_df["file_stem"], folds_df["fold"].astype(int)))
    stems = skens.available_oof_stems(MEMBERS, fold_by_stem, None)
    train_paths = {p.stem: p for p in skio.iter_train_images()}
    disk_info_by_stem = {s: skpre.load_or_build(s, train_paths[s])[1] for s in stems}
    instance_stats = pd.read_csv(INTERIM / "instance_stats.csv")
    quartile_edges = instance_stats["area"].quantile([0.25, 0.5, 0.75]).tolist()
    log(f"{experiment}: min_area={min_area} members={MEMBERS} images={len(stems)}")

    res = cvp.run_p5(MEMBERS, stems, day_by_stem, site_by_stem, disk_info_by_stem, quartile_edges,
                      workers=3, cv_folds=None, weight_grid=None, scorer_grid=False, enable_merge=False,
                      save_details=False, gallery_stems=None, gallery_dir=None, log=log)

    per_image_path = INTERIM / f"{experiment}_p5_per_image.csv"
    res.per_image.to_csv(per_image_path, index=False)
    cv_rows = [{k: v for k, v in f.items() if k not in ("loss_decomposition", "regrow_grid_pq", "merge_grid_pq", "scorer_selection")} | {"cfg1": cvp.cfg_key(skpp.PostprocessConfig.from_dict(f["cfg1"]))} for f in res.per_fold]
    pd.DataFrame(cv_rows).to_csv(INTERIM / f"{experiment}_p5_cv.csv", index=False)

    summary = {
        "experiment": experiment, "protocol": "P5", "members": MEMBERS, "min_area": min_area,
        "n_images": len(stems), **res.summary, "per_fold": res.per_fold,
    }

    ref = pd.read_csv(INTERIM / f"{REFERENCE}_p5_per_image.csv")
    cmp_cv = cvp.paired_comparison(res.per_image, ref, "cv_fold")
    summary["vs_reference"] = {"reference": REFERENCE, "by_cv_fold": cmp_cv}
    summary["vs_reference"]["decision"] = decide(BAR, cmp_cv)
    summary["runtime_minutes"] = round((time.time() - t0) / 60, 1)

    out = INTERIM / f"{experiment}_p5_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log(f"{experiment}: CV PQ {summary.get('cv_pq_mean')} vs {REFERENCE} delta_mean={cmp_cv['mean_group_delta_pq']:.4f} "
        f"positive_folds={cmp_cv['n_groups_positive']} passes={summary['vs_reference']['decision']['passes']} "
        f"({summary['runtime_minutes']} min) -> {out.relative_to(REPO_ROOT)}")
    return summary


if __name__ == "__main__":
    results = {}
    for ma in (150, 250):
        results[ma] = run_one(ma)
    print(json.dumps({k: v["vs_reference"]["decision"] for k, v in results.items()}, indent=2))
