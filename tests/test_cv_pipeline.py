"""CODING_SPEC_05_ENSEMBLE.md section 6: P5's cached per-image statistics
must reproduce aggregate_evaluation exactly, and P5's fitting stages must be
deterministic across processes (hash seeds) and repeated runs."""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_k import cv_pipeline as cvp
from solar_k import evaluate as skeval
from solar_k import metric as skmetric
from solar_k import postprocess as skpp
from solar_k import components as skcomp

REPO_ROOT = Path(__file__).resolve().parents[1]
S = 160  # small synthetic image side


def _blob(rng, shape, n):
    m = np.zeros(shape, dtype=np.uint8)
    for _ in range(n):
        y, x = rng.randint(10, shape[0] - 10, size=2)
        h, w = rng.randint(2, 6), rng.randint(6, 30)
        m[y : y + h, x : x + w] = 1
    return m


def _synthetic_image(seed: int):
    rng = np.random.RandomState(seed)
    probs = rng.rand(S, S).astype(np.float32) * 0.4
    gt = {}
    fil = []
    for _ in range(6):
        m = np.zeros((S, S), dtype=np.uint8)
        y, x = rng.randint(10, S - 20, size=2)
        m[y : y + rng.randint(2, 5), x : x + rng.randint(10, 40)] = 1
        fil.append(m)
        probs[m.astype(bool)] = 0.6 + 0.4 * rng.rand()
    # annotator A labels most filaments, B a subset (plus one missed by the model), C none
    gt["A"] = np.stack(fil[:5])
    extra = _blob(rng, (S, S), 1)
    gt["B"] = np.stack([fil[0], fil[2], extra])
    gt["C"] = np.zeros((0, S, S), dtype=np.uint8)
    # some noise components the model fires on
    noise = _blob(rng, (S, S), 4).astype(bool)
    probs[noise] = 0.7
    processed = rng.rand(S, S).astype(np.float32)
    disk_info = (S / 2, S / 2, S / 2)
    return probs, processed, disk_info, gt


def _tables_and_preds(n_images=4, cfg=None):
    cfg = cfg or skpp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=1, min_area=1)
    out = []
    for i in range(n_images):
        probs, processed, disk_info, gt = _synthetic_image(i)
        stem = f"img{i}"
        t = cvp.component_table(stem, probs, processed, disk_info, cfg, gt)
        labels = cvp.component_labels(probs, cvp.components_cfg(cfg))
        sparse = skcomp.components_to_sparse(labels, t.features_df["label"].tolist())
        out.append((t, sparse, gt))
    return out


def test_cached_stats_equal_aggregate_evaluation():
    data = _tables_and_preds()
    index = cvp.StatsIndex([t for t, _, _ in data])
    rng = np.random.RandomState(0)
    all_stems = index.stem_mask(index.stems)
    for trial in range(25):
        keep = rng.rand(index.n_comp) < rng.rand()
        if trial == 0:
            keep[:] = True
        if trial == 1:
            keep[:] = False
        sub = rng.rand(len(index.stems)) < 0.7 if trial > 2 else all_stems
        results = []
        for si, (t, sparse, gt) in enumerate(data):
            if not sub[si]:
                continue
            k = keep[index.comp_offset[si] : index.comp_offset[si] + len(t.features_df)]
            pred = sparse[k]
            for a, g in gt.items():
                results.append(skeval.evaluate_annotator_image(t.stem, a, pred, g.reshape(g.shape[0], S, S)))
        o = skeval.aggregate_evaluation(results, quartile_edges=[10, 50, 100], noisy_stems=set()).overall
        s_iou, tp, fp, fn = index.stats(keep, sub)
        assert (tp, fp, fn) == (o.tp, o.fp, o.fn)
        assert s_iou == pytest.approx(sum(o.tp_ious), abs=1e-12)
        assert index.pq(keep, sub) == pytest.approx(o.pq, abs=1e-12)


def test_component_table_matches_label_components():
    probs, processed, disk_info, gt = _synthetic_image(7)
    cfg = skpp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=1, min_area=1)
    t = cvp.component_table("x", probs, processed, disk_info, cfg, gt)
    feats, labels = skcomp.extract_components("x", probs, processed, disk_info, cvp.components_cfg(cfg))
    skcomp.label_components(feats, labels, gt)
    ref = skcomp.to_dataframe(feats)
    pd.testing.assert_frame_equal(t.features_df, ref)


def test_cfg1_stats_match_grid_search_matches():
    probs, _, _, gt = _synthetic_image(3)
    cfgs = [skpp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=1, min_area=5),
            skpp.PostprocessConfig(mode="hysteresis", t_low=0.5, t_high=0.9, closing_radius=2, min_area=5)]
    stats = cvp.cfg1_stats_for_image(probs, gt, cfgs)
    for i, cfg in enumerate(cfgs):
        pred = skpp.postprocess_sparse(probs, cfg)
        ms = [skeval.evaluate_annotator_image("s", a, pred, g).match for a, g in gt.items()]
        o = skmetric.aggregate_matches(ms)
        assert stats[i, 1:].tolist() == [o.tp, o.fp, o.fn]
        assert stats[i, 0] == pytest.approx(sum(o.tp_ious), abs=1e-12)


def test_cv_grid_has_66_configs():
    # CODING_SPEC_06 E018 widened grid: 10 thresholds x 3 closing radii (30)
    # + 4 t_low x 3 t_high x 3 closing radii (36) = 66.
    assert len(cvp.CFG1_GRID) == 66
    assert len({cvp.cfg_key(c) for c in cvp.CFG1_GRID}) == 66


def test_agreement_features_match_manual_computation_on_toy_members():
    # One component (a 3x4 block); two toy "members" with known per-pixel
    # probabilities under it, so agree_mean/min/max/std and the majority
    # fraction can be checked against a hand computation.
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:8, 5:9] = 1
    avg_probs = np.full((20, 20), 0.5, dtype=np.float32)
    member_a = np.full((20, 20), 0.9, dtype=np.float32)  # all of mask > 0.5
    member_b = np.full((20, 20), 0.2, dtype=np.float32)  # none of mask > 0.5
    feats = skcomp._agreement_features(mask, {"A": member_a, "B": member_b}, avg_probs)
    assert feats["agree_mean"] == pytest.approx(0.55)
    assert feats["agree_min"] == pytest.approx(0.2)
    assert feats["agree_max"] == pytest.approx(0.9)
    assert feats["agree_std"] == pytest.approx(0.35)
    assert feats["agree_frac_majority"] == pytest.approx(0.5)  # only member A clears 50%


def test_agreement_features_without_member_probs_is_nan_but_ring_still_set():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:8, 5:9] = 1
    avg_probs = np.random.RandomState(0).rand(20, 20).astype(np.float32)
    feats = skcomp._agreement_features(mask, None, avg_probs)
    assert np.isnan(feats["agree_mean"])
    assert not np.isnan(feats["boundary_ring_prob"])


def test_extract_components_v2_columns_present_and_nan_without_members():
    probs, processed, disk_info, gt = _synthetic_image(2)
    cfg = skpp.PostprocessConfig(mode="threshold", t=0.5, closing_radius=1, min_area=1)
    feats, _ = skcomp.extract_components("x", probs, processed, disk_info, cvp.components_cfg(cfg))
    df = skcomp.to_dataframe(feats)
    for col in skcomp.AGREEMENT_FEATURE_COLS:
        assert col in df.columns
    assert df["agree_mean"].isna().all()


def test_day_split_is_seeded_day_grouped_and_disjoint():
    day = {f"s{i:03d}": f"d{i // 7}" for i in range(400)}
    train_a, val_a = cvp.day_split(list(day), day, val_frac=0.2, seed=0)
    train_b, val_b = cvp.day_split(list(reversed(list(day))), day, val_frac=0.2, seed=0)
    assert train_a == train_b and val_a == val_b
    assert set(train_a).isdisjoint(val_a)
    assert set(train_a) | set(val_a) == set(day)
    for s in val_a:
        assert day[s] not in {day[t] for t in train_a}


def test_choose_scorer_version_returns_v1_or_v2_with_selection_scores():
    data = _tables_and_preds(n_images=10)
    index = cvp.StatsIndex([t for t, _, _ in data])
    day_by_stem = {f"img{i}": f"d{i % 3}" for i in range(10)}
    version, selection = cvp.choose_scorer_version(index, index.stems, day_by_stem, log=lambda *a: None)
    assert version in ("v1", "v2")
    assert set(selection) == {"v1", "v2"}


def test_day_grouped_subsample_is_seeded_and_day_grouped():
    day = {f"s{i:03d}": f"d{i // 7}" for i in range(400)}
    a = cvp.day_grouped_subsample(list(day), day, 150, seed=0)
    b = cvp.day_grouped_subsample(list(reversed(list(day))), day, 150, seed=0)
    assert a == b and len(a) <= 150
    days = {day[s] for s in a}
    assert all(sum(day[s] == d for s in a) == sum(v == d for v in day.values()) for d in days)


_FIT_SCRIPT = r"""
import pickle, sys, json
import numpy as np
sys.path.insert(0, sys.argv[2])
from solar_k import cv_pipeline as cvp
cvp.MIN_AREA = 1  # synthetic components are tiny
tables = pickle.load(open(sys.argv[1], 'rb'))
index = cvp.StatsIndex(tables)
rule = cvp.fit_keep_rule(index, sorted(index.stems)[:-2])
print(json.dumps({"gate": rule["shape_gate"], "c": rule["scorer_c"], "pq": rule["c_train_pq"],
                  "keep": np.flatnonzero(rule["keep"]).tolist(),
                  "scores": cvp.sksc.keep_score(rule["scorer"], index.features_df).round(15).tolist()}))
"""


def test_fit_keep_rule_deterministic_across_processes(tmp_path):
    # enough components for a non-trivial scorer
    tables = [t for t, _, _ in _tables_and_preds(n_images=12)]
    pkl = tmp_path / "tables.pkl"
    pkl.write_bytes(pickle.dumps(tables))
    script = tmp_path / "fit.py"
    script.write_text(_FIT_SCRIPT)
    outs = []
    for hashseed, threads in (("1", "1"), ("12345", "4"), ("random", "8")):
        env = {**os.environ, "PYTHONHASHSEED": hashseed, "OMP_NUM_THREADS": threads}
        r = subprocess.run([sys.executable, str(script), str(pkl), str(REPO_ROOT / "src")], capture_output=True, text=True, env=env, check=True)
        outs.append(r.stdout)
    assert outs[0] == outs[1] == outs[2]
    assert json.loads(outs[0])["keep"]


def test_paired_bootstrap_identity_is_zero():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"stem": [f"s{i}" for i in range(50)], "cv_fold": np.arange(50) % 5,
                       "sum_iou": rng.rand(50) * 3, "tp": rng.randint(0, 5, 50), "fp": rng.randint(0, 3, 50), "fn": rng.randint(0, 3, 50)})
    r = cvp.paired_comparison(df, df.copy())
    assert r["pooled_delta_pq"] == 0 and r["bootstrap_ci95_pooled_delta_pq"] == [0.0, 0.0]
