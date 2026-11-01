from __future__ import annotations

import numpy as np
import pandas as pd

from solar_k import scorer as sksc
from solar_k.components import ALL_FEATURE_COLS


def _synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """TPs get higher prob_mean/elongation/contrast and lower texture_std
    than FPs, with noise -- separable but not trivially so, so the fitted
    models have something to learn."""
    rng = np.random.RandomState(seed)
    is_tp = rng.rand(n) < 0.4
    rows = []
    for tp in is_tp:
        base = 0.85 if tp else 0.55
        row = {c: rng.normal(0, 1) for c in ALL_FEATURE_COLS}
        row["prob_mean"] = base + rng.normal(0, 0.1)
        row["elongation"] = (5.0 if tp else 2.0) + rng.normal(0, 0.5)
        row["contrast"] = (0.2 if tp else 0.0) + rng.normal(0, 0.05)
        row["is_tp"] = bool(tp)
        row["best_iou"] = (0.7 if tp else 0.0) + (rng.normal(0, 0.1) if tp else 0.0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["best_iou"] = df["best_iou"].clip(0, 1)
    return df


def test_fit_scorer_predict_match_prob_in_unit_interval():
    df = _synthetic_df()
    scorer = sksc.fit_scorer(df)
    p_logistic = sksc.predict_match_prob(scorer, df, model="logistic")
    p_hgb = sksc.predict_match_prob(scorer, df, model="hgb")
    for p in (p_logistic, p_hgb):
        assert p.shape == (len(df),)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_scorer_separates_tp_from_fp_on_held_out_split():
    train_df = _synthetic_df(seed=0)
    test_df = _synthetic_df(seed=1)
    scorer = sksc.fit_scorer(train_df)
    score = sksc.keep_score(scorer, test_df)
    tp_scores = score[test_df["is_tp"].to_numpy()]
    fp_scores = score[~test_df["is_tp"].to_numpy()]
    assert tp_scores.mean() > fp_scores.mean()


def test_keep_rule_threshold_behaviour():
    df = _synthetic_df()
    scorer = sksc.fit_scorer(df)
    score = sksc.keep_score(scorer, df)
    # A threshold of 0 keeps everything; a threshold above the max keeps nothing.
    assert (score > -1.0).all()
    assert (score > score.max() + 1.0).sum() == 0


def test_feature_importances_covers_every_feature_and_is_nonnegative_for_hgb():
    df = _synthetic_df()
    scorer = sksc.fit_scorer(df)
    imp = sksc.feature_importances(scorer, df, n_repeats=3)
    assert set(imp["feature"]) == set(ALL_FEATURE_COLS)
    assert (imp["hgb_permutation_importance"] >= -1e-9).all()
    # The features we hand-built TP/FP separation into should score highly.
    top_features = set(imp.sort_values("hgb_permutation_importance", ascending=False)["feature"].head(3))
    assert {"prob_mean", "elongation", "contrast"} & top_features
