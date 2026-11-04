"""Learned component scorer (CODING_SPEC_03_FP_AND_FOLDS.md, E003 step 4).

Two classifiers for P(match) -- is this predicted component a true
positive, matched IoU > 0.5 for at least one annotator -- (logistic
regression and a small ``HistGradientBoostingClassifier``, depth <= 3), and
a regressor for E[IoU | match] (``HistGradientBoostingRegressor``, fit only
on matched components). The keep rule thresholds ``P(match) * E[IoU]``
against a cutoff ``c`` tuned on the ``tune`` split (``scripts/run_e003.py``
does the per-stem reconstruction and PQ scoring that tuning needs; this
module only fits/predicts/reports importances).

Always fit on `tune`, scored on `report` -- same rule the rest of
CODING_SPEC_03 follows, so nothing here is fit on data it is evaluated
against (CLAUDE.md's no-leakage rule).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .components import ALL_FEATURE_COLS


@dataclass
class FittedScorer:
    logistic: LogisticRegression
    scaler: StandardScaler
    hgb_clf: HistGradientBoostingClassifier
    hgb_reg: HistGradientBoostingRegressor | None
    feature_cols: list[str]


def _feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return df[feature_cols].fillna(0.0).to_numpy(dtype=np.float64)


def fit_scorer(train_df: pd.DataFrame, feature_cols: list[str] = ALL_FEATURE_COLS, seed: int = 0, version: str = "v1") -> FittedScorer:
    """``version="v2"`` (CODING_SPEC_06 E019) swaps the match classifier for
    ``HistGradientBoostingClassifier(random_state=seed, max_iter=300,
    early_stopping=True)`` -- no depth cap, but early-stopped, so it can use
    the wider (agreement-feature) ``feature_cols`` without overfitting the
    small per-fold train portion."""
    X = _feature_matrix(train_df, feature_cols)
    y = train_df["is_tp"].to_numpy(dtype=int)

    scaler = StandardScaler().fit(X)
    logistic = LogisticRegression(max_iter=1000, random_state=seed).fit(scaler.transform(X), y)
    if version == "v2":
        hgb_clf = HistGradientBoostingClassifier(random_state=seed, max_iter=300, early_stopping=True).fit(X, y)
    else:
        hgb_clf = HistGradientBoostingClassifier(max_depth=3, random_state=seed).fit(X, y)

    matched = train_df[train_df["is_tp"]]
    hgb_reg = None
    if len(matched) >= 5:
        Xm = _feature_matrix(matched, feature_cols)
        ym = matched["best_iou"].to_numpy(dtype=float)
        hgb_reg = HistGradientBoostingRegressor(max_depth=3, random_state=seed).fit(Xm, ym)

    return FittedScorer(logistic=logistic, scaler=scaler, hgb_clf=hgb_clf, hgb_reg=hgb_reg, feature_cols=feature_cols)


def predict_match_prob(scorer: FittedScorer, df: pd.DataFrame, model: str = "hgb") -> np.ndarray:
    X = _feature_matrix(df, scorer.feature_cols)
    if model == "logistic":
        return scorer.logistic.predict_proba(scorer.scaler.transform(X))[:, 1]
    return scorer.hgb_clf.predict_proba(X)[:, 1]


def predict_expected_iou(scorer: FittedScorer, df: pd.DataFrame) -> np.ndarray:
    if scorer.hgb_reg is None or len(df) == 0:
        return np.zeros(len(df))
    X = _feature_matrix(df, scorer.feature_cols)
    return np.clip(scorer.hgb_reg.predict(X), 0.0, 1.0)


def keep_score(scorer: FittedScorer, df: pd.DataFrame, model: str = "hgb") -> np.ndarray:
    """``P(match) * E[IoU | match]`` -- the quantity the keep rule thresholds
    against ``c``. Theory (``c`` tuned, but centred near PQ/2) says a
    component worth keeping should contribute about as much to the PQ
    numerator in expectation as a borderline-accepted match costs the
    denominator."""
    if len(df) == 0:
        return np.zeros(0)
    return predict_match_prob(scorer, df, model=model) * predict_expected_iou(scorer, df)


def feature_importances(scorer: FittedScorer, df: pd.DataFrame, n_repeats: int = 10, seed: int = 0) -> pd.DataFrame:
    """Logistic: |standardised coefficient|. HGB: permutation importance on
    ROC AUC (``HistGradientBoostingClassifier`` has no built-in
    ``feature_importances_``)."""
    X = _feature_matrix(df, scorer.feature_cols)
    y = df["is_tp"].to_numpy(dtype=int)
    logistic_importance = np.abs(scorer.logistic.coef_[0])

    perm = permutation_importance(scorer.hgb_clf, X, y, n_repeats=n_repeats, random_state=seed, scoring="roc_auc")

    out = pd.DataFrame(
        {
            "feature": scorer.feature_cols,
            "logistic_abs_coef": logistic_importance,
            "hgb_permutation_importance": perm.importances_mean,
        }
    ).sort_values("hgb_permutation_importance", ascending=False).reset_index(drop=True)
    return out
