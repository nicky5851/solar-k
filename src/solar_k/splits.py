"""Leakage-safe cross-validation folds, grouped by observation day.

Frames from the same day must share a fold (same-day frames are visually
near-duplicate), so folds are built by greedily assigning whole days to the
fold with the fewest images so far, which also balances image count well in
practice. Days are processed largest-first so the greedy choice has the most
impact early, when it still has balancing to do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_folds(images: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """``images`` needs columns ``file_stem, site, date, n_annotators`` (one row
    per physical image, ``date`` truncated to the observation day). Returns a
    copy with an added ``fold`` column in ``[0, n_folds)``.
    """
    df = images.copy()
    day_groups = df.groupby("date").size().sort_values(ascending=False)

    rng = np.random.RandomState(seed)
    # Shuffle within equal-size groups only, so ties don't always break the
    # same way, while largest-first ordering still drives the balancing.
    order = day_groups.index.to_numpy()
    sizes = day_groups.to_numpy()
    perm = rng.permutation(len(order))
    order, sizes = order[perm], sizes[perm]
    sort_idx = np.argsort(-sizes, kind="stable")
    order = order[sort_idx]

    fold_counts = np.zeros(n_folds, dtype=int)
    day_to_fold: dict = {}
    for day in order:
        n = int(day_groups.loc[day])
        fold = int(np.argmin(fold_counts))
        day_to_fold[day] = fold
        fold_counts[fold] += n

    df["fold"] = df["date"].map(day_to_fold)
    return df


def validate_folds(df: pd.DataFrame, expected_stems: set[str] | None = None) -> list[str]:
    """Return a list of problem descriptions; empty means the split is clean."""
    issues: list[str] = []

    dup_stems = df["file_stem"][df["file_stem"].duplicated()].unique()
    if len(dup_stems) > 0:
        issues.append(f"{len(dup_stems)} file_stem(s) appear more than once: {sorted(dup_stems)[:5]}...")

    day_fold_counts = df.groupby("date")["fold"].nunique()
    bad_days = day_fold_counts[day_fold_counts > 1]
    if len(bad_days) > 0:
        issues.append(f"{len(bad_days)} day(s) span more than one fold: {list(bad_days.index)[:5]}...")

    if expected_stems is not None:
        missing = expected_stems - set(df["file_stem"])
        if missing:
            issues.append(f"{len(missing)} expected image(s) missing from the split: {sorted(missing)[:5]}...")
        extra = set(df["file_stem"]) - expected_stems
        if extra:
            issues.append(f"{len(extra)} image(s) in the split are not in the expected set: {sorted(extra)[:5]}...")

    if df["fold"].isna().any():
        issues.append(f"{df['fold'].isna().sum()} image(s) have no fold assigned")

    return issues


def fold_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold counts of images, instances (if present), and sites."""
    agg = {"file_stem": "count"}
    if "n_instances" in df.columns:
        agg["n_instances"] = "sum"
    out = df.groupby("fold").agg(agg).rename(columns={"file_stem": "n_images"})
    site_counts = df.groupby(["fold", "site"]).size().unstack(fill_value=0)
    return out.join(site_counts)
