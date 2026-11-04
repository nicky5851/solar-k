"""Streaming equal-weight probability-map averaging over ensemble members
(CODING_SPEC_05_ENSEMBLE.md, section 5).

A *member* is an experiment family trained on all 5 model folds, e.g.
``E004`` -> ``outputs/E004-f{0..4}/{oof,test_probs}/<stem>.png``. Each fold's
model was trained on every image whose ``folds_v1.csv`` fold differs from
its own, so:

  - OOF: an image in model fold ``k`` gets ``mean(member-fk for each
    member)`` -- the only maps produced by models that never trained on it.
  - Test: the mean over every member x every fold (e.g. 5 E004 + 5 E007).

Weights are equal and pre-registered (no tuning). Only one image's maps are
ever in memory at a time (CLAUDE.md's 8 GB streaming rule).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import io as skio

N_MODEL_FOLDS = 5


def load_prob_png(path: Path) -> np.ndarray:
    """Same decoding as ``scripts/run_e002_eval.load_prob_png`` (uint8 PNG / 255)."""
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return arr.astype(np.float32) / 255.0


def model_fold_by_stem(folds_csv: Path | None = None) -> dict[str, int]:
    folds_csv = folds_csv or skio.data_root() / "splits" / "folds_v1.csv"
    df = pd.read_csv(folds_csv)
    return dict(zip(df["file_stem"], df["fold"].astype(int)))


def member_dir(member: str, fold: int) -> Path:
    return skio.output_root() / f"{member}-f{fold}"


def oof_sources(members: list[str], stem: str, fold_by_stem: dict[str, int]) -> list[Path]:
    """The OOF maps that may be averaged for ``stem``: one per member, all from
    the model fold that held ``stem`` out. Never any other fold's map."""
    k = fold_by_stem[stem]
    return [member_dir(m, k) / "oof" / f"{stem}.png" for m in members]


def test_sources(members: list[str], stem: str, folds: list[int] | None = None) -> list[Path]:
    folds = list(range(N_MODEL_FOLDS)) if folds is None else folds
    return [member_dir(m, f) / "test_probs" / f"{stem}.png" for m in members for f in folds]


def mean_of(paths: list[Path], weights: list[float] | None = None) -> np.ndarray:
    """Running (weighted) mean, so peak memory is two maps regardless of
    member count. Summation order is the order of ``paths`` -- fixed, so
    bit-reproducible. ``weights`` (CODING_SPEC_06 E018), if given, must align
    1:1 with ``paths`` and need not sum to 1; equal weights reproduce the
    original unweighted mean exactly."""
    if not paths:
        raise ValueError("no probability maps to average")
    if weights is not None and len(weights) != len(paths):
        raise ValueError(f"{len(weights)} weights for {len(paths)} paths")
    acc = None
    wsum = 0.0
    for i, p in enumerate(paths):
        arr = load_prob_png(p)
        w = 1.0 if weights is None else float(weights[i])
        wsum += w
        if acc is None:
            acc = arr.astype(np.float32, copy=True) * np.float32(w)
        else:
            acc += arr * np.float32(w)
        del arr
    acc /= np.float32(wsum)
    return acc


def load_oof_mean(members: list[str], stem: str, fold_by_stem: dict[str, int], weights: list[float] | None = None) -> np.ndarray:
    return mean_of(oof_sources(members, stem, fold_by_stem), weights)


def load_oof_members(members: list[str], stem: str, fold_by_stem: dict[str, int]) -> dict[str, np.ndarray]:
    """CODING_SPEC_06 E019: each member's own OOF map for ``stem`` (the
    fold-matched member set), for agreement features -- never any other
    fold's map, same restriction as ``oof_sources``."""
    return {m: load_prob_png(p) for m, p in zip(members, oof_sources(members, stem, fold_by_stem))}


def load_test_mean(members: list[str], stem: str, folds: list[int] | None = None, weights: list[float] | None = None) -> np.ndarray:
    """``weights``, if given, is one weight per *member* (not per map); it is
    expanded across that member's folds, matching ``test_sources``' member-outer,
    fold-inner order, so a member's weight is split equally over its 5 folds."""
    n_folds = len(folds) if folds is not None else N_MODEL_FOLDS
    expanded = None if weights is None else [w for w in weights for _ in range(n_folds)]
    return mean_of(test_sources(members, stem, folds), expanded)


def available_oof_stems(members: list[str], fold_by_stem: dict[str, int], model_folds: list[int] | None = None) -> list[str]:
    """Sorted stems whose OOF map exists for *every* member (restricted to
    ``model_folds`` if given)."""
    out = []
    for stem, k in sorted(fold_by_stem.items()):
        if model_folds is not None and k not in model_folds:
            continue
        if all(p.exists() for p in oof_sources(members, stem, fold_by_stem)):
            out.append(stem)
    return out
