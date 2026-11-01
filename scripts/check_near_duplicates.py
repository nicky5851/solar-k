#!/usr/bin/env python3
"""Stage 0 (Spec 02): fixed near-duplicate check.

The Spec 01 near-dup check (raw 64x64 correlation > 0.99) was not
selective: every full-disk H-alpha frame shares the same bright-disk /
limb-darkening shape, so arbitrary unrelated pairs already correlate
~0.97-0.98. This version:
  (a) only considers candidate pairs with a real prior for being the same
      event: same GONG site, < 60 minutes apart;
  (b) correlates limb-darkening-corrected, disk-masked, high-pass 256x256
      images instead of raw pixels, for both the candidates and a random
      background sample;
  (c) picks a threshold from the gap between the two distributions (Otsu
      on the pooled correlation values) instead of an arbitrary constant;
  (d) checks every flagged train-train pair lands in one fold, and counts
      (does not act on) flagged train-test pairs.

Usage: python scripts/check_near_duplicates.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import ids as skids
from solar_k import io as skio
from solar_k import preprocess as skpp

THUMB_SIZE = 256
MAX_DT_MINUTES = 60
HIGHPASS_SIGMA = 8.0
BACKGROUND_SAMPLE_PAIRS = 2000
SEED = 0


def load_thumbnail(path: Path, cache: dict[str, np.ndarray]) -> np.ndarray:
    stem = path.stem
    if stem in cache:
        return cache[stem]
    img = skio.load_grayscale(path)
    result = skpp.preprocess_image(img)
    processed = result["processed"]
    resized = cv2.resize(processed, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
    hp = skpp.highpass(resized, sigma=HIGHPASS_SIGMA)
    cache[stem] = hp
    return hp


def correlate(a: np.ndarray, b: np.ndarray) -> float:
    a0, b0 = a.ravel() - a.mean(), b.ravel() - b.mean()
    na, nb = np.linalg.norm(a0), np.linalg.norm(b0)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a0, b0) / (na * nb))


def gather_images() -> list[dict]:
    rows = []
    for split, it in (("train", skio.iter_train_images()), ("test", skio.iter_test_images())):
        for path in it:
            parsed = skids.parse_stem(path.stem)
            if parsed is None:
                continue
            rows.append({"stem": path.stem, "path": path, "split": split, "site": parsed.site, "ts": parsed.timestamp})
    return rows


def find_candidate_pairs(rows: list[dict]) -> list[tuple[dict, dict, float]]:
    by_site: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_site[r["site"]].append(r)
    pairs = []
    for site, items in by_site.items():
        items = sorted(items, key=lambda r: r["ts"])
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                dt = (items[j]["ts"] - items[i]["ts"]).total_seconds() / 60
                if dt > MAX_DT_MINUTES:
                    break
                pairs.append((items[i], items[j], dt))
    return pairs


def sample_background_pairs(rows: list[dict], n: int, exclude: set[tuple[str, str]], seed: int) -> list[tuple[dict, dict]]:
    rng = np.random.RandomState(seed)
    pairs = []
    tries = 0
    max_tries = n * 20
    while len(pairs) < n and tries < max_tries:
        tries += 1
        a, b = rng.choice(len(rows), size=2, replace=False)
        ra, rb = rows[a], rows[b]
        key = tuple(sorted((ra["stem"], rb["stem"])))
        if key in exclude:
            continue
        exclude.add(key)
        pairs.append((ra, rb))
    return pairs


def pick_threshold(background: np.ndarray, candidate: np.ndarray) -> float:
    """Otsu threshold on the pooled correlation values -- the gap between
    the two distributions, found automatically rather than guessed."""
    pooled = np.concatenate([background, candidate]) if len(candidate) else background
    if len(pooled) < 2 or pooled.max() == pooled.min():
        return float(np.percentile(background, 99)) if len(background) else 0.99
    scaled = ((pooled - pooled.min()) / (pooled.max() - pooled.min()) * 255).astype(np.uint8)
    th, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(pooled.min() + (th / 255.0) * (pooled.max() - pooled.min()))


def main() -> None:
    rows = gather_images()
    by_stem = {r["stem"]: r for r in rows}

    candidate_pairs = find_candidate_pairs(rows)
    print(f"Candidate pairs (same site, <{MAX_DT_MINUTES} min apart): {len(candidate_pairs)}")

    exclude = {tuple(sorted((a["stem"], b["stem"]))) for a, b, _ in candidate_pairs}
    background_pairs = sample_background_pairs(rows, BACKGROUND_SAMPLE_PAIRS, exclude, SEED)
    print(f"Background (random, excluded from candidates) pairs: {len(background_pairs)}")

    cache: dict[str, np.ndarray] = {}

    candidate_corrs = []
    candidate_rows = []
    for a, b, dt in candidate_pairs:
        ta, tb = load_thumbnail(a["path"], cache), load_thumbnail(b["path"], cache)
        corr = correlate(ta, tb)
        candidate_corrs.append(corr)
        candidate_rows.append(
            {
                "stem_a": a["stem"],
                "stem_b": b["stem"],
                "site": a["site"],
                "dt_minutes": dt,
                "split_a": a["split"],
                "split_b": b["split"],
                "corr": corr,
            }
        )

    background_corrs = []
    for a, b in background_pairs:
        ta, tb = load_thumbnail(a["path"], cache), load_thumbnail(b["path"], cache)
        background_corrs.append(correlate(ta, tb))

    background_corrs = np.array(background_corrs)
    candidate_corrs = np.array(candidate_corrs)

    threshold = pick_threshold(background_corrs, candidate_corrs)
    print(f"Background corr: mean={background_corrs.mean():.4f} median={np.median(background_corrs):.4f} "
          f"p95={np.percentile(background_corrs, 95):.4f} p99={np.percentile(background_corrs, 99):.4f} "
          f"max={background_corrs.max():.4f}")
    if len(candidate_corrs):
        print(f"Candidate corr: mean={candidate_corrs.mean():.4f} min={candidate_corrs.min():.4f} max={candidate_corrs.max():.4f}")
    print(f"Chosen threshold (Otsu on pooled values): {threshold:.4f}")

    for row, corr in zip(candidate_rows, candidate_corrs):
        row["flagged_near_duplicate"] = bool(corr > threshold)

    df = pd.DataFrame(candidate_rows, columns=["stem_a", "stem_b", "site", "dt_minutes", "split_a", "split_b", "corr", "flagged_near_duplicate"])
    out_csv = skio.data_root() / "interim" / "near_duplicates_v2.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} candidate-pair rows to {out_csv}")

    flagged = df[df["flagged_near_duplicate"]] if len(df) else df
    train_train_flagged = flagged[(flagged["split_a"] == "train") & (flagged["split_b"] == "train")]
    train_test_flagged = flagged[
        ((flagged["split_a"] == "train") & (flagged["split_b"] == "test"))
        | ((flagged["split_a"] == "test") & (flagged["split_b"] == "train"))
    ]

    fold_violations = []
    folds_path = skio.data_root() / "splits" / "folds_v1.csv"
    if folds_path.exists() and len(train_train_flagged):
        folds = pd.read_csv(folds_path).set_index("file_stem")["fold"]
        for _, row in train_train_flagged.iterrows():
            fa, fb = folds.get(row["stem_a"]), folds.get(row["stem_b"])
            if fa is not None and fb is not None and fa != fb:
                fold_violations.append((row["stem_a"], row["stem_b"], int(fa), int(fb)))

    print(f"Flagged near-duplicates: {len(flagged)} total, {len(train_train_flagged)} train-train, "
          f"{len(train_test_flagged)} train-test (information only, not acted on).")
    print(f"Fold violations among train-train flagged pairs: {len(fold_violations)} {fold_violations[:5]}")

    summary = {
        "n_candidate_pairs": len(candidate_pairs),
        "n_background_pairs": len(background_pairs),
        "background_corr_stats": {
            "mean": float(background_corrs.mean()) if len(background_corrs) else None,
            "median": float(np.median(background_corrs)) if len(background_corrs) else None,
            "p95": float(np.percentile(background_corrs, 95)) if len(background_corrs) else None,
            "p99": float(np.percentile(background_corrs, 99)) if len(background_corrs) else None,
            "max": float(background_corrs.max()) if len(background_corrs) else None,
        },
        "candidate_corr_stats": {
            "mean": float(candidate_corrs.mean()) if len(candidate_corrs) else None,
            "min": float(candidate_corrs.min()) if len(candidate_corrs) else None,
            "max": float(candidate_corrs.max()) if len(candidate_corrs) else None,
        },
        "threshold": threshold,
        "n_flagged": int(len(flagged)),
        "n_flagged_train_train": int(len(train_train_flagged)),
        "n_flagged_train_test": int(len(train_test_flagged)),
        "fold_violations": fold_violations,
    }
    summary_path = skio.data_root() / "interim" / "near_duplicates_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
