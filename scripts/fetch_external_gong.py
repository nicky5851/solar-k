#!/usr/bin/env python3
"""CODING_SPEC_07 E028: download ~2000 unlabelled GONG H-alpha full-disk
JPEGs from the public NSO GONG archive for self-training.

Source: the competition's own train annotations embed each image's origin
URL (e.g. "https://gong2.nso.edu/HA/hag/201406/20140609/20140609195854Bh.jpg"),
confirmed live. Site codes used here are restricted to the six that appear
in our own train/test data (B, C, L, M, T, U) -- the "hag" daily index lists
other codes too (S, Z, ...) that aren't part of the competition's six
GONG network sites, so they're excluded to keep the external pool
distributionally consistent with what the model was trained on.

Leakage safety (the whole point of this script): every train *and* test
image's date is loaded, expanded +/-3 days, and any candidate external date
inside that window is skipped outright -- the exclusion is applied before
any site-level selection, so it can't be bypassed by picking a different
site on an excluded day.

Resumable: already-downloaded stems (present in the manifest) are skipped.
Only the manifest (data/external/gong_unlabelled_manifest.csv) is meant to
be committed; downloaded images live under data/external/images/ and are
gitignored, same as data/raw.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solar_k import io as skio

SITES = ["B", "C", "L", "M", "T", "U"]
STEM_RE = re.compile(r"^(\d{14})([A-Za-z])h$")
BASE_URL = "https://gong2.nso.edu/HA/hag"
MANIFEST_COLUMNS = ["stem", "site", "date", "url", "downloaded_at_utc"]


def load_excluded_dates() -> set[date]:
    """Every date in our existing train or test split, +/-3 days."""
    import pandas as pd

    folds = pd.read_csv(skio.data_root() / "splits" / "folds_v1.csv")
    train_dates = {date.fromisoformat(d) for d in folds["date"].astype(str)}

    test_dates = set()
    for p in skio.iter_test_images():
        m = STEM_RE.match(p.stem)
        if m:
            d = m.group(1)
            test_dates.add(date(int(d[0:4]), int(d[4:6]), int(d[6:8])))

    all_dates = train_dates | test_dates
    excluded = set()
    for d in all_dates:
        for delta in range(-3, 4):
            excluded.add(d + timedelta(days=delta))
    return excluded


def load_existing_manifest(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open() as f:
        return {row["stem"] for row in csv.DictReader(f)}


def list_day_files(session: requests.Session, d: date, timeout: float = 15.0) -> list[str]:
    url = f"{BASE_URL}/{d.strftime('%Y%m')}/{d.strftime('%Y%m%d')}/"
    try:
        r = session.get(url, timeout=timeout)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    return re.findall(r'href="([^"]+\.jpg)"', r.text)


def pick_one_per_site(filenames: list[str], sites: list[str]) -> dict[str, str]:
    """First available file per desired site code on this day."""
    chosen: dict[str, str] = {}
    for fn in sorted(filenames):
        stem = fn[:-4]  # strip ".jpg"
        m = STEM_RE.match(stem)
        if not m:
            continue
        site = m.group(2).upper()
        if site in sites and site not in chosen:
            chosen[site] = fn
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=2000)
    ap.add_argument("--start-date", default="2011-01-01")
    ap.add_argument("--end-date", default="2022-12-31")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-site-per-day", type=int, default=1)
    ap.add_argument("--request-delay", type=float, default=0.3, help="seconds between HTTP requests, politeness")
    ap.add_argument("--out-dir", type=Path, default=skio.data_root() / "external" / "images")
    ap.add_argument("--manifest", type=Path, default=skio.repo_root() / "data" / "external" / "gong_unlabelled_manifest.csv")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    excluded = load_excluded_dates()
    print(f"Excluded {len(excluded)} dates (train/test +/-3 days)")

    already = load_existing_manifest(args.manifest)
    print(f"{len(already)} stems already in manifest, resuming")

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    all_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    candidate_days = [d for d in all_days if d not in excluded]
    rng = random.Random(args.seed)
    rng.shuffle(candidate_days)
    print(f"{len(candidate_days)}/{len(all_days)} candidate days after exclusion")

    session = requests.Session()
    manifest_is_new = not args.manifest.exists()
    n_downloaded = len(already)
    with args.manifest.open("a", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_COLUMNS)
        if manifest_is_new:
            writer.writeheader()

        for d in candidate_days:
            if n_downloaded >= args.n_target:
                break
            assert d not in excluded, f"BUG: candidate day {d} is in the excluded set"
            filenames = list_day_files(session, d)
            time.sleep(args.request_delay)
            if not filenames:
                continue
            chosen = pick_one_per_site(filenames, SITES)
            for site, fn in chosen.items():
                if n_downloaded >= args.n_target:
                    break
                stem = fn[:-4]
                if stem in already:
                    continue
                url = f"{BASE_URL}/{d.strftime('%Y%m')}/{d.strftime('%Y%m%d')}/{fn}"
                out_path = args.out_dir / fn
                try:
                    r = session.get(url, timeout=20.0)
                    time.sleep(args.request_delay)
                    if r.status_code != 200 or len(r.content) < 1000:
                        continue
                    out_path.write_bytes(r.content)
                except requests.RequestException:
                    continue
                writer.writerow({
                    "stem": stem, "site": site, "date": d.isoformat(), "url": url,
                    "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                mf.flush()
                already.add(stem)
                n_downloaded += 1
                if n_downloaded % 50 == 0:
                    print(f"{n_downloaded}/{args.n_target} downloaded...", flush=True)

    print(f"Done. {n_downloaded} external images downloaded, manifest at {args.manifest}")


if __name__ == "__main__":
    main()
