"""Disk fit, limb-darkening correction, and robust normalisation.

Shared by Stage 0's fixed near-duplicate check and Stage A's cached
preprocessed images (CODING_SPEC_02_BASELINE.md). Deterministic given the
raw image; the disk-fit method reuses the approach validated in Spec 01's
EDA (Otsu threshold + largest connected component).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import io as skio

PREPROCESS_VERSION = "v1"


def fit_disk(img: np.ndarray) -> tuple[float, float, float]:
    """Otsu threshold + largest connected component -> (cx, cy, radius)."""
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    h, w = img.shape
    if n <= 1:
        return w / 2, h / 2, min(h, w) / 2
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    cx, cy = centroids[biggest]
    radius = float(np.sqrt(stats[biggest, cv2.CC_STAT_AREA] / np.pi))
    return float(cx), float(cy), radius


def disk_mask_and_r(shape: tuple[int, int], cx: float, cy: float, radius: float):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return r <= radius, r


def radial_median_profile(
    img: np.ndarray, disk_mask: np.ndarray, r: np.ndarray, radius: float, n_bins: int = 100, min_pixels: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    bin_idx = np.clip((r / max(radius, 1e-6) * n_bins).astype(int), 0, n_bins - 1)
    prof = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        sel = (bin_idx == b) & disk_mask
        if sel.sum() >= min_pixels:
            prof[b] = np.median(img[sel])
    valid = ~np.isnan(prof)
    if not valid.any():
        fallback = float(np.median(img[disk_mask])) if disk_mask.any() else 1.0
        prof[:] = fallback
    else:
        idxs = np.arange(n_bins)
        prof = np.interp(idxs, idxs[valid], prof[valid])
    return prof, bin_idx


def limb_correct(img: np.ndarray, disk_mask: np.ndarray, r: np.ndarray, radius: float, n_bins: int = 100) -> np.ndarray:
    """Divide by the per-image radial median profile (limb-darkening correction)."""
    prof, bin_idx = radial_median_profile(img, disk_mask, r, radius, n_bins)
    # A too-small floor lets near-zero edge bins blow the ratio up to absurd
    # values; 10% of the profile's own low end is a sane floor in practice.
    floor = max(1.0, float(np.percentile(prof, 5)) * 0.1)
    return img.astype(np.float32) / np.maximum(prof[bin_idx], floor)


def normalize_disk(corrected: np.ndarray, disk_mask: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Robust (percentile) normalisation to ~[0,1] inside the disk, 0 outside."""
    out = np.zeros_like(corrected, dtype=np.float32)
    if not disk_mask.any():
        return out
    vals = corrected[disk_mask]
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    normed = np.clip((corrected - lo) / (hi - lo), 0.0, 1.0)
    out[disk_mask] = normed[disk_mask]
    return out


def preprocess_image(img: np.ndarray) -> dict:
    """Full deterministic pipeline: disk fit -> limb correction -> normalise."""
    cx, cy, radius = fit_disk(img)
    disk_mask, r = disk_mask_and_r(img.shape, cx, cy, radius)
    corrected = limb_correct(img, disk_mask, r, radius)
    normed = normalize_disk(corrected, disk_mask)
    return {
        "processed": normed.astype(np.float32),
        "disk_mask": disk_mask,
        "cx": cx,
        "cy": cy,
        "radius": radius,
    }


def processed_cache_dir() -> Path:
    return skio.cache_root() / PREPROCESS_VERSION


def processed_path(stem: str) -> Path:
    return processed_cache_dir() / f"{stem}.npy"


def save_processed(stem: str, arr: np.ndarray) -> Path:
    processed_cache_dir().mkdir(parents=True, exist_ok=True)
    path = processed_path(stem)
    np.save(path, arr.astype(np.float16))
    return path


def load_processed(stem: str) -> np.ndarray:
    return np.load(processed_path(stem)).astype(np.float32)


def disk_info_path(stem: str) -> Path:
    return processed_cache_dir() / f"{stem}.disk.json"


def save_disk_info(stem: str, cx: float, cy: float, radius: float) -> Path:
    """Cache the disk-fit params alongside the processed array, so a cache
    hit can reconstruct ``disk_mask`` from geometry instead of thresholding
    the already-normalised array (where in-disk pixels can legitimately be
    exactly 0.0 after the percentile clip, making ``processed > 0`` unsound
    as a disk-mask stand-in)."""
    processed_cache_dir().mkdir(parents=True, exist_ok=True)
    path = disk_info_path(stem)
    path.write_text(json.dumps({"cx": cx, "cy": cy, "radius": radius}))
    return path


def load_disk_info(stem: str) -> tuple[float, float, float] | None:
    path = disk_info_path(stem)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d["cx"], d["cy"], d["radius"]


def load_or_build(stem: str, raw_path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Cache-aware preprocessing: ``(processed, (cx, cy, radius))``, computing
    and caching only on a miss. The one place this logic lives, so the
    prebuild step (``build_cache``) and training/inference agree exactly on
    what counts as "cached"."""
    disk_info = load_disk_info(stem)
    if disk_info is not None and processed_path(stem).exists():
        return load_processed(stem), disk_info
    raw = skio.load_grayscale(raw_path)
    result = preprocess_image(raw)
    save_processed(stem, result["processed"])
    save_disk_info(stem, result["cx"], result["cy"], result["radius"])
    return result["processed"], (result["cx"], result["cy"], result["radius"])


def _build_cache_one(args: tuple[str, Path]) -> str:
    stem, path = args
    load_or_build(stem, path)
    return stem


def build_cache(stems_with_paths: list[tuple[str, Path]], workers: int = 1) -> int:
    """Warm the processed-image cache for every (stem, raw_path) pair that
    isn't already cached. Multi-process by default on Kaggle's multi-core
    CPU, since this runs once up front before training (CODING_SPEC_02_BASELINE.md's
    Kaggle path: "build the preprocessing cache at the start")."""
    todo = [(s, p) for s, p in stems_with_paths if not (load_disk_info(s) is not None and processed_path(s).exists())]
    if not todo:
        return 0
    if workers <= 1:
        for args in todo:
            _build_cache_one(args)
    else:
        import multiprocessing as mp

        with mp.Pool(workers) as pool:
            for _ in pool.imap_unordered(_build_cache_one, todo):
                pass
    return len(todo)


def highpass(img: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Subtract a Gaussian-blurred version to isolate fine (non-smooth) structure."""
    img = img.astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return img - blur
