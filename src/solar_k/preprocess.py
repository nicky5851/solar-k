"""Disk fit, limb-darkening correction, and robust normalisation.

Shared by Stage 0's fixed near-duplicate check and Stage A's cached
preprocessed images (CODING_SPEC_02_BASELINE.md). Deterministic given the
raw image; the disk-fit method reuses the approach validated in Spec 01's
EDA (Otsu threshold + largest connected component).
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import io as skio

PREPROCESS_VERSION = "v1"


def fit_disk(img: np.ndarray) -> tuple[float, float, float]:
    """Otsu threshold + largest connected component -> (cx, cy, radius)."""
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
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


def processed_cache_dir(scale: float = 1.0) -> Path:
    if scale == 1.0:
        return skio.cache_root() / PREPROCESS_VERSION
    return skio.cache_root() / f"{PREPROCESS_VERSION}_scale{scale:g}"


def processed_path(stem: str, scale: float = 1.0) -> Path:
    return processed_cache_dir(scale) / f"{stem}.npy"


def save_processed(stem: str, arr: np.ndarray, scale: float = 1.0) -> Path:
    """Writes via a same-directory temp file + ``os.replace`` (atomic within
    a filesystem) so concurrent DataLoader workers racing to build the same
    missing cache entry -- e.g. external pseudo-labelled stems seen for the
    first time, CODING_SPEC_07 E028 -- never observe a partially-written
    file."""
    processed_cache_dir(scale).mkdir(parents=True, exist_ok=True)
    path = processed_path(stem, scale)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npy")
    np.save(tmp_path, arr.astype(np.float16))
    os.replace(tmp_path, path)
    return path


def load_processed(stem: str, scale: float = 1.0) -> np.ndarray:
    return np.load(processed_path(stem, scale)).astype(np.float32)


def disk_info_path(stem: str, scale: float = 1.0) -> Path:
    return processed_cache_dir(scale) / f"{stem}.disk.json"


def save_disk_info(stem: str, cx: float, cy: float, radius: float, scale: float = 1.0) -> Path:
    """Cache the disk-fit params alongside the processed array, so a cache
    hit can reconstruct ``disk_mask`` from geometry instead of thresholding
    the already-normalised array (where in-disk pixels can legitimately be
    exactly 0.0 after the percentile clip, making ``processed > 0`` unsound
    as a disk-mask stand-in)."""
    processed_cache_dir(scale).mkdir(parents=True, exist_ok=True)
    path = disk_info_path(stem, scale)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps({"cx": cx, "cy": cy, "radius": radius}))
    os.replace(tmp_path, path)
    return path


def load_disk_info(stem: str, scale: float = 1.0) -> tuple[float, float, float] | None:
    path = disk_info_path(stem, scale)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d["cx"], d["cy"], d["radius"]


def load_or_build(stem: str, raw_path: Path, scale: float = 1.0) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Cache-aware preprocessing: ``(processed, (cx, cy, radius))``, computing
    and caching only on a miss. The one place this logic lives, so the
    prebuild step (``build_cache``) and training/inference agree exactly on
    what counts as "cached".

    ``scale`` (CODING_SPEC_07_RESOLUTION_AND_SELF_TRAINING.md E027): >1.0
    upsamples the *already-correct* scale-1.0 processed array (bilinear),
    rather than re-running disk-fit/limb-correction on a resized raw image
    -- mathematically equivalent for this pipeline (limb correction and
    percentile normalisation commute with a uniform resize) and much
    cheaper. Cached separately per scale so scale=1.0 call sites (every
    pre-E027 caller) are untouched -- same cache dir, same files, same
    return value -- and bit-for-bit reproduce their pre-E027 behaviour."""
    if scale == 1.0:
        disk_info = load_disk_info(stem)
        if disk_info is not None and processed_path(stem).exists():
            return load_processed(stem), disk_info
        raw = skio.load_grayscale(raw_path)
        result = preprocess_image(raw)
        save_processed(stem, result["processed"])
        save_disk_info(stem, result["cx"], result["cy"], result["radius"])
        # Return exactly what a later cache hit returns (float16-rounded, then
        # float32), so features never depend on whether the cache was warm
        # (CODING_SPEC_05_ENSEMBLE.md section 2.1, determinism).
        processed = result["processed"].astype(np.float16).astype(np.float32)
        return processed, (result["cx"], result["cy"], result["radius"])

    disk_info = load_disk_info(stem, scale)
    if disk_info is not None and processed_path(stem, scale).exists():
        return load_processed(stem, scale), disk_info
    base_processed, (cx, cy, radius) = load_or_build(stem, raw_path, scale=1.0)
    h, w = base_processed.shape
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    scaled = cv2.resize(base_processed, (nw, nh), interpolation=cv2.INTER_LINEAR)
    save_processed(stem, scaled, scale)
    scaled_info = (cx * scale, cy * scale, radius * scale)
    save_disk_info(stem, *scaled_info, scale=scale)
    scaled_f16 = scaled.astype(np.float16).astype(np.float32)
    return scaled_f16, scaled_info


def _build_cache_one(args: tuple[str, Path, float]) -> str:
    stem, path, scale = args
    load_or_build(stem, path, scale=scale)
    return stem


def build_cache(stems_with_paths: list[tuple[str, Path]], workers: int = 1, scale: float = 1.0) -> int:
    """Warm the processed-image cache for every (stem, raw_path) pair that
    isn't already cached. Multi-process by default on Kaggle's multi-core
    CPU, since this runs once up front before training (CODING_SPEC_02_BASELINE.md's
    Kaggle path: "build the preprocessing cache at the start").

    ``scale`` (CODING_SPEC_07_RESOLUTION_AND_SELF_TRAINING.md E027): warms
    the scale-specific cache instead of (or as well as, via a second call)
    the default one. Doing this single-process-pool-serialised up front,
    before the training DataLoader's workers start, avoids several worker
    processes racing to build-and-cache the same first-seen stem at once."""
    todo = [(s, p, scale) for s, p in stems_with_paths if not (load_disk_info(s, scale) is not None and processed_path(s, scale).exists())]
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
