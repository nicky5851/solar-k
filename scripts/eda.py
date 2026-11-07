#!/usr/bin/env python3
"""Spec 01 EDA: dataset audit, metric validation, inter-annotator agreement.

Streams one physical image (and, for a given image, one annotator's small
instance-mask stack) at a time -- never the whole dataset or all masks at
once -- to stay inside an 8 GB Mac's memory. Writes all CSV/figure outputs
and reports/01_dataset_report.md. No model training.

Usage: python scripts/eda.py
"""
from __future__ import annotations

import collections
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import ids as skids
from solar_k import io as skio
from solar_k import masks as skmasks
from solar_k import metric as skmetric

REPO_ROOT = skio.repo_root()
FIG_DIR = REPO_ROOT / "reports" / "figures"
INTERIM_DIR = skio.data_root() / "interim"
H = W = 2048
IOU_THRESHOLD = skmetric.IOU_THRESHOLD


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Task 1: inventory
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task1_inventory() -> dict:
    log("Task 1: inventory, hashes, duplicates")
    rows = []
    thumbs = {}  # stem -> 64x64 float32 flattened thumbnail
    for split, directory in (("train", skio.train_images_dir()), ("test", skio.test_images_dir())):
        for path in skio.iter_image_paths(directory):
            from PIL import Image

            with Image.open(path) as im:
                raw_mode = im.mode
                arr = np.array(im.convert("L"))
                if raw_mode == "RGB":
                    rgb = np.array(im)
                    truly_gray = bool(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]))
                else:
                    truly_gray = raw_mode in ("L", "1", "I", "I;16")
            digest = sha256_of(path)
            thumb = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            thumbs[path.stem] = thumb.ravel()
            rows.append(
                {
                    "split": split,
                    "file_stem": path.stem,
                    "file_name": path.name,
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "format": "JPEG",
                    "mode": raw_mode,
                    "dtype": str(arr.dtype),
                    "height": arr.shape[0],
                    "width": arr.shape[1],
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                    "truly_grayscale": truly_gray,
                }
            )
            del arr

    df = pd.DataFrame(rows)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[]).to_csv(INTERIM_DIR / "hashes.csv", index=False)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    def dup_groups(sub: pd.DataFrame) -> list[tuple[str, list[str]]]:
        groups = sub.groupby("sha256")["file_stem"].apply(list)
        return [(h, stems) for h, stems in groups.items() if len(stems) > 1]

    dups_train = dup_groups(train_df)
    dups_test = dup_groups(test_df)
    cross_hashes = set(train_df["sha256"]) & set(test_df["sha256"])
    dups_cross = []
    if cross_hashes:
        for h in cross_hashes:
            t_stems = train_df.loc[train_df["sha256"] == h, "file_stem"].tolist()
            v_stems = test_df.loc[test_df["sha256"] == h, "file_stem"].tolist()
            dups_cross.append((h, t_stems, v_stems))

    # Near-duplicate check via 64x64-thumbnail correlation, train-train and train-test.
    train_stems = train_df["file_stem"].tolist()
    test_stems = test_df["file_stem"].tolist()
    train_mat = np.stack([thumbs[s] for s in train_stems])
    test_mat = np.stack([thumbs[s] for s in test_stems])

    def corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a0 = a - a.mean(axis=1, keepdims=True)
        b0 = b - b.mean(axis=1, keepdims=True)
        a0 /= np.linalg.norm(a0, axis=1, keepdims=True) + 1e-8
        b0 /= np.linalg.norm(b0, axis=1, keepdims=True) + 1e-8
        return a0 @ b0.T

    tt_corr = corr_matrix(train_mat, train_mat)
    # Background rate: correlation between ARBITRARY (almost certainly
    # unrelated) train pairs, before thresholding. Every full-disk H-alpha
    # frame shares the same bright-disk/limb-darkening shape, so this number
    # tells us whether 0.99 is actually selective for this dataset.
    off_diag = tt_corr[~np.eye(tt_corr.shape[0], dtype=bool)]
    background_corr_stats = {
        "mean": float(off_diag.mean()),
        "median": float(np.median(off_diag)),
        "p95": float(np.percentile(off_diag, 95)),
        "p99": float(np.percentile(off_diag, 99)),
    }
    np.fill_diagonal(tt_corr, 0.0)
    tt_pairs = np.argwhere(tt_corr > 0.99)
    near_dup_train_train = sorted(
        {
            (train_stems[i], train_stems[j], round(float(tt_corr[i, j]), 4))
            for i, j in tt_pairs
            if i < j
        }
    )

    tv_corr = corr_matrix(train_mat, test_mat)
    tv_pairs = np.argwhere(tv_corr > 0.99)
    near_dup_train_test = sorted(
        {(train_stems[i], test_stems[j], round(float(tv_corr[i, j]), 4)) for i, j in tv_pairs}
    )

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "total_size_bytes": int(df["size_bytes"].sum()),
        "formats": df["format"].value_counts().to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "dtypes": df["dtype"].value_counts().to_dict(),
        "shapes": df[["height", "width"]].drop_duplicates().values.tolist(),
        "min_overall": int(df["min"].min()),
        "max_overall": int(df["max"].max()),
        "all_truly_grayscale": bool(df["truly_grayscale"].all()),
        "dups_train": dups_train,
        "dups_test": dups_test,
        "dups_cross": dups_cross,
        "near_dup_train_train": near_dup_train_train,
        "near_dup_train_test": near_dup_train_test,
        "background_corr_stats": background_corr_stats,
    }


# ---------------------------------------------------------------------------
# Task 2: annotation schema
# ---------------------------------------------------------------------------


def _key_type_counts(records: list[dict]) -> dict:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        for k, v in r.items():
            counts[k][type(v).__name__] += 1
    n = len(records)
    return {k: {"types": dict(c), "present_frac": sum(c.values()) / n} for k, c in counts.items()}


def task2_annotation_schema(ann: dict) -> dict:
    log("Task 2: annotation schema")
    images, annotations = ann["images"], ann["annotations"]

    sample_img = dict(images[0])
    sample_ann = dict(annotations[0])
    if isinstance(sample_ann.get("segmentation"), list):
        sample_ann["segmentation"] = [p[:8] + ["..."] for p in sample_ann["segmentation"]]
    if isinstance(sample_ann.get("spine"), list) and len(sample_ann["spine"]) > 8:
        sample_ann["spine"] = sample_ann["spine"][:8] + ["..."]

    n_parts = [len(a["segmentation"]) for a in annotations]
    vertex_counts = [len(p) // 2 for a in annotations for p in a["segmentation"]]

    by_file: dict[str, list[str]] = collections.defaultdict(list)
    for img in images:
        by_file[img["file_name"]].append(img["id"])
    entries_per_file = collections.Counter(len(v) for v in by_file.values())

    files_on_disk = {p.name for p in skio.iter_train_images()}
    files_in_json = set(by_file.keys())
    missing_files = files_in_json - files_on_disk
    files_no_entry = files_on_disk - files_in_json

    cat_names = {c["id"]: c["name"] for c in ann["categories"]}
    cat_counts = collections.Counter(a["category_id"] for a in annotations)
    cat_by_annotator = collections.defaultdict(collections.Counter)
    for a in annotations:
        annotator, _ = skids.parse_image_id(a["image_id"])
        cat_by_annotator[annotator][a["category_id"]] += 1

    return {
        "info": ann["info"],
        "licenses": ann["licenses"],
        "categories": ann["categories"],
        "sample_image": sample_img,
        "sample_annotation": sample_ann,
        "image_keys": _key_type_counts(images),
        "annotation_keys": _key_type_counts(annotations),
        "n_parts_hist": collections.Counter(n_parts),
        "vertex_count_stats": {
            "min": int(np.min(vertex_counts)),
            "max": int(np.max(vertex_counts)),
            "mean": float(np.mean(vertex_counts)),
            "median": float(np.median(vertex_counts)),
        },
        "n_images_entries": len(images),
        "n_unique_files": len(by_file),
        "entries_per_file_hist": dict(entries_per_file),
        "missing_files": sorted(missing_files),
        "files_no_entry": sorted(files_no_entry),
        "category_names": cat_names,
        "category_counts": {cat_names[k]: v for k, v in cat_counts.items()},
        "category_counts_by_annotator": {
            a: {cat_names[k]: v for k, v in c.items()} for a, c in cat_by_annotator.items()
        },
    }


# ---------------------------------------------------------------------------
# Task 3: identifiers, time, site
# ---------------------------------------------------------------------------


def task3_identifiers() -> dict:
    log("Task 3: identifiers, time, site")

    def parse_all(stems: list[str]) -> tuple[list, list]:
        ok, bad = [], []
        for s in stems:
            p = skids.parse_stem(s)
            (ok if p else bad).append(p or s)
        return ok, bad

    train_stems = [p.stem for p in skio.iter_train_images()]
    test_stems = [p.stem for p in skio.iter_test_images()]
    train_ok, train_bad = parse_all(train_stems)
    test_ok, test_bad = parse_all(test_stems)

    def dist(parsed: list) -> dict:
        years = collections.Counter(p.timestamp.year for p in parsed)
        months = collections.Counter(p.timestamp.month for p in parsed)
        days = collections.Counter(p.timestamp.day for p in parsed)
        sites = collections.Counter(p.site for p in parsed)
        return {"year": dict(years), "month": dict(months), "day": dict(days), "site": dict(sites)}

    train_days = {p.timestamp.date() for p in train_ok}
    same_day = sum(1 for p in test_ok if p.timestamp.date() in train_days)

    train_times = [p.timestamp for p in train_ok]
    within_1h = 0
    for p in test_ok:
        if any(abs((p.timestamp - t).total_seconds()) <= 3600 for t in train_times):
            within_1h += 1

    return {
        "n_train_parsed": len(train_ok),
        "n_train_unparsed": len(train_bad),
        "train_unparsed": train_bad,
        "n_test_parsed": len(test_ok),
        "n_test_unparsed": len(test_bad),
        "test_unparsed": test_bad,
        "train_dist": dist(train_ok),
        "test_dist": dist(test_ok),
        "test_frames_same_day_as_train": same_day,
        "test_frames_within_1h_of_train": within_1h,
        "n_test": len(test_ok) + len(test_bad),
    }


# ---------------------------------------------------------------------------
# Disk fitting, shared by tasks 4 and 5
# ---------------------------------------------------------------------------


def fit_disk(img: np.ndarray) -> tuple[float, float, float]:
    """Otsu threshold + largest connected component -> (cx, cy, radius)."""
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return W / 2, H / 2, min(H, W) / 2
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    cx, cy = centroids[biggest]
    radius = float(np.sqrt(stats[biggest, cv2.CC_STAT_AREA] / np.pi))
    return float(cx), float(cy), radius


# ---------------------------------------------------------------------------
# Tasks 4 & 5: per-image loop (instance stats + image stats), one image at a time
# ---------------------------------------------------------------------------


def task4_5_instance_and_image_stats(ann: dict) -> dict:
    log("Tasks 4 & 5: instance and image statistics (streaming, one image at a time)")

    anns_by_image_id: dict[str, list[dict]] = collections.defaultdict(list)
    for a in ann["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)

    images_by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for img in ann["images"]:
        images_by_file[img["file_name"]].append(img)

    instance_rows = []
    image_rows = []
    n_bins = 60
    radial_sum: dict[str, np.ndarray] = collections.defaultdict(lambda: np.zeros(n_bins))
    radial_count: dict[str, np.ndarray] = collections.defaultdict(lambda: np.zeros(n_bins))
    radial_edges = np.linspace(0, 1.5, n_bins + 1)

    gallery_candidates: list[tuple[str, dict]] = []

    paths = list(skio.iter_train_images())
    for idx, path in enumerate(paths):
        stem = path.stem
        img_entries = images_by_file.get(path.name, [])
        if not img_entries:
            continue
        parsed = skids.parse_stem(stem)
        site = parsed.site if parsed else "?"

        img = skio.load_grayscale(path)
        cx, cy, radius = fit_disk(img)

        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1e-6)
        disk_mask = r <= 1.0

        contrast = float(img[disk_mask].std() / (img[disk_mask].mean() + 1e-6)) if disk_mask.any() else float("nan")
        noise = float(cv2.Laplacian(img, cv2.CV_64F).std())

        r_flat = r.ravel()
        v_flat = img.ravel().astype(np.float64)
        bin_idx = np.digitize(r_flat, radial_edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        np.add.at(radial_sum[site], bin_idx[valid], v_flat[valid])
        np.add.at(radial_count[site], bin_idx[valid], 1)

        image_rows.append(
            {
                "file_stem": stem,
                "site": site,
                "disk_cx": cx,
                "disk_cy": cy,
                "disk_radius": radius,
                "contrast": contrast,
                "noise": noise,
            }
        )

        len(img_entries)
        if idx % max(1, len(paths) // 12) == 0 and len(gallery_candidates) < 12:
            gallery_candidates.append((stem, {"path": path}))

        for img_entry in img_entries:
            annotator, _ = skids.parse_image_id(img_entry["id"])
            anns = anns_by_image_id.get(img_entry["id"], [])
            # Every per-instance array op below works on a bbox crop, not the
            # full 2048x2048 frame -- skeletonize/distance-transform on the
            # full array per instance was the EDA's dominant cost (~25x
            # slower; see docs/specs report notes) for no accuracy benefit.
            inst_masks = []
            inst_boxes: list[tuple[int, int, int, int] | None] = []
            for a in anns:
                m = skmasks.polygon_to_mask(a["segmentation"], H, W)
                inst_masks.append(m)
                ys, xs = np.nonzero(m)
                inst_boxes.append((ys.min(), ys.max(), xs.min(), xs.max())) if len(ys) else inst_boxes.append(None)

            # same-annotator touch/overlap: dilate by 1px and check intersection,
            # only over the small region where two instances' boxes could touch.
            touch_flags = [False] * len(inst_masks)
            for i in range(len(inst_masks)):
                bi = inst_boxes[i]
                if bi is None:
                    continue
                for j in range(i + 1, len(inst_masks)):
                    bj = inst_boxes[j]
                    if bj is None:
                        continue
                    if (
                        bi[0] - 1 > bj[1] + 1
                        or bj[0] - 1 > bi[1] + 1
                        or bi[2] - 1 > bj[3] + 1
                        or bj[2] - 1 > bi[3] + 1
                    ):
                        continue  # boxes can't come within 1px of each other
                    uy0 = max(0, min(bi[0], bj[0]) - 1)
                    uy1 = min(H - 1, max(bi[1], bj[1]) + 1)
                    ux0 = max(0, min(bi[2], bj[2]) - 1)
                    ux1 = min(W - 1, max(bi[3], bj[3]) + 1)
                    ci = cv2.dilate(inst_masks[i][uy0 : uy1 + 1, ux0 : ux1 + 1], np.ones((3, 3), np.uint8))
                    cj = cv2.dilate(inst_masks[j][uy0 : uy1 + 1, ux0 : ux1 + 1], np.ones((3, 3), np.uint8))
                    if np.any(ci & cj):
                        touch_flags[i] = True
                        touch_flags[j] = True

            for a, m, box, touches in zip(anns, inst_masks, inst_boxes, touch_flags):
                area = int(m.sum())
                if area == 0 or box is None:
                    continue
                y0b, y1b, x0b, x1b = box
                bbox_h, bbox_w = (y1b - y0b + 1), (x1b - x0b + 1)

                pad = 2
                y0p, y1p = max(0, y0b - pad), min(H - 1, y1b + pad)
                x0p, x1p = max(0, x0b - pad), min(W - 1, x1b + pad)
                crop = m[y0p : y1p + 1, x0p : x1p + 1]

                skel = skeletonize(crop.astype(bool))
                skel_len = int(skel.sum())
                elongation = skel_len / np.sqrt(area) if area > 0 else float("nan")
                if skel_len > 0:
                    dt = distance_transform_edt(crop)
                    min_width = float(2 * dt[skel].min())
                else:
                    min_width = float("nan")

                cys, cxs = np.nonzero(crop)
                xs_mean, ys_mean = cxs.mean() + x0p, cys.mean() + y0p
                cen_r = float(np.sqrt((xs_mean - cx) ** 2 + (ys_mean - cy) ** 2) / max(radius, 1e-6))
                dist_from_center = np.sqrt((cxs + x0p - cx) ** 2 + (cys + y0p - cy) ** 2)
                frac_off_disk = float((dist_from_center > radius).mean())
                instance_rows.append(
                    {
                        "file_stem": stem,
                        "annotator": annotator,
                        "ann_id": a["id"],
                        "category": a["category_id"],
                        "area": area,
                        "bbox_h": bbox_h,
                        "bbox_w": bbox_w,
                        "elongation": elongation,
                        "min_width": min_width,
                        "centroid_r_over_R": cen_r,
                        "frac_pixels_off_disk": frac_off_disk,
                        "fully_off_disk": frac_off_disk >= 0.999,
                        "touches_same_annotator_instance": touches,
                    }
                )

        if (idx + 1) % 100 == 0:
            log(f"  ...processed {idx + 1}/{len(paths)} train images")

    instance_df = pd.DataFrame(instance_rows)
    image_df = pd.DataFrame(image_rows)
    instance_df.to_csv(INTERIM_DIR / "instance_stats.csv", index=False)
    image_df.to_csv(INTERIM_DIR / "image_stats.csv", index=False)

    radial_profiles = {
        site: (radial_sum[site] / np.maximum(radial_count[site], 1)).tolist() for site in radial_sum
    }

    worst5 = image_df.sort_values("noise", ascending=False).head(5)[["file_stem", "noise", "contrast"]]

    make_figures(image_df, instance_df, radial_profiles, radial_edges, gallery_candidates, ann)

    return {
        "instances_per_image_hist": instance_df.groupby("file_stem").size().describe().to_dict(),
        "n_instances": len(instance_df),
        "area_stats": instance_df["area"].describe().to_dict(),
        "elongation_stats": instance_df["elongation"].describe().to_dict(),
        "min_width_stats": instance_df["min_width"].describe().to_dict(),
        "centroid_r_stats": instance_df["centroid_r_over_R"].describe().to_dict(),
        "frac_touching_same_annotator": float(instance_df["touches_same_annotator_instance"].mean()),
        "frac_fully_off_disk": float(instance_df["fully_off_disk"].mean()),
        "frac_partly_off_disk": float((instance_df["frac_pixels_off_disk"] > 0).mean()),
        "disk_radius_stats": image_df["disk_radius"].describe().to_dict(),
        "contrast_stats": image_df["contrast"].describe().to_dict(),
        "noise_stats": image_df["noise"].describe().to_dict(),
        "worst5_noise": worst5.to_dict("records"),
        "radial_profiles_by_site": radial_profiles,
    }


def make_figures(image_df, instance_df, radial_profiles, radial_edges, gallery_candidates, ann):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    instance_df.groupby("file_stem").size().plot.hist(bins=30, ax=ax)
    ax.set_xlabel("instances per image")
    ax.set_title("Instances per image (train)")
    fig.savefig(FIG_DIR / "01_instances_per_image.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(np.log10(instance_df["area"].clip(lower=1)), bins=40)
    ax.set_xlabel("log10(area in px)")
    ax.set_title("Instance area distribution (train)")
    fig.savefig(FIG_DIR / "01_instance_area.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    centers = (radial_edges[:-1] + radial_edges[1:]) / 2
    for site, profile in sorted(radial_profiles.items()):
        ax.plot(centers, profile, label=site)
    ax.axvline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("r / disk radius")
    ax.set_ylabel("mean intensity")
    ax.set_title("Radial intensity profile by site (limb darkening)")
    ax.legend(fontsize=7)
    fig.savefig(FIG_DIR / "01_radial_profile.png", dpi=120)
    plt.close(fig)

    # Gallery with GT overlays, one panel per annotator when there are several.
    anns_by_image_id: dict[str, list[dict]] = collections.defaultdict(list)
    for a in ann["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)
    images_by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for img in ann["images"]:
        images_by_file[img["file_name"]].append(img)

    candidates = gallery_candidates[:12]
    n = len(candidates)
    if n > 0:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        cmap = plt.get_cmap("tab20")
        for ax, (stem, info) in zip(axes, candidates):
            path = info["path"]
            img_arr = skio.load_grayscale(path)
            ax.imshow(img_arr, cmap="gray", vmin=0, vmax=255)
            entries = images_by_file.get(path.name, [])
            entry = entries[0] if entries else None
            if entry is not None:
                for k, a in enumerate(anns_by_image_id.get(entry["id"], [])):
                    m = skmasks.polygon_to_mask(a["segmentation"], H, W)
                    ys, xs = np.nonzero(m)
                    if len(xs) == 0:
                        continue
                    ax.scatter(xs[::37], ys[::37], s=1, color=cmap(k % 20))
            n_ann = len(entries)
            ax.set_title(f"{stem} (annotators={n_ann})", fontsize=8)
            ax.axis("off")
            del img_arr
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Train gallery: GT overlays coloured per instance")
        fig.savefig(FIG_DIR / "01_gallery.png", dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Task 6: metric and submission plumbing
# ---------------------------------------------------------------------------


def task6_metric_submission(ann: dict) -> dict:
    log("Task 6: metric + submission round-trip")

    images_by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for img in ann["images"]:
        images_by_file[img["file_name"]].append(img)
    anns_by_image_id: dict[str, list[dict]] = collections.defaultdict(list)
    for a in ann["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)

    single_annotator_files = [f for f, entries in images_by_file.items() if len(entries) == 1]

    # One image's GT masks at a time: rasterise, encode, immediately decode
    # back and compare, then score and discard, instead of holding every
    # single-annotator image's masks (several GB) in memory at once.
    sub_rows = []
    pixel_exact = True
    match_results: list[skmetric.ImageMatch] = []
    for file_name in single_annotator_files:
        stem = Path(file_name).stem
        entry = images_by_file[file_name][0]
        anns = anns_by_image_id.get(entry["id"], [])

        gt_list, pred_list = [], []
        for k, a in enumerate(anns):
            m = skmasks.polygon_to_mask(a["segmentation"], H, W)
            if m.sum() == 0:
                continue
            counts = skmasks.encode_rle(m)
            sub_rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": counts})
            decoded_back = skmasks.decode_rle(counts, H, W)
            if not np.array_equal(m, decoded_back):
                pixel_exact = False
            gt_list.append(m)
            pred_list.append(decoded_back)

        gt_stack = np.stack(gt_list) if gt_list else np.zeros((0, H, W), dtype=np.uint8)
        pred_stack = np.stack(pred_list) if pred_list else np.zeros((0, H, W), dtype=np.uint8)
        match_results.append(skmetric.score_image(gt_stack, pred_stack))
        del gt_list, pred_list, gt_stack, pred_stack

    sub_df = pd.DataFrame(sub_rows, columns=["filament_id", "segmentation_rle"])
    sub_csv = INTERIM_DIR / "roundtrip_submission.csv"
    sub_df.to_csv(sub_csv, index=False)
    del sub_rows

    import validate_submission as vs

    errors = vs.validate(sub_csv, H, W, image_dir=skio.train_images_dir())

    roundtrip_pq = skmetric.aggregate_matches(match_results)

    return {
        "n_single_annotator_images": len(single_annotator_files),
        "n_submission_rows": len(sub_df),
        "validate_submission_errors": errors,
        "pixel_exact_round_trip": pixel_exact,
        "roundtrip_pq": roundtrip_pq.pq,
        "roundtrip_tp_fp_fn": (roundtrip_pq.tp, roundtrip_pq.fp, roundtrip_pq.fn),
    }


# ---------------------------------------------------------------------------
# Task 7: inter-annotator agreement
# ---------------------------------------------------------------------------


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return (mask.astype(bool)) & (~eroded.astype(bool))


def boundary_distance(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Average symmetric boundary distance, computed on a cropped union bbox
    (not the full 2048x2048 frame -- same cost mistake as the task 4/5 loop,
    fixed the same way: erode/distance-transform scale with crop size)."""
    ys, xs = np.nonzero(gt_mask | pred_mask)
    if len(ys) == 0:
        return float("nan")
    pad = 3
    y0, y1 = max(0, ys.min() - pad), min(gt_mask.shape[0] - 1, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(gt_mask.shape[1] - 1, xs.max() + pad)
    gt_c = gt_mask[y0 : y1 + 1, x0 : x1 + 1]
    pred_c = pred_mask[y0 : y1 + 1, x0 : x1 + 1]

    gt_b = boundary_pixels(gt_c)
    pred_b = boundary_pixels(pred_c)
    if not gt_b.any() or not pred_b.any():
        return float("nan")
    dt_pred = distance_transform_edt(~pred_b)
    dt_gt = distance_transform_edt(~gt_b)
    return float((dt_pred[gt_b].mean() + dt_gt[pred_b].mean()) / 2)


def task7_inter_annotator(ann: dict) -> dict:
    log("Task 7: inter-annotator agreement")

    images_by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for img in ann["images"]:
        images_by_file[img["file_name"]].append(img)
    anns_by_image_id: dict[str, list[dict]] = collections.defaultdict(list)
    for a in ann["annotations"]:
        anns_by_image_id[a["image_id"]].append(a)

    multi_files = [f for f, e in images_by_file.items() if len(e) >= 2]

    directed_matches = []
    boundary_dists = []
    unmatched_sizes = []
    gt_degree_counts = collections.Counter()
    pred_degree_counts = collections.Counter()

    for file_name in multi_files:
        entries = images_by_file[file_name]
        masks_by_annotator = {}
        for entry in entries:
            annotator, _ = skids.parse_image_id(entry["id"])
            anns = anns_by_image_id.get(entry["id"], [])
            masks_by_annotator[annotator] = [
                skmasks.polygon_to_mask(a["segmentation"], H, W) for a in anns
            ]

        annotators = list(masks_by_annotator.keys())
        for i in range(len(annotators)):
            for j in range(len(annotators)):
                if i == j:
                    continue
                gt_masks = masks_by_annotator[annotators[i]]
                pred_masks = masks_by_annotator[annotators[j]]
                gt_stack = np.stack(gt_masks) if gt_masks else np.zeros((0, H, W), np.uint8)
                pred_stack = np.stack(pred_masks) if pred_masks else np.zeros((0, H, W), np.uint8)

                if gt_stack.shape[0] == 0 or pred_stack.shape[0] == 0:
                    match = skmetric.match_scores(
                        np.zeros((0, 0)), np.zeros((0, 0)), gt_stack.shape[0], pred_stack.shape[0]
                    )
                    directed_matches.append(match)
                    continue

                iou, dice = skmetric.pairwise_iou_dice(gt_stack, pred_stack)
                match = skmetric.match_scores(iou, dice, gt_stack.shape[0], pred_stack.shape[0])
                directed_matches.append(match)

                hit = iou > IOU_THRESHOLD
                for gi in range(iou.shape[0]):
                    for pi in range(iou.shape[1]):
                        if hit[gi, pi]:
                            boundary_dists.append(
                                boundary_distance(gt_masks[gi], pred_masks[pi])
                            )

                hit_any = iou > 0
                gt_degrees = hit_any.sum(axis=1)
                pred_degrees = hit_any.sum(axis=0)
                for gi, deg in enumerate(gt_degrees):
                    gt_degree_counts[int(deg)] += 1
                    if deg == 0:
                        unmatched_sizes.append(
                            {"role": "gt_unmatched", "area": int(gt_masks[gi].sum())}
                        )
                for pi, deg in enumerate(pred_degrees):
                    pred_degree_counts[int(deg)] += 1
                    if deg == 0:
                        unmatched_sizes.append(
                            {"role": "pred_unmatched", "area": int(pred_masks[pi].sum())}
                        )

    result = skmetric.aggregate_matches(directed_matches)
    unmatched_df = pd.DataFrame(unmatched_sizes)
    if len(unmatched_df):
        unmatched_df.to_csv(INTERIM_DIR / "interannotator_unmatched.csv", index=False)

    boundary_arr = np.array([d for d in boundary_dists if not np.isnan(d)])

    one_to_many = sum(c for deg, c in gt_degree_counts.items() if deg > 1)
    many_to_one = sum(c for deg, c in pred_degree_counts.items() if deg > 1)

    return {
        "n_multi_annotator_images": len(multi_files),
        "n_directed_pairs": len(directed_matches),
        "pq": result.pq,
        "sq": result.sq,
        "rq": result.rq,
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
        "tp_iou_stats": pd.Series(result.tp_ious).describe().to_dict() if result.tp_ious else {},
        "tp_dice_stats": pd.Series(result.tp_dices).describe().to_dict() if result.tp_dices else {},
        "boundary_distance_stats": pd.Series(boundary_arr).describe().to_dict()
        if len(boundary_arr)
        else {},
        "unmatched_count": len(unmatched_df),
        "unmatched_area_stats": unmatched_df["area"].describe().to_dict() if len(unmatched_df) else {},
        "gt_degree_counts": dict(gt_degree_counts),
        "pred_degree_counts": dict(pred_degree_counts),
        "n_one_to_many_gt": one_to_many,
        "n_many_to_one_pred": many_to_one,
    }


# ---------------------------------------------------------------------------
# Task 8: validation split (delegates to make_splits.py, then reports on it)
# ---------------------------------------------------------------------------


def task8_validation_split() -> dict:
    log("Task 8: validation split")
    import make_splits

    from solar_k import splits as sksplits

    images = make_splits.build_image_table()
    folded = sksplits.assign_folds(images, n_folds=5, seed=0)
    expected_stems = {p.stem for p in skio.iter_train_images()}
    issues = sksplits.validate_folds(folded, expected_stems=expected_stems)

    out_path = skio.data_root() / "splits" / "folds_v1.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    folded[["file_stem", "fold", "site", "date", "n_annotators"]].sort_values("file_stem").to_csv(
        out_path, index=False
    )

    instance_csv = INTERIM_DIR / "instance_stats.csv"
    if instance_csv.exists():
        inst = pd.read_csv(instance_csv)
        n_instances = inst.groupby("file_stem").size().rename("n_instances")
        folded = folded.join(n_instances, on="file_stem")
        folded["n_instances"] = folded["n_instances"].fillna(0).astype(int)

    report = sksplits.fold_report(folded)

    return {
        "n_images": len(folded),
        "n_folds": 5,
        "issues": issues,
        "fold_report": report.to_dict("index"),
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(results: dict) -> None:
    from report_template import render

    out_path = REPO_ROOT / "reports" / "01_dataset_report.md"
    out_path.write_text(render(results))
    log(f"Wrote {out_path}")


def main() -> None:
    t0 = time.time()
    ann = skio.load_coco_annotations()

    results = {}
    results["task1"] = task1_inventory()
    results["task2"] = task2_annotation_schema(ann)
    results["task3"] = task3_identifiers()
    results["task45"] = task4_5_instance_and_image_stats(ann)
    results["task6"] = task6_metric_submission(ann)
    results["task7"] = task7_inter_annotator(ann)
    results["task8"] = task8_validation_split()

    write_report(results)

    elapsed = time.time() - t0
    log(f"Done in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
