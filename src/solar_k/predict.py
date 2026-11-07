"""Run any probability-producing model over a folder of images -> instances -> CSV.

``prob_fn`` is the one thing that varies between E001 (classical) and E002
(U-Net): a callable ``(processed_image, disk_mask) -> probs in [0,1]``, same
H x W as the input. Everything else -- preprocessing, postprocessing,
RLE/CSV writing -- is shared.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import io as skio
from . import masks as skmasks
from . import postprocess as skpp
from . import preprocess as skpre

ProbFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def predict_image(prob_fn: ProbFn, processed: np.ndarray, disk_mask: np.ndarray, cfg: skpp.PostprocessConfig) -> list[np.ndarray]:
    probs = prob_fn(processed, disk_mask)
    return skpp.postprocess(probs, cfg)


def run_over_paths(
    prob_fn: ProbFn,
    image_paths: Iterable[Path],
    cfg: skpp.PostprocessConfig,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Stream one image at a time: load -> preprocess (cache permitting) ->
    predict -> encode RLE rows -> discard, never holding the whole folder's
    masks or images in memory."""
    rows = []
    timings: dict[str, float] = {}
    for path in image_paths:
        stem = path.stem
        t0 = time.time()
        processed = None
        disk_info = skpre.load_disk_info(stem) if use_cache else None
        if use_cache and disk_info is not None and skpre.processed_path(stem).exists():
            processed = skpre.load_processed(stem)
            disk_mask, _ = skpre.disk_mask_and_r(processed.shape, *disk_info)
        if processed is None:
            raw = skio.load_grayscale(path)
            result = skpre.preprocess_image(raw)
            processed, disk_mask = result["processed"], result["disk_mask"]
            if use_cache:
                skpre.save_processed(stem, processed)
                skpre.save_disk_info(stem, result["cx"], result["cy"], result["radius"])
            del raw

        instances = predict_image(prob_fn, processed, disk_mask, cfg)
        timings[stem] = time.time() - t0

        for k, m in enumerate(instances):
            if m.sum() == 0:
                continue
            counts = skmasks.encode_rle(m)
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": counts})
        del processed, instances

    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    return df, timings


def write_submission(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
