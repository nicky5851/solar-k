"""Path resolution and streaming I/O for images and COCO-style annotations.

Never load the whole dataset into memory: callers should iterate over
``iter_train_images`` / ``iter_test_images`` and load one image at a time.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image


def repo_root() -> Path:
    """Resolve the repo root from this file's location, not the cwd."""
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """``SOLAR_K_DATA`` env var if set, else ``<repo_root>/data``."""
    env = os.environ.get("SOLAR_K_DATA")
    return Path(env).resolve() if env else repo_root() / "data"


def train_images_dir() -> Path:
    return data_root() / "raw" / "train" / "train_images"


def test_images_dir() -> Path:
    return data_root() / "raw" / "test" / "test_images"


def train_annotations_path() -> Path:
    return data_root() / "raw" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def external_dir() -> Path:
    return data_root() / "external"


def output_root() -> Path:
    """``SOLAR_K_OUTPUT`` env var if set, else ``<repo_root>/outputs``.

    On Kaggle this should point at ``/kaggle/working/outputs`` so
    checkpoints, logs, and probability maps land directly in the persisted
    working directory instead of needing a copy step at the end of a run.
    """
    env = os.environ.get("SOLAR_K_OUTPUT")
    return Path(env).resolve() if env else repo_root() / "outputs"


def cache_root() -> Path:
    """``SOLAR_K_CACHE`` env var if set, else ``<data_root>/processed``.

    Separate from ``data_root()`` because the preprocessing cache needs a
    writable, fast local disk -- on Kaggle, the competition data mount is
    read-only, so this should point at ``/kaggle/temp`` instead.
    """
    env = os.environ.get("SOLAR_K_CACHE")
    return Path(env).resolve() if env else data_root() / "processed"


IMAGE_SUFFIXES = (".jpg", ".jpeg")


def iter_image_paths(directory: Path) -> Iterator[Path]:
    """Yield image paths in a directory, sorted by file stem, skipping dotfiles."""
    paths = [
        p
        for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    ]
    yield from sorted(paths, key=lambda p: p.stem)


def iter_train_images() -> Iterator[Path]:
    yield from iter_image_paths(train_images_dir())


def iter_test_images() -> Iterator[Path]:
    yield from iter_image_paths(test_images_dir())


def load_grayscale(path: Path) -> np.ndarray:
    """Load one image as a 2D uint8 array. Caller is responsible for discarding it."""
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def load_coco_annotations() -> dict:
    """Load the full COCO-style annotation JSON (metadata only, no pixels)."""
    with open(train_annotations_path(), "r") as f:
        return json.load(f)
