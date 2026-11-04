#!/usr/bin/env python3
"""Generate per-fold Kaggle notebooks + kernel-metadata files for E021
(CODING_SPEC_05_ENSEMBLE.md): second E007 seed (seed 1337).
Folds 3 and 4 train on Kaggle (2 parallel T4 sessions); folds 0-2 train on
the RunPod 4090 via scripts/run_queue_runpod.sh. Same structure as the
E006 generator, pointed at configs/E021-fN.yaml.

This only writes files -- it never calls `kaggle kernels push`.

Usage: python scripts/gen_e021_kaggle_artifacts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_k import io as skio

REPO_ROOT = skio.repo_root()
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
KAGGLE_USER = "potcharapanchumrak"
PACKAGE_DATASET = f"{KAGGLE_USER}/solar-k-package"
COMPETITION = "filament-segmentation-2026"

SETUP_SOURCE = """import os
import time

PKG_DIR = '/kaggle/input/datasets/potcharapanchumrak/solar-k-package'

for attempt in range(10):
    if os.path.isdir(PKG_DIR):
        break
    print(f'{PKG_DIR} not mounted yet, waiting (attempt {attempt + 1}/10)...')
    time.sleep(3)
else:
    raise FileNotFoundError(f'{PKG_DIR} never appeared after 30s -- check the dataset is attached')

os.chdir(PKG_DIR)
print(os.listdir('.'))"""

PIP_SOURCE = "!pip install -q -r requirements.txt"

ENV_SOURCE = """import os

COMP_ROOT = '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
PKG_SPLITS = os.path.join(PKG_DIR, 'data', 'splits')
DATA_LINK_ROOT = '/kaggle/working/data_root'
os.makedirs(DATA_LINK_ROOT, exist_ok=True)
for name, target in [('raw', COMP_ROOT), ('splits', PKG_SPLITS)]:
    link = os.path.join(DATA_LINK_ROOT, name)
    if not os.path.islink(link):
        os.symlink(target, link)

os.environ['SOLAR_K_DATA'] = DATA_LINK_ROOT
os.environ['SOLAR_K_OUTPUT'] = '/kaggle/working/outputs'
os.environ['SOLAR_K_CACHE'] = '/kaggle/temp/solar_k_cache'

RESUME_FROM = None
MAX_HOURS = 8.5"""

TRAIN_SOURCE_TMPL = """import subprocess

cmd = ['python', 'scripts/train.py', '--config', 'configs/E021-f{fold}.yaml', '--max_hours', str(MAX_HOURS)]
if RESUME_FROM:
    cmd += ['--resume', RESUME_FROM]

# E007 took ~46 min/fold on a 4090; estimated ~2.5-3 h/fold on a T4,
# well under MAX_HOURS; train.py checkpoints every 1000 iters, so a session
# that dies can resume via RESUME_FROM.
subprocess.run(cmd, check=True)"""


def make_notebook(fold: int) -> dict:
    markdown = f"""# E021 training on Kaggle -- fold {fold} (third ensemble member)

CODING_SPEC_05_ENSEMBLE.md: E007's exact recipe with seed 1337 instead of 42 (lowest-risk added member) -- see
configs/E021-f{fold}.yaml.

Setup (manual, one time):
1. Re-package with `scripts/package_kaggle.py` (so configs/E021-f*.yaml are
   in it), push a new version of `{PACKAGE_DATASET}`, attach it and the
   `{COMPETITION}` competition.
2. Turn on a GPU accelerator and Internet access (pip + timm ImageNet weights).

Checkpoints/logs/probs write to `/kaggle/working/outputs`."""

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": markdown.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": SETUP_SOURCE.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [PIP_SOURCE]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ENV_SOURCE.splitlines(keepends=True)},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": TRAIN_SOURCE_TMPL.format(fold=fold).splitlines(keepends=True)},
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_kernel_metadata(fold: int) -> dict:
    return {
        "id": f"{KAGGLE_USER}/solar-k-e021-f{fold}-train",
        "title": f"solar-k-e021-f{fold}-train",
        "code_file": f"kaggle_train_e021_f{fold}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [PACKAGE_DATASET],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [],
    }


def main() -> None:
    for fold in (0, 1):  # gate; folds 2-4 follow if the gate passes
        fold_dir = NOTEBOOKS_DIR / f"e021_f{fold}"
        fold_dir.mkdir(exist_ok=True)
        nb_path = fold_dir / f"kaggle_train_e021_f{fold}.ipynb"
        nb_path.write_text(json.dumps(make_notebook(fold), indent=1))
        km_path = fold_dir / "kernel-metadata.json"
        km = make_kernel_metadata(fold)
        km["code_file"] = nb_path.name
        km_path.write_text(json.dumps(km, indent=2) + "\n")
        print(f"Wrote {nb_path.relative_to(REPO_ROOT)} and {km_path.relative_to(REPO_ROOT)}")
    print("\nNot pushed yet. Launch with: kaggle kernels push -p notebooks/e021_fN")


if __name__ == "__main__":
    main()
