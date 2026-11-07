#!/usr/bin/env python3
"""Generate the 2 per-fold Kaggle notebooks + kernel-metadata files for E006
(CODING_SPEC_04_MODEL_SIDE.md): hard-negative mining, folds 0 and 1 only
(the spec's decision rule only needs these two to judge keep/reject).
Near-identical to scripts/gen_e004_kaggle_artifacts.py, pointed at
configs/E006-fN.yaml instead.

This only writes files -- it never calls `kaggle kernels push`.

Usage: python scripts/gen_e006_kaggle_artifacts.py
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

cmd = ['python', 'scripts/train.py', '--config', 'configs/E006-f{fold}.yaml', '--max_hours', str(MAX_HOURS)]
if RESUME_FROM:
    cmd += ['--resume', RESUME_FROM]

# data/interim/hard_negatives_v1.csv must be in the package dataset (it's
# under data/interim/, not included in scripts/package_kaggle.py's default
# INCLUDE list -- add it there, or this run falls back to no hard negatives
# since train.py prints a warning and continues with 0 catalogued stems
# rather than failing).
subprocess.run(cmd, check=True)"""


def make_notebook(fold: int) -> dict:
    markdown = f"""# E006 training on Kaggle -- fold {fold} (hard-negative mining)

CODING_SPEC_04_MODEL_SIDE.md: identical recipe to E004-f{fold} except the
crop sampler mix (45% positive / 30% hard-negative / 25% random, see
configs/E006-f{fold}.yaml) -- the single variable this experiment isolates.

Setup (manual, one time):
1. Attach the `{PACKAGE_DATASET}` dataset (make sure
   `data/interim/hard_negatives_v1.csv` is included -- rebuild+repush with
   `scripts/package_kaggle.py` after `scripts/build_hard_negatives.py` has
   run on all 5 folds' OOF) and the `{COMPETITION}` competition.
2. Turn on a GPU accelerator and Internet access.

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
        "id": f"{KAGGLE_USER}/solar-k-e006-f{fold}-train",
        "title": f"solar-k-e006-f{fold}-train",
        "code_file": f"kaggle_train_e006_f{fold}.ipynb",
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
    for fold in (0, 1):
        fold_dir = NOTEBOOKS_DIR / f"e006_f{fold}"
        fold_dir.mkdir(exist_ok=True)
        nb_path = fold_dir / f"kaggle_train_e006_f{fold}.ipynb"
        nb_path.write_text(json.dumps(make_notebook(fold), indent=1))
        km_path = fold_dir / "kernel-metadata.json"
        km = make_kernel_metadata(fold)
        km["code_file"] = nb_path.name
        km_path.write_text(json.dumps(km, indent=2) + "\n")
        print(f"Wrote {nb_path.relative_to(REPO_ROOT)} and {km_path.relative_to(REPO_ROOT)}")
    print("\nNot pushed yet. Launch with: kaggle kernels push -p notebooks/e006_fN")


if __name__ == "__main__":
    main()
