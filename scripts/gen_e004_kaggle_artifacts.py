#!/usr/bin/env python3
"""Generate the 5 per-fold Kaggle notebooks + kernel-metadata files for E004
(CODING_SPEC_03_FP_AND_FOLDS.md): one Kaggle run per fold, each a near-copy
of notebooks/kaggle_train.ipynb pointed at its own configs/E004-fN.yaml and
its own kernel slug, so they can be pushed (and launched) independently.

This only writes files -- it never calls `kaggle kernels push` (that
launches a GPU run immediately), so running this script is safe to do
ahead of actually starting training.

Usage: python scripts/gen_e004_kaggle_artifacts.py
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

# On this Kaggle layout, attached sources are nested under /kaggle/input/datasets/
# and /kaggle/input/competitions/ (not flat /kaggle/input/<slug>/).
PKG_DIR = '/kaggle/input/datasets/potcharapanchumrak/solar-k-package'
os.chdir(PKG_DIR)
print(os.listdir('.'))"""

PIP_SOURCE = "!pip install -q -r requirements.txt"

ENV_SOURCE = """import os

# The competition mount is .../MAGFiLO_1.0_Kaggle_2026/{train,test}/..., but
# solar_k.io expects <SOLAR_K_DATA>/raw/{train,test}/... (matching the local
# data/raw/ layout) -- bridge the one missing "raw" level with a symlink
# rather than changing io.py's path assumptions (which also serve local runs).
# splits/folds_v1.csv similarly comes from the uploaded package, not the
# competition mount, so it needs its own symlink alongside raw/.
COMP_ROOT = '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
PKG_SPLITS = os.path.join(PKG_DIR, 'data', 'splits')
DATA_LINK_ROOT = '/kaggle/working/data_root'
os.makedirs(DATA_LINK_ROOT, exist_ok=True)
for name, target in [('raw', COMP_ROOT), ('splits', PKG_SPLITS)]:
    link = os.path.join(DATA_LINK_ROOT, name)
    if not os.path.islink(link):
        os.symlink(target, link)

os.environ['SOLAR_K_DATA'] = DATA_LINK_ROOT

# Checkpoints/logs/probs go straight into the persisted working directory.
os.environ['SOLAR_K_OUTPUT'] = '/kaggle/working/outputs'

# Preprocessing cache on fast, writable local disk (the competition mount is read-only).
os.environ['SOLAR_K_CACHE'] = '/kaggle/temp/solar_k_cache'

# Set to a last.pt path (e.g. under /kaggle/input/<prior-version-dataset>/outputs/E004-f{fold}/checkpoints/last.pt)
# to continue a previous notebook version's run instead of starting fresh.
RESUME_FROM = None

# E004's 4000 iterations finish well under this; kept as a safety cap in
# case a Kaggle session is slower than expected.
MAX_HOURS = 8.5"""

TRAIN_SOURCE_TMPL = """import subprocess

cmd = ['python', 'scripts/train.py', '--config', 'configs/E004-f{fold}.yaml', '--max_hours', str(MAX_HOURS)]
if RESUME_FROM:
    cmd += ['--resume', RESUME_FROM]

# One invocation covers the whole run: training (EMA weights, checkpoint
# selection by instance PQ every 1000 iters on a 40-image subset, see
# configs/E004-f{fold}.yaml), then -- in the same process -- flip-TTA OOF
# (fold {fold} val) and test-set probability maps saved as compressed uint8
# PNGs. No postprocessing/threshold tuning happens here; that runs locally
# against these saved probs (scripts/run_e005_eval.py, once all 5 folds are in).
subprocess.run(cmd, check=True)"""


def make_notebook(fold: int) -> dict:
    markdown = f"""# E004 training on Kaggle -- fold {fold}

CODING_SPEC_03_FP_AND_FOLDS.md: same model as E002 (resnet34/imagenet U-Net)
with EMA of weights (decay 0.999) and checkpoint selection by instance PQ
(every 1000 iterations, on a fixed 40-image subset of fold {fold}'s val
split) instead of pixel Dice, 4000 iterations. One of five near-identical
notebooks (fold 0-4), each pushed and run as its own Kaggle kernel so the
five folds can train in parallel across separate Kaggle sessions.

Designed to run unattended top to bottom (**Save & Run All**), with Internet
**on** (for `pip install` and the ImageNet encoder weights).

Setup (manual, one time, same package dataset as E002 -- rebuild/push it
first with `scripts/package_kaggle.py` if `configs/E004-f{fold}.yaml` or any
`src/`/`scripts/` change isn't in it yet):
1. Attach the `{PACKAGE_DATASET}` dataset and the `{COMPETITION}` competition.
2. Turn on a GPU accelerator and Internet access.
3. If resuming from a previous notebook version's output, attach that
   version's output as a dataset too and set `RESUME_FROM` below to its
   `last.pt` path; otherwise leave it `None`.

Checkpoints, logs, OOF/test probability maps all write directly to
`/kaggle/working/outputs` (the persisted working directory) -- no
copy-at-the-end step needed. Never put credentials in this notebook or in
the uploaded package."""

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
        "id": f"{KAGGLE_USER}/solar-k-e004-f{fold}-train",
        "title": f"solar-k-e004-f{fold}-train",
        "code_file": f"kaggle_train_e004_f{fold}.ipynb",
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
    # The Kaggle CLI's `kernels push -p <dir>` looks for a file literally
    # named kernel-metadata.json inside <dir> -- one subdirectory per fold
    # (rather than 5 differently-named metadata files in notebooks/) is what
    # actually lets each fold be pushed independently.
    for fold in range(5):
        fold_dir = NOTEBOOKS_DIR / f"e004_f{fold}"
        fold_dir.mkdir(exist_ok=True)
        nb_path = fold_dir / f"kaggle_train_e004_f{fold}.ipynb"
        nb_path.write_text(json.dumps(make_notebook(fold), indent=1))
        km_path = fold_dir / "kernel-metadata.json"
        km = make_kernel_metadata(fold)
        km["code_file"] = nb_path.name
        km_path.write_text(json.dumps(km, indent=2) + "\n")
        print(f"Wrote {nb_path.relative_to(REPO_ROOT)} and {km_path.relative_to(REPO_ROOT)}")
    print(
        "\nThese were only written, not pushed (pushing launches a GPU run "
        "immediately). To launch fold N's training, after confirming the "
        "outputs/kaggle_pkg dataset is up to date (scripts/package_kaggle.py "
        "+ `kaggle datasets version -p outputs/kaggle_pkg/ --dir-mode zip -m ...`):\n"
        "  kaggle kernels push -p notebooks/e004_fN"
    )


if __name__ == "__main__":
    main()
