# scaffold.md — instructions for Claude Code

**Goal:** create a minimal, empty project skeleton for our Kaggle *Solar Filament Segmentation Challenge 2026* entry, so the dataset can be dropped into an organised `data/` folder. **Do not write any model, training, or pipeline code yet.** That comes later from `coding_spec.md`.

Run these steps from the repository root.

---

## 1. Create this folder structure

```
.
├── data/                      # ALL contents git-ignored (CC BY-NC data, never pushed)
│   ├── raw/                   # untouched Kaggle download, exactly as extracted
│   │   └── .gitkeep
│   ├── interim/               # intermediate stuff: decoded masks, consensus labels, caches
│   │   └── .gitkeep
│   ├── processed/             # final model-ready inputs (resized/normalised images, tiles, masks)
│   │   └── .gitkeep
│   ├── splits/                # CV fold assignments (small CSVs; also ignored by default)
│   │   └── .gitkeep
│   └── README.md              # explains each subfolder (this file IS committed)
├── notebooks/                 # exploration notebooks (EDA, quick checks)
│   └── .gitkeep
├── src/                       # Python package code (empty for now)
│   └── __init__.py
├── configs/                   # experiment configs (yaml)
│   └── .gitkeep
├── scripts/                   # CLI entry points (download, prepare, train, predict)
│   └── .gitkeep
├── outputs/                   # git-ignored: checkpoints, logs, predictions, submissions
│   └── .gitkeep
├── reports/                   # written reports (dataset report, final 4-page tech report)
│   └── .gitkeep
├── docs/
│   └── Notes.md               # copy of our decisions log (paste in from the chat)
├── .gitignore
├── README.md
└── requirements.txt           # empty placeholder for now
```

## 2. Write `.gitignore` with exactly this content

```gitignore
# ---- Dataset: never commit (CC BY-NC 4.0, not ours to redistribute) ----
data/**
!data/
!data/README.md
!data/*/
!data/*/.gitkeep

# Kaggle archives / raw downloads anywhere in the repo
*.zip
*.tar
*.tar.gz
*.7z
MAGFiLO*/
**/train_images/
**/test_images/

# Images and masks (in case they end up outside data/)
*.jpg
*.jpeg
*.png
*.fits
*.fits.gz
*.npy
*.npz
*.h5
*.hdf5
*.parquet

# ---- Large artifacts ----
outputs/**
!outputs/.gitkeep
*.pt
*.pth
*.ckpt
*.onnx
*.safetensors
*.bin
submission*.csv
wandb/
lightning_logs/
mlruns/
runs/

# ---- Credentials ----
kaggle.json
.env
*.env

# ---- Python / tooling ----
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.DS_Store
.vscode/
.idea/
```

> Note: `*.png` is ignored globally to keep images out. If we later want figures in `reports/`, add `!reports/**/*.png`.

## 3. Write `data/README.md`

```markdown
# data/

Everything in here except this README and the `.gitkeep` files is git-ignored.
Data: MAGFiLO 1.0 (Kaggle 2026), CC BY-NC 4.0. Do not upload it anywhere.

- `raw/`       Kaggle download, extracted as-is. Never edit files here.
               Expected: data/raw/<extracted Kaggle folder>/ (e.g. MAGFiLO_1.0_Kaggle_2026/ with train/ and test/)
- `interim/`   Derived but not final: decoded masks, consensus labels, caches.
- `processed/` Model-ready inputs produced by our scripts.
- `splits/`    Cross-validation fold files.

Download (needs Kaggle API token in ~/.kaggle/kaggle.json and rules accepted on the site):
    kaggle competitions download -c filament-segmentation-2026 -p data/raw
    unzip data/raw/filament-segmentation-2026.zip -d data/raw/
```

## 4. Write a short `README.md`

Title "Solar_K — Solar Filament Segmentation Challenge 2026", one paragraph saying this is our entry (instance segmentation of filaments in GONG H-alpha images, metric Panoptic Quality), the competition link https://www.kaggle.com/competitions/filament-segmentation-2026, and a one-line pointer to `data/README.md` for where the dataset goes. Nothing else yet.

## 5. Leave `requirements.txt` empty and `src/__init__.py` empty.

## 6. Verify the dataset is really ignored

After creating everything, run:

```bash
mkdir -p data/raw/_probe && touch data/raw/_probe/x.jpeg data/raw/fake.zip outputs/model.pt
git status --porcelain --ignored | grep -E "data/|outputs/"
git check-ignore -v data/raw/_probe/x.jpeg data/raw/fake.zip outputs/model.pt
rm -rf data/raw/_probe data/raw/fake.zip outputs/model.pt
```

All three probe files must be reported as ignored, and `git status` must show only `data/README.md` and the `.gitkeep` files as trackable under `data/`. Report the result.

## 7. Commit

Commit the skeleton with message `Scaffold project structure`. Do **not** push unless asked.

---

### After the scaffold (next step, for reference only)

Once the dataset is extracted into `data/raw/`, we'll ask Claude Code for a **read-only dataset report** in `reports/dataset_report.md`: file tree, image count/sizes/bit depth, annotation JSON schema, categories, instances per image, annotators per image, mask area and length stats, train/test date and GONG site distribution, and any oddities. That report goes back into the chat before we write `coding_spec.md`.
