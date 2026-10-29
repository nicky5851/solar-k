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
