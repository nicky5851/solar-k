#!/usr/bin/env bash
# Idempotent environment setup for a fresh RunPod (or any bare Linux+CUDA)
# GPU pod (CODING_SPEC_04_MODEL_SIDE.md E007). Re-running on a pod that
# already has some of this done just skips/overwrites in place -- no manual
# cleanup needed between pods.
#
# Usage (from the repo root, already cloned/rsynced onto the pod):
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...   # or have ~/.kaggle/kaggle.json already
#   bash scripts/setup_remote.sh
#
# Never commit a Kaggle token. This script only ever reads one from the
# environment or an existing ~/.kaggle/kaggle.json -- it does not write
# credentials into the repo or print them.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== CPU/GPU info =="
nproc || true
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not found -- is this a GPU pod?"

echo "== Python venv =="
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== Kaggle auth =="
mkdir -p ~/.kaggle
if [ -n "${KAGGLE_KEY:-}" ] && [ -n "${KAGGLE_USERNAME:-}" ]; then
  printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
elif [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f ~/.kaggle/access_token ]; then
  echo "No Kaggle credentials found. Set KAGGLE_USERNAME/KAGGLE_KEY, or copy"
  echo "~/.kaggle/kaggle.json (or access_token) onto this pod, then re-run."
  exit 1
fi

echo "== Data: competition raw images + annotations =="
# Mirrors the local data/raw/{train,test} layout that solar_k.io expects
# (SOLAR_K_DATA/raw/..., SOLAR_K_DATA/splits/...) -- not the Kaggle-notebook
# nested-mount layout, since this is a plain pod, not a Kaggle kernel.
DATA_ROOT="${SOLAR_K_DATA:-$REPO_ROOT/data}"
mkdir -p "$DATA_ROOT/raw"
if [ ! -d "$DATA_ROOT/raw/train" ] || [ -z "$(ls -A "$DATA_ROOT/raw/train" 2>/dev/null)" ]; then
  kaggle competitions download -c filament-segmentation-2026 -p /tmp/solar_k_comp_dl
  python3 - <<'PY'
import zipfile, glob, shutil, pathlib
dl_dir = pathlib.Path("/tmp/solar_k_comp_dl")
zips = list(dl_dir.glob("*.zip"))
assert zips, f"no competition zip found in {dl_dir}"
with zipfile.ZipFile(zips[0]) as zf:
    zf.extractall(dl_dir / "extracted")
# The competition zip's top-level folder name has shown up as
# MAGFiLO_1.0_Kaggle_2026 historically (see notebooks/e004_f*/*.ipynb's
# Kaggle-mount comment) -- locate train/ and test/ wherever they landed and
# copy them to data/raw/{train,test}, same layout the Mac/Kaggle code expects.
extracted = dl_dir / "extracted"
candidates = [p for p in extracted.rglob("train") if p.is_dir()]
assert candidates, f"couldn't find a train/ directory under {extracted}"
comp_root = candidates[0].parent
import os
data_root = pathlib.Path(os.environ.get("SOLAR_K_DATA", "")) / "raw" if os.environ.get("SOLAR_K_DATA") else pathlib.Path("data/raw")
data_root.mkdir(parents=True, exist_ok=True)
for name in ("train", "test"):
    src, dst = comp_root / name, data_root / name
    if dst.exists():
        continue
    shutil.copytree(src, dst)
print(f"Copied train/test from {comp_root} to {data_root}")
PY
else
  echo "data/raw/train already populated, skipping download"
fi

echo "== Splits + E002/E003 package contents (configs, src, the repo code) =="
# This script assumes the repo itself (this clone/rsync) already has
# data/splits/folds_v1.csv checked in -- nothing to fetch separately.
test -f "$DATA_ROOT/splits/folds_v1.csv" || echo "WARNING: $DATA_ROOT/splits/folds_v1.csv missing -- rsync/clone the full repo, not just scripts/"

echo "== Env vars for this shell (add to your pod's profile/.bashrc to persist) =="
echo "export SOLAR_K_DATA=$DATA_ROOT"
echo "export SOLAR_K_OUTPUT=$REPO_ROOT/outputs"
echo "export SOLAR_K_CACHE=/tmp/solar_k_cache"

echo "== Done. Smoke-test with: =="
echo "  source .venv/bin/activate && SOLAR_K_DATA=$DATA_ROOT SOLAR_K_OUTPUT=$REPO_ROOT/outputs SOLAR_K_CACHE=/tmp/solar_k_cache python3 -m pytest -q"
echo "== Then train, e.g.: =="
echo "  SOLAR_K_DATA=$DATA_ROOT SOLAR_K_OUTPUT=$REPO_ROOT/outputs SOLAR_K_CACHE=/tmp/solar_k_cache python3 scripts/train.py --config configs/E007-f0.yaml"
echo "== Sync results back to your machine when a fold finishes, e.g.: =="
echo "  rsync -avz <pod-host>:$REPO_ROOT/outputs/E007-f0/ ./outputs/E007-f0/"
