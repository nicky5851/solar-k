#!/usr/bin/env bash
# CODING_SPEC_05_ENSEMBLE.md section 4: the RunPod 4090 training queue.
# Runs each config in order, one at a time, on a single pod:
#   E007-f2, E007-f3, E007-f4  (E012: finish the E007 ensemble member)
#   E015-f0, E015-f1, E015-f2  (E015: third member; folds 3-4 run on Kaggle)
# Idempotent/resumable: a run whose test_probs/ already has all 180 maps is
# skipped; a run with a last.pt but no finished probs resumes from it. So if
# the pod dies, start a new pod, run setup_remote.sh, rsync outputs/ back up,
# and re-run this script.
#
# Usage (repo root on the pod, after `bash scripts/setup_remote.sh`):
#   nohup bash scripts/run_queue_runpod.sh > outputs/runpod_queue.log 2>&1 &
# Optionally pass a subset: bash scripts/run_queue_runpod.sh E015-f0 E015-f1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

export SOLAR_K_DATA="${SOLAR_K_DATA:-$REPO_ROOT/data}"
export SOLAR_K_OUTPUT="${SOLAR_K_OUTPUT:-$REPO_ROOT/outputs}"
export SOLAR_K_CACHE="${SOLAR_K_CACHE:-/tmp/solar_k_cache}"

QUEUE=("$@")
if [ ${#QUEUE[@]} -eq 0 ]; then
  QUEUE=(E007-f2 E007-f3 E007-f4 E015-f0 E015-f1 E015-f2)
fi
N_TEST=180

# E015 fails fast if the encoder has no stride-2 skip (spec section 3).
if printf '%s\n' "${QUEUE[@]}" | grep -q '^E015'; then
  python3 -m pytest -q tests/test_encoder_stride.py || { echo "encoder stride test failed -- fix configs/E015-f*.yaml (fallback resnext50_32x4d) before training"; exit 1; }
fi

for EXP in "${QUEUE[@]}"; do
  CFG="configs/${EXP}.yaml"
  OUT="$SOLAR_K_OUTPUT/$EXP"
  if [ ! -f "$CFG" ]; then echo "[$(date -u +%FT%TZ)] $EXP: $CFG missing, skipping"; continue; fi
  n_done=$(ls "$OUT/test_probs" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n_done" -ge "$N_TEST" ]; then echo "[$(date -u +%FT%TZ)] $EXP: already finished ($n_done test maps), skipping"; continue; fi
  ARGS=(--config "$CFG")
  if [ -f "$OUT/checkpoints/last.pt" ]; then ARGS+=(--resume); fi
  echo "[$(date -u +%FT%TZ)] $EXP: starting ${ARGS[*]}"
  mkdir -p "$OUT"
  if python3 scripts/train.py "${ARGS[@]}" 2>&1 | tee -a "$OUT/train.log"; then
    echo "[$(date -u +%FT%TZ)] $EXP: done"
  else
    echo "[$(date -u +%FT%TZ)] $EXP: FAILED (see $OUT/train.log) -- continuing with the next run"
  fi
done
echo "[$(date -u +%FT%TZ)] queue finished. Sync back, then STOP THE POD:"
echo "  rsync -avz <pod-host>:$SOLAR_K_OUTPUT/{E007-f2,E007-f3,E007-f4,E015-f0,E015-f1,E015-f2} ./outputs/"
