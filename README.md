# Solar_K — Solar Filament Segmentation Challenge 2026

Our entry to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026), instance segmentation of filaments in GONG H-alpha images, scored by Panoptic Quality (PQ).

See `data/README.md` for where to put the dataset (CC BY-NC 4.0 — not redistributed in this repo).

## Final result

**Primary: `outputs/submissions/E030_primary.csv`** — U-Net ensemble of `efficientnet-b4` (E007) + `tu-regnety_040` (E015), 4-flip + 1.25x-upscale TTA, 5-fold OOF-fitted post-processing. Local day-grouped 5-fold CV PQ **0.4298** (SQ 0.681, RQ 0.631). Public LB 0.37.
**Backup: `outputs/submissions/E013_backup.csv`** — `resnet34` (E004) + `efficientnet-b4` (E007), 3-flip TTA. Local CV PQ **0.4176** (rebuilt Nov 7 after three E004 folds were retrained; the original E013 scored 0.4086 locally). Public LB 0.37 (original E013).

E030 was submitted to Kaggle on Nov 7 and scored 0.37 public, the same as E024, which it reproduces. Final selection: E030 primary, E013 backup. Submission was always a human decision, never automated here. Full write-up, by-site/by-area breakdowns, failure taxonomy and the rejected-experiment table: **`reports/08_final_writeup.md`**.

| | PQ (local CV) | Public LB |
|---|---|---|
| E010 (E004 alone) | 0.3948 ± 0.0067 | — |
| E013 (E004+E007, original) | 0.4086 ± 0.0050 | 0.37 |
| E013 backup (rebuilt from shipped weights) | 0.4176 ± 0.0060 | — |
| E024 (E007tta4s+E015tta4s) | 0.4302 | 0.37 |
| **E030 primary** (E007tta4s+E015tta4sR, rebuilt from shipped weights) | **0.4297 ± 0.0048** | **0.37** |

## Reports

`reports/01_dataset_report.md` through `reports/08_final_writeup.md` are the dated, evidence-backed write-ups, one per `docs/specs/CODING_SPEC_0*.md`. **`reports/08_final_writeup.md` is the final deliverable** — task/metric background, the inter-annotator PQ ceiling (0.34), the end-to-end pipeline, P5 validation protocol, quantitative results, morphology/failure cases, what didn't work (Specs 05–07), the leaderboard-integrity note, and exact reproduce commands.

## Reproduce

Two separate paths, for two separate purposes:

### A. Reproduce the submission from shipped weights (no GPU training needed)

This is what a judge runs. Weights (15 fp16 EMA checkpoints, ~700 MB total) are **not** in git — they ship as a GitHub Release asset (`v1.0-final`), checksummed against `weights/MANIFEST.json`.

```
python3 scripts/download_weights.py                        # fetch + sha256-verify weights/*.pt
python3 scripts/predict.py --pipeline primary  --out outputs/submissions/E030_primary.csv
python3 scripts/predict.py --pipeline backup   --out outputs/submissions/E013_backup.csv
python3 scripts/validate_submission.py outputs/submissions/E030_primary.csv
```

`predict.py` runs fresh TTA inference from the shipped weights and applies the frozen post-processing chain in `configs/final_postproc_{primary,backup}.json` (cfg1, shape gate, the fitted scorer, cutoff, watershed regrow) — nothing is refit at this stage. It is not bit-exact: on Nov 7 it matched instance counts on 172/176 test images for the primary (174/174 for the backup) with mean IoU 0.9998 on matched images. See `reports/08_final_writeup.md` §8. This needs a GPU for inference speed but not for training; `torch.device("mps")`/CPU both work, just slower.

### B. Reproduce from scratch (training, the actual development path)

All training ran on Kaggle (T4) or RunPod GPUs; `src/` and the CPU-only post-processing/evaluation scripts (`scripts/run_p5.py`, `scripts/validate_submission.py`, the test suite) run fine on a laptop.

1. **Data + splits.**
   ```
   python3 scripts/make_splits.py        # -> data/splits/folds_v1.csv (day-grouped 5-fold)
   ```
   (per-image preprocessing caches are built lazily by `src/solar_k/preprocess.py:load_or_build`.)

2. **Train each ensemble member's 5 folds** (`configs/<id>-f{0..4}.yaml`):
   ```
   python3 scripts/train.py --config configs/E007-f0.yaml   # repeat per fold, per member; GPU required
   ```
   `scripts/gen_e0*_kaggle_artifacts.py` package the same configs to run as Kaggle kernels, which is how most of this project's training actually happened.

3. **Regenerate OOF/test probability maps with TTA** from each fold's `best.pt` EMA weights (no retraining):
   ```
   python3 scripts/infer.py --config configs/E007-f0.yaml --checkpoint outputs/E007-f0/best.pt \
       --tta-views 4 --tta-scales 1.0,1.25 --out-experiment E007tta4s-f0   # repeat per fold, per member
   ```

4. **Fit post-processing and evaluate under the P5 protocol** (nested day-grouped 5-fold CV — see
   `reports/05_ensemble_report.md` §2–3 for why this, not a single train/report split, is the decision metric):
   ```
   python3 scripts/run_p5.py --members E007tta4s E015tta4sR --experiment <id>
   ```
   Writes `data/interim/<id>_p5_summary.json` (PQ/SQ/RQ, loss decomposition) and `<id>_p5_per_image.csv`.
   `--reference <other_id> --bar pipeline|ensemble|early` gives a paired-bootstrap comparison and a
   pre-registered keep/reject decision; `--save-details` adds the site/size-quartile and per-image-Dice
   breakdowns used in the final report.

5. **Build and validate the final submission** (refits post-processing on all 707 OOF images, no CV
   split held out, then applies it to the test set's mean probability map):
   ```
   python3 scripts/build_final.py --members E007tta4s E015tta4sR --experiment E030_primary
   python3 scripts/validate_submission.py outputs/submissions/E030_primary.csv
   ```

6. **Tests.** `python3 -m pytest` — CPU-only, no GPU or network, small synthetic fixtures (157 tests).

`experiments.csv` is the append-only experiment log — every run above, and every one tried and rejected, has a row with its hypothesis, decision, and the exact numbers behind it.

## Repo layout

```
src/solar_k/      library code (preprocessing, models, post-processing, metric, ensembling)
scripts/          CLI entry points (train/infer/predict/evaluate), call into src/
configs/          per-experiment YAML configs + the two frozen post-processing JSONs
weights/          MANIFEST.json only -- actual .pt files come from the GitHub Release
tests/            unit tests, esp. for the metric, RLE encoding, and splits
docs/specs/       the numbered specs this project was built from
reports/          one dated write-up per spec; 08 is the final deliverable
data/, outputs/   gitignored (raw data, caches, checkpoints, maps -- see data/README.md),
                  except outputs/submissions/{E030_primary,E013_backup}.csv
```

## Licence / citation

Code: MIT (see `LICENSE`). Dataset: CC BY-NC 4.0 per the competition terms, not redistributed here — see `data/README.md`.
