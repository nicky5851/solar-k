# Solar_K — Solar Filament Segmentation Challenge 2026

This is our entry to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026), an instance segmentation task for filaments in GONG H-alpha images, scored with Panoptic Quality (PQ).

See `data/README.md` for where to put the dataset.

## Reports

`reports/01_dataset_report.md` through `reports/06_final_report.md` are the dated, evidence-backed
write-ups, one per `docs/specs/CODING_SPEC_0*.md`. `reports/06_final_report.md` is the final
deliverable and the right place to start — it has the leaderboard-trust caveat, the inter-annotator
PQ ceiling (0.34), and the full progression from the first baseline to the frozen pipeline.

## Reproduce from scratch

All training and GPU inference ran on Kaggle (T4) or a RunPod pod; `src/` and the CPU-only
post-processing/evaluation scripts (`scripts/run_p5.py`, `scripts/validate_submission.py`, the test
suite) run fine on a laptop. Nothing here submits to Kaggle on your behalf.

1. **Data.** Download and extract per `data/README.md`. Then build the leakage-safe, day-grouped
   5-fold split and the cached per-image preprocessing (disk center/radius, etc.):
   ```
   python3 scripts/make_splits.py        # -> data/splits/folds_v1.csv
   ```
   (preprocessing caches are built lazily by `src/solar_k/preprocess.py:load_or_build` the first
   time any script reads an image).

2. **Train the ensemble members.** Each member is 5 independently-trained folds (`configs/<id>-f{0..4}.yaml`).
   The two kept in the final pipeline are E007 (resnet34) and E015 (efficientnet-b4):
   ```
   python3 scripts/train.py --config configs/E007-f0.yaml   # repeat per fold, per member; GPU required
   ```
   `notebooks/kaggle_train.ipynb` / `scripts/gen_e0*_kaggle_artifacts.py` package the same configs to
   run as Kaggle kernels, which is how most of this project's training actually happened.

3. **Regenerate OOF/test probability maps with TTA.** The frozen pipeline uses 4-flip-view +
   1.25x-upscale test-time augmentation (`src/solar_k/unet.py:make_prob_fn`), applied at inference
   time from each fold's `best.pt` EMA weights — it does not require retraining:
   ```
   python3 scripts/infer.py --config configs/E007-f0.yaml --checkpoint outputs/E007-f0/best.pt \
       --tta-views 4 --tta-scales 1.0,1.25 --out-experiment E007tta4s-f0   # repeat per fold, per member
   ```

4. **Fit post-processing and evaluate under the P5 protocol** (nested day-grouped 5-fold CV; see
   `reports/05_ensemble_report.md` §2–3 for why this, and not a single train/report split, is the
   decision metric):
   ```
   python3 scripts/run_p5.py --members E007tta4s E015tta4s --experiment <id>
   ```
   This writes `data/interim/<id>_p5_summary.json` (PQ/SQ/RQ, loss decomposition) and
   `<id>_p5_per_image.csv`. Pass `--reference <other_id> --bar pipeline|ensemble|early` to get a
   paired bootstrap comparison and a keep/reject decision against a prior experiment, and
   `--save-details` for the per-component (site/size-quartile) and per-image-Dice breakdowns used in
   the final report.

5. **Build and validate the final submission** (fits post-processing on all 707 OOF images, no CV
   split, then applies it to the mean of the test-set probability maps):
   ```
   python3 scripts/build_final.py --members E007tta4s E015tta4s --experiment E024_primary
   python3 scripts/validate_submission.py outputs/submissions/E024_primary.csv
   ```

6. **Tests.** `python3 -m pytest` (CPU-only, no GPU or network needed; a few tests build small
   synthetic fixtures rather than touching the real dataset).

`experiments.csv` is the append-only experiment log — every run above (and every one we tried and
rejected) has a row there with its hypothesis, decision, and the exact numbers behind it.
