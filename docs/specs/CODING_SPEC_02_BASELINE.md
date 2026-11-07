# CODING_SPEC_02_BASELINE — first end-to-end pipeline and Kaggle submission

**Phase 1. Time box: Nov 1–3. Training is allowed in this spec, but only as described here.**
Read `CLAUDE.md` and `reports/01_dataset_report.md` first. Where this spec conflicts with the report's verified facts, stop and ask instead of guessing.

## Goal
1. A **reproducible end-to-end pipeline**: image → preprocessing → probability map → instances → validated CSV.
2. Two reference results on the frozen CV split: a **classical** baseline (E001) and a **semantic U-Net + connected-components** baseline (E002).
3. The **first valid Kaggle submission**.
4. Evidence on **H1 (revised after EDA)**. Find which of these failure modes costs the most PQ, because that decides Spec 03:
   - **(a) Detection:** missed or spurious small, faint filaments (FN/FP by size bucket).
   - **(b) Fragmentation:** one GT filament broken into several predicted components.
   - **(c) Near-miss IoU:** the right filament found, but IoU just under 0.5 on thin structures, so one error counts as both an FP and an FN.
   - **(d) Merging:** expected to be rare, since only 0.5% of GT instances touch a neighbour.

## Decisions taken from the EDA (2026-11-01)
- **Training label = soft consensus**: each pixel's value is the fraction of that image's annotators who marked it; single-annotator images give a binary target. Scoring averages over annotators, so the best semantic prediction is the consensus. With BCE, soft consensus has the same expected gradient as sampling a random annotator each epoch, but with less noise. **Union is rejected**: it inflates masks, which lowers IoU against each individual annotator.
- **Noisy or partial frames:** keep them in training and in validation, and report them as a separate slice.
- **Model:** a semantic U-Net plus post-processing. A true instance model (Mask R-CNN, YOLO-seg, Mask2Former) is **not** planned unless the baseline shows merging (d) dominating, which the EDA makes unlikely.

No architecture search, no ensembling, no TTA beyond flips, no leaderboard tuning.

## Stage 0 — housekeeping (≤ 30 min)
- Un-ignore the split files: add `!data/splits/*.csv` to `.gitignore`. They hold only file stems and folds, no image data, and the public repo needs them for reproducibility. Commit `folds_v1.csv`.
- **Near-duplicate check, fixed:** replace the raw thumbnail correlation. (a) Pair frames by timestamp: same site, < 60 min apart. (b) For those pairs and a random background sample, correlate **limb-darkening-corrected, disk-masked, high-pass** 256×256 images. Report the background vs candidate distributions and pick a threshold from the gap. Check that every near-duplicate pair falls in the same fold, and count train–test near-duplicates (information only).

## Stage A — shared pipeline pieces (`src/solar_k/`)
- `preprocess.py`: disk fit (reuse the EDA code), limb-darkening correction (divide by the per-image radial median profile), robust normalisation to roughly [0, 1] inside the disk, off-disk set to 0. Cache the preprocessed float16 images in `data/processed/v1/`, one file per physical image. Generating them must be deterministic and versioned.
- `targets.py`: per physical image, build a training target from all its annotators. Record the strategy in the config. Strategy: **soft consensus** (see Decisions above). Keep per-annotator instance masks available for evaluation.
- `postprocess.py`: `probs → binary` via either a single threshold `t` or **hysteresis** (`t_high` seeds, grown into `t_low` regions). Hysteresis targets both faint-filament FN and fragmentation. Then optional morphological closing with radius `r` → connected components (8-connectivity) → drop components < `min_area` → exclusive masks. All parameters come from config.
- `evaluate.py`: given instances and GT for a set of images, return the official PQ/SQ/RQ (through `metric.py`) plus **pixel Dice/IoU** (against each annotator and against the union), FP and FN counted by GT/pred area bucket (use the EDA quartiles), **one-to-many (fragmentation) and many-to-one (merge) counts** (pred↔GT overlap graph with IoU > 0.1), **near-misses** (pred/GT pairs whose best IoU is in [0.3, 0.5)), PQ by GT size quartile, PQ by disk position (`r/R` bins: centre, middle, limb), per-site PQ, the noisy-frame slice, and inference time per image.
  Also add a **PQ-loss decomposition** that assigns each FP and FN to exactly one of (a)–(d) above, so the report can state what share of lost PQ each failure mode costs.
- `predict.py` + `scripts/predict.py`: run any model over the test folder → instances → CSV → `validate_submission.py`.

## Stage B — E001 classical baseline (time box 2 h)
On the preprocessed image: a dark-feature map (e.g. black top-hat or a multi-scale ridge filter, pick one and justify it in a sentence), then the same `postprocess.py`. Tune `t`, `r`, and `min_area` on fold 0 as described in the tuning protocol below. Write a test CSV and validate it.
Its job is to prove the pipeline end to end on the Mac and to give the report a reference number. **This CSV can be the first Kaggle submission.**

## Stage C — E002 semantic U-Net
- Model: `segmentation_models_pytorch.Unet`, encoder `resnet34` with ImageNet weights, 1 input channel (the preprocessed image), 1 output logit.
- Training data: **full-resolution random 512×512 crops** from the disk area. Two thirds of the crops are centred on a random GT filament pixel and one third is random. Augmentations: flips, 90° rotations, mild brightness/contrast/gamma. No resizing.
- Loss: BCE + soft Dice on the target from `targets.py`. Optimiser AdamW at lr 3e-4 with weight decay 1e-4, cosine schedule with 300 warm-up steps, batch 8, AMP on CUDA. Train for a fixed number of iterations (start at 6k; record it). Seed 42.
- Validation every 500 iterations on fold-0 val: pixel Dice at t = 0.5 on downsampled full images, which is cheap. Keep the best checkpoint by that metric.
- Inference: the whole 2048×2048 image in one pass (or 1024 tiles with 128 px overlap if memory requires it), averaged over h/v flips. Save val probabilities as compressed uint8 in `outputs/E002/oof/`.
- **Folds:** train **fold 0 only** in this spec. Folds 1–4 come later, when a pipeline is a candidate for the final submission.
- **Portability:** `scripts/train.py --config configs/E002.yaml`. Device = CUDA > MPS > CPU. Checkpoint every 500 iterations, with `--resume` restoring model, optimiser, scheduler, RNG state, and iteration count. Paths come from config or env.
- **Mac smoke test first:** 50 iterations on MPS with a tiny config. It must run train → resume → infer → evaluate on 4 val images before anything goes to Kaggle.
- **Kaggle:** `scripts/package_kaggle.py` zips `src/`, `scripts/`, `configs/`, and `requirements.txt` into `outputs/kaggle_pkg/` for upload as a private Kaggle dataset. Add `notebooks/kaggle_train.ipynb`, which installs the package and trains on the attached competition data. Nicky runs the upload and the notebook manually; write the steps in the report. Never put credentials in code.

## Tuning protocol (both experiments)
Split fold-0 val **by observation day** into two halves, `tune` and `report`, with a fixed seed. Grid-search single `t ∈ {0.3…0.7 by 0.05}` **and** hysteresis `(t_low, t_high) ∈ {0.2, 0.3, 0.4} × {0.5, 0.6, 0.7}`, `r ∈ {0, 1, 2, 3}`, `min_area ∈ {EDA 1st percentile ×0.5, ×1, ×2}` by PQ on `tune`, then report every metric on `report`. Also report the PQ that pixel-optimal settings would give (best pixel Dice threshold), to show the gap between pixel and instance objectives.

## Outputs
- `outputs/submissions/E001.csv` and `E002.csv`, both passing `validate_submission.py`.
- Rows E001 and E002 in `experiments.csv`, with every field from CLAUDE.md plus the Kaggle public score, filled in by hand after Nicky submits.
- `reports/02_baseline_report.md`:
  - a results table (E001, E002, the inter-annotator reference) with PQ/SQ/RQ, pixel Dice, FP/FN by size bucket, one-to-many and many-to-one counts, per-site PQ, and runtime
  - **H1 verdict:** the PQ-loss decomposition across (a) detection, (b) fragmentation, (c) near-miss IoU, and (d) merging, with SQ and RQ compared to the inter-annotator reference (SQ 0.634, RQ 0.536)
  - **H2 check:** PQ by `r/R` bin, to judge whether limb correction is doing its job (it's always on in this spec; the ablation comes later)
  - a gallery of the 8 worst and 4 best `report` images (GT vs pred, instances coloured), with a failure-mode tag on each
  - 3–5 ranked hypotheses for Spec 03, each with the evidence behind it and a proposed test
  - a reproducibility checklist: commands to regenerate every number

## Acceptance criteria
- `pytest` passes, with new tests for `postprocess` (exclusive, non-empty masks; a known toy case) and for resuming from a checkpoint.
- The Mac smoke test passes.
- Both CSVs validate, and at least one has been submitted to Kaggle.
- No threshold was chosen on the `report` half.
- Nothing from `data/` or `outputs/` is committed except split CSVs, `.gitkeep`, and READMEs.
