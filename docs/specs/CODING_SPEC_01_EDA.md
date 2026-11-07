# CODING_SPEC_01_EDA — dataset audit, metric, and validation split

**Phase 0. Time box: about 1 day. No model training in this spec.**
Read `CLAUDE.md` first. Put this file in `docs/specs/`.

## Goal
Establish verified facts about the data. Build the three foundations everything else depends on: a **local scorer** that matches the official one, a **submission writer and validator**, and a **leakage-safe CV split**. Also measure the **inter-annotator PQ ceiling**.

## Prerequisite
The organisers' self-evaluation notebook is in `data/external/` (git-ignored). Read it carefully before writing `metric.py`. Document in the report its exact matching rule, per-annotator handling, aggregation across images, and edge cases.

## Outputs
- `src/solar_k/` modules: `io.py` (loading images and annotations), `ids.py` (filename parsing), `masks.py` (polygon→mask, RLE encode/decode), `metric.py` (PQ scorer), `splits.py`
- `scripts/eda.py`, `scripts/make_splits.py`, `scripts/validate_submission.py`
- `tests/test_metric.py`, `tests/test_rle.py`, `tests/test_splits.py`
- `data/splits/folds_v1.csv`
- `reports/01_dataset_report.md` plus figures in `reports/figures/01_*`
- Pinned `requirements.txt` (numpy, pandas, pycocotools, opencv-python-headless, scikit-image, matplotlib, pyyaml, pytest)

## Tasks

### 1. Inventory
- The file tree of `data/raw/` (2 levels), counts, total sizes, and a SHA-256 of each file (store the hashes in `data/interim/hashes.csv`).
- For every image: format, mode, dtype, shape, min/max, and whether it's truly grayscale.
- **Exact duplicate check by hash** within train, within test, and **between train and test**. Also a near-duplicate check (downsampled 64×64 correlation > 0.99) for train-train and train-test pairs.

### 2. Annotation schema
- Print the structure of `info`, `licenses`, `categories`, one full `images` entry, and one full `annotations` entry (truncate long polygon lists).
- List every key that appears in `images` and `annotations`, with its type and how often it's present. Look especially for annotator, spine, chirality, and observation-time fields.
- Segmentation encoding: polygon vs RLE, the number of polygon parts per annotation, and the vertex count distribution.
- **Explain the 1154 `images` entries vs 707 JPEGs.** Map every `images` entry to a physical file (by `file_name`). Report the count of entries per physical file (expected: 1, 2, or 3 annotators), entries whose file is missing, and files with no entry. State clearly whether repeated entries are different annotators of the same image.
- Category counts overall and per annotator.

### 3. Identifiers, time, and site
- Parse filename stems. Expected forms look like `20150125172714Mh` (timestamp + GONG site letter + `h`), possibly with a numeric prefix. Verify this on every train and test file and report any that don't parse.
- Distributions of year, month, day, and site for train vs test. Count how many test frames fall on a day (or within ±1 h) of a train frame. Report this only as a leakage risk; **don't use it**.

### 4. Instance statistics (train)
Per annotation record:
- instances per image (histogram)
- area, bbox size, elongation (skeleton length ÷ sqrt(area)), and minimum width
- distance from disk centre ÷ disk radius
- the fraction of instances that touch or overlap another instance from the **same** annotator (this matters for whether "connected components = instances" can work at all)
- instances partly or fully off-disk

### 5. Image statistics
- Fit the solar disk (centre, radius) per image and report the distribution.
- Radial intensity profile (limb darkening), with mean profiles per site.
- Global contrast and noise per image. Flag outliers such as clouds, saturation, or partial frames, and show the five worst.
- A gallery of 12 train images covering sites and years, with GT overlays coloured per instance and one panel per annotator when there are several.

### 6. Metric and submission plumbing
- `metric.py`: the PQ exactly as in the official notebook. Cover matching (IoU > 0.5), per-annotator averaging, and edge cases (no GT, no predictions, both empty). Also return SQ, RQ, TP, FP, FN, and per-match IoU/Dice.
- Tests: GT vs itself → PQ = 1; empty prediction → 0; a hand-built 2-instance case with a known answer; a merged-instance case; and a fragmented case.
- RLE round-trip: rasterise all train GT → encode → write a CSV in the submission format → `validate_submission.py` → decode → compare pixel-exact. Then score that CSV with `metric.py` against GT (expected PQ = 1 for a single-annotator image).
- `validate_submission.py` checks: exact columns, unique IDs, ID→image mapping against the test folder, every RLE decodes to 2048×2048, no empty masks, no overlapping masks within an image, and a clean reload. Report images with zero rows.

### 7. Inter-annotator agreement
For every physical image with ≥2 annotators, score annotator A as "prediction" against B (and the reverse). Report the mean and distribution of PQ, SQ, and RQ. This is our **human ceiling** reference.
Separate the two kinds of disagreement:
- **Boundaries:** IoU and Dice of matched pairs (SQ), and boundary distance.
- **Instance identity:** unmatched instances with their size and contrast, one-to-many cases (one annotator splits what another merges), and many-to-one cases.

End with an evidence-based recommendation for the training-label strategy: union, soft consensus, separate samples, or something else. Don't assume any of these is best.

### 8. Validation split
- `folds_v1.csv`: one row per **physical image**, with columns `file_stem, fold, site, date, n_annotators`.
- 5 folds, grouped by **observation day** (so frames from the same day share a fold), with site and annotator count balanced as well as possible.
- Tests: no physical image or day appears in two folds, and every train image is assigned.
- Report per-fold counts of images, instances, and sites.

### 9. Report: `reports/01_dataset_report.md`
Sections: Verified facts (each with the script and function that produced it) · Assumptions and unknowns · Answers to tasks 1–8 · Leakage risks · Implications for modelling (input resolution, instance separability, label strategy for multiple annotators, preprocessing needs) · **Prioritised hypotheses for Spec 02** · Open questions for Nicky.

## Acceptance criteria
- `pytest` passes.
- The GT-as-submission round-trip is pixel-exact and scores PQ = 1 on single-annotator images.
- `git status` shows no data, masks, or CSVs staged.
- The report states the inter-annotator PQ and the reason for the 1154-vs-707 gap, backed by evidence.
- Runs on CPU in under about 15 minutes on an 8 GB M2 Mac, streaming one image at a time.
