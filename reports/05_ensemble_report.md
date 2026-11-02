# Report 05 — ensemble, per-fold post-processing refit (P5), third architecture

Covers `docs/specs/CODING_SPEC_05_ENSEMBLE.md`. **Status: in progress (started 2026-11-02).** Sections marked
PENDING are filled in as runs finish.

## 1. Read this first: the public leaderboard is not evidence of what is achievable

This section repeats Spec 05 §1.1 so this report (and the final write-up) can't be read as underperforming
a legitimate 0.55 or 0.70. All items were found by Nicky on 2026-11-02 with `kaggle kernels pull` on public
notebooks for this competition. **They were not reproduced in this repo, and the notebooks are not committed.**

1. **Evaluator defect** (artkomissar, "Please fix the Leaderboard"). Submitting only 5 of 180 test images with
   *empty* masks scored public **1.00**. One empty mask per image for all 180 images scored **0.88**, and seven
   per image scored **0.70**. The public evaluator doesn't penalise missing or near-empty predictions the way
   instance PQ should.
2. **The defect caps genuine entries too** (phuongncn, Mask R-CNN + U-Net refiner + TTA, LB 0.70). The author
   reports a known empty/missing-prediction defect, and says that many different post-processing variants all
   tied at 0.70. Near that plateau the score stops tracking model quality.
3. **Fabricated submission** (lamhuy8904, "0.55+"). The notebook has **no training loop**. Inference looks up
   precomputed RLE strings in an LZMA/base85 blob called `CHAMPION_PAYLOAD` and fills any missing image with a
   trivial mask. The full MAGFiLO dataset is public, so this looks like leaked-label exploitation, not modelling.
4. **Score displays don't reconcile** (prvsiyan audit). One notebook showed 0.93 in one snapshot and 0.34 in
   another. The same author's honest pipeline scored an official public 0.12.
5. **Independent confirmation of our ceiling** (dariushafshar, "Filament PQ: Two Experts Agree at 0.34").
   This matches E008's inter-annotator result. Their transparent 10-model ensemble (5-fold ResNet34 + 5-fold
   EfficientNetV2-S U-Nets, heavy TTA) reports **OOF PQ 0.45 / public 0.39**. That's the loose reference point
   for this spec. Their OOF protocol isn't ours, so the comparison is loose.

Our own submissions track local CV in **rank order but not level**: E005 local 0.3395 → public 0.28, E009 local
0.3662 → public 0.31, and E003 (local 0.3819 on one report half) → public 0.32. The public split is about 90
images, so ±0.01–0.03 there is noise. **Operating rule:** every displayed public score above about 0.45 is
unverified. We never optimise toward the public LB or validate against it, and never submit sparse, empty, or
near-empty predictions. Local 5-fold day-grouped CV is the only decision metric in this spec.

## 2. Evaluation determinism (Spec 05 §2.1 precondition)

### 2.1 The discrepancy
In Spec 04, the same E004-f0 OOF through the same `run_e007_eval.full_path` scored **0.3886** in the E006 run
(c = 0.325, TP/FP/FN 402/215/388) and **0.3808** in the E007 run (c = 0.35, 385/201/405). cfg1 and the shape gate
matched. Both runs used identical code: the Spec 04 session transcript shows no edits to `src/` or
`scripts/run_e00*.py` between the two launches (00:38 and 02:07 UTC on Nov 2), and the launch commands were
identical. Inputs on disk were also unchanged: the newest preprocessing-cache file is from Nov 1 22:55 and the
E004-f0 OOF maps are from Nov 1 18:19.

### 2.2 Root cause (verified)
The component feature `orientation` is the angle from `cv2.fitEllipse` (`components._shape_features`). For a
**degenerate contour** (all points collinear, e.g. a 1-px-wide straight component), OpenCV 5.0.0 returns
values that depend on uninitialised scratch memory, so they change with whatever OpenCV work ran earlier in
the same process. Evidence, with the commands in this session's scratch scripts:
- Same component, three passes in one process: angles {22.4986, 0.0, 0.0} with axes ≈ 0.001 px. Fitting one
  diagonal contour 20 times, with unrelated fits in between, gave **11 distinct angles** (0.0 to 135.04).
- Building E004-f0's fold-0 component tables, then E006-f0's, then E004-f0's again changed `orientation`
  (only that column) on 2 of the first 25 tune images.
- **Replaying the E006 run's in-process order** (E006-f0 candidate first, then E004-f0, cfg1/gate fixed to
  the values both runs chose) reproduced **both** E006-run numbers exactly: E006-f0 0.4013 (427/229/363) and
  E004-f0 **0.3886 (402/215/388), c = 0.325**. Running E004-f0 alone in a fresh process reproduces the E007
  run's **0.3808 (385/201/405), c = 0.35** exactly.
- Ruled out by direct tests: `fit_scorer` (bit-identical under OMP_NUM_THREADS 1/2/3/4/8, six concurrent
  processes, shuffled row order, and after an unrelated fit in the same process), `cv2.connectedComponents`
  labelling (identical across `cv2.setNumThreads` 0/1/2/8), and the tune/report split.

Why a few feature values swing PQ by 0.008: on E004-f0 fold 0 the tune-half PQ-vs-c curve is flat at the
top (0.41817 at c = 0.300, 0.41818 at 0.325, 0.41958 at 0.350). A small change in the scorer moves the
argmax by one grid step, and report-half PQ moves by about 0.005 per step. This is also why P5 averages over
5 CV folds rather than 2 report halves.

### 2.3 Fixes and tests
- `components._orientation`: if `cv2.minAreaRect` of the contour has a side < 1 px, use the second-moment
  angle of the contour points (deterministic, same [0, 180) range). Well-conditioned fits are unchanged.
  Test: `tests/test_components.py::test_orientation_deterministic_for_degenerate_contours` (it churns OpenCV
  memory between calls; raw `fitEllipse` fails that check on the same contour).
- Two latent hazards found on the way and fixed, even though neither caused this discrepancy:
  `preprocess.load_or_build` returned unrounded float32 on a cache miss but the float16-rounded array on a
  hit, so features depended on cache warmth (`test_load_or_build_miss_and_hit_are_bit_identical`).
  `tuning.grid_search` iterated a `set` of stems, so float summation order depended on `PYTHONHASHSEED`
  (`test_grid_search_independent_of_stem_iteration_order`).
- P5 itself: sorted stems everywhere, ties go to the first grid entry, worker results are reassembled by
  stem, and seeds are explicit. `test_fit_keep_rule_deterministic_across_processes` runs the gate/scorer/c
  fit in three processes with different hash seeds and thread counts and requires identical output.
- Run-twice check on real data (spec §6): PENDING, after E011 (`run_p5.py --cv-folds 0` twice with separate
  cold caches).

**Consequence for earlier numbers:** E003/E005/E009 were computed with the noisy feature, so their scorer
cutoffs may not reproduce to the last digit. The E009 per-image regeneration (§5) measures this.

## 3. Protocol P5 — implementation

`src/solar_k/cv_pipeline.py`, `src/solar_k/ensemble.py`, `scripts/run_p5.py`. PENDING detail.

## 4. E011 — E004 + E007 early read on model folds 0–1

PENDING.

## 5. E010 — E004 alone under P5 vs E009

PENDING.

## 6. E012 / E015 training

PENDING.

## 7. E013 — 5-fold E004 + E007 ensemble

PENDING.

## 8. E014 / E015 evaluation

PENDING.

## 9. Compute used

PENDING.

## 10. Verified facts vs assumptions

PENDING.
