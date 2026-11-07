# CODING_SPEC_03 — suppress confident false positives, and complete the 5 folds

**Phase 3, cycle 1. Time box: Nov 1–5.** Read `CLAUDE.md` and `reports/02_baseline_report.md` first.

## Evidence this spec acts on (from report 02)
- E002: local PQ 0.269, **public LB 0.24**. Local validation tracks the LB (E001: 0.035 local vs 0.03 public).
- SQ (0.651) is at the human level. The gap is RQ, which is lost mainly to **`isolated` FPs**: 807 of 1075, median prob 0.957, often on bright, textured active-region (plage) areas. Thresholding alone can't remove them.
- PQ vs `t` was still rising at the grid edge (t = 0.70).
- Val Dice is very noisy (20 images, pixel metric). The best checkpoint was at iteration 2000 of 6000.

**Caveat for the analysis:** on **single-annotator** images, an `isolated` FP can't be told apart from a real filament that the one annotator skipped. Report the `isolated` share separately for single- and multi-annotator images.

## Experiments

### E003 — post-processing only (local, no training, fold-0 OOF probs)
Run the steps cumulatively and report each one's increment on the `report` half (tuning stays on `tune`):
1. **Wider threshold grid:** single `t` from 0.50 to 0.95 in steps of 0.025, plus the hysteresis grid widened to the same range.
2. **Component feature table:** one row per predicted component, built in `src/solar_k/components.py`. Features:
   - shape: area, skeleton length, elongation, mean and max width, solidity, orientation
   - probability: mean, max, and p90 under the mask
   - **context:** mean preprocessed intensity inside the component, mean intensity in a 15 px ring around it, their contrast, the local texture (std) in a 64 px window, and the fraction of bright pixels nearby (a plage indicator)
   - position: r/R
   
   Label each component as a TP (matched IoU > 0.5 for at least one annotator), and record its best IoU per annotator.
3. **Shape gate (from report 02, H1):** tune minimum elongation and minimum length thresholds.
4. **Learned component scorer:** logistic regression and a small gradient-boosted model (sklearn `HistGradientBoostingClassifier`, depth ≤ 3). Target: P(match). A second regressor predicts E[IoU | match]. Keep a component if **P(match) × E[IoU] > c**, with `c` tuned on `tune`; theory says about PQ/2. Fit on `tune`, evaluate on `report`. Report feature importances, and say whether context features beat shape features.
5. Wire `pixel_iou_dice` into `aggregate_evaluation` (report 02 gap).

Write `outputs/submissions/E003.csv`, using the fold-0 model on test with the best post-processing. **Don't submit it yet**; wait for Nicky.

### E004 — complete 5-fold training (Kaggle), same model as E002 with two changes
- **EMA of weights** (decay 0.999), used for evaluation and saving.
- **Checkpoint selection by instance PQ**: every 1000 iterations, run full-resolution inference on a fixed 40-image subset of the fold's val split, post-process with E002's tuned settings (t = 0.7, r = 2, min_area = 418), and score PQ. Keep the best by PQ, and also keep the last checkpoint.
- Iterations: 4000.
- Train folds 1–4, plus fold 0 again under the new settings (E004-f0), so all five are comparable. One Kaggle run per fold, or several folds per run within `--max_hours`. Each run saves OOF probs for its fold and test probs.
- Result: **5-fold OOF probabilities for all 707 train images** and 5 test probability maps.

### E005 — ensemble + component scorer on full OOF (local)
- Test probs = the mean of the 5 fold models' probs.
- Re-fit the E003 post-processing and component scorer on **all 5-fold OOF**, with grouped CV over folds: fit on 4 folds, evaluate on the 5th. Report mean ± std PQ across folds. This is our main CV number from now on.
- **Known mismatch:** the scorer is fit on single-model OOF but applied to ensemble test probs, which are smoother. Report a check: on fold 0, compare the component feature distributions from the single model against a 2-model average if one is available. Otherwise flag it as a limitation.
- Write `outputs/submissions/E005.csv`.

## Not in this spec
Hard-negative training on plage regions, limb-correction ablation, and encoder upgrades. Those are Spec 04 candidates, decided by the E003 to E005 evidence.

## Outputs
- `src/solar_k/components.py`, `src/solar_k/scorer.py`, tests (feature extraction on a toy mask; the scorer's keep rule)
- Rows E003, E004-f0…f4, and E005 in `experiments.csv`, including the Kaggle public scores Nicky reports (E001 = 0.03, E002 = 0.24; fill these in now)
- `reports/03_fp_report.md`:
  - an increments table for E003 steps 1–4
  - the scorer's feature importances
  - FP class shares before and after, split by single- and multi-annotator images
  - per-fold PQ for E004 and E005
  - a gallery of the remaining worst FPs
  - ranked hypotheses for Spec 04

## Acceptance
- `pytest` passes.
- No parameter or scorer is fit on the data it's evaluated on.
- All CSVs validate.
- Fix report 02's acceptance note: both E001 and E002 **were** submitted.
