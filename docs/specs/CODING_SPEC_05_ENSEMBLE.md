# CODING_SPEC_05_ENSEMBLE — ensemble the two architectures we already have, refit post-processing per fold, and push local CV PQ past 0.3662

**Phase 3, cycle 3. Time box: Nov 2–10, freeze Nov 11.** Read `CLAUDE.md`, `reports/04_model_side_report.md`, `scripts/run_e005.py`, `scripts/run_e009.py`, and `scripts/run_e007_eval.py` first.
Training is allowed in this spec, but only as described here (E012 always; E015 only with Nicky's written go-ahead, see §7).

## 1. Goal and reality check

**Operational goal:** push **local 5-fold day-grouped CV PQ** as far past **E009's 0.3662 ± 0.0026** as genuine modelling allows. Local CV is the only promotion and decision metric in this spec. The Kaggle public LB is not part of any decision (§1.1).

**Reference point, not a target or a ceiling:** the most transparent outside result we found is a 10-model ensemble (5-fold ResNet34 + 5-fold EfficientNetV2-S U-Nets, heavy TTA) reporting **OOF PQ 0.45 / public 0.39** (§1.1, item 5). That shows what a bigger, more diverse ensemble can plausibly add. If local CV keeps improving past it, keep going. If we plateau below it, that is an honest result, not a failure. Their OOF protocol isn't ours (scoring per annotator, folds, and IoU handling may differ), so compare loosely.

What the repo says about where the headroom is (all verified from `data/interim/*.json` and `experiments.csv`):
- E009 pooled: **SQ 0.6773, RQ 0.5407** (TP 3825 / FP 2124 / FN 4374). The human SQ ceiling is **0.6366** (E008, 296 multi-annotator images, 743 comparisons). SQ is already above it, so **the remaining headroom is almost entirely RQ**: there are about 2× as many FN as FP.
- The E006/E007 raw paths show both models find more real filaments than E004 (raw TP +60/−19 for E006, **+53/+52 for E007** on folds 0/1 with identical post-processing). Averaging two architecturally different models is the textbook way to keep those extra TPs while cancelling each model's own idiosyncratic FPs. **That has never been tested.** Spec 04 only asked whether E007 could *replace* E004.
- E005/E009 keep **cfg1 and the shape gate fixed from E003** (`threshold t=0.925, closing 2, min_area 418`; gate elongation ≥ 1.5). E003 tuned those on **E002's** fold-0 probabilities, not E004's. When `run_e007_eval.py` refit them for E004 itself, it chose very different settings (fold 0: `threshold t=0.575, closing 3`; fold 1: `hysteresis 0.6/0.95, closing 2`). So the production pipeline is running E004 through another model's thresholds. Report 04 ranked fixing this as hypothesis #1 for Spec 05.

### 1.1 Why the public leaderboard is not a target (record this; don't re-chase it)

Our own submissions track local CV in **rank order but not level**: E005 local 0.3395 → public **0.28**; E009 local 0.3662 → public **0.31**; E003 (single fold-0 model, local 0.3819 on one report half) → public **0.32** (Spec 04 §1). The public split is about 90 images, so differences of ±0.01–0.03 there are noise. The leaderboard's displayed top scores (about 0.60–0.70 for ranks 1–5, about 0.55–0.60 below that) are **not evidence of what's achievable**. On 2026-11-02 Nicky pulled public notebooks for this competition with `kaggle kernels pull` and found the following. This was not reproduced in this repo, and the notebooks are not committed.

1. **Evaluator defect (artkomissar, "Please fix the Leaderboard", 53 votes).** Submitting only 5 of 180 test images with *empty* masks scored **public 1.00**. One empty mask per image for all 180 images scored **0.88**, and seven empty masks per image scored **0.70**. The public evaluator does not penalise missing or near-empty predictions the way instance PQ should.
2. **The defect caps genuine entries too (phuongncn, Mask R-CNN + U-Net refiner + TTA, LB 0.70).** The author writes that the LB "is preliminary and currently has a known empty/missing-prediction defect", and that "many deterministic morphology, threshold, merge/split, confidence-fusion, and consensus variants tied 0.70". Near that plateau, the score stops tracking model quality.
3. **Fabricated submission (lamhuy8904, "0.55+" in the title).** The notebook contains **no training loop**. Its U-Net class and dataset loader are never trained or called. Inference looks up precomputed RLE strings in an LZMA/base85-encoded blob named `CHAMPION_PAYLOAD` ("verified 0.55 on Leaderboard"), and gives any image missing from the blob a trivial filler mask to fake 180/180 coverage. The full MAGFiLO dataset is public (Spec 04 §1), so this looks like leaked-label exploitation, not modelling.
4. **The score displays don't reconcile (prvsiyan audit).** One notebook showed "0.93" in one Code-tab snapshot and "0.34" in another. The same author's straightforward honest pipeline scored an official public **0.12**.
5. **Independent confirmation of our ceiling (dariushafshar, "Filament PQ: Two Experts Agree at 0.34").** This matches E008's inter-annotator result. Their transparent 10-model ensemble scored **OOF 0.45 / public 0.39**, the reference point in §1.

**Operating rule:** treat every displayed public score above about 0.45 as unverified, and some of them as fabricated. Never optimise toward the public LB or validate against it. **Never submit sparse, empty, or near-empty predictions, or anything that resembles exploiting the defect.** That breaks competition integrity, risks disqualification, and would evaporate if the organisers patch the evaluator before final judging. The public LB is used only as a smoke test: a validated CSV that scores far below its local CV (say < 0.20) suggests a bug in our pipeline.

## 2. Comparison protocol P5 and decision rule (applies to every experiment)

### 2.1 Protocol P5: nested, per-CV-fold refit of the whole post-processing chain
P5 extends E005/E009's day-grouped 5-fold CV over the **707 OOF images** (same `GroupKFold(n_splits=5)` over capture date, same stem ordering, so the CV folds are identical to `E005_cv_results.csv`). Inside each CV fold, **everything** is fit on the 4/5 train portion and scored on the held-out 1/5:
1. **cfg1** from a reduced grid built around every winner seen so far: threshold `t ∈ {0.55, 0.575, 0.6, 0.625, 0.65, 0.7, 0.8, 0.925}`; hysteresis `(t_low, t_high) ∈ {0.5, 0.6} × {0.9, 0.95, 0.975}`; `closing_radius ∈ {1, 2, 3}`; `min_area = 418`. That's 42 configs instead of the full 372. `t_high = 0.975` is new, because 0.95 sat at the grid edge in every hysteresis winner.
2. **Shape gate** with the same elongation × length grid as `run_e007_eval.py`.
3. **Scorer** `fit_scorer(seed=0)` and cutoff `c` from `C_GRID`.
4. **Re-growing** `(t_mask, d)` over `{0.5, 0.6} × {8, 12, 16}`. `d = 8` was at the grid edge in E008. To bound cost, tune on a fixed, seeded, day-grouped subsample of at most 150 train-portion images.

**Cost control, which is required to make P5 feasible on the Mac.** Matches at IoU > 0.5 are unique. So for a fixed cfg1, compute each image's component table **once**: per component and per annotator, the matched IoU or none, plus the GT count. PQ for *any* keep-subset (shape gate, scorer cutoff) then comes from summing per-component statistics, with no re-rasterising. A fold's choice of cfg1/gate/c becomes a sum over cached per-image statistics. Only regrow needs fresh mask work. Estimate per model variant: about 1 h for the cfg1 sweep (42 configs × 707 images), minutes for gate/scorer/c, about 80 min for regrow tuning, and about 20 min for the final held-out pass. **Roughly 2.5–3 h of Mac CPU per variant.** Stream one image at a time (`CLAUDE.md`).

**Reported per variant:** PQ/SQ/RQ mean ± std over the 5 CV folds, pooled PQ/SQ/RQ with TP/FP/FN (exactly how E005/E009 report), per-fold chosen cfg1/gate/c/regrow, and FP/FN class counts.

**Never share post-processing between variants.** Each variant (E004 alone, E004+E007, and so on) gets its own full P5 refit, because averaging changes probability calibration. Baseline and candidate always use the **same CV folds and the same held-out images**, so differences are paired.

**Determinism is a precondition.** In Spec 04, the *same* E004-f0 OOF through the *same* `full_path` scored **0.3886** in E006's run and **0.3808** in E007's run (`E006_decision.json` vs `E007_decision.json`: same cfg1 and gate, but `c` came out 0.325 vs 0.35). That's a 0.008 swing from the evaluation alone, which is as large as some effects we want to detect. Before E010 is reported, find the source (scorer fitting, set ordering, threading) and fix it. P5 run twice on one CV fold must give bit-identical results (§6).

### 2.2 Decision rule (pre-registered)
Paired deltas against the current reference (E010 once it exists), on the same 5 CV folds:

| Change type | Promote only if **all** hold |
|---|---|
| **Adds an ensemble member** (E013, E015): doubles inference cost and adds reproducibility burden | mean paired ΔPQ ≥ **+0.010**; ΔPQ > 0 on ≥ **4 of 5** CV folds; pooled SQ falls by no more than **0.005**; image-level paired bootstrap (2000 resamples, seed 0) 95% CI of pooled ΔPQ excludes 0 |
| **Pipeline-only, no new parameters at inference** (E010 refit, E014 4th TTA view) | mean paired ΔPQ ≥ **+0.005**; > 0 on ≥ 4/5 folds; SQ tolerance 0.005; bootstrap CI excludes 0 |
| **Early read** (E011, on model folds 0–1 only; no longer gates spend, §4) | paired ΔPQ ≥ **+0.008** on the 284-image subset **and** > 0 on both model folds |

Why these bars and not Spec 04's +0.02:
- Spec 04 compared **2 report halves of about 71 images each**. P5 pairs **all 707 images over 5 folds**, which is far more power, so a smaller bar is still clearly above noise. E009's CV std is 0.0026 and E005's is 0.0041. +0.010 is about 4× E009's std and above the 0.008 evaluation swing above. The swing will be fixed, but the bar shouldn't depend on that.
- **SQ tolerance instead of "SQ must not fall":** Spec 04 rejected E007-f0 at **+0.0243 PQ** for an SQ dip of **−0.0029**, which is smaller than E009's cv_sq_std (0.0031). SQ is already above the human ceiling (0.677 vs 0.637). A small SQ dip paid for with more TPs is a good trade under PQ. A drop beyond 0.005 still blocks promotion.
- **Public LB is not in the rule.** Not as a tiebreak either.

## 3. Experiments (ranked by expected value per GPU-hour)

### E010 — P5 re-baseline of E004 alone (Mac, no GPU, ~3 h) — do this first
**Hypothesis:** refitting cfg1 and the shape gate per CV fold for E004's own probabilities beats E009's E003-inherited `t = 0.925` pipeline.
- Implement P5 (§2.1) and run it on E004's 5-fold OOF.
- Compare to E009 under the pipeline-only bar. **Whatever the outcome, E010 becomes the reference for every later comparison in this spec**, because it's the same model under the same protocol the ensembles will use. If E010 < E009, report it plainly. The final submission can still use E009's pipeline for E004-only, but ensemble comparisons are against E010.
- Also produce `outputs/submissions/E010.csv`. The test path is the same as E009's: 5-fold mean of test probs, final post-processing fit on all 707 images.

### E011 — E004+E007 ensemble gate on model folds 0–1 (Mac, no GPU, ~1.5–2 h)
**Hypothesis:** averaging E004's and E007's probability maps beats either alone through the full pipeline. Two folds are enough to decide whether to spend GPU money on E007 folds 2–4.
- Verified state: `outputs/E007-f0/` and `outputs/E007-f1/` each have `checkpoints/{best,last}.pt`, 142 OOF maps, and 180 test maps (3-view flip TTA, EMA weights). E004 has all five folds.
- Restrict to the **284 images of model folds 0 and 1**. Run P5's nested refit with a day-grouped 5-fold CV inside that subset on three variants: E004 alone, E007 alone, and **mean(E004, E007)** (equal weights, pre-registered). Each gets its own refit, on the same inner folds.
- Early read (§2.2): report whether mean ≥ E004 + 0.008 paired and > 0 on both model folds. Training no longer waits on this, since Nicky approved launching everything on 2026-11-02.
- **Calibration diagnostic (report only, no tuning):** compare the scorer's input features (component max/mean probability, area, elongation) between single-member OOF and the two-member mean on the same 284 images. Report KS statistics. The earlier analysis of E005/E009's ~0.06 local-to-public gap blamed this mismatch: scorers are fit on single-model OOF but applied to 5-model-averaged test maps (verified in `run_e005.py`/`run_e009.py`: `np.mean(probs_list)` at test time vs one fold's OOF at fit time). This measures how big the shift is. **It is a hypothesis, not a verified cause.**
- Optional if the Mac is idle: the same comparison with E006 (same architecture as E004, hard negatives) as the second member. We expect low diversity, so it is not a priority.

### E012 — finish E007 on folds 2, 3, 4 (RunPod RTX 4090, ~1 h per fold, ~$2–2.5 total; approved, $7.59 available, §7)
- `configs/E007-f{2,3,4}.yaml`: copies of `E007-f0.yaml` with only `experiment_id`/`fold` changed (efficientnet-b4, 6000 iterations, EMA 0.999, batch 8, crop 512, AMP, seed 42, PQ checkpoint selection every 1000 iterations, `infer_tile_size 1024`, `overlap 128`). Keep **3-view TTA**, identical to f0/f1, so the five folds stay comparable.
- Timing from the logs: E007-f0 trained in **2753 s** and E007-f1 in **2684 s** (about 46 min each), plus about 3.2 min of OOF/test inference and pod setup. Report 04 puts both folds at about $1.5–2. Nicky's figure is about $2.40 spent so far. Reconcile these in the report.
- **Order:** all three folds start now, back to back on one pod via `scripts/run_queue_runpod.sh`, followed by E015 f0–f2 (Nicky approved on 2026-11-02). Stop the pod when idle. `rsync` outputs back as in Spec 04.
- One `experiments.csv` row per fold, with the checkpoint-selection val_pq. The real evaluation is E013.

### E013 — full 5-fold E004+E007 ensemble under P5 (Mac, ~3 h)
- OOF member mapping: an image in model fold *k* gets `mean(E004-fk, E007-fk)`, so neither model has seen it. Test: mean over all 10 maps (5 E004 + 5 E007).
- Run P5 and compare to E010 under the **ensemble-member bar**. If it's promoted, it's the new reference, and `outputs/submissions/E013.csv` is written and validated.
- Report the ensemble's per-fold chosen cfg1 next to E010's, to show where averaging moved the optimal thresholds.

### E014 — 4th TTA view (h+v flip) (GPU inference only, ~1.5 h T4 + ~20 min 4090; only if time allows)
**Verified current state:** 3-view flip TTA (identity, h-flip, v-flip, averaged) is **hard-coded on** in `scripts/train.py` (`make_prob_fn(..., tta_flips=True)` at both the checkpoint-selection and the OOF/test inference call sites). It isn't a config key. So every saved E004/E006/E007 OOF and test map already uses 3-view TTA. There's no separate Kaggle inference config: test maps are written at the end of each training run, and submissions are built locally.
- Add a `tta_views: 3 | 4` config key (default 3, so old configs reproduce bit-for-bit) to `unet.make_prob_fn`, and a `scripts/infer.py --config ... --checkpoint best.pt --tta_views 4` that regenerates OOF and test maps into `outputs/<id>-tta4/` from existing checkpoints, using EMA weights the same way `train.py` does.
- Cost: E004 OOF and test inference took about 11.4 min per fold on a T4 with 3 views, so about 15 min per fold with 4, about 75 min for 5 folds on Kaggle. E007 takes about 4.3 min per fold on the 4090, so run it on the same pod session as E012.
- Expected gain is small (+0.000 to +0.005), because the 4th view mostly duplicates the information from the first three. Promote under the pipeline-only bar against whichever of E010/E013 is current. **Rotations aren't in scope**, because the solar images have a fixed N–S orientation and we haven't checked that rotation is label-preserving for filament morphology.

### E015 — a third, architecturally distinct model on all 5 folds (APPROVED by Nicky 2026-11-02: train now, in parallel with E012)
Nicky chose to train E015 immediately instead of waiting for E013, to use the idle GPUs. Training no longer depends on E013, but **promotion still does**: E015 joins the ensemble only if it clears the ensemble-member bar against E013 (§2.2). If E013 itself isn't promoted, evaluate E004+E007+E015 against E010 instead.
- Diminishing returns are expected. A third member typically adds a fraction of what the second did. If E013's gain was about +0.010, E015 is unlikely to clear its own +0.010 bar. The +0.010 bar applies to E015 against E013.
- Candidate (verify before configuring): U-Net with a **RegNet encoder via smp's timm-universal interface (`tu-regnety_040`)**. That's a different family from ResNet34 and EfficientNet-B4, at a similar parameter count. Fallback: `resnext50_32x4d` (closer to E004, so less diversity). **Hard constraint carried over from Spec 04:** the encoder must expose a **stride-2 feature map**, which rules out ConvNeXt, Swin, and MiT/SegFormer. Add a unit test that builds the model with `smp==0.6.0` and asserts the reduction of the first skip feature is 2 before any training run.
- Same recipe as E007 (6000 iterations, EMA, PQ selection, seed 42, 3-view TTA). `configs/E015-f{0..4}.yaml` use `encoder_name: tu-regnety_040`. `tests/test_encoder_stride.py` checks the stride-2 skip, and `scripts/run_queue_runpod.sh` runs it before any E015 training. RunPod trains folds 0–2 after E007 f2–f4. Kaggle trains folds 3–4 in two parallel sessions (`scripts/gen_e015_kaggle_artifacts.py`).

### E016 — final build (Mac, ~1 h)
- Rebuild the promoted pipeline (whichever of E010, E013, or E013+E014/E015 won) from scratch with one command. Validate the CSV with `scripts/validate_submission.py`. **Don't submit.** Nicky chooses the two final selections: the primary is the highest P5 result, and the backup is the best E004-only pipeline (E010 or E009).

### Not in this spec, and why
- **Exploiting the evaluator defect in any form** (§1.1).
- **Instance-vote or fold-wise test-time combination** in place of probability averaging. It's a plausible fix for the calibration mismatch, but local CV can't validate a 5-fold test-time combiner, and we've stopped using the public LB to choose. Revisit only if E011's diagnostic shows a large feature shift (KS > 0.2 on the probability features).
- **Learned ensemble weights or stacking:** only equal weights are pre-registered. A weight sweep on the train portion is a cheap add-on only if E013 is promoted and time remains.
- **Bigger crops or a global-context channel** (report 04 hypothesis #3): needs new training on every fold. Spec 06, if there is time.

## 4. Parallel schedule

| Slot | RunPod 4090 | Kaggle | Mac (one heavy job at a time, 8 GB) |
|---|---|---|---|
| Nov 2 (now) | One pod, sequential queue (`scripts/run_queue_runpod.sh`): E007-f2 → f3 → f4 → E015-f0 → f1 → f2. About 1 h each, about 6–7 h total | Two parallel T4 sessions: E015-f3, E015-f4 (estimated 2.5–3 h each) | Fix eval determinism; implement P5 + tests; **E011** |
| Nov 3 | Stop the pod as soon as the queue log says finished; rsync the 6 output dirs back | Free (E014 E004 re-inference, about 75 min, once `scripts/infer.py` exists) | **E010** |
| Nov 3–4 | Optional short pod: E007/E015 4-view inference (E014) | — | **E013** (needs E007 f0–f4 synced) |
| Nov 4–6 | — | — | **E015** under P5 (E004+E007+E015); **E014** eval |
| Nov 7–9 | — | — | Report drafting; buffer for re-runs |
| Nov 10 | — | — | **E016** final build; report done |
| Nov 11 | — | — | **Freeze** (RESEARCH_PLAN §4) |

E011 no longer gates GPU spend, because Nicky approved training everything now. It still runs first on the Mac as an early read on whether the ensemble helps. **Order dependencies:** E010, E011, E013, and E015's evaluation all need P5 and the determinism fix. E013 needs E012.
**Compute budget (estimates):** RunPod about 6–7 pod-hours for E012 + E015 f0–f2, about $4.5–6 of the $7.59 left (Nicky will top up if needed). Kaggle about 5–6 GPU-h for E015 f3–f4, plus about 1.5 GPU-h for E014, out of about 15.3 h left this week. Mac about 12–15 h of CPU in total.

## 5. Deliverables
- `src/solar_k/ensemble.py`: streaming per-stem member averaging (equal weights; member → fold mapping for OOF; all-folds mapping for test). Never holds more than one image's maps.
- `src/solar_k/cv_pipeline.py`: P5, with per-image cached sufficient statistics, nested refit of cfg1/gate/scorer/c/regrow, paired-delta + bootstrap reporting. `scripts/run_p5.py --experiment E0xx --members E004 [E007 ...] [--model-folds 0 1]`.
- `tta_views` config key + `scripts/infer.py` (E014). `configs/E007-f{2,3,4}.yaml`, `configs/E015-f{0..4}.yaml`, `scripts/run_queue_runpod.sh`, `scripts/gen_e015_kaggle_artifacts.py`, `tests/test_encoder_stride.py` (written 2026-11-02 for the launch).
- `experiments.csv`: rows E010–E016, decisions filled per §2.2. **Backfill the blank `kaggle_public_score` cells**: E003 0.32, E005 0.28, E009 0.31. Add a note to E005/E009's rows that public scores aren't decision inputs (§1.1).
- `data/interim/E0xx_p5_summary.json` and `E0xx_p5_cv.csv` per variant, in the same fields as `E009_eval_summary.json` plus per-fold choices and paired deltas.
- `reports/05_ensemble_report.md`:
  - **must open by repeating §1.1's leaderboard-trust findings** (notebook refs, the fabricated-payload finding, the 0.45/0.39 reference), so the final writeup can't be read as underperforming a legitimate 0.55 or 0.70. Those numbers are not trustworthy evidence of what's achievable.
  - the determinism root cause and fix
  - E010 vs E009 (how much the per-fold refit is worth)
  - the E011 gate table and calibration diagnostic
  - the E013 ensemble vs E010, per CV fold, with bootstrap CI
  - E014/E015 results or the reason they were skipped
  - compute used (GPU-h per platform, RunPod $), verified facts separated from assumptions (`CLAUDE.md`)
  - ranked hypotheses for anything after the freeze

## 6. Acceptance criteria
- `pytest` passes, including new tests for: (a) PQ from summed per-image cached statistics equals `aggregate_evaluation` on the same predictions; (b) `ensemble.py` member/fold mapping never gives an OOF image a model that trained on it; (c) the encoder stride-2 assertion (E015 only); (d) `tta_views=3` reproduces the existing `make_prob_fn` output exactly.
- **Determinism:** P5 run twice on the same input gives identical JSON, and the Spec 04 0.3886/0.3808 discrepancy is explained in the report.
- Every comparison is paired on identical CV folds. Nothing (cfg1, gate, scorer, c, regrow, ensemble weights) is fit on images it's scored on. No post-processing config is shared between variants.
- Every training run has an E### id, an explicit seed (42), a config in `configs/`, and an `experiments.csv` row. The Mac streams one image at a time.
- No data, probability maps, checkpoints, or credentials are committed (`git status` checked before every commit, per `CLAUDE.md`).
- Every CSV passes `validate_submission.py`. **Nothing is submitted to Kaggle without Nicky's go-ahead.** The public LB is never used to choose between variants.

## 7. Decisions from Nicky (2026-11-02) and what's still open
1. **RunPod budget: $7.59 remaining** (confirmed by Nicky). Spending is approved, and Nicky will top up if more is needed. **E012 (about $2–2.5) is approved and can start now.** E012 plus E015 (about $4) fits in $7.59 with about $1 to spare. Stop the pod whenever it's idle.
2. **Kaggle GPU: 14:41 of 30 h used this week**, which leaves about 15.3 h (read as used/quota; confirm in the Kaggle UI before launching). That covers E014's E004 re-inference (about 1.5 h) and E015 folds 3–4 (about 3.5 h).
3. **Third architecture (E015): approved to train now** (Nicky, 2026-11-02), along with everything else, with the training queue organised across RunPod and Kaggle (§4). Promotion into the final ensemble is still decided only by §2.2.

## 8. Addendum (2026-11-04): the official rubric and the E013/E015 public scores

### 8.1 Rubric (competition Overview → Evaluation, updated Aug 7, 2026; screenshots from Nicky)
- **Quantitative, 70%:** (a) Panoptic Quality (the leaderboard metric; it replaced the struck-through "Mean Dice"); (b) the **distribution of Dice scores**; (c) the **distribution of IoU scores**; (d) the **distribution of one-to-many and many-to-one relations** between GT and predicted segmentations, i.e. fragmentation and merging.
- **Qualitative, 30%:** (a) a detailed description of the whole pipeline, from preprocessing to the final prediction, including the architecture; (b) the **apparent morphology of predicted segmentations on H-alpha images**; (c) **code quality: modularity and documentation**.
- The leaderboard is "one of the considerations". The Evaluation page also notes further verification for edge cases, which may close the empty-mask defect (§1.1).

Two consequences for this spec. First, public PQ is only one of four quantitative items, which is one more reason not to chase the board. Second, fragmentation/merging and prediction morphology are graded directly, so they become reported quantities and a tiebreak, not just error-analysis curiosities.

### 8.2 Changes to the protocol and deliverables
1. **P5 reports three more things per variant** (OOF, held-out CV folds, no tuning): per-image Dice distribution (`pixel_dice_mean` already exists in `evaluate.py`; add histogram and quantiles); per-TP IoU distribution (histogram plus share of TPs with IoU in 0.5–0.6, which are near-misses that a boundary shift loses); **split/merge counts**, meaning the number of GT instances overlapped by ≥ 2 predictions and the number of predictions overlapping ≥ 2 GT instances. Overlap is defined as intersection ≥ 20% of the smaller segment, fixed before looking at results. Count per annotator, the same way PQ does.
2. **Tiebreak in §2.2:** when two variants are within 0.005 PQ of each other, prefer the one with fewer split + merge events, then the simpler one (fewer ensemble members).
3. **New cheap experiment E017, fragment merging (Mac, no GPU, only if 8.2.1 shows fragmentation is common, say ≥ 5% of matched GT have ≥ 2 predictions):** join kept instances whose endpoints are within `g` px and whose spine directions differ by < `θ` (RESEARCH_PLAN H5). Grid `g ∈ {5, 10, 20}`, `θ ∈ {20°, 40°}`, tuned inside P5. Pipeline-only promotion bar, and it must not increase merges.
4. **Qualitative deliverables, owned by E016 and the report:**
   - an overlay gallery on H-alpha (extend `scripts/run_e003_gallery.py`): about 12 fixed, seeded OOF images covering each GONG site, quiet and active disk, and limb, showing GT vs prediction, plus the worst split and the worst merge case. Train OOF only, never test images.
   - a one-figure pipeline diagram: preprocess → 3 U-Nets with flip TTA → probability mean → threshold/hysteresis → shape gate → learned scorer → watershed re-growing → RLE.
   - code quality: module docstrings in `src/solar_k/`, a README "reproduce from scratch" section, one end-to-end notebook that only imports from `src/`, and `pytest` green. Remove dead scripts or mark them superseded.
5. **Report sections** map one-to-one onto the rubric's seven bullets, so a grader can find each one.

### 8.3 What the E013 / E015 public 0.37 means
Both scored **public 0.37**: the E004+E007 ensemble (E013) and the three-model ensemble (E015). That's +0.06 over E009's 0.31 and +0.05 over E003's 0.32, our previous best.
- **The ensemble gain is real and shows up off-CV.** Rank order still holds, which is consistent with local CV. Read the P5 numbers for E013 and E015 against E010 before attributing it: the public split is about 90 images, so ±0.03 is noise there.
- **The third model adds nothing visible at public resolution.** Decide E015 vs E013 **only** on P5 (§2.2: +0.010 bar, plus the split/merge tiebreak in 8.2.2). If P5 doesn't clear the bar, keep **E013** as the primary. It's simpler, cheaper to reproduce, and easier to describe, which counts in the 30%.
- **Next steps, in order:** (1) finish the P5 rows for E010, E013, and E015 in `experiments.csv`, plus `kaggle_public_score` 0.37 for E013/E015; (2) add 8.2.1's distributions and split/merge counts for the leading variant; (3) run E017 only if fragmentation clears the threshold; (4) E014 (4th TTA view) only if time remains; (5) E016 final build, gallery, and report by Nov 10, freeze Nov 11.
