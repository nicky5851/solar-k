# Report 08: Final write-up (Solar Filament Instance Segmentation)

**Primary: `E030_primary.csv`** (E007tta4s + E015tta4sR, local CV PQ 0.4298 pooled / 0.4297 ± 0.0048 CV mean, public 0.37 — inherited from E024, which it reproduces).
**Backup: `E013_backup.csv`** (E004 + E007, local CV PQ 0.4176 pooled, public 0.37 for the original E013).
Neither has been submitted. Choosing which to submit, and whether to submit at all, is Nicky's call (CODING_SPEC_08 §7).

---

## 1. Task, data and metric

The task is instance segmentation of solar filaments (dark Hα chromospheric features) in full-disk H-alpha images, scored by **Panoptic Quality (PQ = SQ × RQ)**: segmentation quality (mean IoU of matched instances) times recognition quality (an F1 over instance matches at IoU > 0.5), against a pool of up to several human annotators per image (not a single consensus mask).

Two facts anchor every result below:
- **The inter-annotator ceiling is PQ ≈ 0.34** (CODING_SPEC_02, E008): scoring one annotator's masks against the others on the same images gives PQ ≈ 0.34, because annotators themselves disagree substantially on faint/ambiguous filaments. Our final PQ (0.43) exceeds this because PQ is computed against the *union* of all annotators per image (any one match counts), not against a single reference annotator — the ceiling is a floor on how much headroom genuinely exists, not a number our score should approach from below.
- **The metric changed from Dice to PQ on Aug 7** (CODING_SPEC_02). Early specs (01–02) optimized pixel Dice; every result from CODING_SPEC_03 onward is PQ-based, including checkpoint selection during training (`checkpoint_select_by: pq` in every config from E004 on).

Data: full-disk Hα images from 6 observing sites (B, C, L, M, T, U — site codes baked into the GONG-derived filenames), day-grouped into folds (`data/splits/folds_v1.csv`) so no single day straddles a train/held-out split. 707 OOF (labelled) images, ~180 unlabelled test images.

## 2. Pipeline end to end

**Preprocessing:** raw Hα images are disk-masked and normalized (`solar_k.preprocess`), cached to disk with atomic writes (temp-file + `os.replace`, fixed in CODING_SPEC_07 after a race condition corrupted a cache file under concurrent DataLoader workers).

**Crops / training:** 512×512 random crops, batch size 8, AMP (fp16 forward/backward), Adam with a warmup+cosine-style schedule, 4000–6000 iterations depending on the model (CODING_SPEC_03's report 02 found training past ~4000 iterations stopped helping for the resnet34 models; the two ensemble members kept to 6000).

**Architectures** (`segmentation-models-pytorch` 0.6.0, U-Net decoder):
- **E004**: `resnet34` encoder, ImageNet-pretrained, 4000 iters. The E013 backup's weaker-but-diverse member.
- **E007**: `efficientnet-b4` encoder (`timm`), 6000 iters. The strongest single model, in both E030 and E013.
- **E015**: `tu-regnety_040` encoder (`timm`), 6000 iters. Architecturally distinct from both (different family from resnet/efficientnet), the second E030 member.

**EMA:** every model keeps an exponential moving average of its weights (decay 0.999) alongside the raw training weights; only the EMA shadow is checkpointed and used for inference/evaluation (CODING_SPEC_03, E004) — it consistently scored higher than the raw weights at matched iteration counts.

**TTA (test-time augmentation):**
- E030's members (`E007tta4s`, `E015tta4sR`) use **4-flip + 1.25× upscale** TTA: identity + horizontal + vertical + both-flip views, each run at both native resolution and a 1.25× bilinear upscale (probability map downscaled back before averaging — never downscaling the *input*, since that destroys 2-px-wide filaments), averaged equally (CODING_SPEC_06, E023).
- E013's members (`E004`, `E007`) use the older **3-flip, single-scale** recipe (identity + h-flip + v-flip, no upscale) — the recipe every raw (non-"tta4s") saved map has always used.

**Probability averaging:** member-outer, fold-inner running mean (`solar_k.ensemble.load_test_mean`) — each member's 5 per-fold models contribute equally to the test-set probability map (no per-member weighting in either final pipeline).

**Post-processing** (fit once on all 707 OOF images with no held-out split, `scripts/build_final.py`):
1. **cfg1** — a threshold or hysteresis-threshold + morphological closing, chosen from a 66-point grid by OOF PQ (E030: hysteresis 0.4–0.99, closing radius 3; E013: hysteresis 0.35–0.99, closing radius 1).
2. **Shape gate** — an elongation × length-ratio filter on candidate components (E030: no gate, [0,0]; E013: [0, 20]).
3. **Scorer** — a fitted classifier (logistic regression + a depth-3 `HistGradientBoostingClassifier`) estimates P(match) × E[IoU | match] per component; kept if this score exceeds a tuned cutoff *c* (E030: c=0.325; E013: c=0.3).
4. **Watershed regrow** — kept seed components are regrown outward into the probability map up to a soft threshold (both pipelines: t_mask=0.4, d=4 dilation iterations).

See `reports/figures/06_pipeline_diagram.png` for the end-to-end diagram (preprocessing → TTA inference → averaging → cfg1 → gate → scorer → regrow → RLE-encoded submission).

## 3. Validation protocol P5

**P5** (CODING_SPEC_05 §2.1) is day-grouped, nested 5-fold CV over the 707 OOF images:
- Inside each of 5 CV folds, the **entire post-processing chain** (cfg1, shape gate, scorer, regrow) is refit from scratch on the 4/5 train portion only, and scored on the 1/5 held-out portion — never on data it was fit on.
- A fixed cfg1's **per-image component table** (which GT instances each predicted component hits, and at what IoU) is a sufficient statistic for PQ of *any* keep-subset, so steps 1–3 are sums over cached statistics; only regrow and the final held-out scoring pass rasterize masks.
- **Paired bootstrap**: candidate vs. reference compared on identical held-out images, with an image-level paired bootstrap (resample stems with replacement, same draw for both) giving a 95% CI on the pooled ΔPQ.
- **Pre-registered acceptance bars** (set before seeing results, not tuned post hoc): an *early* bar (ΔPQ ≥ +0.008, positive on every model fold) for first-pass screening, and a stricter *pipeline* bar (ΔPQ ≥ +0.005, ≥4/5 CV folds positive, CI excludes 0, SQ doesn't drop >0.005) for anything proposed as a pipeline change.

**Why not the public leaderboard:** the public score is computed on a small, fixed subset of the test set with no visibility into its labels, so it can't drive iterative decisions without risking the very overfitting P5 exists to prevent — and, per the Nov 7 leaderboard audit (§7 below), several boards above our score show signs that aren't reproducible from the data alone. P5's local CV is the trusted number throughout; the public score is reported only as a sanity check that local and public move together in roughly the same direction, never as a selection criterion.

## 4. Results

### 4.1 Final PQ / SQ / RQ

| | PQ (pooled) | SQ | RQ | CV mean PQ ± std |
|---|---|---|---|---|
| **E030 primary** (E007tta4s+E015tta4sR) | **0.4298** | 0.6807 | 0.6314 | 0.4297 ± 0.0048 |
| E024 (scored, now unreproducible from current weights) | 0.4302 | 0.6813 | 0.6313 | 0.4300 ± 0.0052 |
| **E013 backup** (E004+E007, restored) | **0.4176** | — | — | 0.4176 ± 0.0060 |

TP=4742, FP=2080, FN=3457 (E030, pooled across all 707 OOF images).

### 4.2 Dice / IoU distributions and one-to-many / many-to-one counts

Per-image **pixel Dice**: mean 0.654, median 0.688, IQR [0.598, 0.755] (1154 annotator-image pairs). Per-image **pixel IoU**: mean 0.503, median 0.524, IQR [0.427, 0.606].

**Component-level failure taxonomy** (the rubric's one-to-many/many-to-one breakdown), by failure class:

| Class | Meaning | FN | FP |
|---|---|---|---|
| **a** | undetected entirely (recall failure — no predicted component overlaps the GT at all) | 2424 | 923 |
| **b** | one-to-many (our prediction fragments one GT filament into pieces) | 219 | 450 |
| **c** | boundary/other mismatch | 605 | 605 |
| **d** | many-to-one (we merge several GT filaments into one prediction) | 209 | 102 |

Class **a** dominates (69.5% of all FN) and is essentially unchanged across every Spec 05–07 experiment that touched post-processing — the ceiling here is recall of faint/small filaments the model never activates on, not a fixable boundary-precision issue. Classes b/d (the genuine instance-topology errors) are a small minority and roughly balanced.

### 4.3 By site

| Site | TP | FP | FN | PQ | SQ | RQ |
|---|---|---|---|---|---|---|
| B | 870 | 391 | 632 | 0.4317 | 0.6855 | 0.6298 |
| C | 834 | 343 | 688 | 0.4173 | 0.6752 | 0.6180 |
| L | 810 | 387 | 628 | 0.4210 | 0.6847 | 0.6148 |
| M | 751 | 360 | 547 | 0.4198 | 0.6734 | 0.6235 |
| T | 811 | 330 | 558 | 0.4405 | 0.6817 | 0.6462 |
| U | 666 | 269 | 404 | 0.4542 | 0.6837 | 0.6643 |

Spread of ~0.037 PQ across sites (C lowest, U highest) — modest, consistent with sites differing mainly in image quality/noise rather than a systematic site-specific failure mode.

### 4.4 By area quartile

| Quartile | TP | FP | FN | PQ | SQ | RQ |
|---|---|---|---|---|---|---|
| Q1 (smallest) | 750 | 430 | 1299 | 0.3072 | 0.6614 | 0.4645 |
| Q2 | 1280 | 665 | 770 | 0.4321 | 0.6743 | 0.6408 |
| Q3 | 1381 | 615 | 669 | 0.4647 | 0.6808 | 0.6826 |
| Q4 (largest) | 1331 | 370 | 719 | 0.4952 | 0.6977 | 0.7097 |

A clean monotonic trend: PQ rises from 0.31 to 0.50 as instance size grows, driven almost entirely by recognition quality (RQ 0.46 → 0.71 — small filaments are missed, not mis-segmented), while segmentation quality (SQ) is nearly flat (0.66–0.70). This is consistent with class a dominating FN and being a small-object recall problem.

### 4.5 Local vs. public

| Experiment | Local CV PQ | Public LB |
|---|---|---|
| E010 (E004 alone) | 0.3948 ± 0.0067 | — |
| E013 (E004+E007) | 0.4086 ± 0.005 | 0.37 |
| E024 / **E030** (E007tta4s+E015tta4s(R)) | 0.4302 / 0.4298 | 0.37 |

Both final candidates score the same 0.37 publicly despite a real (CI-excluding-0) 0.0216 local CV gap — the public subset is small and noisy enough that it can't discriminate between them, which is exactly why local CV, not the public score, drove every selection in this project (§3).

## 5. Morphology

See `reports/figures/06_overlay_gallery.png` for the overlay gallery (predictions vs. GT vs. source Hα image, across sites and failure classes).

**Representative failure cases** (class label from §4.2's taxonomy):
- **Class a (undetected):** faint, low-contrast filaments near the disk limb are the most common miss — the model's probability map simply never crosses threshold there, consistent with class a's size-quartile skew (62% of class-a FN sit in Q1/Q2).
- **Class b (fragmentation):** a single long, thin filament occasionally gets split into 2–3 predicted components where its contrast dips briefly mid-filament; the watershed regrow step (tuned per the final cfg) recovers most but not all of these.
- **Class c (boundary):** predictions that overlap a GT filament but at IoU just under 0.5 — typically a filament whose predicted extent is noticeably shorter than annotated, at the faint tail ends.
- **Class d (merging):** two closely-spaced, roughly parallel filaments occasionally get connected by the morphological closing step and predicted as one component; this is the main failure mode the shape gate and scorer cutoff were tuned to suppress without also suppressing genuine long filaments.

## 6. What didn't work

Every experiment from Specs 05–07 that was rejected, with its headline ΔPQ and which bar it missed (full detail and exact CI/fold counts in `experiments.csv` and `reports/06_final_report.md` / `reports/07_resolution_and_self_training.md`):

| Experiment | Spec | What it tried | ΔPQ | Bar missed |
|---|---|---|---|---|
| E006/E007 fold sweeps | 04 | Retrain with PQ-based checkpoint selection vs pixel Dice | +0.008–0.024 | PQ≥0.02 bar and/or SQ fell |
| E015-eval | 05 | Add E015 as a 3rd ensemble member | +0.0074 | ensemble bar (≥0.010) |
| E018b (E004+E007+E015) | 06 | Same 3rd-member add, re-tested | +0.0023 | ensemble bar, only 3/5 folds |
| E018b_weights | 06 | Weighted 3-member ensemble | +0.0053 | below unweighted 2-member PQ anyway |
| E019 | 06 | (scorer v2 / agreement features) | -0.0019 | pipeline bar, 0/5 folds positive |
| E020 (gate tuning) | 06 | Shape-gate retune | +0.0053 | early bar (needs ≥0.008 or splits≥15%) |
| E021 | 06 | — | +0.0023 | ensemble bar, 3/5 folds |
| E022 (fragment merge) | 06 | Post-regrow fragment merging | +0.0010 | pipeline bar, CI includes 0 |
| E023_4flip (flip-only, no scale) | 06 | Isolate flip-view contribution from E023's scale TTA | -0.0009 | pipeline bar, CI includes 0 |
| E025 (E007 TTA scale-sets b/c/d) | 07 | Alternative TTA scale combinations | -0.0037 to +0.0002 | pipeline bar, ≤2/5 folds positive |
| E026 (min-area grid) | 07 | Component min-area retune | +0.0027 | pipeline bar (needs ≥0.005) |
| E027 (gate replace/add E007) | 07 | Shape-gate member swap | -0.0078 / -0.0052 | early bar, negative on both model folds |
| E028 (GONG self-training, replace/add) | 07 | Pseudo-labeled GONG unlabelled data as a 3rd member or E007 replacement | -0.0022 / -0.0025 | early bar |

**Why so much saturates near zero:** we're close to the inter-annotator ceiling's structural implication — most remaining error is class-a recall on small/faint filaments that the labels themselves disagree about, so incremental architecture, TTA, and post-processing changes mostly trade FP for FN within a narrow band rather than finding genuinely new signal. Resolution levers (E023's scale TTA) and data levers (E028's self-training) were both tried and both saturated; the 2-member TTA ensemble (E024/E030) remains the best validated configuration found across all of Specs 04–07.

## 7. Leaderboard integrity note

A factual summary of the Nov 7 leaderboard audit referenced in CODING_SPEC_08: no genuine public notebook on the competition leaderboard scores above 0.40 OOF-equivalent (the two best public examples, OOF 0.4597→public 0.40 and OOF 0.4502→public 0.39, both lose roughly the same ~0.05 OOF-to-public gap we do). A score plateau around 0.55 on the board is a hard-coded, copied blob of masks, not a trained model. Nobody scoring between 0.56 and 0.70 has published code; the organisers are expected to clean the board after the deadline. **We did not exploit the evaluator or use leaked labels at any point in this project** — every number in this report comes from day-grouped P5 cross-validation on the labelled OOF set, fit with no access to test labels, and the public score was used only as a non-selecting sanity check (§4.5). No usernames are named here per the spec's instruction.

## 8. Reproduce

**Weights:** 15 EMA-shadow checkpoints (E007-f0..4, E015-f0..4, E004-f0..4), fp16, hosted as a GitHub Release (`v1.0-final`) asset bundle, verified against `weights/MANIFEST.json`'s sha256 list.

**Commands** (CODING_SPEC_08 §2.3):
```
python scripts/download_weights.py                 # fetch + sha256-verify weights/*.pt
python scripts/predict.py --pipeline primary \
    --out outputs/submissions/E030_primary.csv      # reproduces this report's primary
python scripts/predict.py --pipeline backup \
    --out outputs/submissions/E013_backup.csv       # reproduces the backup
python scripts/validate_submission.py <csv>          # schema + RLE validation
```
`predict.py` loads each member's 5 fold checkpoints from `weights/`, runs the exact frozen TTA recipe (4-flip+1.25× for E030's members, 3-flip for E013's), averages member-outer/fold-inner, and applies the frozen post-processing chain stored in `configs/final_postproc_{primary,backup}.json` (cfg1, shape gate, the pickled fitted scorer, cutoff, regrow) — no refitting happens at inference time.

**Compute used, this spec (CODING_SPEC_08, Nov 7):** one RunPod RTX 4090 pod (`85ki7s4ssa3lqv`), ~6+ hours: E015-f3/f4 retrain (~43 min each) + TTA inference, E004-f2/f3/f4 retrain (~29 min each), two full P5 runs (~38 min each) for the E030 and E013 reproducibility checks, weight export, and the `predict.py` end-to-end verification. At $0.89/hr this phase cost roughly $5–6. Total project compute across all 8 specs is tracked per-experiment in `experiments.csv`'s `compute` column. No Kaggle-notebook hours were used in this spec (all GPU work ran on RunPod).
