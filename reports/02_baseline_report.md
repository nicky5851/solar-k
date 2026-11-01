# Baseline report — Spec 02

Covers `docs/CODING_SPEC_02_BASELINE.md`: the classical baseline (E001), the
semantic U-Net baseline (E002), and the H1 failure-mode analysis that
decides Spec 03's priorities. Generated from `scripts/run_e001.py`,
`scripts/run_e002_eval.py`, `scripts/run_e002_deepdive.py`, and the training
log of the Kaggle kernel `potcharapanchumrak/solar-k-e002-train`.

## Results table

All PQ/SQ/RQ numbers are on fold-0 val's **report** half (71 images, never
used to choose a threshold). The inter-annotator reference is Spec 01's
296-image, 1196-comparison inter-annotator agreement (`reports/01_dataset_report.md`).

| Experiment | PQ | SQ | RQ | TP | FP | FN | Kaggle LB | Notes |
|---|---|---|---|---|---|---|---|---|
| Inter-annotator ref. | 0.3399 | 0.6337 | 0.5364 | 4292 | 3709 | 3709 | — | human ceiling, full train set (1196 A-vs-B comparisons), not the report half |
| E001 (classical dark-feature + CC) | 0.0346 | 0.6317 | 0.0547 | 52 | 1059 | 738 | — | tuned t=0.5, closing_radius=0, min_area=418 |
| E002 (U-Net resnet34 + CC) | 0.2691 | 0.6509 | 0.4134 | 486 | 1075 | 304 | — | tuned t=0.7, closing_radius=2, min_area=418, seed=42, 6000 iters |

`kaggle_public_score` is blank in `experiments.csv` for both — neither CSV
has been submitted to the competition leaderboard yet.

Pixel Dice/IoU was **not** separately aggregated by `evaluate.py`'s report
scripts for either experiment (the function `pixel_iou_dice` exists but
isn't wired into `aggregate_evaluation`'s output); the only pixel-level
number on record is E002's own training-time val Dice at t=0.5 used for
checkpoint selection (see the training curve below). This is a gap against
the spec's "pixel Dice/IoU" ask, noted here rather than filled with an
invented number — a 15-minute follow-up (call `pixel_iou_dice` per
report-half image, union of annotators vs union of predictions) would close
it.

Mean inference time: E001 0.046 s/image (classical, Mac CPU). E002 isn't
timed per-image by `run_e002_eval.py` (it loads cached PNGs, not the
model), but the Kaggle training log gives the end-to-end GPU inference
rate: 142 OOF images in 5.1 min (2.15 s/image) and 180 test images in 6.4
min (2.13 s/image), both flip-TTA at 1024×1024 tiles with 128 px overlap.

## H1 verdict: where does E002 lose PQ?

**SQ is close to the human reference (0.651 vs 0.634) — boundary quality is
not the bottleneck.** Once a prediction is matched, it's about as tight a
fit as two human annotators are to each other. **RQ is the gap (0.413 vs
0.536)** — the loss is almost entirely about which instances get matched at
all, not how well-shaped the matches are.

The spec's (a)-(d) decomposition, from `evaluate.py`'s weak-overlap-graph
classifier, on the report half:

| Mode | FN | FP | share of FN | share of FP |
|---|---|---|---|---|
| (a) detection | 104 | 830 | 34% | 77% |
| (b) fragmentation | 53 | 108 | 17% | 10% |
| (c) near-miss IoU | 127 | 127 | 42% | 12% |
| (d) merging | 20 | 10 | 7% | 1% |

Compared to E001 (FN: a=665/12/59/2, FP: a=974/25/59/1 — detection-dominated
on **both** sides, the model barely finds anything), E002's FN count
collapsed 738→304 (detection got much better) while FP count stayed flat at
1059→1075. **The dominant failure mode flipped from under-detection to
over-prediction.**

### FP/FN deep-dive (`scripts/run_e002_deepdive.py`, report half only)

The (a)-(d) decomposition says *how* an error connects to GT in the overlap
graph; it doesn't say *why* a given FP exists or what kind of FN it is. This
breaks every FP/FN instance into a more specific cause and estimates each
cause's effect on PQ: the counterfactual `PQ` if that class's errors were
fixed (their count removed from the FP/FN side of the denominator, `tp_ious`
held fixed) minus the actual PQ of 0.2691 — a per-class, additive-not-exact
but directly interpretable sensitivity number.

**FP classes** (1075 total; checked in this order, first match wins —
details in the script's docstring):

| Class | Count | Share | PQ gain if eliminated | Median area (px²) | Median length (px) | Median prob | Median r/R |
|---|---|---|---|---|---|---|---|
| isolated | 807 | 75% | **+0.1406** | 726 | 53 | 0.957 | 0.747 |
| near_miss | 183 | 17% | +0.0227 | 1472 | 111 | 0.963 | 0.551 |
| annotator_disagreement | 74 | 7% | +0.0087 | 942 | 67 | 0.968 | 0.731 |
| fragment | 11 | 1% | +0.0013 | 1777 | 111 | 0.966 | 0.582 |

(Median of each instance's own mean-probability-under-its-mask, not the
mean across instances — consistent with the area/length columns, and more
robust to the handful of very large or very bright outlier instances.)

**`isolated` is the whole story on the FP side: 75% of FPs, and eliminating
them alone would take report-half PQ from 0.269 to roughly 0.41 — past the
inter-annotator ceiling of 0.34.** Two things stand out:

1. **These are not low-confidence noise.** Median probability under an
   `isolated` FP mask is 0.957, barely below the TP population and far
   above the tuned threshold (t=0.7, closing_radius=2, min_area=418, over
   `t ∈ {0.30…0.70}` — see the PQ-vs-threshold curve below). Raising `t`
   further would cost real detections before it touches these: the model
   is **confidently wrong**, not uncertain.
2. **Visually (`reports/figures/E002_gallery.png`), `isolated` FPs
   repeatedly land on plage/network regions** — the bright, mottled
   facular patches near active regions, not scattered speckle. The model
   appears to have learned "dark-ish elongated structure in a textured
   region" rather than the true filament signature, and over-generalises
   to other textured bright features. A probability-threshold-only
   postprocess cannot fix a confident misclassification; shape filtering
   (the EDA's own instance stats put median filament elongation at 3.75
   and median min-width at 2 px — a stricter elongation/min-width gate in
   `postprocess.py`) or harder negative examples at train time are the two
   candidate fixes.

`near_miss` (17% of FPs, median IoU in [0.3, 0.5)) is the second-largest
share but worth 6x less PQ (+0.023) — these are mostly *already* close
enough that their paired FN is also already counted once, so fixing them
buys less than their raw count suggests. `annotator_disagreement` (7%,
IoU>0.5 with a *different* annotator on the same image) says the model
agrees with somebody on 74 of its "false" positives — a metric artifact of
scoring against one annotator at a time, not a model failure.
`fragment` (1%) confirms fragmentation is genuinely rare, consistent with
the EDA's 0.5% same-annotator touching-instance rate.

**FN classes** (304 total):

| Class | Count | Share | PQ gain if eliminated | Median area (px²) | Median prob (under GT mask) |
|---|---|---|---|---|---|
| missed_by_all | 240 | 79% | +0.0306 | 975 | 0.683 |
| missed_only_this_annotator | 64 | 21% | +0.0075 | 1476 | 0.841 |

`missed_by_all` (unmatched for every annotator who drew something there,
including trivially every single-annotator image) is the large majority.
Median probability under these GT masks is 0.683 — below the t=0.7
threshold but not by much and not near 0: the model partially sees these
filaments, it just doesn't clear the bar or the binary mask doesn't connect
into one component ≥ min_area. `missed_only_this_annotator` (unmatched for
this annotator, but a *different* annotator's overlapping GT instance in
the same image *was* matched) has higher median prob (0.841) and larger
median area (1476 vs 975 px²) — these read as instance-identity/boundary
disagreements between annotators that the model happened to side with one
annotator over another, not detection misses.

**Bottom line for Spec 03**: FN is no longer the problem (FN share of lost
PQ ≈ 0.038 combined vs FP's ≈ 0.173 combined, dominated by `isolated`).
The single highest-leverage fix is suppressing confident, shape-plausible
FPs on bright/textured non-filament regions — not more detection recall,
not fragmentation handling, and not threshold retuning (the model is
already confident past the tuned threshold on these).

### Was the best checkpoint at the last iteration?

**No.** `reports/figures/E002_val_dice_vs_iter.png` (full, unedited series
from the Kaggle training log, every 500-iteration checkpoint eval):

| iter | val_dice | best so far |
|---|---|---|
| 500 | 0.1733 | 0.1733 |
| 1000 | 0.0551 | 0.1733 |
| 1500 | 0.0774 | 0.1733 |
| **2000** | **0.1822** | **0.1822** |
| 2500 | 0.1352 | 0.1822 |
| 3000 | 0.0725 | 0.1822 |
| 3500 | 0.1326 | 0.1822 |
| 4000 | 0.0946 | 0.1822 |
| 4500 | 0.0881 | 0.1822 |
| 5000 | 0.0998 | 0.1822 |
| 5500 | 0.0644 | 0.1822 |
| 6000 | 0.0813 | 0.1822 |

The best checkpoint (pixel Dice @ t=0.5) was iteration 2000/6000 — a third
of the way through training — and `train.py`'s checkpoint-by-best-val-dice
logic correctly kept it (`best.pt`, used for every OOF/test prob map in
this report). Val Dice is noisy and never recovers afterward, which looks
like overfitting on a tiny (20-image) validation set measured by a
pixel metric at a single fixed threshold, rather than genuine capacity
exhaustion — the training loss itself (not plotted here, see the raw
Kaggle log) kept decreasing. **6000 iterations was not undertraining; if
anything it was more than needed for this val-Dice signal, and the
noisiness argues for either a larger/more frequent val sample, checkpoint
selection by instance PQ instead of pixel Dice, or both, in Spec 03.**

### PQ vs threshold, and the PQ/2 keep-rule check

`reports/figures/E002_pq_vs_threshold.png` (tune half, closing_radius=2,
min_area=418 held at their tuned values, sweeping only `t`):

PQ rises monotonically across the entire grid, t=0.30 (0.193) to t=0.70
(0.275), **with no sign of turning over by the grid's upper edge.** The
grid searched in `tuning.py` (`SINGLE_T_GRID`) stops at t=0.70 by spec, so
this result is a legitimate finding, not an error: **the grid wasn't wide
enough — a wider sweep (t up to ~0.85-0.9) is a cheap, high-value first
experiment for Spec 03**, especially given the `isolated`-FP finding above
(these FPs sit at mean prob 0.95, so t would need to climb well past 0.7
before touching them, likely trading a few TPs for many FPs along the way
until the curve does eventually turn over).

**PQ/2 keep-rule check**: PQ's IoU>0.5 strict matching rule guarantees a
*unique* matching — no GT instance can have IoU>0.5 with two different
predictions simultaneously (and vice versa), because two masks that each
cover more than half of a third mask's union would have to overlap each
other by more than they could occupy. `run_e002_deepdive.py` verifies this
empirically on every report-half annotator-image hit matrix (1075+486=1561
matched/unmatched instance checks): **0 uniqueness violations** — the
implementation matches the guarantee the IoU>0.5 threshold is supposed to
provide.

## H2 check: PQ by r/R bin (limb correction)

Limb-darkening correction is always on in this spec (no ablation yet).
Per-r/R-bin PQ, report half:

| r/R bin | E001 PQ | E002 PQ |
|---|---|---|
| centre (r/R < 1/3) | 0.0756 | 0.2902 |
| middle (1/3 ≤ r/R < 2/3) | 0.0579 | 0.2903 |
| limb (r/R ≥ 2/3) | 0.0176 | 0.2455 |

E001 drops sharply toward the limb (0.076 → 0.018, a 4.3x falloff) —
expected for a classical dark-feature filter on a classical normalisation
that still leaves some residual limb-darkening gradient. **E002 is much
flatter** (centre and middle are statistically indistinguishable at
0.2902/0.2903; limb is lower at 0.2455 but only an ~18% relative drop, not
4x). This is consistent with (not proof of) limb correction doing its job
for the learned model — a CNN with a large receptive field is also better
positioned than a hand-tuned filter to absorb a residual gradient the
correction doesn't fully remove. **No ablation (limb correction on vs off)
exists yet to isolate the correction's own contribution from the model's**;
that's the real H2 test and is listed as a Spec 03 hypothesis below. The
inter-annotator reference has no r/R breakdown in `reports/01_dataset_report.md`,
so no human-ceiling comparison is available for this slice.

## Gallery

`reports/figures/E002_gallery.png`: the 8 worst-PQ and 4 best-PQ report-half
images (ranked by per-image PQ, aggregated over all annotators of that
image), grayscale frame with the GT union's boundary in black and each
predicted instance in its own colour, titled with per-image PQ and the
image's single most common FP/FN class.

Seven of the eight worst images are dominated by `isolated` FPs — visibly,
colour-filled blobs with no nearby black GT outline, several sitting on the
bright mottled plage patches near active regions rather than on quiet
photosphere. The worst two (`20171008144250Bh`, `20200806053830Lh`,
PQ=0.000) each have only one or two real GT filaments and the model's
prediction misses both while adding spurious instances elsewhere — small
images with few instances are disproportionately punished by PQ's
all-or-nothing per-instance scoring. The best four still show `isolated`
FPs (e.g. `20130609123334Ch`, `20121118092514Th`) even at their best, which
matches the deep-dive: the FP problem is pervasive, not concentrated in a
handful of bad images.

## Prioritised hypotheses for Spec 03

1. **Suppress confident, shape-implausible FPs with a stricter
   elongation/min-width postprocess gate**, evidenced by: `isolated` FPs
   are 75% of all FPs, mean prob 0.952 (model is confident, so threshold
   tuning alone can't remove them), and median length 53 px vs the EDA's
   instance elongation median of 3.75 (length/sqrt(area)) — a
   shape-consistency filter on top of the existing area filter is the
   natural next postprocess knob. Test: add an elongation/min-width
   threshold to `postprocess.py`, re-run the tune/report grid, compare FP
   count and PQ to this report's numbers.
2. **Widen the single-threshold grid past t=0.70**, evidenced by: PQ vs
   threshold rises monotonically to the grid's edge with no sign of a
   turnover (`reports/figures/E002_pq_vs_threshold.png`). Test: extend
   `tuning.SINGLE_T_GRID` to ~0.9, re-run `grid_search`, confirm PQ turns
   over and report the new optimum.
3. **Select the training checkpoint by instance PQ (not pixel Dice @
   t=0.5) on a larger validation sample**, evidenced by: val Dice is noisy
   after iteration 2000/6000 and never recovers, while train loss (not
   checked here) presumably kept falling — a 20-image pixel-Dice signal is
   a weak and possibly misleading proxy for the thing actually being
   optimised. Test: checkpoint every 500 iters as now, but score each on
   full fold-0 val instance PQ (not just 20 images' pixel Dice) and compare
   which checkpoint the two criteria pick.
4. **Run the limb-correction ablation** (correction on vs off, same model
   and postprocess otherwise), evidenced by: E002's PQ is much flatter
   across r/R bins than E001's, but this is confounded with "the model is
   just better overall" rather than isolated to the correction step. Test:
   retrain (or re-infer, if the raw image is cheap to re-preprocess without
   correction) one variant without limb correction, compare per-r/R-bin PQ
   to this report's table.
5. **Fill the pixel Dice/IoU gap in `evaluate.py`'s reported output**,
   evidenced by: the spec asks for it, `pixel_iou_dice` already exists, and
   it isn't wired into `aggregate_evaluation`. Low-risk, ~15-minute
   follow-up, not a modelling hypothesis — listed because the acceptance
   criteria below can't fully close without it.

## Reproducibility checklist

```bash
source .venv/bin/activate

# Spec 01 EDA (dataset report + figures this report's inter-annotator
# reference and EDA-derived numbers come from)
python scripts/eda.py

# E001 classical baseline: tune/report split, grid search, test CSV,
# experiments.csv row, data/interim/E001_eval_summary.json
python scripts/run_e001.py

# E002 package + Kaggle training (manual steps; see notebooks/kaggle_train.ipynb)
python scripts/package_kaggle.py
# upload outputs/kaggle_pkg/solar_k_package.zip as dataset potcharapanchumrak/solar-k-package
# kaggle kernels push -p notebooks/   (from notebooks/, with notebooks/kernel-metadata.json)
# kaggle kernels output potcharapanchumrak/solar-k-e002-train -p <dir>
# copy <dir>/outputs/E002/{checkpoints,oof,test_probs} -> outputs/E002/

# E002 local tune/report evaluation against the Kaggle-saved OOF/test probs:
# test CSV, experiments.csv row, data/interim/E002_eval_summary.json,
# data/interim/E002_grid_search.csv
python scripts/run_e002_eval.py

# This report's FP/FN deep-dive, training-curve/threshold figures, gallery:
# data/interim/E002_{fp,fn}_deepdive.csv, E002_deepdive_summary.json,
# reports/figures/E002_{val_dice_vs_iter,pq_vs_threshold,gallery}.png
python scripts/run_e002_deepdive.py

# Validate either submission CSV standalone
python scripts/validate_submission.py outputs/submissions/E001.csv
python scripts/validate_submission.py outputs/submissions/E002.csv

pytest
```

## Acceptance criteria status

- `pytest` passes (66 tests as of this report; postprocess and
  resume-from-checkpoint tests included from earlier Spec 02 work).
- Mac smoke test (50-iter MPS train → resume → infer → evaluate on 4 val
  images) passed before the full Kaggle run, per the earlier Spec 02 work.
- Both `E001.csv` and `E002.csv` validate via `validate_submission.py`.
  **Neither has been submitted to Kaggle yet** — the acceptance criterion
  "at least one has been submitted" is **not yet met**; `kaggle competitions
  submit` was not run this session.
- No threshold was chosen on the `report` half for either experiment (the
  deep-dive's PQ-vs-threshold figure uses the `tune` half only).
- Nothing from `data/` or `outputs/` was committed except split CSVs,
  `.gitkeep`, and READMEs (checked via `git status` before every commit
  this session).
