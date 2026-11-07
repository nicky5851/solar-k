# Report 07 — resolution and self-training: four more ideas, zero promotions

Covers `docs/specs/CODING_SPEC_07_RESOLUTION_AND_SELF_TRAINING.md`. **Status: final (2026-11-07).**
Every experiment in the spec's queue (E025–E029) is run and recorded in `experiments.csv`. No code
change in this spec was promoted; the frozen pipeline from `reports/06_final_report.md` (E024_primary,
local CV PQ 0.4302) is unchanged, re-validated byte-for-byte reproducible, and remains both the primary
candidate and the only candidate under active consideration. No submission has been made to Kaggle.

## 1. Read this first: the public leaderboard is not evidence

Repeated from `reports/05_ensemble_report.md` §1 and `reports/06_final_report.md` §1 because it is still
true and still the single most important caveat on every number below. Two expert annotators agree with
each other at only **PQ 0.3399** on this dataset (`reports/01_dataset_report.md`). The public leaderboard
has a documented evaluator defect (empty-mask submissions scoring 1.00, a fabricated submission scoring
0.55+ from precomputed RLE lookups, score displays not reconciling between snapshots) — every displayed
public score above about 0.45 is unverified, and **local 5-fold day-grouped CV (protocol P5) is the only
decision metric used in this spec, as in every prior one.** The frozen pipeline's local 0.4302 would be
expected to land near public 0.39 based on the historical local-to-public gap (E010 0.3948→0.35, E013
0.4086→0.37, about −0.04) — not meaningfully past the independently-confirmed 0.34–0.39 ceiling.

## 2. Where Spec 07 started and what it tried

Spec 06 closed with E024 (E007+E015, 4-flip + 1.25x-upscale TTA, local CV PQ 0.4302) as the frozen
pipeline, with recall on small/faint filaments (class-a, undetected-with-no-overlap FN) still the
largest remaining error class. Spec 07's priority order was: (A) push TTA scale further, since the
single scale added in E023 was the single biggest win of Spec 06 (§2.3 asked "does more help more?");
(B) a cheap min_area/shape-gate recheck; (C) train-time (not just test-time) resolution, via an
upsampled preprocessing cache; (D) more data, via self-training on external unlabelled GONG H-alpha
images with per-fold pseudo-labels; (E) final build and deliverables.

| Experiment | Change | Result vs. REF | Bar (needed) | Decision |
|---|---|---|---|---|
| E025 scale-b | E007 TTA scales [1.0,1.5] (drop 1.25x, jump to 1.5x) | pooled ΔPQ **−0.0037**, CI [−0.0067,−0.0008], 1/5 folds positive | pipeline (≥+0.005, ≥4/5 positive, CI excl. 0) | reject |
| E025 scale-c | E007 TTA scales [1.0,1.25,1.5] (keep 1.25x, add 1.5x) | pooled ΔPQ **+0.0002**, CI [−0.0024,0.0027], 2/5 folds positive | pipeline | reject |
| E025 scale-d | E007 TTA scales [1.0,1.25,1.5,1.75] (4 scales) | pooled ΔPQ **−0.0026**, CI [−0.0056,0.0002], 1/5 folds positive | pipeline | reject |
| E026 | min_area∈{150,250} vs REF's 418 | PQ +0.0027 (both values, identical), 5/5 folds positive, CI excludes 0 | pipeline (≥+0.005) | reject — real but too small |
| E027 (gate, replace E007) | E007 retrained + inferred at 1.5x resolution (both train- and test-time), model folds 0–1 | pooled ΔPQ **−0.0078**, CI [−0.0151,−0.0006], 0/2 model folds positive | early (≥+0.008, every fold positive) | reject — folds 2–4 never trained |
| E027 (gate, add as 3rd member) | same E027 checkpoint, added alongside E007+E015 | pooled ΔPQ **−0.0052**, CI [−0.0123,0.0019], 0/2 model folds positive | early | reject |
| E028 (gate, replace E007) | E007 fine-tuned 3000 iters on 50/50 real + external-GONG-pseudo-labelled crops, model folds 0–1 | pooled ΔPQ **−0.0022**, CI [−0.0074,0.0035], 0/2 model folds positive | early | reject — folds 2–4 never trained |
| E028 (gate, add as 3rd member) | same E028 checkpoints, added alongside E007+E015 | pooled ΔPQ **−0.0025**, CI [−0.0087,0.0039], 1/2 model folds positive | early | reject |
| E029 | final build re-validation: `build_final.py` rerun on E007tta4s+E015tta4s | byte-identical to existing `E024_primary.csv` (0 diff lines, 1008 instances) | — | confirms E024_primary unchanged |

**All eight experimental decisions this spec were rejections.** Every one failed a real,
pre-registered decision bar (P5's "early"/"pipeline" bars, same protocol as Spec 06 §2) — see §3.

## 3. Why none of the four ideas made it

**More TTA scales didn't help, and the direction of the effect is informative (E025).** Spec 06's
E023 found that adding a *single* 1.25x upscale pass to flip-only TTA was the single largest win of
that spec (+0.0082 pooled). Spec 07 tested three ways to push further: dropping 1.25x in favour of a
single larger jump to 1.5x (scale-b), keeping 1.25x and adding 1.5x on top (scale-c), and keeping 1.25x
while adding both 1.5x and 1.75x (scale-d). None cleared the bar, and the pattern across the three is
informative rather than noisy: scale-b (which *removes* 1.25x) is the clear loser (−0.0037, the most
negative of the three); scale-c (which keeps 1.25x and adds one more scale) is closest to neutral
(+0.0002); scale-d (keeps 1.25x, adds two more scales) is negative again (−0.0026), suggesting the extra
scales beyond 1.25x add inference cost and TTA-averaging noise without new signal. The conclusion is
specific, not just "TTA scales don't matter": **1.25x is doing real work and should not be dropped;
anything beyond it is not paying for itself.** E023's original two-scale [1.0, 1.25] configuration
remains, by a clear margin across all three alternatives tried, the best known TTA configuration.

**The min_area/shape-gate recheck found a real, small, below-bar effect (E026).** Lowering `min_area`
from REF's 418 to 150 or 250 gave an identical +0.0027 pooled PQ gain at both values, positive on
every fold, with a CI that excludes zero — this is not noise, it is a genuine, reproducible, small
improvement from letting slightly smaller detected components survive the shape gate. It is roughly a
third of the +0.005 the pipeline bar requires, so it stays rejected under the same pre-registered
threshold every other change in this project has been held to. (This run used the then-current best TTA
maps, since E025 was still mid-flight when E026 ran, per the spec's own contingency note — rerunning
against E025's eventual best-known maps would be cheap if ever warranted, but since none of E025's three
scale-sets beat the existing REF, there is nothing new to rerun this check against.)

**Higher training-time resolution (not just test-time) made things worse, not better (E027).**
E023 (Spec 06) established that letting the model's test-time TTA include a 1.25x upscale pass helped.
E027 asked whether *baking* a similar resolution increase into training itself — training E007's
encoder at 1.5x crop resolution (768px, matching the 512px@1x field of view) rather than only upscaling
at inference — would help further. It was tested on the cheap gate (model folds 0–1 only, so folds 2–4
were never trained if it failed) both as a straight replacement for E007 and as a third ensemble member.
Both configurations failed clearly and in the same direction: **negative on every model fold tested**,
with the replacement configuration's confidence interval excluding zero in the negative direction
(−0.0151 to −0.0006) — this is not an ambiguous or marginal result, retraining at a higher crop
resolution genuinely hurt this architecture on this data. The most likely explanation: E023's test-time
upscale gain comes from TTA ensembling across the model's existing receptive-field response at multiple
scales, not from the model itself needing to see higher-resolution crops during training — baking the
resolution change into training removes the multi-scale averaging that was the actual source of E023's
benefit, while adding the usual risks of training at a different effective receptive field (less
context per crop, different augmentation statistics) with no offsetting benefit observed.

**External self-training on unlabelled GONG images did not transfer useful signal (E028).** This was
the most structurally different idea in the spec: rather than changing the model or its TTA, it added
~2000 unlabelled GONG H-alpha images (public NSO archive, 2011–2022, all six sites, every observation
within ±3 days of any train/test date excluded — manifest committed at
`data/external/gong_unlabelled_manifest.csv`, images themselves gitignored per project rules) and
fine-tuned E007 for 3000 extra iterations on a 50/50 mix of the fold's own labelled crops and
pseudo-labels generated **only by that same fold's own E007+E015 models** (never the full ensemble, and
never any other fold's models — the leakage argument is in §4). Training itself behaved normally: loss
decreased noisily but without divergence, and the internal validation PQ tracked during fine-tuning
(fold 0 best 0.3580, fold 1 best 0.3766) was in a plausible range, not a collapse. But on the held-out
P5 gate, both configurations failed, with the "add as 3rd member" configuration's confidence interval
([−0.0087, 0.0039]) straddling zero and even its best individual fold ([model fold 1] essentially flat
at +0.0002) showing no real signal. The most likely explanation: the external GONG archive images,
while the same instrument family and wavelength as the competition's MAGFiLO-derived dataset, differ
enough in acquisition/calibration/preprocessing characteristics that pseudo-labels generated on them —
even after confidence-based pixel exclusion in [0.3, 0.7] — carried domain-shift noise rather than
transferable signal for this specific segmentation task. **More unlabelled data from a related but not
identical source does not substitute for more labelled data from the same source**, at least not at the
scale (2000 images, 287-288 kept per fold after confidence filtering) and fine-tuning budget (3000
iterations) tried here.

**Final build (E029) found no new combination to build.** Per the spec's own rule, E029 only assembles
a new combined pipeline if at least one of E027/E028 is promoted; since neither was, there was nothing
new to build. The check that remained was re-validating that the existing `E024_primary.csv` is still
exactly reproducible from the current codebase — it is: rerunning `scripts/build_final.py` with the
same members (E007tta4s, E015tta4s) produced a byte-for-byte identical submission (1008 instances, 0
diff lines against the file already on disk from Spec 06).

## 4. E028's leakage argument and the external-data rules

**Rules basis.** Nicky confirmed on 2026-11-06 that the competition's Rules tab allows unlabelled
external data ("participants may use other external data sources") and forbids only other ground-truth
metadata or annotation sets. E028 used images only — no MAGFiLO labels, no other annotation sets, no
catalogue metadata (observation timestamps were used solely to apply the ±3-day exclusion window, which
is a near-duplicate-avoidance check, not a label).

**Leakage argument.** Three separate guarantees apply:
1. **Temporal exclusion.** Every external GONG frame within ±3 days of *any* image in the train *or*
   test split was excluded before pseudo-labelling, so no external frame can be a near-duplicate of a
   frame the model is ever evaluated against. The manifest (`data/external/gong_unlabelled_manifest.csv`,
   committed) lists the exact URLs and dates kept, so this is independently checkable.
2. **Per-fold-only pseudo-labelling.** Pseudo-labels for fold *k*'s external crops were generated using
   **only fold *k*'s own E007-f*k*+E015-f*k* models** — never the full 5-model ensemble, and never any
   other fold's models. This means fold *k*'s OOF evaluation (which always uses fold *k*'s own held-out
   real images) never saw any signal derived from a model that had seen information from outside fold
   *k*'s own training set; the pseudo-labelled external data for fold *k* is exactly as "fold-*k*-only"
   as fold *k*'s own real training crops are.
3. **Confidence masking.** Pseudo-label pixels with model confidence in [0.3, 0.7] — the band where the
   fold-*k* models themselves are most uncertain — were excluded from the fine-tuning loss via a
   per-pixel weight mask, rather than being treated as confident labels. This reduces (does not
   eliminate) the risk of the fine-tune reinforcing the model's own existing mistakes.

Taken together, this is the same evaluation discipline P5 has used throughout the project (strict
per-fold separation, no information crossing fold boundaries) extended to the one new data source this
spec introduced, and the null result in §3 is a real, leakage-safe null result, not an artifact of
accidentally-easy pseudo-labels.

## 5. The frozen pipeline (unchanged)

No experiment this spec was promoted, so the frozen pipeline is **exactly** the one described in
`reports/06_final_report.md` §4: members E007 (efficientnet-b4) + E015 (tu-regnety_040), 4-flip +
1.25x-upscale TTA, hysteresis cfg1 (`hys0.4-0.99_r3`), scorer cutoff c=0.325, regrow (t_mask=0.4, d=4).

- **Primary candidate — `E024_primary.csv`**: E007tta4s+E015tta4s, 1008 instances, local CV PQ
  **0.4302**. Re-validated this spec (`scripts/build_final.py`, E029) as byte-for-byte reproducible from
  the current codebase.
- **Backup candidate — `E013.csv`**: Spec 05's E004+E007 ensemble, 1029 instances, local CV PQ 0.4086,
  public LB **0.37**.

Both were already validated with `scripts/validate_submission.py` in Spec 06. **Neither has been
submitted.** This remains explicitly Nicky's call; this report takes no further action on either CSV.

### 5.1 Quantitative breakdown (unchanged from Spec 06)

Since the frozen pipeline's members and post-processing are unchanged, the full quantitative breakdown
in `reports/06_final_report.md` §4.1 — PQ/SQ/RQ (0.4302/0.6813/0.6313), per-image pixel Dice, per-TP IoU,
split/merge error counts, by-site PQ (U 0.4527 down to C 0.4143), and by-GT-area-quartile PQ (q1 0.3091
up to q4 0.4943) — applies unchanged and is not recomputed here; see that report for the full table and
`data/interim/E024_quant_p5_details.csv` / `E024_quant_p5_pixel_dice.csv` for the underlying per-image
and per-component data.

### 5.2 Qualitative items

- **Pipeline diagram** and **H-alpha overlay gallery**: `reports/figures/06_pipeline_diagram.png` and
  `reports/figures/06_overlay_gallery.png` (regenerated this spec via `scripts/make_final_figures.py`
  after a prior session left the gallery ungenerated — both now match the unchanged frozen pipeline's
  actual OOF predictions, four cases: typical, worst split, worst merge, recovered faint filament vs.
  E013). Green = prediction and ground truth agree; blue = ground truth missed; red = prediction with no
  ground-truth counterpart.

## 6. Compute used this spec

RunPod: three pods (two RTX 4090s, one L4), used in parallel per the spec's lane plan — E025's TTA
inference and E026's min_area/P5 sweep on the L4 and one 4090, E027's two gate folds' training +
inference on a 4090, E028's external-image download/preprocessing/pseudo-labelling/fine-tuning/P5-gate
on the third pod/4090. A pre-existing non-atomic cache-write race (`src/solar_k/preprocess.py`
`save_processed`/`save_disk_info`, surfaced only once E028's fine-tune was the first job to build the
preprocessing cache for never-before-seen stems under `num_workers>1` concurrently) was found and fixed
with an atomic temp-file+`os.replace` write pattern — a genuine correctness fix, not specific to E028,
committed alongside the experiment results. One of the three pods (`85ki7s4ssa3lqv`) was capacity-
blocked by RunPod for the device/datacenter combination requested for the entire spec and was never
used; the other two pods carried the full workload. Kaggle was not needed — E027's and E028's gate folds
both fit comfortably on the pod GPUs, and since both gates failed, folds 2–4 (which the spec's plan had
reserved for Kaggle, pending the weekly quota reset) were never trained for either branch.

## 7. Verified facts vs. assumptions

**Verified, with evidence cited above:** every decision-bar pass/fail in §2 (each has an
`experiments.csv` row with the exact pooled delta, bootstrap CI, and fold-positivity count); E029's
byte-identical rebuild of `E024_primary.csv`; E028's three-part leakage argument (§4); the directional
pattern across E025's three scale-sets (§3) isolating that 1.25x specifically, not "more TTA scales" in
general, is where Spec 06's TTA gain came from.

**Assumptions carried forward, not re-verified this spec:** the day-grouped 5-fold CV split
(`data/splits/folds_v1.csv`) and the P5 run-twice determinism check, both already established in prior
specs and relied upon unchanged; the inter-annotator PQ ceiling and every leaderboard-defect finding
from `reports/06_final_report.md` §1, repeated here but not re-derived.

## 8. Status and next action

Everything in `docs/specs/CODING_SPEC_07_RESOLUTION_AND_SELF_TRAINING.md`'s experiment queue is run and
decided — all eight tried changes were rejected against the same pre-registered decision bars used
throughout this project, and the final build step confirms nothing changed. The frozen pipeline is
exactly `reports/06_final_report.md`'s E024_primary, unchanged and re-validated. Code for this spec
(`tta_scales` extension, the min_area/shape-gate grid script, `train_scale`/`infer_scale` plumbing and
the upsampled cache builder for E027, `scripts/fetch_external_gong.py` plus the manifest, the pseudo-
label generator with confidence masking, and the mixed sampler for E028, plus the preprocessing
cache-write race fix) is committed; `experiments.csv` rows E025–E029 are recorded.

**The only action remaining is Nicky's: pick primary vs. backup (or neither) between `E024_primary.csv`
and `E013.csv`, and decide whether and when to submit to Kaggle.** No code in this repository will do
that automatically.
