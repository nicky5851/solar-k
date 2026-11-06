# Report 06 — final report: recall, connectivity, and the frozen pipeline

Covers `docs/specs/CODING_SPEC_06_RECALL_AND_CONNECTIVITY.md`. **Status: final (2026-11-05).** This is the
project's last report: every experiment in the spec's queue (E018–E024) is run and recorded in
`experiments.csv`, nothing further is queued, and no submission has been made to Kaggle.

## 1. Read this first: the public leaderboard is not evidence, and 0.34 is the real ceiling

This repeats `reports/05_ensemble_report.md` §1 because it is still true and still the single most
important caveat on every number below.

Two expert annotators agree with each other at only **PQ 0.3399** on this dataset (`reports/01_dataset_report.md`,
296 images, 1196 pairwise comparisons; SQ 0.634, RQ 0.536). That is not a property of our model — it is
how much two humans who both tried hard disagree about where filament boundaries and instance splits
are. A model cannot be "more correct" than that without the comparison becoming meaningless (whose
annotation is it being more correct *against*?).

The public leaderboard is not a reliable second opinion. On 2026-11-02, Nicky found (`kaggle kernels
pull` on public notebooks for this competition, not reproduced in this repo, notebooks not committed):
1. **Evaluator defect**: submitting 5/180 images with empty masks scored public **1.00**; all-empty
   scored **0.88**; near-empty scored **0.70**. The evaluator does not penalise missing predictions the
   way instance PQ should.
2. **The defect caps genuine entries too**: an honest Mask R-CNN + U-Net + TTA submission's author
   reports many different post-processing variants all plateauing at public **0.70**.
3. **A fabricated submission** scored "0.55+" by looking up precomputed RLE strings from an
   LZMA/base85 blob for every image in the (public) MAGFiLO dataset — no training loop.
4. **Score displays don't reconcile**: one notebook showed 0.93 in one snapshot, 0.34 in another; the
   same author's honest pipeline scored an official public 0.12.
5. **Independent confirmation of the 0.34 ceiling**: a transparent 10-model ensemble (5-fold ResNet34 +
   5-fold EfficientNetV2-S, heavy TTA) reports OOF PQ 0.45 / public 0.39 — matching our inter-annotator
   number and giving a loose outside reference point.

**Operating rule, unchanged through this spec:** every displayed public score above about 0.45 is
unverified; we never optimise toward or validate against the public LB; local 5-fold day-grouped CV
(protocol P5, §2) is the only decision metric. Our own local-to-public gap has stayed fairly stable —
E010 local 0.3948 → public 0.35, E013 local 0.4086 → public 0.37 — about −0.04, so the frozen pipeline's
local 0.4302 would be expected to land near public 0.39, i.e. at the transparent outside result, not
meaningfully past it. That is a good outcome given the 0.34 ceiling, not a disappointing one.

## 2. Where Spec 06 started and what it tried

Spec 05 closed with **E013** (E004+E007 mean ensemble, local CV PQ 0.4086 ± 0.0050, public 0.37) as the
primary submission. E013's loss decomposition showed 68% of its 3675 FN had **no overlapping
prediction at all** (class a) — recall of faint/small filaments, not boundary precision, was the
biggest lever left. Spec 06's priority order (§2) was: (A) widen grids/weights/scorer (CPU, cheap);
(B) connectivity post-processing (CPU); (C) a connectivity-trained model + a second seed (GPU); (D)
inference-only TTA (GPU); (E) final build and deliverables.

| Experiment | Change | Local CV PQ (mean ± std) | vs. prior REF | Decision |
|---|---|---|---|---|
| E018a | widen cfg1/regrow grid, E004+E007+E015 members, vs E013 | 0.4161 ± 0.0058 | +0.0075 | **keep** |
| E018b_E007E015 | drop E004, refit widened grid on E007+E015 | 0.4217 ± 0.0045 | +0.0057 | **keep** — new REF |
| E018b_E004E007E015 | add E004 back under widened grid | 0.4184 ± 0.0058 | +0.0023 (ensemble bar) | reject |
| E018b_weights | per-fold member-weight search incl. E004 | 0.4213 ± 0.0034 | +0.0053 (ensemble bar) | reject |
| E019 | scorer v2: cross-member agreement features + HGB classifier | 0.4199 ± 0.0060 | −0.0019 | reject (inner-split overfitting) |
| E020 (gate only) | connectivity encoder (efficientnet-b4) + skeleton-recall loss, model folds 0–1 | 0.419 | +0.0053 (needs +0.008) | reject — folds 2–4 never trained |
| E021 (gate) / E021 (final) | second E007 seed (1337) | gate +0.0084 → final 0.4243 ± 0.0047 | +0.0023 (needs +0.010, ensemble bar) | reject |
| E022 | fragment-merging post-process (skeleton-endpoint join) | 0.4227 ± 0.0041 | +0.0010 (needs +0.005) | reject — also raised merge errors |
| E023_4flip | TTA: 4-flip views only, isolating the scale effect | 0.4209 ± 0.0062 | −0.0009 | reject |
| **E023_4flip_scale** | TTA: 4-flip + 1.25x upscale (no downscale) | **0.4302 ± 0.0052** | **+0.0082** (clears pipeline bar) | **keep — new REF** |
| E024 | final build: fit on all 707 OOF, apply to test mean | 0.4302 (local CV, same members) | — | frozen; **neither candidate submitted** |

Six of ten tried changes were rejected. Every rejection was a real, pre-registered decision bar (P5's
"early"/"ensemble"/"pipeline" bars from `reports/05_ensemble_report.md` §2) failing, not a vibe call —
see §3.

## 3. Why six of ten ideas didn't make it

**Widening the grid (E018a) worked; adding members and weighting them back in (E018b_E004E007E015,
E018b_weights) didn't.** E004 is weaker than E007/E015 and, once the postprocessing grid is wide enough
to use E007+E015 well, E004 is net-negative at every weight the search tried, including the lowest
(0.15). The 2-member REF (E007+E015, 0.4217) beat every 3-member variant. This confirms Spec 05's §2
finding (E007 alone ≈ E004+E007 on folds 0–1) at the ensemble level, not just per-model.

**Scorer v2 (E019) won the inner-split selection in 4/5 folds but lost the outer one.** The
cross-member-agreement + boundary-contrast features, with an early-stopped HGB classifier, beat the
existing linear scorer on each fold's own inner validation split by small margins (e.g. 0.4297 vs
0.4252) — but on the held-out CV fold the mean delta was **−0.0019** and 0/5 folds were positive. That
pattern (wins inner, loses outer) is the signature of overfitting the inner split's own idiosyncrasies,
not finding a real signal. The FN-by-size-quartile breakdown it produced is still useful: 37% of all FN
are in the smallest GT-area quartile vs ~20% expected under uniform area — faint/small filaments really
are disproportionately missed — but scorer v2 did not close that gap.

**The connectivity-trained model (E020) failed its own early gate before folds 2–4 were ever trained.**
E020 (efficientnet-b4 encoder + a skeleton-recall loss term, meant to reward connected rather than
fragmented predictions) was tested on model folds 0–1 only, per the spec's two-stage protocol — train
the cheap subset first, only pay for the rest if it clears a bar. Pooled delta vs REF was +0.0053 (bar
needs +0.008) and the SQ delta was *negative* (−0.0022); the split-error count fell only 6.2% where the
early-bar's alternate path needed ≥15%. Folds 2–4 were never trained — this saved an estimated three to
four GPU-hours for a change that, on the evidence available, wasn't going to clear the full bar either.

**A second E007 seed (E021) passed its cheap gate and still didn't clear the real bar.** This is the
clearest illustration of why Spec 05's two-tier bar design (a cheap "early" gate that justifies spending
the rest of the compute, and a stricter "ensemble" bar for the final decision) matters: E021 passed the
model-folds-0-1 early gate convincingly (pooled delta +0.0084, both folds positive), which correctly
justified training folds 2–4 on the pod. But the full 5-fold ensemble promotion bar needs a mean delta
≥0.010 with ≥4/5 CV-folds positive and a bootstrap CI excluding 0; E021's full result was +0.0023 mean,
only 3/5 folds positive, CI [−0.0011, 0.006] including 0. A second same-architecture seed adds some
variance reduction but not a different enough error profile to move the ensemble meaningfully — exactly
what Spec 06 §2.3 predicted going in ("diversity has to be in what the model is trained to get right").
Finding this cost the early-gate compute plus three pod-trained folds — a few GPU-hours, not wasted
exactly, but a genuine example of a cheap gate not being a perfect predictor of the expensive bar.

**Fragment merging (E022) traded the wrong error for another.** The hypothesis — join kept instances
whose skeleton endpoints are close, aligned, and connected by high-probability pixels — worked exactly
as designed on the error type it targeted: one-to-many (fragmentation) FN fell 201→184 and FP fell
415→381. But many-to-one (merge) errors *rose* (FN 266→280, FP 131→138), net against the spec's
explicit guardrail that this change must not increase merges. The net PQ effect (+0.0010 mean, CI
including 0) was also below the pipeline bar regardless. The code (`merge_fragments` in
`postprocess.py`, `MERGE_GRID`/`choose_merge` in `cv_pipeline.py`) stayed in the codebase — it's correct
and tested, just not adopted.

**TTA flip views alone did nothing; adding a scale did everything (E023).** This was the one experiment
designed as a controlled pair specifically to isolate which part of a TTA change was doing the work.
4-flip-views-only (identity + h + v + combined-flip, vs. the pre-existing 3-view default) changed PQ by
−0.0009 — noise. 4-flip + a 1.25x upscale-only pass (never downscale — these filaments are often 2px
wide, and downscaling destroys that) moved PQ by **+0.0082 pooled, 5/5 folds positive, CI [0.0042,
0.0127]**, and also improved SQ (+0.0017) rather than trading it away. The two runs used identical code
and identical members (E007+E015), differing only in whether `tta_scales` included 1.25 — the
flip-only run isolates that the gain is specifically a resolution effect (the model sees more detail in
thin structures at a larger effective receptive field per pixel), not from the extra flip view. This
became the pipeline's REF, superseding E018b_E007E015 (0.4217 → 0.4302) and, with it, E013 as the
project's best local-CV result.

## 4. The frozen pipeline (E024)

Members **E007 (efficientnet-b4) + E015 (tu-regnety_040)**, both evaluated with 4-flip + 1.25x-upscale TTA
inference from each fold's `best.pt` EMA weights (`scripts/infer.py`, outputs renamed `E007tta4s`/
`E015tta4s` so `src/solar_k/ensemble.py`'s existing member-resolution logic picked them up with no new
plumbing). Post-processing: hysteresis cfg1 (`hys0.4-0.99_r3`), no shape gate, scorer cutoff c=0.325,
regrow (t_mask=0.4, d=4) — all fit per-fold under P5 for the CV number, then refit once more on **all
707 OOF images with no CV split** (`scripts/build_final.py`, which factors the same fit-and-write logic
`scripts/run_p5.py --write-submission` already used) for the actual submission build.

- **Primary candidate — `E024_primary.csv`**: E007tta4s+E015tta4s, 1008 instances, local CV PQ **0.4302**.
- **Backup candidate — `E013.csv`**: Spec 05's original E004+E007 ensemble, 1029 instances, local CV PQ
  0.4086, public LB **0.37** (the only one of the two with a real public score, since it was already
  submitted in Spec 05).

Both were validated with `scripts/validate_submission.py`. **Neither has been submitted.** Choosing
between them — and whether to submit either — is explicitly Nicky's call (spec §7); this report records
the facts needed to make that call and takes no further action on either CSV.

### 4.1 Quantitative breakdown (OOF, protocol P5, E007tta4s+E015tta4s)

*Numbers in this subsection come from `data/interim/E024_quant_p5_summary.json`,
`E024_quant_p5_details.csv`, and `E024_quant_p5_pixel_dice.csv`, produced by
`scripts/run_p5.py --members E007tta4s E015tta4s --experiment E024_quant --save-details` on the RunPod
pod — a rerun of the same E023_4flip_scale/E024 pipeline with per-component and per-image detail
persisted instead of discarded. By-site and by-area-quartile PQ below are aggregated directly from that
run's per-image and per-component rows (pooled TP/FP/FN and summed TP IoU within each slice), not
re-fit per slice.*

**PQ / SQ / RQ:** PQ 0.4302 (CV mean 0.4300 ± 0.0052), SQ 0.6813, RQ 0.6313 (pooled across all 707 OOF
images, TP=4765, FP=2131, FN=3434).

**Per-image pixel Dice:** mean 0.6472, median 0.6794, p10/p90 0.4745 / 0.7968 (per-image mean across
annotators, of the semantic union of kept instances vs. semantic union of each annotator's GT, per
`src/solar_k/evaluate.py:pixel_iou_dice`; 1154 annotator-image pairs over 707 images).

**Per-TP IoU:** mean 0.6813, median 0.6859, p10/p90 0.5590 / 0.7914 (n=4765 matched pairs). The mean
here equals pooled SQ by construction (SQ is the mean matched-pair IoU).

**One-to-many (split, class b) / many-to-one (merge, class d):** FN/FP 215/445 (b) and 215/105 (d)
(pooled across all 707 OOF images), vs. E013's 243/502 (b) and 210/102 (d) — both error classes are
essentially flat to slightly improved, not the main story. The real shift is in class a (undetected,
no overlapping prediction): FN fell from 2513 (68.4% of E013's 3675 total FN) to 2386 — but FN also fell
overall (3434 total), so class a's **share** of FN is 69.5%, not lower. The recall gain is a reduction in
the *absolute count* of missed filaments (−127 class-a FN, −241 FN overall, −6.6%), not a change in which
error type dominates; class a remains, as in every prior spec, the large majority of what's still missed.

**By GONG site:** (B=Big Bear, C=Cerro Tololo, L=Learmonth, M=Mauna Loa, T=El Teide, U=Udaipur)

| Site | n images | PQ | SQ | RQ |
|---|---|---|---|---|
| U (Udaipur) | 102 | 0.4527 | 0.6852 | 0.6607 |
| T (El Teide) | 110 | 0.4432 | 0.6807 | 0.6511 |
| B (Big Bear) | 118 | 0.4336 | 0.6865 | 0.6316 |
| L (Learmonth) | 127 | 0.4223 | 0.6854 | 0.6162 |
| M (Mauna Loa) | 125 | 0.4202 | 0.6745 | 0.6230 |
| C (Cerro Tololo) | 125 | 0.4143 | 0.6757 | 0.6131 |

A ~0.04 PQ spread across sites (0.4143–0.4527), driven mostly by RQ (recall/precision), not SQ — SQ is
flat (0.674–0.687) across every site. No site is an outlier; this reads as ordinary station-to-station
image-quality variation, not a site the pipeline fails on.

**By GT area quartile:** PQ 0.3091 (q1 smallest) / 0.4373 / 0.4611 / 0.4943 (q4 largest). The smallest
quartile remains by far the weakest slice (RQ 0.4672 vs 0.70+ in q3/q4) — a direct echo of E019's finding
(§3) that 37% of all FN concentrate in the smallest-area quartile. A same-granularity q1-q4 PQ breakdown
for E013 wasn't computed (E013's P5 run predates the `--save-details` flag added for this analysis), so
the *size* of the gap-narrowing vs. E013 isn't quantified here — only the direction (recall improved
overall, §3) is established.

### 4.2 Qualitative items

- **Pipeline diagram**: `reports/figures/06_pipeline_diagram.png` — image → per-fold U-Net (5-fold CV,
  two encoders) → TTA inference → member-mean probability map → cfg1/shape-gate/scorer → regrow →
  instance masks, with the per-fold-refit box called out.
- **H-alpha overlay gallery** (OOF only): `reports/figures/06_overlay_gallery.png` — four cases, picked
  from the per-image error counts in `data/interim/E013_p5_per_image.csv` (for split/merge extremes) and
  cross-checked against `E024_quant_p5_per_image.csv`: a **typical** image (`20140509071314Th`), the
  **worst split/fragmentation** case (`20111215063134Lh`, FP−FN=+14 under E013), the **worst merge**
  case (`20110914063134Lh`, FN=41 with only FP=10 under E013), and a **recovered faint filament** case
  (`20140216230134Lh`) where the TTA pipeline's recall gain over E013 is visible directly in the
  overlay. Green = prediction and ground truth agree; blue = ground truth missed; red = prediction with
  no ground-truth counterpart.

## 5. Compute used this spec

Training: E020 folds 0-1 (RunPod RTX PRO 4500 Blackwell, $0.72/h) + E021 folds 0-1 (Kaggle T4, free) +
E021 folds 2-4 (RunPod RTX 4090). Postprocessing/evaluation: everything in the table in §2 that isn't
training, split between Mac CPU (E018/E019, cheap grid searches on cached component tables) and RunPod
RTX 4090 CPU-side P5 passes (E020-E024, once members needed pod-resident OOF caches). TTA inference for
E023/E024 (regenerating OOF+test probability maps from `best.pt` for both members, all 5 folds, two TTA
variants) ran on the RunPod 4090's GPU. Total new RunPod spend this spec stayed within the ~$10 buffer
`docs/specs/CODING_SPEC_06_RECALL_AND_CONNECTIVITY.md` §5 budgeted after E020/E021.

## 6. Verified facts vs. assumptions

**Verified, with evidence cited above:** the inter-annotator PQ ceiling (0.3399); every leaderboard
defect in §1; every decision-bar pass/fail in §2 (each has a `experiments.csv` row with the exact
pooled delta, CI, and fold-positivity count); the E022 merge-class regression (before/after FN/FP
counts from its own summary JSON); the E023 flip-vs-scale isolation (two runs, one changed variable).

**Assumptions carried forward, not re-verified this spec:** the day-grouped 5-fold CV split
(`data/splits/folds_v1.csv`) still assumed leakage-safe per Spec 01's construction; the P5 run-twice
determinism check (Spec 05 §2.3) was not re-run for every new experiment this spec, only relied upon as
already-established; `instance_stats.csv`'s GT-area quartile edges are the same ones computed in Spec
05 and reused unchanged for every size-bucket breakdown in this report.

## 7. Status and next action

Everything in `docs/specs/CODING_SPEC_06_RECALL_AND_CONNECTIVITY.md`'s experiment queue is run and
decided. The deliverables in §6 of that spec are complete per this report and the repository state:
code (grid/weights, scorer v2, fragment merging, skeleton-recall loss, TTA views+scales,
`scripts/infer.py`, `scripts/build_final.py`, E020/E021 configs, tests — all committed, `pytest` green,
132 tests passing), the quantitative breakdown above, the qualitative artifacts in §4.2, this report,
`README.md`'s "reproduce from scratch" section, and `notebooks/06_pipeline_demo.ipynb` (imports only
`src/solar_k`, runs one CV fold of the real P5 protocol CPU-only against the checked-in OOF probability
maps, no GPU or network required).

**The only action remaining is Nicky's: pick primary vs. backup (or neither), and decide whether and
when to submit to Kaggle.** No code in this repository will do that automatically.
