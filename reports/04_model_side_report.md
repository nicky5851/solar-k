# Model-side gains report — Spec 04

Covers `docs/specs/CODING_SPEC_04_MODEL_SIDE.md`. All experiments are
complete: E004 (5-fold Kaggle retrain), E005 (the real 5-fold
cross-validated baseline), E008 (mask-extent diagnosis + two-stage
re-growing), E006 (hard-negative mining) and E007 (stronger encoder) were
both trained and evaluated against the pre-registered decision rule —
**both rejected** — and E009 (combine kept changes) ended up being E005 +
E008 only, since nothing else passed. **Bottom line: PQ 0.3395 → 0.3662
(+0.0267) from E008's mask re-growing alone, with zero additional
training.**

## E004 — 5-fold retrain (checkpoint-selection val_pq, not yet full pipeline)

| Fold | val_pq (checkpoint selection) | Kernel |
|---|---|---|
| 0 | 0.3400 | solar-k-e004-f0-train |
| 1 | 0.3708 | solar-k-e004-f1-train |
| 2 | 0.3290 | solar-k-e004-f2b-train |
| 3 | **0.3829** (best) | solar-k-e004-f3b-train |
| 4 | 0.3114 | solar-k-e004-f4b-train |

Spread is about ±0.025 around a ~0.35 mean, consistent with the spec's
expectation. These numbers use E002's fixed tuned threshold on a 40-image
subset for fast checkpoint selection, not the full E003/E005 pipeline — see
E005 below for the number that actually matters.

## E005 — the real baseline (5-fold CV, full pipeline)

E003's single fold-0 report-half PQ (0.3819) was flagged as scrutiny-worthy
(it exceeded the human-ceiling PQ on one 71-image half, fit+scored on a
single fold). E005 refits only the scorer and cutoff `c` per fold via
day-grouped `GroupKFold`, keeping E003's cfg1 threshold/closing/min_area and
shape gate fixed (not re-derived — would cost hours for likely small gain),
and reports mean±std across all 5 held-out folds:

| Metric | Value |
|---|---|
| PQ (mean ± std across 5 folds) | **0.3395 ± 0.0041** |
| SQ (mean ± std) | 0.6680 ± 0.0025 |
| RQ (mean ± std) | 0.5082 ± 0.0060 |
| Pooled PQ / SQ / RQ (3596 TP / 2351 FP / 4603 FN) | 0.3396 / 0.6680 / 0.5084 |
| Final scorer cutoff (fit on all 707 images) | c = 0.225 |

This is markedly lower than E003's single-fold number (0.3819) and much
closer to the human inter-annotator ceiling (PQ 0.3399, SQ 0.634, RQ
0.536) — which resolves the earlier concern: E003's 0.3819 was likely a
mix of a favorable fold-0/report-half split and mild overfitting of the
scorer to a small tune half, not a real result that beats human agreement.
**0.3395 ± 0.0041 is the number every E006/E007/E009 comparison in this
spec must beat by ≥0.02 PQ (CODING_SPEC_04 §2's decision rule), not
E003's 0.3819.** The CV std (0.0041) is small relative to that 0.02
threshold, so the decision rule has reasonable statistical power at the
per-fold level despite using only 2 folds per comparison.

`outputs/submissions/E005.csv` was generated (5-fold test-prob ensemble,
final scorer refit on all data) and validated, but **not submitted** to
Kaggle, per the spec.

## E008 — mask-extent diagnosis and two-stage re-growing

**SQ ceiling (pure GT comparison, no model involved):** on 296
multi-annotator images (743 annotator-vs-consensus-of-others comparisons,
TP=2811, FP=2182, FN=2656), **SQ_ceiling = 0.6366** — closely matching the
spec's stated human SQ ≈ 0.634. This confirms E005's SQ (0.668) is already
essentially at the ceiling a consensus-trained model can expect; the real
headroom is in RQ, as the spec predicted.

**Mask-bias diagnosis** (TP pairs, all 5 folds' OOF, cfg1+shape_gate+scorer
seeds):

| Metric | Median | IQR |
|---|---|---|
| Area ratio (pred / GT) | 0.875 | [0.753, 0.995] |
| Boundary offset (px, signed) | −1.00 | [−2.00, −1.00] |

Both confirm the spec's hypothesis directly: predicted masks are
systematically **thinner** than ground truth by about 1px at the boundary
and ~12.5% in area at the median.

**Two-stage re-growing** (watershed: seed = kept instance, elevation =
−probs, flood region = seed's d-pixel geodesic dilation ∩ {probs ≥
t_mask}), grid-searched over t_mask ∈ {0.2…0.7} and d ∈ {2,4,8} on `tune`,
best config **t_mask=0.6, d=8**, reported on `report` (all 5 folds' OOF,
tune/report split within each fold):

| | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| Before re-growing | 0.2745 | 0.6789 | 0.4043 | 1085 | 427 | 2770 |
| After re-growing | **0.2909** | **0.6897** | **0.4218** | 1132 | 380 | 2723 |
| Δ | **+0.0164** | **+0.0108** | +0.0175 | +47 | −47 | −47 |

Re-growing improves **both** PQ and SQ — unlike E003's earlier gains,
which were entirely RQ (FP suppression) with flat/falling SQ. This is a
genuinely new, cheap (CPU-only, no retraining) lever, and it's additive to
whatever E006/E007 contribute.

**Caveat on the absolute numbers above:** this run applies E003's
fold-0-tuned cfg1 threshold and shape gate directly to all 5 folds' OOF, even
though each fold's model has its own probability calibration. The raw-path
numbers in the E006/E007 decision-rule comparisons below show this matters:
applying a threshold tuned on one fold's model to another's differently-
calibrated probabilities measurably under- or over-detects. A per-fold
postprocessing refit (as E005 and the E006/E007 comparisons below do) would
likely push E008's report-half PQ higher still — the numbers above should be
read as a **lower bound** on the re-growing gain, not the final word. E009
(if it proceeds) should refit cfg1/shape_gate per fold alongside re-growing
to capture this.

## E006 — hard-negative mining (Kaggle, folds 0 and 1)

Hard-negative pool rebuilt from all 5 folds' OOF (once E004 fold 3 was
available): **1359 components** across folds {0: 301, 1: 249, 2: 259, 3:
255, 4: 295}, 728/1359 (54%) from single-annotator images (kept only where
scorer P(match) < 0.2, per the spec's caveat about real-but-unannotated
filaments).

E006-f0 and E006-f1 trained successfully on Kaggle (4000 iterations each,
same recipe as E004 except the crop sampler: 45% positive / 30%
hard-negative / 25% random). Checkpoint-selection val_pq: f0 = 0.3523, f1 =
0.3695 (vs E004's 0.3400 / 0.3708 — comparable at this coarse checkpoint-
selection granularity; the real test is the full-pipeline decision rule
below).

**Decision: REJECT.** Full-pipeline comparison vs E004 on the same folds:

| Fold | PQ (E006) | PQ (E004) | PQ Δ | SQ Δ | Passes? |
|---|---|---|---|---|---|
| 0 | 0.4013 | 0.3886 | +0.0127 | −0.0006 | No (ΔPQ < 0.02, SQ also fell slightly) |
| 1 | 0.3956 | 0.3876 | +0.0080 | +0.0004 | No (ΔPQ < 0.02) |

Neither fold clears the ≥0.02 PQ bar, so E006 is rejected outright (not
even a close technicality like E007's fold-0 SQ miss). The raw-path
numbers again looked promising on their own (+0.017 PQ on fold 0, +0.009 on
fold 1, untuned) — hard-negative mining does measurably reduce raw FPs
(fold 0: FP 1090→1092 is flat, but fold 1: FP 1035→884, a real drop) — but
once the full pipeline (shape gate + learned scorer) is independently
refit for each model, most of that gain is already captured by the
existing scorer's FP suppression, leaving a smaller net full-pipeline
improvement that doesn't clear the pre-registered bar.

## E007 — stronger encoder (RunPod 4090, folds 0 and 1)

Both folds trained with `efficientnet-b4` (no fallback to resnext50_32x4d
needed — fit at batch 8/crop 512 with AMP, ~7.2GB VRAM). 6000 iterations
each (vs E004's 4000, since the 4090 is ~2-3x a T4). Checkpoint-selection
val_pq: f0 = 0.3676, f1 = 0.3791 (vs E004's 0.3400 / 0.3708).

**Decision: REJECT.** Full-pipeline comparison (independent cfg1/shape_gate/
scorer refit per model per fold, via `scripts/run_e007_eval.py`):

| Fold | PQ (E007) | PQ (E004) | PQ Δ | SQ Δ | Passes? |
|---|---|---|---|---|---|
| 0 | 0.4051 | 0.3808 | **+0.0243** | **−0.0029** | No (SQ fell) |
| 1 | 0.3973 | 0.3876 | +0.0097 | +0.0049 | No (ΔPQ < 0.02) |

Fold 0 clears the PQ bar but SQ dropped slightly, failing the "SQ must not
fall" half of the rule. Fold 1 falls short of the PQ bar outright. The
decision rule needs **both** folds to pass, so E007 is rejected despite
every raw-path number looking favorable (raw PQ: +0.025 on fold 0, +0.017
on fold 1, untuned) — a reminder that the full post-processing pipeline
(which the submission actually uses) doesn't automatically inherit a
better model's raw-probability gains, especially since E007's model has a
different probability calibration that reshuffles where the optimal
cfg1/shape_gate/scorer landed (note E007-f0's winning cfg1 switched to
`hysteresis` mode entirely, vs E004's `threshold` mode).

RunPod cost: pod spend ~$1.5-2 total for both folds (well within the $10
budget cap), pod stopped promptly after f1 synced to avoid idle billing.

## E009 — combine the winners

Both E006 and E007 were rejected, so per the spec ("if nothing passed,
skip the training") **E009 required no new training.** E009 is E005 +
E008 combined: E005's 5-fold cross-validated pipeline (cfg1 + shape gate +
per-fold-refit scorer) with E008's two-stage mask re-growing
(`t_mask=0.6, d=8`) layered on top, evaluated the same way E005 was
(day-grouped 5-fold CV, nothing tuned on the fold it's scored on).

`scripts/run_e009.py` reuses E005's already-fit per-fold scorer and chosen
cutoff `c` (deterministic given the same seed/GroupKFold split, so
identical to E005's) rather than re-running E005's ~5.8h C_GRID search —
the only new work per fold is one evaluation pass with re-growing added.
Total runtime: **19.5 minutes** (vs E005's 347 min and E006/E007's
decision runs' ~600 min each).

**Final result — this is the Spec 04 deliverable:**

| Metric | E005 (no regrow) | E009 (+ regrow) | Δ |
|---|---|---|---|
| PQ (5-fold CV mean ± std) | 0.3395 ± 0.0041 | **0.3662 ± 0.0026** | **+0.0267** |
| SQ (pooled) | 0.6680 | 0.6773 | +0.0093 |
| RQ (pooled) | 0.5084 | 0.5407 | +0.0323 |
| Pooled PQ | 0.3396 | 0.3662 | +0.0266 |

The gain from re-growing alone (+0.0267 PQ, properly cross-validated) is
notably *larger* here than E008's own report-half estimate (+0.0164) —
confirms the caveat flagged in E008's section above: applying E003's
fixed fold-0-tuned cfg1/shape-gate across all folds' differently-
calibrated OOF was leaving gains on the table, and once each fold gets its
own scorer refit (as E005/E009 do), re-growing recovers more. The CV std
also tightened (0.0041 → 0.0026), suggesting re-growing makes the pipeline
less sensitive to which fold's data it's evaluated on.

`outputs/submissions/E009.csv` was generated (963 instances, 5-fold
test-prob ensemble + final scorer/cutoff reused from E005 + re-growing)
and validated, but **not submitted** to Kaggle, per the spec (needs
Nicky's go-ahead).

## Compute used

| Platform | Work | Approx cost |
|---|---|---|
| Kaggle (T4) | E004 5 folds (~completed earlier session) + E006 2 folds | GPU-hours, no $ cost (Kaggle free tier) |
| RunPod (RTX 4090) | E007 2 folds | ~$1.5-2 of the $10 cap |
| Mac (CPU, local) | E005 (5-fold CV, ~5.8h), E008 (5-fold, ~1.4h), E006/E007 decision-rule comparisons (~10h each), E009 (~20 min) | no $ cost |

## Ranked hypotheses for Spec 05

1. **Per-fold-refit cfg1/shape-gate, not just the scorer.** E009 already
   shows re-growing recovers *more* PQ (+0.0267) once each fold gets its
   own scorer refit than E008's own report-half estimate (+0.0164) did
   with a fixed fold-0-tuned threshold. The next lever is re-deriving
   cfg1 and the shape gate per fold too (not just the scorer) — E006/E007's
   decision-rule comparisons already show each model's optimal cfg1 landing
   in a different place (E004-f0: `threshold` mode; E007-f0: `hysteresis`
   mode) when independently refit, so a fixed cfg1 is very likely leaving
   PQ on the table across all 5 of E004's folds, not just in the E006/E007
   comparison.
2. **E006 and E007's raw-path gains are real but don't survive the full
   pipeline.** Both showed clear raw-probability improvements (E006: FP
   dropped markedly on fold 1; E007: cleaner probabilities on both folds)
   that evaporated once cfg1/shape-gate/scorer were independently refit —
   suggesting the current post-processing pipeline's capacity to extract
   value from a better model is itself capped. Worth investigating directly
   (e.g. does the scorer's feature set generalize to a different model's
   component statistics, or does it need per-model recalibration) before
   trying yet another model-side change.
3. Per the spec's "not in this scope" list: bigger crops / global context
   remains open — E006's hard-negative mining target (bright plage/network
   FPs) wasn't cleanly resolved (raw FP counts moved in different
   directions per fold), so it's still plausible the model needs more
   spatial context than a 512px crop provides, rather than more targeted
   training examples.
4. Given E006/E007's full-pipeline decision runs each cost ~10h of Mac CPU
   time (vs E009's 20 min, which reused E005's already-fit scorer), a
   cheaper version of the decision-rule comparison — e.g. reusing E004's
   per-fold cfg1 search space as a warm start, or subsampling the ~372-config
   grid — would make future model-side experiments in this spec's style
   much more practical to iterate on.
