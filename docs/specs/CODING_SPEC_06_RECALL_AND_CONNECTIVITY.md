# CODING_SPEC_06_RECALL_AND_CONNECTIVITY — the final push: find the filaments we miss entirely, keep them in one piece, and finish the deliverables

**Phase 3, cycle 4 + Phase 4. Time box: Nov 4–10, freeze Nov 11.** Read `CLAUDE.md`, `reports/05_ensemble_report.md`, and `docs/specs/CODING_SPEC_05_ENSEMBLE.md` §2 (the P5 protocol, which this spec reuses unchanged) and §8 (the official rubric).
Spec 05 is **finished**: E013 is the current primary submission and E016 has been built. This spec starts from there. Training is allowed here, but only as described.

## 0. Where we are (verified from `data/interim/*_p5_summary.json` and `experiments.csv`, 2026-11-04)

| Variant | Members | P5 local PQ (mean ± std) | pooled SQ / RQ | TP / FP / FN | Public |
|---|---|---|---|---|---|
| E009 (old pipeline) | E004 | 0.3662 ± 0.0026 | 0.677 / 0.541 | 3825 / 2124 / 4374 | 0.31 |
| E010 | E004 | 0.3948 ± 0.0067 | 0.676 / 0.584 | 4321 / 2275 / 3878 | 0.35 |
| **E013 (primary)** | E004 + E007 | **0.4086 ± 0.0050** | 0.680 / 0.601 | 4524 / 2333 / 3675 | **0.37** |
| E015_eval (rejected, +0.0074 < 0.010 bar) | + E015 | 0.4159 ± 0.0070 | 0.680 / 0.612 | 4627 / 2301 / 3572 | 0.37 |

E011 (284 images, model folds 0–1): **E004 alone 0.3892, E007 alone 0.4128, mean(E004, E007) 0.4138.**

E013 loss decomposition (FN/FP by error class, from `E013_p5_summary.json`): **a: FN 2513 / FP 1020** (no overlapping counterpart at all); **b: FN 243 / FP 502** (fragmentation, several predictions on one GT); **c: 709 / 709** (near-miss pairs, IoU < 0.5); **d: FN 210 / FP 102** (merging).

## 1. Goal and reality check: can we reach 0.50?

**Honest answer: public 0.50 is very unlikely, and local 0.50 is unlikely. Aim for local 0.43–0.46. Expect public about 0.39–0.42.**
- Local CV went 0.366 → 0.409 in Spec 05, with a nearly constant local-to-public gap of about 0.04 (E010 0.395 → 0.35; E013 0.409 → 0.37). A local 0.45 would land near public 0.41.
- Two expert annotators agree at **PQ 0.34**. The best transparent outside result is **OOF 0.45 / public 0.39**. Public 0.50 would need us to beat that by +0.11 on about 90 images.
- PQ is one of four quantitative rubric items, and 30% is qualitative (Spec 05 §8.1). Polishing the Dice/IoU distributions, split/merge behaviour and the writeup is worth real points, while the last +0.01 PQ is worth few.

**Operational goal:** the largest P5 gain over **E013 (0.4086)** by Nov 9, then a final build and writeup that cover all seven rubric items.

## 2. What we learned, connected (this is the strategy)
1. **Post-processing fitted to the model is the biggest lever we've found.** Refitting post-processing per CV fold took E004 from 0.366 to **0.395 (+0.029)**. Ensembling added +0.014, the third model +0.007. **Every P5 winner sits at a grid edge:** hysteresis `t_high = 0.975` (the maximum) on every fold of every variant, `t_low = 0.5` (the minimum) on every ensemble fold, and re-growing `d = 8` (the minimum) on every fold. The optimum is probably outside the grid. Widening it costs CPU minutes.
2. **The bottleneck is filaments we never detect.** 2513 of E013's 3675 FN (68%) have **no overlapping prediction at all**. Near-misses account for 709 more, fragmentation-related FN 243, merges 210. SQ (0.680) is already above the human ceiling (0.637). Recall of faint or small filaments is where the points are.
3. **E007 (efficientnet-b4) is the strongest model, and E004 mostly dilutes it.** On folds 0–1, E007 alone (0.4128) is as good as E004+E007 (0.4138), and the mean has *more* fragmentation FP (176 vs 143). E015 helped a little (+0.007). So the next members should be **strong, b4-class models with a different error profile**, not more weak ones, and E004's weight should be allowed to fall.
4. **Diversity has to be in what the model is trained to get right.** E015 (a different encoder, same loss and crop) added little. A member trained for **thin-structure connectivity and more context** attacks classes a, b and c directly.
5. **The ensemble's member-agreement signal is unused.** A faint component that two members see but one doesn't is exactly the class-a/FP battleground. The scorer currently sees only the averaged map.

Priority order: **(A)** widen the grids, member weights and subsets, a scorer v2 with agreement features (all CPU, cheap); **(B)** connectivity post-processing (CPU); **(C)** one new connectivity-trained b4 model plus a second E007 seed (GPU); **(D)** inference-only TTA (GPU); **(E)** final build and rubric deliverables.

## 3. Protocol and decision rule
Unchanged from Spec 05 §2.1 (P5: day-grouped 5-fold CV over 707 OOF images, nested refit, paired, deterministic), with Spec 05 §8.2's extra reporting (Dice/IoU distributions, split/merge counts). **REF = E013** until something beats it.
- Pipeline-only change: mean paired ΔPQ ≥ **+0.005**, > 0 on ≥ 4/5 folds, SQ drop ≤ 0.005, bootstrap CI excludes 0, splits + merges not up by more than 5%.
- New member / member swap: ΔPQ ≥ **+0.010**, same other conditions. Exception: dropping a member, or replacing E004, needs only ΔPQ ≥ 0 with fewer splits + merges, because simpler is better under the rubric.
- **Stacking rule:** each promoted change becomes the new REF before the next experiment is compared.
- Public LB is never a decision input.

## 4. Experiments (ranked by expected value per hour)

### E018 — widen the grids, member weights and subsets (Mac, CPU, ~2–3 h total) — do first
- **Grid extension** inside P5: hysteresis `t_low ∈ {0.35, 0.4, 0.45, 0.5}`, `t_high ∈ {0.975, 0.985, 0.99}`; threshold mode `t ∈ {0.45, 0.5}` added; re-growing `d ∈ {4, 6, 8}` and `t_mask ∈ {0.4, 0.5, 0.6}`. Rerun E013's members (E018a). A promoted grid becomes the default for all later variants.
- **Member subsets and weights** with the widened grid: {E007}, {E007, E015}, {E004, E007, E015}, and weighted mean `w_E004 ∈ {0, 0.15, 0.3}` with the remainder split equally between E007 and E015. **Weights are chosen inside each P5 train portion only** (add `weights` to the nested refit), so this is still leakage-free. E015 is allowed back in here: it was rejected against E013 at +0.0074, but against an E004-down-weighted reference it may clear.
- Every run is about 25–35 min of CPU (E013 took 25.9 min). Run them **one at a time** (§5).

### E019 — scorer v2: agreement features + recall-oriented admission (Mac, CPU, ~2 h)
- Per-component features from each member's own map, alongside the averaged-map features: mean / min / max across members of the component's mean probability, **std across members**, the fraction of members with ≥ 50% of the component above 0.5, and the mean probability on a 1-px ring outside the boundary (contrast).
- Classifier: current scorer vs `HistGradientBoostingClassifier(random_state=0, max_iter=300, early_stopping=True)` on a day-grouped inner split. Choose inside the P5 train portion.
- Agreement features come only from the members that exist for that OOF image (fold-matched). At test time, compute them per fold and then average, so train and test feature distributions match. Document this.
- Report FN class a by GT area quartile, to show whether faint or small filaments are recovered.

### E020 — connectivity-trained b4 model (GPU: RunPod; gate on folds 0–1, then all 5)
**Hypothesis:** a b4 U-Net trained with a skeleton-recall loss and more context detects more of the class-a filaments and fragments less (class b), giving the ensemble a genuinely different error profile.
- Base: E007's recipe exactly (efficientnet-b4, EMA 0.999, PQ checkpoint selection, seed 42, 3-view TTA for saved maps). Two changes, pre-registered together because there's time for only one GPU cycle:
  - **Loss:** BCE + soft Dice **+ skeleton-recall loss** (Kirchhoff et al., ECCV 2024). This is a recall term on a precomputed, 1-px-dilated skeleton of the soft-consensus target binarised at 0.5, with `λ = 1.0`. Precompute skeletons once with `skimage.morphology.skeletonize` into the preprocessing cache **on the GPU machine, not the Mac**. Tests: a straight-line toy keeps 1-px connectivity; the loss is 0 for a perfect prediction.
  - **Crop 768** at batch 4 with AMP, 8000 iterations. Check VRAM in the first 50 iterations. If it doesn't fit, use crop 640 at batch 6 and record it.
- `configs/E020-f{0..4}.yaml`. **Gate:** folds 0 and 1 first. Run P5 on model folds 0–1, current REF members vs REF members + E020 (and vs E020 replacing E004). Continue to folds 2–4 if paired ΔPQ ≥ +0.008, **or** splits fall ≥ 15% with ΔPQ ≥ 0.
- Cost estimate: E007 took about 46 min per fold on a 4090 at crop 512 and 6000 iterations, so E020 at crop 768 and 8000 iterations is roughly 2.5–3× that. About **2–2.5 h per fold on a 4090** (about $2 per fold at $0.89/h), or about 3.5–4 h on an L4 ($0.59/h, about $2.2 per fold). Five folds come to about $10–11.

### E021 — second E007 seed (GPU: Kaggle T4, 5 folds, runs in parallel with E020)
- `configs/E021-f{0..4}.yaml`: E007 exactly, with **seed 1337** instead of 42. This is the lowest-risk way to add a strong member. Seed ensembles of the best model reliably reduce variance, and E007 is our best model.
- Estimated about 2.5–3 h per fold on a T4. Use two parallel sessions, about 13–15 Kaggle GPU-h in total. **Check the weekly quota first.** If it doesn't cover all 5 folds, run folds 0–1 as a gate (same rule as E020), and then run folds 2–4 on whichever GPU frees up first.
- Promotion as a new member (+0.010), or as a replacement for E004 (ΔPQ ≥ 0, fewer splits + merges).

### E022 — fragment merging + re-growing retune (Mac, CPU, ~1.5 h)
- Join kept instances whose endpoints are within `g ∈ {5, 10, 20}` px, whose skeleton directions at those endpoints differ by < `θ ∈ {20°, 40°}`, and whose connecting path has ensemble probability ≥ `p ∈ {0.2, 0.3}`. Tune inside P5, and it must not increase many-to-one merges. Report the one-to-many and many-to-one distributions before and after (rubric item).

### E023 — 4-flip + 1.25× scale TTA (GPU inference only, ~1–2 h on the pod after E020)
- Add `tta_views: 3 | 4` and `tta_scales: [1.0] | [1.0, 1.25]` to `unet.make_prob_fn` (defaults reproduce current maps exactly; add a test). Add `scripts/infer.py` to regenerate OOF and test maps from `best.pt` with EMA weights. **Run it only on RunPod or Kaggle, never the Mac.** No downscaling, because it destroys 2-px filaments.
- Regenerate maps for the final members only. Run P5 with a 4-flip-only row first, so the scale effect is separated out. Pipeline-only bar.

### E024 — final build and submissions (Mac, ~1 h, Nov 9–10)
- `scripts/build_final.py`: one command from saved probability maps → CSV → `validate_submission.py`. The final post-processing is fit on all 707 OOF images. **Don't submit.** Two final selections: **primary** = the best P5 variant at freeze, **backup** = E013 (local 0.4086, public 0.37, simplest strong ensemble). Nicky delegated this choice; record it in `experiments.csv`.

### Not doing, and why
Pseudo-labelling test images or any use of test for training (`CLAUDE.md`); external MAGFiLO labels (forbidden); the empty-mask evaluator defect (integrity, Spec 05 §1.1); Mask R-CNN / Mask2Former (merging is only 210 FN, and a new framework is high-risk with 6 days left); learned stacking beyond member weights.

## 5. Who runs what (minimum strain on the 8 GB Mac with a nearly full SSD)

**Mac rules:** run CPU-only P5 evaluations, **one at a time** (each about 25–35 min, streaming one image at a time). Never train, never run inference, never build preprocessing or skeleton caches locally. **Sync only `oof/`, `test_probs/` and `train.log` back from GPU machines, never checkpoints** (each `best.pt`/`last.pt` is 300–400 MB). Leave checkpoints on the pod volume or in Kaggle outputs. Before each sync, check `df -h ~`. If less than 5 GB would remain, stop and tell Nicky. Delete nothing without Nicky's go-ahead.

| Day | RunPod (one GPU pod: 4090 if available, else L4) | Kaggle T4 (2 parallel sessions) | Mac (CPU, one job at a time) |
|---|---|---|---|
| Nov 4 | E020-f0 → f1 (gate) | E021-f0, f1 (gate) | **E018** (grids, weights, subsets) |
| Nov 5 | E020-f2 → f3 → f4 if the gate passes; otherwise stop the pod | E021-f2, f3 if its gate passes | **E019**; then E020/E021 gate evals on folds 0–1 |
| Nov 6 | E020 continues; then **E023** TTA re-inference for the final members | E021-f4 | **E022** |
| Nov 7 | Stop the pod | — | Full 5-fold P5 with E020 and/or E021 |
| Nov 8 | — | — | **E023** eval; overlay gallery and figures (OOF only) |
| Nov 9–10 | — | — | **E024** final build; report and code-quality pass |
| Nov 11 | — | — | **Freeze** |

**Budget:** RunPod has $23.38 (Nicky, 2026-11-04). E020's 5 folds plus E023 come to about $11–13, leaving a $10 buffer. **Reuse Nicky's existing pod.** Create a new one only with Nicky's explicit go-ahead (standing instruction from Nov 2). Stop the pod whenever the queue is idle. Kaggle: check the weekly quota before E021.

## 6. Deliverables (rubric-mapped)
- **Code:** grid/weights in `cv_pipeline.py`; scorer v2 in `scorer.py`; fragment merging in `postprocess.py`; `losses.py` (skeleton recall) plus the skeleton cache; `tta_views`/`tta_scales` and `scripts/infer.py`; `scripts/build_final.py`; `configs/E020-f*.yaml`, `configs/E021-f*.yaml`. Tests for each.
- **Quantitative rubric items, for the final pipeline on OOF:** PQ/SQ/RQ; per-image **Dice** distribution; per-TP **IoU** distribution; **one-to-many / many-to-one** distributions. All broken down by GONG site and GT area quartile.
- **Qualitative rubric items:** a pipeline diagram; an H-alpha overlay gallery (OOF only) with typical cases, the worst split, the worst merge, and recovered faint filaments; module docstrings across `src/solar_k/`; a README "reproduce from scratch" section; one end-to-end notebook importing only `src/`; `pytest` green.
- `experiments.csv` rows E018–E024 with decisions; public scores only where Nicky submitted.
- `reports/06_final_report.md`: opens with the leaderboard-trust caveat (Spec 05 §1.1) and the 0.34 human ceiling, then the progression table in §0, then sections mapping one-to-one onto the seven rubric bullets. This becomes the draft for the 4-page paper.

## 7. Decisions
- **Settled (Nicky, 2026-11-04):** RunPod budget is $23.38. Nicky delegated the remaining choices: primary = best P5 at freeze, backup = E013, both GPU members (E020, E021) are approved within this budget, and no new pod is created without Nicky's go-ahead.
- **Still open:** Kaggle's weekly GPU quota, so check it before E021.

## 8. Acceptance criteria
- `pytest` passes, including the new tests in §6. Existing defaults reproduce their old outputs exactly.
- Every comparison is P5-paired and deterministic. Nothing is fit on images it's scored on. Test images are used only for inference.
- Every run has an E### id, seed, config and `experiments.csv` row. The Mac runs one heavy job at a time and never stores checkpoints.
- No data, probability maps, checkpoints or credentials are committed. Every CSV passes `validate_submission.py`. Nothing is submitted without Nicky.
