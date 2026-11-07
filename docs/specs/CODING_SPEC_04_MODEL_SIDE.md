# CODING_SPEC_04_MODEL_SIDE — model-side gains: hard negatives, a stronger encoder, and the mask-extent threshold

**Phase 3, cycle 2. Time box: Nov 2–6.** Read `CLAUDE.md`, `reports/02_baseline_report.md`, and `reports/03_fp_report.md` first.
Training is allowed in this spec, but only as described here.

## 1. Goal and reality check

**Stretch goal:** beat the public #1 (PQ 0.70). **Operational goal:** the largest *legitimate, cross-validated* PQ gain per GPU-hour, measured by 5-fold CV, not the public LB.

Why 0.70 is treated as a stretch, not a plan:
- PQ = SQ × RQ. Reaching 0.70 needs, for example, SQ 0.80 with RQ 0.875, or SQ 0.75 with RQ 0.93.
- Two expert annotators agree with each other at **SQ 0.634, RQ 0.536, PQ 0.340** (Spec 01). Their boundaries differ by about 2.5 px on structures that are often 2–10 px wide. Each image is scored against the annotator who drew it, and a model can't know which annotator's boundary style to copy. So an SQ near 0.80 against individual annotators would mean matching each one's private style.
- The full MAGFiLO dataset (about 1,593 observations with labels) is published. The competition uses about 887 of them. The test labels may therefore be publicly obtainable, which the rules forbid using. We **cannot verify** how any leaderboard score was produced, and we won't speculate about specific entries. We also won't chase the public LB: it covers only about 90 images, and the organisers judge reproducibility and method (30%).

**What the numbers say about where the headroom is** (E003: SQ 0.667, RQ 0.573, PQ 0.382):
- SQ is already *above* the human SQ. Its realistic headroom is about +0.03–0.06.
- RQ has more headroom: 0.57 → 0.70 would mean PQ ≈ 0.47 at the current SQ.
- **The two are coupled.** Near-miss pairs (IoU 0.3–0.5) count as one FP plus one FN. Better masks turn them into TPs, so mask quality raises **RQ** as well as SQ. That's the real case for model-side work.

Also note: E003's local 0.382 vs public 0.32 is a larger gap than before (E002: 0.269 vs 0.24). The scorer may be slightly overfitting the small tune half. **E005's 5-fold CV number is the baseline every experiment here must beat.**

## 2. Baseline and comparison protocol (applies to every experiment)

- **Reference:** E004 (ResNet34 U-Net, EMA, PQ checkpoint selection). Its val PQ per fold on the 40-image selection subset was f0 0.340, f1 0.371, f2 0.329, f3 = pending, f4 0.311. The fold-to-fold spread is about ±0.025.
- **Comparison unit:** a new model is compared to E004 **on the same folds**, using full fold-val OOF probabilities, through two post-processing paths:
  1. **raw:** E002-style threshold + connected components, with the threshold tuned on the fold's `tune` half and reported on its `report` half
  2. **full:** the E003/E005 pipeline (shape gate + learned scorer), refit for that model, with fits on `tune` and reporting on `report`
  
  Report PQ/SQ/RQ, near-miss count, FP classes, and FN classes for both paths. *raw* isolates the model's own gain; *full* tells us what reaches the submission.
- **Decision rule (pre-registered):** a change is **kept** only if its *full* PQ improves by **≥ 0.02 on each of 2 folds** (fold 0 and fold 1) and SQ doesn't fall. Otherwise it's rejected or marked inconclusive. Only kept changes get trained on all 5 folds (E009).
- Nothing is tuned on the `report` half, and no scorer is fit on the fold it's evaluated on.

## 3. Experiments (ranked by expected value per GPU-hour)

### E008 — mask-extent threshold and mask-bias diagnosis (local Mac, no training, about 2 h) — do this first
**Hypothesis:** the threshold that's best for *detection* (about 0.7+, chosen to kill FPs) isn't the threshold that gives the best *mask extent*. Masks cut at t ≈ 0.7 are systematically thinner than the annotators' masks, which caps SQ and creates near-misses.
- On E004/E005 OOF (all 5 folds), measure each matched pair's **area ratio** (pred/GT) and its signed boundary offset. Report the median and IQR per fold.
- Implement **two-stage post-processing**: detect and keep components with the current pipeline (seed threshold `t_det` + scorer), then **re-grow each kept instance's mask** to a lower threshold `t_mask` within a `d`-pixel geodesic dilation of the seed, without merging into other instances. Tune `t_mask ∈ {0.2…0.7}` and `d ∈ {2, 4, 8}` on `tune`, and report on `report`.
- **SQ ceiling estimate:** on multi-annotator images, compute SQ of annotator A against the *consensus of the others*. That's the most a consensus-trained model can expect, and it tells us how much SQ headroom really exists.
- Cost: CPU only. **Runs in parallel with everything else.**

### E006 — hard-negative mining (Kaggle, 2 concurrent sessions: fold 0 and fold 1, about 1.5 h each)
**Hypothesis:** the remaining confident FPs on bright plage and network regions exist because random and positive-centred crops rarely show the model those regions. Training on them directly cuts FPs at the source and makes the probability map cleaner, which helps the scorer and E008's re-growing.
- Same architecture and recipe as E004 (ResNet34, EMA, 4000 iterations, PQ checkpoint selection). **Only the crop sampler changes**, so the effect is isolated.
- Hard-negative pool: from the **5-fold OOF** (each image's FPs come from a model that never trained on it), take FP components classed `isolated` in the E003 taxonomy. Restrict to training images of the fold being trained. On **single-annotator images**, include only FPs whose scorer P(match) < 0.2, since an "FP" there may be a real filament the annotator skipped. Save as `data/interim/hard_negatives_v1.csv` (stem, component bbox, centroid, class, P(match)).
- Sampling mix: 45% positive-centred, 30% hard-negative-centred (random jitter of ±128 px), 25% random on-disk.
- Report: FP class counts against E004 on the same folds, PQ through both paths, and whether `isolated` FPs fell without FN rising.

### E007 — stronger encoder, keeping high-resolution skips (RunPod RTX 4090, folds 0 and 1 back to back, about 1.5–2.5 h each)
**Hypothesis:** a stronger encoder improves both recognition (fewer confident FPs, better recall of faint filaments) and mask fidelity, provided the decoder keeps a stride-2 skip connection. Filaments are often 2–4 px wide, and those details only survive in the highest-resolution features.
- Encoder: **`efficientnet-b4`** (smp, ImageNet), U-Net decoder. **Don't use ConvNeXt- or Swin-style encoders** in this spec: their first feature map is at stride 4, so the U-Net has no stride-2 skip, which is the wrong trade for 2 px structures. If b4 doesn't fit at batch 8 with 512 crops on 24 GB with AMP, use `resnext50_32x4d`. Record the choice.
- The crop sampler is the **same as E004** (not E006's), so E006 and E007 test one change each. Training: 6000 iterations (the 4090 is about 2–3× a T4), EMA 0.999, PQ checkpoint selection every 1000 iterations, same augmentations, same loss. Seed 42.
- The RunPod setup must be scripted: `scripts/setup_remote.sh` (venv, pinned `requirements.txt`, Kaggle data download via the user's token from an env var or `~/.kaggle`, never committed, and the `SOLAR_K_DATA` / `SOLAR_K_OUTPUT` / `SOLAR_K_CACHE` layout). It must be idempotent, so re-running on a new pod just works. Sync outputs back with `rsync`/`scp`; document the command.
- If fold 0 finishes and the pod is still up, start fold 1 straight away without waiting for analysis.

### E009 — combine the winners (assignment depends on what passed)
After the E006/E007 decisions are written into `experiments.csv`:
- Apply every kept change at once (e.g. b4 + hard negatives), and train **all 5 folds**: folds 0–2 on RunPod and folds 3–4 on Kaggle in 2 parallel sessions.
- Then rerun the E005 procedure: 5-fold ensemble, post-processing and scorer refit with grouped CV, and E008's mask re-growing if it was kept. Report mean ± std PQ across folds and write `outputs/submissions/E009.csv`.
- If **nothing** passed, skip the training. E009 is then E005 + E008 only.

### Not in this spec, and why
- **True instance models** (Mask R-CNN, Mask2Former): merging is 1% of FPs and fragmentation 1–10%. Instance identity isn't where PQ is lost, the setup cost is high, and thin-structure masks from box-based heads tend to be *worse*. Revisit only if E008 shows that near-misses come from instance splits.
- **Limb-correction ablation:** E002's limb drop is small (0.29 → 0.25). It's a cheap report-only ablation for later.
- **Bigger crops or a global context channel:** a Spec 05 candidate if E006 shows that plage FPs survive hard negatives (which would suggest the model needs more context than a 512 px crop).

## 4. Parallel schedule

| Slot | Kaggle session A | Kaggle session B | RunPod 4090 | Mac (local) |
|---|---|---|---|---|
| Now | E004 fold 3 finishing | E006 f0 (start once hard negatives exist) | `setup_remote.sh`, then E007 f0 | E005, then build `hard_negatives_v1.csv`, then E008 |
| +2 h | E006 f1 | — | E007 f1 | E005/E008 report; evaluate E006 f0 |
| +4 h | (free) | — | (free or stop pod) | Evaluate E006/E007 against the decision rule |
| Next | E009 f3 | E009 f4 | E009 f0, f1, f2 | E009 ensemble + submission |

Kaggle budget: E006 ≈ 3 GPU-h, E009 ≈ 3 GPU-h, so about 6 h of the ~24 h left. Stop the RunPod pod whenever it's idle.

**Order dependency:** E006 needs the 5-fold OOF (E004 complete) to build its hard-negative pool. E007 and E008 depend on nothing new and can start immediately.

## 5. Deliverables
- `src/solar_k/sampler.py` (a pluggable crop sampler: `positive`, `random`, `hard_negative`, configurable mix) with tests
- `src/solar_k/postprocess.py`: two-stage re-growing (`t_det`, `t_mask`, `d`) with tests (non-merging, exclusive masks, a toy case)
- Encoder choice wired through the config, needing no code change per encoder
- `scripts/build_hard_negatives.py`, `scripts/setup_remote.sh`, `configs/E006.yaml`, `configs/E007.yaml`, an updated Kaggle notebook/package
- `experiments.csv` rows: E005, E006-f0/f1, E007-f0/f1, E008, E009-*, each with the decision filled in according to the rule in §2
- `reports/04_model_side_report.md`:
  - E008's area-ratio and boundary-offset results, the SQ-ceiling estimate, and the `t_mask` gains
  - a comparison table of E004 vs E006 vs E007 for each fold, through both paths
  - FP/FN class shifts
  - compute used (GPU-hours per platform, and RunPod cost)
  - a keep/reject decision for each experiment, with its evidence
  - ranked hypotheses for Spec 05

## 6. Acceptance criteria
- `pytest` passes, including new tests for the sampler and for re-growing.
- Every model comparison uses the same folds and the same tune/report halves as E004. No parameter, threshold, or scorer is fit on the data it's reported on.
- The hard-negative pool for each fold uses only that fold's **training** images and OOF predictions from models that never saw them.
- `setup_remote.sh` reproduces the environment on a fresh pod with no manual steps beyond providing the Kaggle token.
- No credentials, data, checkpoints, or probability maps are committed.
- Every CSV passes `validate_submission.py`. Nothing is submitted to Kaggle without Nicky's go-ahead.
