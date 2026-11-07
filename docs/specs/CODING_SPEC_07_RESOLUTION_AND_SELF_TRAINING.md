# CODING_SPEC_07_RESOLUTION_AND_SELF_TRAINING — follow the one lead that worked (resolution), then try a bigger, legitimate data lever

**Phase 3, cycle 5. Time box: Nov 6–11, freeze Nov 12** (Nov 13–14 deliverables, Nov 15 buffer). Read `CLAUDE.md`, `reports/06_final_report.md`, and `docs/specs/CODING_SPEC_05_ENSEMBLE.md` §2 (P5) and §8 (rubric).
Training is allowed in this spec, but only as described here. **REF = E024** (E007tta4s + E015tta4s, P5 local 0.4302 ± 0.0052, public 0.37).

## 0. First, finish and correct report 06 (≤ 2 h, on the pod's CPU, in parallel with the first GPU jobs)
`reports/06_final_report.md` isn't finished, and parts of it are wrong. These are graded rubric items, so fix them first:
1. §4.1 is all "???". Fill PQ/SQ/RQ, per-image Dice, per-TP IoU, split/merge counts, the class-a share of FN, by-site and by-area-quartile PQ from `data/interim/E024_quant_p5_*`. If those files don't exist, rerun `scripts/run_p5.py --members E007tta4s E015tta4s --experiment E024_quant --save-details` on the pod.
2. §4 has the encoders wrong. **E007 = efficientnet-b4, E015 = tu-regnety_040** (`configs/E007-f0.yaml`, `configs/E015-f0.yaml`).
3. §3 says "0.4137 → 0.4302", but the table and `experiments.csv` say the previous REF was **0.4217**. Fix it, and the matching note in the E024 row of `experiments.csv`.
4. §7 says the notebook runs on "checked-in OOF probability maps". Run `git ls-files | grep -Ei '\.(png|npy|npz|pt|jpe?g)$|outputs/|data/'`. If any probability maps or data are tracked, `git rm --cached` them and make the notebook load from `outputs/`, per `CLAUDE.md`.
5. Add E024's public score (0.37) to `experiments.csv`.

## 1. Goal and reality check
- Public 0.37 for both E013 and E024 is **expected**, not a failure. Their +0.022 local gap is inside the noise of a roughly 90-image public split (±0.03). Local P5 stays the only decision metric (Spec 05 §1.1).
- Realistic upside: **+0.01–0.03 local** across everything here. That would be local ~0.44–0.46 and public ~0.39–0.41 at the observed −0.04 gap. Public 0.50 is still not realistic (two experts agree at 0.34; the best transparent outside result is 0.45 OOF / 0.39 public).
- The decision rule is unchanged from Spec 06 §3 (pipeline bar +0.005, member bar +0.010, early-gate bar +0.008, stacking rule, bootstrap CI, split/merge guardrail).

## 2. What the evidence says (why these experiments)
1. **Resolution is the only lever that moved in Spec 06.** The 1.25× upscale in test-time augmentation gave +0.0082 (5/5 folds, CI [0.0042, 0.0127]) and improved SQ. The extra flip view in the same controlled pair gave −0.0009. The model sees thin filaments better when they're bigger in pixels.
2. **We miss the small and faint filaments.** 37% of FN are in the smallest GT-area quartile (vs about 20% expected). Upscaling helps exactly these.
3. **Same-recipe diversity is exhausted.** A second seed (E021: +0.0023), a third encoder (E015 first pass: +0.0074), a connectivity loss (E020: +0.0053) and scorer v2 (E019: −0.0019) all fell short. Another architecture at the same resolution and on the same data is not worth GPU money.
4. So the two remaining levers are **(a) resolution at training time**, not just at test time, and **(b) more images**. Only 707 labelled images exist, so the legitimate source of more images is unlabelled external GONG H-alpha with pseudo-labels (§E028). The rules allow external data as long as no ground-truth metadata is used; Nicky confirmed this against the Rules tab on Nov 6.

## 3. Experiments (ranked by expected value per hour)

### E025 — larger TTA scales (pod GPU inference + pod CPU P5, ~2 h, ~$2)
- Regenerate OOF and test maps for E007 and E015 with `tta_scales` = `[1.0, 1.5]`, `[1.0, 1.25, 1.5]` and `[1.0, 1.25, 1.5, 1.75]` (4 flips each). Downscaling stays excluded. Infer with tiles of 1024 px in the upscaled image, `overlap 128`. At 1.75× (3584 px) that's 16 tiles per view, so run it on the pod only.
- P5 under the pipeline bar vs REF. If scales keep paying, that's direct evidence for E026.

### E026 — min-area and small-object grid (pod CPU, ~1 h)
- `min_area` has been fixed at 418 px (the GT 1st percentile) since Spec 03. Add `min_area ∈ {150, 250, 418}` and allow the shape gate's length minimum to go to 0 at small areas. Run on the best E025 maps. Pipeline bar. It's a cheap check that the small-filament FN aren't simply being filtered out after detection.

### E027 — high-resolution training of E007 (pod GPU; gate on folds 0–1, then all 5)
**Hypothesis:** the gain from 1.25× test-time upscaling means the model resolves 2–4 px filaments better when they're bigger. Training at 1.5× scale should give that gain at its source, and give a model that sees scale-consistent filaments at inference.
- Data: upsample each preprocessed image and its soft-consensus target by **1.5×** (bilinear for the image, bilinear then the same soft target, so 2048 → 3072). **Crop 768** at 1.5× covers the same field of view as a 512 crop at 1×. efficientnet-b4, batch 6 (8 if VRAM allows; check in the first 50 iterations), AMP, EMA 0.999, **6000 iterations**, seed 42, the same sampler, augmentation and loss as E007. Build the upsampled cache on the machine doing the training: the pod volume, or `/kaggle/temp` for Kaggle folds. Never build it on the Mac. At float16 it's about 18 MB per image, about 13 GB for 707 images, so put it on the pod volume.
- Inference: run at 1.5× (3072 px; tiles of 1536, overlap 192), downsample probabilities back to 2048 with area interpolation, and use 4 flips with scales {1.0, 1.25}, relative to the 1.5× input. Save maps at 2048 as usual so P5 and the ensemble code don't change.
- `configs/E027-f{0..4}.yaml` with new keys `train_scale: 1.5` and `infer_scale: 1.5`. Old configs default to 1.0, and a test checks the defaults reproduce existing behaviour exactly.
- **Gate on model folds 0–1** (P5 on 284 images): continue to folds 2–4 if E027 **replacing E007** gives ΔPQ ≥ +0.008, or **adding** it gives ΔPQ ≥ +0.008.
- Cost estimate (4090): about 2.25× E007's compute per iteration plus larger inference, so about **1.75–2 h per fold on a 4090** (about $1.6–1.8 per fold, about $9 for 5 folds).
- **E027b (only if E027 passes the full bar):** the same treatment for E015 (`tu-regnety_040`), another ~$9.

### E028 — self-training on external, unlabelled GONG H-alpha (stretch; pod; gate on folds 0–1)
**Hypothesis:** 707 labelled images limit recall of rare, faint filaments. Pseudo-labelling a few thousand extra full-disk GONG H-alpha images, with confidence masking, adds examples of those filaments without any extra human labels.
- **Rules (settled, §7):** Nicky confirmed on Nov 6 that the Rules tab allows external unlabelled images (the Evaluation page says "participants may use other external data sources" and forbids only other ground-truth metadata). **No MAGFiLO labels, no other annotation sets, no catalogue metadata.** Images only.
- **Data:** download about 2000 GONG H-alpha full-disk JPEGs from the public NSO GONG archive (`scripts/fetch_external_gong.py`, on the pod), spread over 2011–2022 and all six sites. **Exclude every observation within ±3 days of any train *or test* image's date**, so no external frame is a near-duplicate of a test frame. Record the exact URL list in `data/external/gong_unlabelled_manifest.csv` (the manifest is committed, the images are not). Preprocess with the same pipeline.
- **Pseudo-labels without leakage:** for model fold *k*, generate pseudo-labels **only with that fold's own members** (E007-f*k* + E015-f*k* with E024's TTA). Those models never saw fold *k*'s validation images, so fold *k*'s OOF stays clean. Never use the all-fold final ensemble to pseudo-label for CV. Targets are the soft probabilities. Pixels with p ∈ [0.3, 0.7] are ignored in the loss, and an image is dropped if its mean confidence is too low (the bottom 20%).
- **Training:** fine-tune from E007-f*k* `best.pt` for 3000 iterations. It deliberately does *not* wait for E027, so the two branches run in parallel. If both are promoted, they're combined as members in E029 on a 50/50 mix of labelled crops and pseudo-labelled crops, with lr 1e-4, EMA, PQ checkpoint selection on the fold's labelled val set as usual, seed 42. `configs/E028-f{0..4}.yaml`.
- **Gate on model folds 0–1** (replacement or addition, +0.008). Full 5-fold only if the gate passes.
- Cost: download and preprocessing about 1–2 h of pod CPU, pseudo-labelling about 1 h of GPU per fold pair, fine-tuning about 1 h per fold on the 4090 (about 2.5 h on a T4). About $6–8 in total for 5 folds.

### E029 — final build (pod CPU, ~1 h, by Nov 11)
- `scripts/build_final.py` with the promoted members, scales and grid. If E027 and E028 are both promoted, run one combined P5 first ({E027 or E007, E015, E028}) and apply the stacking rule. Validate, **don't submit**. **Primary** = best P5 at freeze. **Backup** = E024 (or E013 if the primary *is* E024). Record both in `experiments.csv`, update `reports/07_*.md` and the paper.

### Not doing
Pseudo-labelling or training on test images (`CLAUDE.md`), external labels, the evaluator defect, Mask R-CNN-style frameworks (merging is a minor error class and setup risk is high), more same-resolution architectures or seeds (§2.3).

## 4. Who runs what: no work on the Mac, and everything independent runs in parallel

**Nicky's rule (2026-11-06): the Mac does no work in this spec.** It runs no training, no inference, no P5, no cache builds, no `pytest`, no report generation and no syncing of results. The only thing running on the Mac is the Claude Code session itself, sending commands over SSH and the `kaggle` CLI.
- **The pod is the hub.** Clone the repo onto the pod volume (`/workspace/Solar`) and do all coding, testing (`pytest`), P5 evaluation, report and paper edits there. Commit and push from the pod to GitHub, never committing data, maps, checkpoints or credentials. Kaggle outputs are downloaded onto the pod with `kaggle kernels output`, not to the Mac. The Mac does a single text-only `git pull` at freeze (Nov 12), and only after checking `df -h ~` (stop if under 5 GB free).
- **Four lanes run at once:** pod GPU, pod CPU (P5 and caches don't need the GPU, so they run alongside it), Kaggle session A and Kaggle session B. Anything without a dependency starts as soon as its lane is free. Dependent steps wait only for what they actually need.
- **Kaggle** needs the inputs packaged as private datasets pushed from the pod: the code package, E028 pseudo-labels per fold, and for E028 fine-tuning the E007-f*k* `best.pt`. Kaggle notebooks build their own upsampled cache in `/kaggle/temp`. T4 estimates: E027 about 5 h per fold (batch 4–6 at crop 768; `--max_hours 8.5` with resume), E028 about 2.5 h per fold.

**Dependencies (the only things that must wait):** the E027 cache before E027 training on the same machine; the E028 download before E028 pseudo-labels, and pseudo-labels before E028 fine-tuning; a gate's P5 needs both gate folds' maps; folds 2–4 of E027/E028 need their gate passed; the combined P5 needs all promoted members; E029 needs everything. E025, E026, the §0 fixes, E027-f0/f1 and the E028 download share no dependencies, so they all start on day 1.

**Kaggle quota (Nicky, Nov 6): only 5 h 12 m left this week.** That's too little for a 5-hour E027 fold, so the E027 gate folds go on the pod GPU. Kaggle's remaining hours are spent on the two shorter E028 gate fine-tunes (about 2.5 h each, run in parallel). The full 30 h comes back at the weekly reset. Check the reset time on Kaggle's quota page; it's expected within about a day. After that, Kaggle takes folds 3–4 of each branch. If the quota runs out mid-run, `--max_hours` checkpointing lets the fold resume after the reset.

| When | Pod GPU | Pod CPU (parallel with GPU) | Kaggle A | Kaggle B |
|---|---|---|---|---|
| Nov 6 | **E025** TTA inference (~1.5 h) → **E028** pseudo-labels f0, f1 (~1 h, once enough images have downloaded) → **E027-f0** → **E027-f1** (~2 h each) | §0 report fixes (E024_quant P5) → **E026** P5 → **E025** P5; **E028** download + preprocess, running from the start; E027 upsampled cache, built before E027-f0 starts | **E028-f0** fine-tune (~2.5 h, after its pseudo-labels are pushed as a Kaggle dataset) | **E028-f1** fine-tune (~2.5 h) |
| Nov 7 (Kaggle quota resets) | E028 pseudo-labels f2–f4 → **E027-f2** *(if E027 gate passes)* → **E028-f2** *(if E028 gate passes)* | **E027 gate** P5 → **E028 gate** P5 | E027-f3 *(if gate)* | E027-f4 *(if gate)* |
| Nov 8 | **E027b** f0 → f1 → f2 *(if E027 promoted)*, else idle → stop the pod | report 07 sections as results land | E028-f3 *(if gate)* | E028-f4 *(if gate)* |
| Nov 9 | E027b remaining folds / idle | **E027 full P5** → **E028 full P5** | E027b-f3 *(if run)* | E027b-f4 *(if run)* |
| Nov 10 | — | combined P5 (all promoted members) → E027b P5 | — | — |
| Nov 11 | — | **E029** final build, validate CSVs, update report 07 and the paper, push | — | — |
| Nov 12 | **stop the pod** | — | — | — |

If a gate fails, drop that branch at once and give its lane to the next queued item. If the Kaggle reset comes later than Nov 7, run E027-f3 on the pod GPU after f2, and keep f4 for Kaggle.

**RunPod budget: $52.24 (Nicky, Nov 6).** Planned pod spend is about **$2 (E025) + $1 (pseudo-labels) + $5.5 (E027-f0–f2) + $1 (E028-f2) + $5.5 (E027b f0–f2) ≈ $15**, plus pod CPU time for the P5 lane. That leaves plenty of headroom, so **E027b is approved if E027 is promoted.** If the Kaggle reset is late, run the Kaggle-assigned folds on the pod GPU instead of waiting. Reuse Nicky's existing pod, and stop it whenever both of its lanes are idle.

## 5. Deliverables
- All code is written and tested on the pod (§4). Code: `tta_scales` extended; `train_scale`/`infer_scale` in the dataset and `unet.py`; the upsampled cache builder; `scripts/fetch_external_gong.py` plus the manifest; the pseudo-label generator with confidence masking; the mixed sampler. Tests for each, plus a defaults-unchanged test.
- `experiments.csv` rows E025–E029 (and E024's public 0.37).
- `reports/07_resolution_and_self_training.md`: opens with the leaderboard caveat; per-experiment gate and full results; an E028 rules note and the leakage argument; the final pipeline's rubric breakdown (Dice/IoU/split/merge, by site and area quartile) and an updated overlay gallery.
- The paper draft updated with the final pipeline.

## 6. Acceptance criteria
- `pytest` green. Existing defaults reproduce their old outputs exactly.
- Every comparison is P5-paired and deterministic. Pseudo-labels for fold *k* come only from fold-*k* models. External frames within ±3 days of any train/test date are excluded, and the manifest proves it.
- No data, external images, probability maps, checkpoints or credentials are committed. Every CSV is validated. Nothing is submitted without Nicky.

## 7. Decisions for Nicky

1. **External unlabelled GONG images (E028): settled.** Nicky confirmed on Nov 6 that the Rules tab allows unlabelled external images. E028 goes ahead with images only: no labels and no catalogue metadata.
2. **Settled:** RunPod $52.24, Kaggle 5 h 12 m left until the weekly reset, and the Mac does no work (all Nicky, Nov 6).
