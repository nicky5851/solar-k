# CODING_SPEC_08_FINAL_PACKAGE: make E024 reproducible from shipped weights, write it up, tidy the repo, hand over for submission

**Phase 4, wrap-up. Time box: Nov 7–11, freeze Nov 12** (Nov 13–14 Nicky submits, Nov 15 deadline/buffer). No new experiments: nothing in this spec is chosen by score.
Read `CLAUDE (3).md`, `reports/06_final_report.md`, `reports/07_resolution_and_self_training.md`, `docs/HANDOFF.md` (Spec 07 closing summary), and `docs/specs/CODING_SPEC_05_ENSEMBLE.md` §1 (leaderboard trust) and §8 (rubric).

**Final pick:** **E024** is primary (E007tta4s + E015tta4s, P5 local 0.4302 ± 0.0052, public 0.37). **E013** is backup (E004 + E007, local 0.4086, public 0.37). Both scored 0.37 public, so the trusted local CV decides.

**Why this spec exists:**
1. **Weights rule.** Award entries must load weights stored in the repository, and the organisers judge segmentation quality, not PQ alone (Kaggle discussion, audited Nov 7).
2. **Weights aren't in git.** All checkpoints are gitignored and live only on the pod volume (`ft28lnkjdp`).
3. **Two E015 checkpoints are gone.** **E015-f3 and E015-f4 `best.pt` were lost on Nov 6** (HANDOFF 18:25), so E024 can't be regenerated from weights today.

**Leaderboard context (Nov 7 audit), for the write-up:**
- No genuine public notebook scores above 0.40. The best are tushhaaarrrr (OOF 0.4597 → public 0.40) and dariushafshar's ten U-Nets (OOF 0.4502 → public 0.39). Both lose the same ~0.05 from OOF to public that we do.
- The 0.55 plateau is a hard-coded, copied blob of masks.
- Nobody between 0.56 and 0.70 has public code. The organisers will clean the board after the deadline.

## 0. Rules carried over
- **The Mac does no work.** All coding, training, inference, P5, `pytest`, report and paper edits, and git pushes happen on the pods (repo at `/workspace/Solar` on the shared volume).
- **Use only the existing pods, never create new ones:**
  - `85ki7s4ssa3lqv` solar-k-training (4090)
  - `e02xjn6s8k7osf` solar-k-training-4090b (4090)
  - `0i75irld5c215s` solar-k-training-l4 (L4)
  Run independent jobs in parallel. Stop every pod as soon as its lane is idle.
- **Commit rules:** never commit data, probability maps, checkpoints or credentials to git history. Weights ship as Release assets (§2), not commits.
- **Never submit to Kaggle.** Nicky makes the final selection and fills in the Google Form.
- `docs/HANDOFF.md` stays out of git (add it to `.gitignore`). Keep appending status there.

## 1. E030: restore E015-f3/f4 and rebuild E024 as a weights-reproducible pipeline

### 1.1 Audit the checkpoints first (L4, CPU, ~10 min)
Before retraining anything, list every `outputs/<model>-f<k>/checkpoints/best.pt` for E004, E007 and E015, folds 0–4. Record size, mtime and sha256 in `weights/CHECKPOINT_AUDIT.md`.

Anything missing besides E015-f3/f4 gets added to §1.2. Missing E004 folds count too, because they block the E013 backup. If any are missing, tell Nicky in HANDOFF before training.

### 1.2 Retrain the missing folds (two 4090s in parallel, ~1.5–2 h each)
- **E015-f3** on `85ki7s4ssa3lqv` and **E015-f4** on `e02xjn6s8k7osf`, using the unchanged `configs/E015-f3.yaml` / `E015-f4.yaml` (tu-regnety_040, 6000 iters, seed 42).
- If a 4090 has no capacity after ~30 min of retries, run that fold on the L4, or on Kaggle (the quota has reset).
- Same seed doesn't mean identical weights (cuDNN nondeterminism). That's expected and is why §1.3 re-measures.

### 1.3 Regenerate maps and re-measure (L4 or whichever pod is free)
1. Run the E015 inference with the frozen E024 TTA (4-flip + 1.25× upscale, EMA weights) for **f3 and f4 only**. This gives new `E015tta4s-f3/f4` OOF and test maps; write them under a new tag `E015tta4sR` so the old maps stay untouched. Folds 0–2 reuse the existing maps.
2. **Determinism check:** regenerate the E015-f0 maps from its existing checkpoint and compare them with the stored `E015tta4s-f0`. The max abs prob diff should be ≤ 1e-3. If it isn't, inference isn't reproducible from weights. Stop and report.
3. Run P5 for **E030** (members E007tta4s + E015tta4sR) the usual way: paired against E024, bootstrap CI, split/merge guardrail.
4. **Acceptance:** |ΔPQ| ≤ 0.005 vs E024 and the split/merge guardrail holds. This is a reproducibility check, not a selection, so a higher score is not a reason to prefer E030.
   - **Pass:** E030 is the final primary. Build `submissions/E030_primary.csv` with `scripts/build_final.py` and validate it with `validate_submission.py`.
   - **Fail:** compare the new fold maps with the old ones (per-fold OOF PQ for f3/f4) and report to Nicky before doing anything else. Don't retrain repeatedly to chase the old number.
5. Log E030 in `experiments.csv`, with its pod ids and cost.

### 1.4 Backup (E013) stays reproducible too
E013 needs E004-f0..4 and E007-f0..4. If all ten checkpoints exist, rebuild `E013_backup.csv` from them and check it's byte-identical to the scored file, or within the §1.3 tolerance if maps were regenerated.

## 2. Ship the weights "in the repository"

### 2.1 Export
- **Contents:** EMA `state_dict` only, no optimizer state, for the 15 checkpoints: E007-f0..4, E015-f0..2 + retrained f3/f4, and E004-f0..4.
- **Name and format:** `weights/<model>-f<k>.pt`, saved in **fp16**.
- **fp16 check:** rerun the final build from fp16-loaded weights, with inference still in fp32. The CSV must match the fp32 build exactly, or the mean per-instance IoU between the two must be ≥ 0.99 with identical instance counts per image. If not, ship fp32.
- **Manifest:** `weights/MANIFEST.json` lists file, model, fold, encoder, sha256, bytes and the training config path. Commit the manifest; don't commit the `.pt` files.

### 2.2 Host
- **Default:** a GitHub Release `v1.0-final` on `nicky5851/solar-k`, with the `.pt` files as assets (≤ 2 GB each). Upload with `gh release create` from the pod.
- **Fallback:** if Nicky decides the organisers need the files literally in git (§7), use Git LFS instead. Check the total size against the free 1 GB LFS storage/bandwidth first.

### 2.3 Load
- **`scripts/download_weights.py`:** fetches the release assets into `weights/` and verifies every sha256 against `MANIFEST.json`.
- **`scripts/predict.py --images <dir> --out <csv> [--pipeline primary|backup]`:** the single entry point. It loads `weights/`, runs the frozen TTA, averages members, and applies the frozen post-processing (cfg1, shape gate, scorer, cutoff, regrow) from a committed `configs/final_postproc_{primary,backup}.json` that `build_final.py` writes. It writes a validated CSV.
- **Tests (pytest, CPU-only, tiny):**
  - the manifest-checksum logic;
  - `predict.py` on 2 test images with a random-init model, checking the CSV schema;
  - the post-processing JSON round-trip.
- **End-to-end (on a pod):** clone the repo fresh into `/tmp/fresh`, run `download_weights.py` then `predict.py` on the full test set. The output must equal `submissions/E030_primary.csv`. This proves the weights in the repository reproduce the submission.

## 3. Write-up (graded at 30%, plus the quantitative distributions)

### 3.1 `reports/08_final_writeup.md`
This is the single document a judge reads. Structure it on the rubric:
1. **Task, data and metric.** Include the inter-annotator ceiling of PQ 0.34 (E008) and the Dice→PQ metric change of Aug 7.
2. **Pipeline end to end:** preprocessing, crops, architectures (U-Net with efficientnet-b4 and tu-regnety_040, smp 0.6.0), loss, training schedule, EMA, TTA, probability averaging, post-processing (cfg1, shape gate, scorer, cutoff, watershed regrow). Include one diagram.
3. **Validation protocol P5:** day-grouped 5-fold, nested refit, paired bootstrap, pre-registered bars. Explain why the public LB isn't used to select.
4. **Results:**
   - final PQ/SQ/RQ;
   - **Dice distribution, IoU distribution, and one-to-many / many-to-one counts** (the rubric's quantitative items, from the `E024_quant` details or rerun for E030);
   - by site and by area quartile;
   - local vs public scores for E010, E013 and E024/E030.
5. **Morphology:** the overlay gallery on H-alpha (`06_overlay_gallery.png`). Add 3–4 annotated failure cases (class a/b/c/d).
6. **What didn't work:** a table of every rejected experiment from Specs 05–07 with its ΔPQ and the bar it missed. A short paragraph on why: we're near the annotator-agreement ceiling, and resolution and data levers are saturated.
7. **Leaderboard integrity note:** a factual summary of the Nov 7 audit. Say plainly that we didn't exploit the evaluator or use leaked labels. Name no users.
8. **Reproduce:** the exact commands from §2.3, and the compute used (pods, Kaggle hours, total $).

### 3.2 Paper
Un-pause `reports/paper/main.tex` (4 pages) and fill it from the write-up. Build the PDF on the pod (`latexmk`) and commit only the `.tex`, `.bib` and figures, not build artefacts.

## 4. Repo tidy (L4 or any pod, CPU)
- **Rename duplicate-suffixed files** with `git mv`, after confirming no duplicate target exists:
  - `CLAUDE (3).md` → `CLAUDE.md`
  - `docs/Notes (1).md` → `docs/Notes.md` (Nicky's decision log; content unchanged)
  - `docs/specs/CODING_SPEC_01_EDA (1).md` → `CODING_SPEC_01_EDA.md`
  Fix any links that pointed at the old names.
- **`README.md`:** a one-paragraph summary, the results table (local and public), and quick start (install, download weights, predict). Then sections on reproducing CV, the repo layout, the data licence (MAGFiLO, CC BY-NC 4.0, not redistributed, see `data/README.md`), the code licence (MIT) and a citation.
- **`requirements.txt` / `pyproject.toml`:** pin exact versions, including smp 0.6.0 and timm. Add a `make`-style list of commands, or document them in the README.
- **Code quality** (also graded):
  - module docstrings on every `src/solar_k/*.py` and `scripts/*.py`;
  - delete commented-out code;
  - `ruff check` clean, with formatting only and no behaviour changes.
  - **Don't delete scripts.** List unused ones in the write-up's appendix, and leave deleting them to Nicky.
- **Checks:**
  - `pytest` is green.
  - `git ls-files` shows no data, maps, checkpoints or credentials.
  - `.gitignore` covers `weights/*.pt`, `outputs/`, `data/` and `docs/HANDOFF.md`.
- **Finish:** tag `v1.0-final` on the commit whose weights and CSV match. The repo stays **private** until Nicky decides (§7).

## 5. Lanes (parallel where independent)

| When | 4090 `85ki…` | 4090b `e02x…` | L4 `0i75…` (CPU-heavy work) |
|---|---|---|---|
| Nov 7 | E015-f3 retrain | E015-f4 retrain | §1.1 audit → §1.3 step 2 determinism check → §4 renames and README skeleton |
| Nov 8 | E015tta4sR-f3 inference → stop | E015tta4sR-f4 inference → stop | E030 P5 → E030 + E013 builds → §2.1 export and fp16 check |
| Nov 9 | — | — | §2.2 release, §2.3 scripts and tests, fresh-clone end-to-end |
| Nov 10 | — | — | §3.1 write-up (rerun quant details for E030 if it passed) |
| Nov 11 | — | — | §3.2 paper, §4 code-quality pass, final `pytest`, tag → **stop** |
| Nov 12 | freeze | | |

**Cost:** two retrains (~2 h × ~$0.69) plus inference, about $4. The L4 runs for about 4 days of mostly CPU work at its hourly rate, which is less than a 4090's. Stop it overnight when nothing is running. Expect about $10–15 in total. Check the RunPod balance first and note it in HANDOFF.

## 6. Acceptance criteria
- [ ] Checkpoint audit committed. E015-f3/f4 retrained, and any other gaps closed or reported.
- [ ] Determinism check passed (≤ 1e-3).
- [ ] E030: |ΔPQ| ≤ 0.005 vs E024 and the guardrail holds. `E030_primary.csv` and `E013_backup.csv` validated. Rows in `experiments.csv`.
- [ ] Release `v1.0-final` has the weights and their checksums. A fresh clone running `download_weights.py` + `predict.py` reproduces `E030_primary.csv` exactly.
- [ ] `reports/08_final_writeup.md` covers all 8 sections, including the Dice/IoU distributions and split/merge counts. The paper PDF builds.
- [ ] README, renames, pins and docstrings done. `pytest` and `ruff` green. Nothing forbidden is tracked.
- [ ] All pods stopped. Nothing submitted to Kaggle.

## 7. Decisions for Nicky
1. **Weights hosting:** a GitHub Release (default) or Git LFS. Only choose LFS if you read "stored in the repository" as literally inside git.
2. **Repo visibility:** the judges need access to the repository. Make it public at submission time (Nov 13–14), or add the organisers as collaborators if they give usernames. Keep it private until then.
3. **Kaggle final selection (you do this):** primary **E030** (or **E024** if §1.3 fails and you decide to keep the scored file), backup **E013**. Then fill in the Google Form.
