# Solar_K — Notes

Running log of facts and decisions for our entry in the **Kaggle Solar Filament Segmentation Challenge 2026**.
Newest entries go at the top of each section. Dates are absolute (YYYY-MM-DD).

Status legend: ✅ verified (local files or official text) · ⚠️ second-hand, verify · ❓ open

---

## 1. Competition facts

| Topic | Fact | Status |
|---|---|---|
| URL | https://www.kaggle.com/competitions/filament-segmentation-2026 | ✅ |
| Hosts | ESAIR Lab / IEEE BigData Cup 2026, sponsored by NSF National Solar Observatory. Judges from NSO, GSU, NJIT, UMSL | ✅ |
| Prizes | Up to $3,000 across 3 winning teams. Presentation at IEEE BigData (Dec 14–17, Phoenix). No Kaggle medals or points | ✅ |
| Timeline | Launched 2026-07-10. **Final deadline 2026-11-15, 11:59 pm US Central.** Winners announced 2026-11-30 | ✅ (Nicky's briefing) |
| Task | Instance segmentation of filaments in 2048×2048 GONG H-alpha images. Barbs count as part of the filament | ✅ |
| Metric | **Panoptic Quality**: PQ = ΣIoU(TP) / (TP + ½FP + ½FN). Switched from Dice on 2026-08-07. Reference: host's self-evaluation notebook (`azimahmadzadeh/self-evaluation-notebook`). Match at IoU > 0.5 and per-annotator scoring are ⚠️ until we read the notebook | ✅ / ⚠️ |
| Judging | 70% quantitative (PQ plus Dice/IoU distributions, one-to-many and many-to-one errors, morphology, efficiency), 30% qualitative (method, preprocessing, morphology, code, reproducibility) | ✅ |
| Score context | Organisers: PQ > 0.35 is "scientifically valuable" (not a prize threshold). Public participant scores seen: 0.12–0.35 | ✅ / ⚠️ |
| Submission | CSV `filament_id,segmentation_rle`, one row per predicted filament. `filament_id = <image_id>_<k>`. RLE = pycocotools compressed `counts` string only (no size, no JSON). Images with no predictions get no rows | ✅ |
| Leaderboard | Public LB ≈ 50% of test, private on the rest | ✅ |
| Limits | 5 submissions/day, 2 final selections | ✅ |
| Data rules | External public data and models allowed. Inference uses test H-alpha images only. No external GT. No hand-labelling of test | ✅ |
| Deliverables | Kaggle CSV · 4-page PDF report (official Overleaf template) · public Git repo (public at close until 2026-11-30) with pinned `requirements.txt` and an end-to-end notebook · Google Form | ✅ |
| Data licence | CC BY-NC 4.0. Never pushed to GitHub | ✅ |

**Time check:** as of 2026-10-29 we have about **17 days** to the deadline.

## 2. Dataset facts (local download)

| Fact | Status |
|---|---|
| `data/raw/train/train_images/` has 707 JPEGs (~850–875 KB each) | ✅ |
| `data/raw/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` is 47 MB, COCO-style: `info`, `licenses` (3), `categories` (4), `images` (1154), `annotations` (8199) | ✅ |
| `data/raw/test/test_images/` has 180 JPEGs with no labels. Total `data/raw` is 745 MB | ✅ |
| **1154 vs 707:** most likely one `images` entry per (physical image × annotator). 1154 − 707 = 447, which exactly matches the "447 repeated frames" another participant reported. So the same JPEG is listed 2–3 times, each with a different annotator's polygons. Consequence: split and dedupe by **file**, never by COCO `image_id` | ⚠️ strong hypothesis, Spec 01 verifies |
| What the 4 categories are (chirality Left/Right/…?) | ❓ Spec 01 |

### Useful references
- MAGFiLO paper: https://doi.org/10.1038/s41597-024-03876-y
- Host self-evaluation notebook: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
- IEEE BigData Cup: https://bigdataieee.org/BigData2026/cup/
- Participant repos with useful notes: [Agnuxo1](https://github.com/Agnuxo1/solar-filament-segmentation-2026), [pragyaangaur](https://github.com/pragyaangaur/NSO-Solar-Filament-Challenge), [MNMANISHNEGI](https://github.com/MNMANISHNEGI/kaggle-solar-filament-segmentation-2026), [mtalafha90](https://github.com/mtalafha90/Solar_Filament_Segmentation_Challenge_2026)

---

## 3. Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-01 | Compute: MacBook Air M2 (8 GB) for EDA, dev, and light experiments. Kaggle GPUs for heavy training, with no fixed quota assumed. Training must be portable, checkpointed, and resumable. Stream images on the Mac | Nicky |
| 2026-11-01 | Repo: GitHub `solar`, private now, made public at the required time | Nicky |
| 2026-11-01 | The official self-evaluation notebook is in `data/external/` and is the reference for `metric.py` | Nicky |
| 2026-11-01 | Multi-annotator label strategy is deferred. The EDA must quantify boundary vs instance-identity disagreement and recommend one | Nicky |
| 2026-11-01 | First NN baseline = semantic U-Net + instance extraction by post-processing (if the EDA supports it). Report pixel metrics and PQ separately. Move to instance or boundary-aware models only if the baseline shows merging or fragmentation dominates | Nicky |
| 2026-11-01 | Still Phase 0: no network code or training until the audit and metric are validated and the baseline spec is agreed | Nicky |
| 2026-11-01 | Planning docs stay local and are git-ignored: every `*.md` except `README.md` files, and the whole `docs/` folder. Claude Code still reads them from disk | Nicky's preference; repo shows only README |
| 2026-11-01 | Overleaf report template lives in `reports/paper/` and is **tracked** (LaTeX build files ignored) | The report is a deliverable, and the public repo backs it |
| 2026-11-01 | Nicky approved the plan, time boxes, and Spec 01 | Chat |
| 2026-10-29 | Team = Nicky, solo | Briefing |
| 2026-10-29 | Docs in repo: `CLAUDE.md` (root, standing rules), `docs/RESEARCH_PLAN.md`, `docs/Notes.md`, numbered specs in `docs/specs/CODING_SPEC_XX_*.md` | Briefing workflow |
| 2026-10-29 | Evidence-driven phases, but **time-boxed**: EDA Oct 29–30 · baseline + first Kaggle submission by Nov 2 · experiment cycles Nov 3–10 · freeze Nov 11 · deliverables Nov 12–14 · buffer Nov 15 | 17 days left; the report is 30% of judging |
| 2026-10-29 | Build the scorer, submission validator, and CV split **before any model**. The scorer must match the official notebook | They're the foundation every decision rests on |
| 2026-10-29 | CV: 5-fold grouped by physical image **and observation day** | Repeated frames and near-duplicate same-day frames would inflate CV |
| 2026-10-29 | Measure inter-annotator PQ as the human ceiling | Calibrates expectations; strong report material |
| 2026-10-29 | Classical baseline capped at half a day, used as a report reference | Public reports put it at PQ ~0.08–0.13 |
| 2026-10-29 | Experiment log = `experiments.csv`, one row per run ID | Reproducibility, report traceability |
| 2026-10-29 | Dataset never goes to GitHub: `data/` contents are git-ignored | User request + CC BY-NC licence |
| 2026-10-29 | Workflow: decisions are made in chat and logged here. Claude Code builds from specs | User's plan |

---

## 4. Open questions

- ❓ Multi-annotator training label strategy: **deferred to the EDA evidence** (boundary vs identity disagreement, Spec 01 §7)
- ❓ Final architecture: decide after baseline results (H1/H6)

---

## 5. Log

- **2026-11-01** — Open questions answered (compute, repo, notebook, labels, architecture). Added Mac memory and portability rules to CLAUDE.md and the annotator-disagreement analysis to Spec 01.
- **2026-11-01** — Plan approved. Decided where the docs and the Overleaf template go, and what gets ignored.
- **2026-10-29** — Scaffold committed and dataset in `data/raw/`. Reviewed Nicky's briefing: agreed, with time boxes added. Wrote `CLAUDE.md`, `RESEARCH_PLAN.md`, `CODING_SPEC_01_EDA.md`.
- **2026-10-29** — Researched the competition, created this file and `scaffold.md`.
