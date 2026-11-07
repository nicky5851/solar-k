# CLAUDE.md — standing instructions for Claude Code in this repo

Project: Solar_K, an entry for the Kaggle **Solar Filament Segmentation Challenge 2026**
(instance segmentation of solar filaments in 2048×2048 GONG H-alpha images, scored by Panoptic Quality).
Deadline: **2026-11-15**. Scientific context and phases are in `docs/RESEARCH_PLAN.md`. The facts and decisions log is `docs/Notes.md`.

## How work arrives
- Work comes as numbered specs: `docs/specs/CODING_SPEC_XX_<name>.md`. Implement **only** what the current spec asks for. If a spec is ambiguous or conflicts with the data, stop and write the question into the spec's report rather than guessing.
- **Never start model training** unless a spec explicitly says so.
- Every spec ends with a report in `reports/` that separates **verified facts** (with the command or code that produced them) from **assumptions**.

## Data rules (hard)
- `data/raw/` is read-only. Never modify, move, or re-encode files there.
- Never commit data, masks, predictions, checkpoints, or credentials. `.gitignore` already covers them. Before every commit, check `git status` and confirm nothing under `data/` or `outputs/` is staged except `.gitkeep`/`README.md`.
- Test images are for inference only. Never inspect test images to tune anything by hand, never label them, and never pull MAGFiLO labels from outside the competition download.
- One physical JPEG can appear several times in the annotation JSON (one `images` entry per annotator). Always split, group, and deduplicate by the **physical file**, never by COCO `image_id`.

## Code conventions
- Python 3.11. Library code lives in `src/solar_k/`. CLI entry points go in `scripts/` and call into `src/`. Notebooks only import from `src/` and never hold core logic.
- Configs are YAML files in `configs/`. Never hard-code paths: resolve from the repo root or a `SOLAR_K_DATA` env var that defaults to `data/`.
- Pin every new dependency with its exact version in `requirements.txt`.
- Set seeds explicitly and record them. Write deterministic outputs where feasible.
- Put small unit tests in `tests/` for anything that touches the metric, RLE encoding, or splits. Those three must be correct before anything else.
- Keep figures under `reports/figures/` (git-ignored PNGs are fine; commit only the ones the report needs).

## Compute
- Development machine: MacBook Air M2 with **8 GB unified memory**. Process images one at a time and stream them, because a single 2048×2048 instance stack adds up quickly. Never load the whole dataset or all masks into memory. Use `torch.device("mps")` when it's available and fall back to CPU.
- Heavy training runs on Kaggle GPUs. Don't assume a fixed quota. Training code must be portable (paths and device come from config or env), must checkpoint regularly, and must resume from the last checkpoint.

## Metric and submission
- The local scorer must reproduce the organisers' self-evaluation notebook (`data/external/self-evaluation-notebook.ipynb` once present). Don't substitute a generic panoptic library.
- Submission CSV: columns `filament_id,segmentation_rle`, `filament_id = <image_stem>_<k>`, RLE = pycocotools compressed `counts` string from a Fortran-ordered uint8 2048×2048 mask, with no size field and no JSON wrapper. No rows for images with no predictions. Validate every CSV with `scripts/validate_submission.py` before it leaves the machine.

## Experiment discipline
- Every run gets an ID (`E###`) and a row in `experiments.csv`: id, date, spec, hypothesis, config path, split/fold, seed, compute, runtime, val PQ (mean ± std over folds), SQ, RQ, secondary metrics, decision (keep/reject), notes.
- Report validation numbers, not training numbers. Never pick thresholds on the fold you report.
- The Kaggle public LB is a sanity check, not the selection criterion.
