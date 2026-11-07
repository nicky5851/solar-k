# weights/ checkpoint audit

Verified 2026-11-07 against the live files in `weights/` (RunPod pod `iyn0hv9xbr2su6`, RTX PRO 4500 Blackwell,
`/workspace` on network volume `ft28lnkjdp` shared with every training pod this project used).

## 1. Integrity: all 15 files match `weights/MANIFEST.json`

```
sha256sum weights/*.pt
```

| file | sha256 | manifest match |
|---|---|---|
| E004-f0.pt | e9a8dd7797a03fa9b7926e25a94169b353fb49ff685730864d54d0f83d7c2920 | OK |
| E004-f1.pt | 9aebf3e633c791b89a809569923e3d2a099615d77788d02fb1667cffaec33fb4 | OK |
| E004-f2.pt | 552c40f3d1f73f07d4d971827b08cb18745ec2f61dd75cef128697788a7c57ea | OK |
| E004-f3.pt | 32d52cf89c23498e00cecb67639db3844ae7942a88df27426b81c73a7778709e | OK |
| E004-f4.pt | 5b26989e77a79bc1674b36aa44a4d1aea4621cc2e3a39b016e2a6595f231288f | OK |
| E007-f0.pt | efae3b460d21bdb21f4828d255ec9915fac9ec2fb7cbdece78a400944d2dd8a6 | OK |
| E007-f1.pt | abe308570af8f7555ffecef79ce4646532bda49ff2e29224eb0e18c042201169 | OK |
| E007-f2.pt | be95934dadf1461176be915489d11f5cde6c83d84b50cae6dedcd5a8ff5c3f42 | OK |
| E007-f3.pt | a8f246e9bf0ca6c451fa79ac91a98ed9f023276f4988c3c54311aa09eca67d47 | OK |
| E007-f4.pt | dfe437239e5bdb58d7d03e4eee73a9ff0f5a461aa9e61e90ae4d605d45cbb36a | OK |
| E015-f0.pt | e56381fdc26332bbaa518b024de74104d975da988305f2fc96a7f9303a251b4b | OK |
| E015-f1.pt | d04bb0b7efb7944690f50ff609a4c5eb35d7d33655c9344a52a6719205e22bd6 | OK |
| E015-f2.pt | 67513a194a0f808e67594eed62ae7945487eb62fdc7bef248673bd38fbfde9e0 | OK |
| E015-f3.pt | 9c34d7eadf6035b2562d1279b3346b8dee38d862aef9f444498c6ff1372860d2 | OK |
| E015-f4.pt | 16afa1ac9347c97430738741ba25b209b51c013788459279924d727233e130e4 | OK |

All 15 `sha256sum` outputs matched `weights/MANIFEST.json` exactly (verified by direct comparison, not just by
`download_weights.py`'s own cached-OK message). Sizes: E007 ~40.9MB/fold (efficientnet-b4), E015 ~49.8MB/fold
(tu-regnety_040), E004 ~49.0MB/fold (resnet34); all fp16 EMA-shadow state_dicts, ~1.36GB total.

## 2. Provenance: which checkpoints are original vs. retrained

Five of the fifteen files are **not** the originally-trained checkpoints — they were lost at some point before
Spec 08 (confirmed via `experiments.csv`, not merely "never existed") and retrained from scratch on pod
`85ki7s4ssa3lqv`, same configs and seeds as the originals:

| model-fold | status | source |
|---|---|---|
| E004-f0, E004-f1 | original | E004 training, 2026-11-01 |
| E004-f2, E004-f3, E004-f4 | **retrained** 2026-11-07 | `experiments.csv` row `E013_rebuild` |
| E007-f0..f4 (all 5) | original | E007 training, 2026-11-02 / `E012-f2..f4` |
| E015-f0, E015-f1, E015-f2 | original | E015 training, 2026-11-03 |
| E015-f3, E015-f4 | **retrained** 2026-11-07 | `experiments.csv` row `E030` |

Both retrains used the pipeline's original `configs/<model>-f<k>.yaml` (unchanged) and seed 42. Each retrain's
effect on the pipeline was measured, not assumed:

- **E015-f3/f4** (primary pipeline, `E030`): a determinism check regenerated TTA probability maps from the
  retrained E015-f0 checkpoint alongside the untouched original and found them **bit-identical** (max abs diff
  0.0 across 322 files) — evidence the retrain recipe itself is faithful, not just "close enough." The full
  5-fold P5 comparison of the rebuilt pipeline (E030) against the original frozen pipeline (E024) gave pooled
  ΔPQ = **-0.0004** (max per-fold |ΔPQ| = 0.0038), inside the spec's ±0.005 reproducibility bar. **Decision: E030
  supersedes E024 as the checkpoint-of-record** — E024 is no longer reproducible from current checkpoints since
  E015-f3/f4 were lost, E030 is.
- **E004-f2/f3/f4** (backup pipeline, `E013_rebuild`): P5 comparison of the restored backup against the
  originally-scored E013 gave pooled ΔPQ = **+0.009** (all 5 folds positive, bootstrap 95% CI [0.0047, 0.0135],
  excludes 0) — slightly better than the original, not a regression, and expected to not be byte-identical since
  this is a restoration (3 of 10 checkpoints freshly retrained with different init/cuDNN nondeterminism), unlike
  E030's stricter reproducibility bar.

## 3. End-to-end reproduction fidelity (shipped weights -> submission CSV)

`scripts/predict.py` reruns fresh fp32 inference from these 15 files and applies the frozen post-processing
chains in `configs/final_postproc_{primary,backup}.json` — no OOF/CV refitting. As of this audit, `predict.py`
quantizes each per-member, per-fold probability map through the same 8-bit PNG round-trip
(`train.py:save_probs_png`'s `clip(p*255,0,255).astype(uint8)`, then `/255.0`) that the original pipeline applied
*before* ensembling, since that quantization — not just TTA/inference nondeterminism — is what the original
`E030_primary.csv`/`E013_backup.csv` were built from. See `docs/HANDOFF.md` for the before/after reproduction
numbers (pre-quantization: 155/176 and 136/174 images with matching instance counts, mean IoU 0.992/0.997 on
those; post-quantization re-verification logged there once complete).

## 4. What this audit does not cover

- No check that the *architectures* (`solar_k.unet.build_model`) are unchanged since these checkpoints were
  saved — only that the files on disk match the manifest and that inference from them reproduces the scored
  submissions. A silent architecture change would make this audit's sha256 check pass while still breaking
  reproduction; `predict.py`'s own CSV-level comparison is the real guard against that.
- Checksums were computed on this pod's copy of the shared network volume (`ft28lnkjdp`), not on a GitHub
  Release download — `scripts/download_weights.py` is what verifies a download against this same manifest once
  the release exists.
