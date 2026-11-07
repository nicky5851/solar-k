#!/usr/bin/env python3
"""CODING_SPEC_08_FINAL_PACKAGE.md 2.3: the single entry point from the
weights shipped in the repository (``weights/<model>-f<k>.pt``, fp16 EMA
state_dicts, downloaded via ``scripts/download_weights.py``) to a validated
submission CSV, with no OOF/CV fitting -- the post-processing chain is
frozen and loaded from ``configs/final_postproc_{primary,backup}.json``
(written by ``scripts/build_final.py --postproc-out ...``).

For each member listed in that JSON (member names are the historical
output-directory convention: an "tta4s"/"tta4sR" suffix means the frozen
4-flip + 1.25x-upscale TTA recipe that CODING_SPEC_06's primary pipeline
uses; no suffix means the plain 3-view flip-only recipe every raw
E004/E007 map used -- both recipes are exactly reproduced here from
``solar_k.unet.make_prob_fn``, same tta_views/tta_scales as
``scripts/infer.py``/``scripts/train.py`` always used), this script loads
that member's 5 per-fold checkpoints (fp16 on disk, upcast to fp32 -- the
spec requires inference in fp32 even though the saved weights are fp16),
runs fresh inference over the test images, and averages member-outer
fold-inner (matching ``solar_k.ensemble.load_test_mean``'s summation order)
to reproduce the same probability maps the original pipeline built from
precomputed PNGs -- modulo the PNG's own 8-bit quantization, which is why
the spec's own acceptance bar is "CSV matches exactly, or mean per-instance
IoU >= 0.99", not necessarily byte-identical.

Usage:
  python scripts/predict.py --images data/raw/test/test_images \
      --out outputs/submissions/E030_rebuilt.csv --pipeline primary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solar_k import components as skcomp
from solar_k import cv_pipeline as cvp
from solar_k import io as skio
from solar_k import masks as skmasks
from solar_k import postprocess as skpp
from solar_k import preprocess as skpre
from solar_k import scorer as sksc
from solar_k import unet as sku

import validate_submission as vs

N_FOLDS = 5


def member_model_and_tta(member: str) -> tuple[str, bool]:
    """Member names are the historical output-directory convention: strip a
    trailing "tta4s" or "tta4sR" to get the model family; its presence means
    the 4-flip + 1.25x-upscale TTA recipe, its absence the plain 3-view
    flip-only recipe (CODING_SPEC_05/06)."""
    for suffix in ("tta4sR", "tta4s"):
        if member.endswith(suffix):
            return member[: -len(suffix)], True
    return member, False


def load_member_config(model: str) -> dict:
    path = skio.repo_root() / "configs" / f"{model}-f0.yaml"
    return yaml.safe_load(path.read_text())


def build_member_fold_model(weights_dir: Path, model: str, fold: int, encoder_name: str, device: torch.device) -> torch.nn.Module:
    net = sku.build_model(encoder_name=encoder_name, encoder_weights=None)
    sd16 = torch.load(weights_dir / f"{model}-f{fold}.pt", map_location="cpu", weights_only=False)
    sd32 = {k: (v.float() if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in sd16.items()}
    net.load_state_dict(sd32, strict=True)
    net.to(device)
    net.eval()
    return net


def predict_test_means(members: list[str], weights_dir: Path, device_str: str, member_weights: list[float] | None, log=print) -> dict[str, np.ndarray]:
    """Fresh-inference equivalent of ``solar_k.ensemble.load_test_mean`` --
    same member-outer, fold-inner running-mean order, but probabilities come
    from loaded weights instead of precomputed PNGs."""
    device = sku.pick_device(device_str)
    test_paths = {p.stem: p for p in skio.iter_test_images()}
    n_members = len(members)
    expanded_weights = None
    if member_weights is not None:
        expanded_weights = [w for w in member_weights for _ in range(N_FOLDS)]

    running: dict[str, np.ndarray] = {}
    running_w = 0.0
    total_maps = n_members * N_FOLDS
    map_i = 0
    for m_i, member in enumerate(members):
        model, is_tta = member_model_and_tta(member)
        cfg = load_member_config(model)
        encoder_name = cfg.get("encoder_name", "resnet34")
        tile_size, overlap = cfg.get("infer_tile_size", 1024), cfg.get("infer_overlap", 128)
        tta_views, tta_scales = (4, (1.0, 1.25)) if is_tta else (3, (1.0,))
        for fold in range(N_FOLDS):
            map_i += 1
            log(f"[{map_i}/{total_maps}] member={member} (model={model} tta={is_tta}) fold={fold}")
            net = build_member_fold_model(weights_dir, model, fold, encoder_name, device)
            prob_fn = sku.make_prob_fn(net, device, tta_flips=True, tile_size=tile_size, overlap=overlap,
                                        tta_views=tta_views, tta_scales=tta_scales)
            w = expanded_weights[m_i * N_FOLDS + fold] if expanded_weights is not None else 1.0
            with torch.no_grad():
                for stem in sorted(test_paths):
                    processed, disk_info = skpre.load_or_build(stem, test_paths[stem])
                    disk_mask, _ = skpre.disk_mask_and_r(processed.shape, *disk_info)
                    probs = prob_fn(processed, disk_mask)
                    if stem not in running:
                        running[stem] = np.zeros_like(probs, dtype=np.float64)
                    running[stem] += probs.astype(np.float64) * w
            del net, prob_fn
            running_w += w if expanded_weights is not None else 1.0
    denom = running_w if expanded_weights is not None else total_maps
    return {stem: (acc / denom).astype(np.float32) for stem, acc in running.items()}


def apply_frozen_postproc(probs_by_stem: dict[str, np.ndarray], postproc: dict, log=print) -> list[dict]:
    """The exact keep/regrow logic ``cv_pipeline.write_submission`` applies
    to the test set, factored out so both share one implementation."""
    cfg1: skpp.PostprocessConfig = postproc["cfg1"]
    ccfg = cvp.components_cfg(cfg1)
    me, ml = postproc["shape_gate"]
    scorer = postproc["scorer"]
    scorer_c = postproc["scorer_c"]
    t_mask, d = postproc["regrow_t_mask"], postproc["regrow_d"]

    test_paths = {p.stem: p for p in skio.iter_test_images()}
    rows = []
    for stem in sorted(probs_by_stem):
        probs = probs_by_stem[stem]
        processed, disk_info = skpre.load_or_build(stem, test_paths[stem])
        features, labels = skcomp.extract_components(stem, probs, processed, disk_info, ccfg)
        df = skcomp.to_dataframe(features)
        if len(df):
            keep = cvp.gate_keep(df, me, ml) & (sksc.keep_score(scorer, df) > scorer_c)
            seeds = [(labels == f.label).astype(np.uint8) for f, k in zip(features, keep) if k]
            grown = skpp.regrow_instances(probs, seeds, t_mask, d) if seeds else []
            for k_i, mask in enumerate(grown):
                if mask.sum() == 0:
                    continue
                rows.append({"filament_id": f"{stem}_{k_i}", "segmentation_rle": skmasks.encode_rle(mask.astype(np.uint8))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=None, help="default: data/raw/test/test_images")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pipeline", choices=("primary", "backup"), required=True)
    ap.add_argument("--weights-dir", type=Path, default=None, help="default: weights/")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    repo = skio.repo_root()
    weights_dir = args.weights_dir or (repo / "weights")
    postproc_path = repo / "configs" / f"final_postproc_{args.pipeline}.json"
    postproc = cvp.load_final_postproc(postproc_path)
    members = postproc["members"]
    member_weights = postproc["member_weights"]

    if args.images is not None and args.images != skio.test_images_dir():
        raise SystemExit("predict.py's frozen pipeline is defined over the competition test set; "
                          "--images must be the test image directory (or omitted).")

    print(f"pipeline={args.pipeline} members={members} weights={member_weights}")
    probs_by_stem = predict_test_means(members, weights_dir, args.device, member_weights)
    rows = apply_frozen_postproc(probs_by_stem, postproc)

    import pandas as pd
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
    errors = vs.validate(args.out, 2048, 2048, image_dir=skio.test_images_dir())
    if errors:
        print("Submission validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"wrote + validated {args.out} ({len(rows)} instances)")


if __name__ == "__main__":
    main()
