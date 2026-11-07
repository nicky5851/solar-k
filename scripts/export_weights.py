#!/usr/bin/env python3
"""CODING_SPEC_08_FINAL_PACKAGE.md 2.1: export the EMA state_dict (no
optimizer/scheduler/rng state) of each of the 15 checkpoints E030/E013 need,
in fp16, to weights/<model>-f<k>.pt, plus weights/MANIFEST.json (file,
model, fold, encoder, sha256, bytes, training config path).

The checkpoint's "ema" field IS a plain state_dict (same keys/shapes as the
model's own state_dict() -- solar_k.unet.EMA.state_dict() just returns its
shadow dict directly), so no unwrapping beyond ckpt["ema"] is needed. This
exported dict is exactly what scripts/infer.py and scripts/train.py already
load via model.load_state_dict(shadow, strict=True) for OOF/test inference.

Usage: python scripts/export_weights.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solar_k import io as skio  # noqa: E402

CHECKPOINTS = (
    [("E007", k, "efficientnet-b4", f"configs/E007-f{k}.yaml") for k in range(5)]
    + [("E015", k, "tu-regnety_040", f"configs/E015-f{k}.yaml") for k in range(5)]
    + [("E004", k, "resnet34", f"configs/E004-f{k}.yaml") for k in range(5)]
)


def main() -> None:
    repo = skio.repo_root()
    out_dir = repo / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for model, fold, encoder, config_path in CHECKPOINTS:
        src = repo / "outputs" / f"{model}-f{fold}" / "checkpoints" / "best.pt"
        if not src.exists():
            raise SystemExit(f"missing checkpoint: {src}")
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        ema = ckpt.get("ema")
        if ema is None:
            raise SystemExit(f"{src} has no saved EMA shadow")
        fp16 = {k: (v.half() if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in ema.items()}

        dst = out_dir / f"{model}-f{fold}.pt"
        torch.save(fp16, dst)

        sha256 = hashlib.sha256(dst.read_bytes()).hexdigest()
        size = dst.stat().st_size
        manifest.append({
            "file": f"weights/{dst.name}",
            "model": model,
            "fold": fold,
            "encoder": encoder,
            "sha256": sha256,
            "bytes": size,
            "config_path": config_path,
        })
        print(f"{dst.name}: {size:,} bytes, sha256={sha256[:12]}...")

    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
