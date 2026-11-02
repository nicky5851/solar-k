"""CODING_SPEC_05_ENSEMBLE.md E015 (and Spec 04's rule): an ensemble
member's encoder must expose a stride-2 skip feature, because filaments are
often 2-4 px wide. Checks every E015 config's encoder before any training
run. No pretrained weights are downloaded (encoder_weights=None)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from solar_k import unet as sku

REPO_ROOT = Path(__file__).resolve().parents[1]
E015_CONFIGS = sorted((REPO_ROOT / "configs").glob("E015-f*.yaml"))


@pytest.mark.parametrize("cfg_path", E015_CONFIGS, ids=lambda p: p.stem)
def test_encoder_has_stride2_skip(cfg_path):
    cfg = yaml.safe_load(cfg_path.read_text())
    model = sku.build_model(encoder_name=cfg["encoder_name"], encoder_weights=None)
    model.eval()
    x = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        feats = model.encoder(x)
    reductions = [256 // f.shape[-1] for f in feats if f.shape[-1] > 0]
    assert 2 in reductions, f"{cfg['encoder_name']}: feature reductions {reductions}, no stride-2 skip"
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 256, 256)
