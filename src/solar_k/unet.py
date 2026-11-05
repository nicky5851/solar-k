"""U-Net wrapper: build, checkpoint (with full resume state), and inference
with optional flip TTA (3 or 4 views), multi-scale upscale-only TTA
(CODING_SPEC_06_RECALL_AND_CONNECTIVITY.md E023), and tiling
(CODING_SPEC_02_BASELINE.md Stage C).

Device = CUDA > MPS > CPU, chosen here so scripts/train.py and
scripts/predict.py don't each reimplement the fallback.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(encoder_name: str = "resnet34", encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    return smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=1, classes=1)


class EMA:
    """Exponential moving average of a model's weights (CODING_SPEC_03_FP_AND_FOLDS.md,
    E004: decay 0.999, used for evaluation and for the saved checkpoints --
    the raw model keeps training/backpropagating as usual, and only this
    shadow copy is what gets validated and written to best.pt/last.pt)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()  # integer buffers (e.g. num_batches_tracked): copy, don't average

    def state_dict(self) -> dict:
        return self.shadow

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def capture_rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch's CPU RNG state must be a CPU ByteTensor; torch.load(...,
    # map_location=device) moves every tensor in the checkpoint (including
    # this one) onto that device, which set_rng_state then rejects.
    torch.random.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, model, optimizer, scheduler, iteration: int, best_metric: float, ema: "EMA | None" = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "iteration": iteration,
            "rng_state": capture_rng_state(),
            "best_metric": best_metric,
            "ema": ema.state_dict() if ema is not None else None,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, map_location=None, restore_rng: bool = True, ema: "EMA | None" = None) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and ckpt.get("rng_state") is not None:
        restore_rng_state(ckpt["rng_state"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    return ckpt


def load_model(checkpoint_path, encoder_name: str = "resnet34", device: str = "auto"):
    dev = pick_device(device)
    model = build_model(encoder_name=encoder_name, encoder_weights=None)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(dev)
    model.eval()
    return model, dev


def _infer_once(model: torch.nn.Module, device: torch.device, arr: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        probs = torch.sigmoid(logits)[0, 0].to("cpu").numpy()
    del t, logits
    return probs


def infer_tiled(model: torch.nn.Module, device: torch.device, arr: np.ndarray, tile_size: int, overlap: int) -> np.ndarray:
    h, w = arr.shape
    if tile_size >= h and tile_size >= w:
        return _infer_once(model, device, arr)
    out = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    step = max(1, tile_size - overlap)

    def axis_starts(size: int) -> list[int]:
        starts = list(range(0, max(size - tile_size, 0) + 1, step))
        if not starts:
            starts = [0]
        if starts[-1] + tile_size < size:
            starts.append(size - tile_size)
        return starts

    for y0 in axis_starts(h):
        for x0 in axis_starts(w):
            y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
            patch = arr[y0:y1, x0:x1]
            p = _infer_once(model, device, patch)
            out[y0:y1, x0:x1] += p
            weight[y0:y1, x0:x1] += 1.0
    return out / np.maximum(weight, 1e-6)


def make_prob_fn(
    model: torch.nn.Module, device: torch.device, tta_flips: bool = True, tile_size: int | None = None, overlap: int = 128,
    tta_views: int = 3, tta_scales: tuple[float, ...] = (1.0,),
):
    """Flip + scale TTA. ``tta_views`` (CODING_SPEC_05_ENSEMBLE.md E014): 3 =
    identity + h-flip + v-flip (the default every saved E004/E006/E007 map
    used, kept bit-for-bit), 4 = those plus the h+v flip. ``tta_flips=False``
    -> 1 view. ``tta_scales`` (CODING_SPEC_06 E023): each scale >= 1.0 is run
    by upscaling the input, inferring, then downscaling the *probability map*
    back to the original resolution before averaging -- upscaling only, never
    downscaling the input itself, since that destroys 2-px-wide filaments.
    Default ``(1.0,)`` reproduces the pre-E023 output exactly."""
    if tta_views not in (3, 4):
        raise ValueError(f"tta_views must be 3 or 4, got {tta_views}")
    if any(s < 1.0 for s in tta_scales):
        raise ValueError(f"tta_scales must be >= 1.0 (upscaling only), got {tta_scales}")

    def infer(arr: np.ndarray) -> np.ndarray:
        if tile_size is not None:
            return infer_tiled(model, device, arr, tile_size, overlap)
        return _infer_once(model, device, arr)

    def infer_at_scale(arr: np.ndarray, scale: float) -> np.ndarray:
        if scale == 1.0:
            return infer(arr)
        h, w = arr.shape
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        up = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        probs_up = infer(up)
        return cv2.resize(probs_up, (w, h), interpolation=cv2.INTER_LINEAR)

    def prob_fn(processed: np.ndarray, disk_mask: np.ndarray) -> np.ndarray:
        scale_probs = []
        for scale in tta_scales:
            probs = infer_at_scale(processed, scale)
            if tta_flips:
                probs_h = infer_at_scale(processed[:, ::-1], scale)[:, ::-1]
                probs_v = infer_at_scale(processed[::-1, :], scale)[::-1, :]
                if tta_views == 4:
                    probs_hv = infer_at_scale(processed[::-1, ::-1], scale)[::-1, ::-1]
                    probs = (probs + probs_h + probs_v + probs_hv) / 4.0
                else:
                    probs = (probs + probs_h + probs_v) / 3.0
            scale_probs.append(probs)
        probs = scale_probs[0] if len(scale_probs) == 1 else sum(scale_probs) / len(scale_probs)
        return probs * disk_mask.astype(np.float32)

    return prob_fn
