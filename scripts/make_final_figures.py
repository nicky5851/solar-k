"""CODING_SPEC_06 final deliverables (qualitative): the pipeline diagram and the
H-alpha overlay gallery (typical / worst-split / worst-merge / recovered-faint-filament),
all on OOF images only. Run after `scripts/run_p5.py --gallery-stems ...` has written
`data/interim/<experiment>_gallery/<stem>.npz` (probs, pred, gt_<annotator> arrays).

Pipeline diagram needs no data; the gallery needs the npz files plus the matching raw
grayscale images (saved alongside, see Spec 06 session notes) since the dataset itself is
not committed to the repo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "reports" / "figures"


def make_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    boxes = [
        (0.2, "H-alpha\nimage", "#dbe9f6"),
        (2.1, "U-Net\n(resnet34 /\nefficientnet-b4)\n5-fold CV", "#cfe8d8"),
        (4.3, "4-flip +\n1.25x-scale\nTTA\n(inference-only)", "#cfe8d8"),
        (6.5, "member-mean\nprob map", "#dbe9f6"),
        (8.3, "cfg1 hysteresis +\nshape gate +\nscorer (c)", "#f6e3cf"),
        (10.5, "regrow\n(watershed)", "#f6e3cf"),
    ]
    w, h = 1.7, 1.5
    y = 1.5
    for x, label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                              fc=color, ec="#444", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    for (x0, _, _), (x1, _, _) in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(x1, y + h / 2), xytext=(x0 + w, y + h / 2),
                     arrowprops=dict(arrowstyle="->", lw=1.3, color="#444"))

    ax.text(6.5 + w / 2, y - 0.35, "fitted per CV fold on the fold's train stems (P5 protocol)",
             ha="center", fontsize=8, style="italic", color="#555")
    last_x = boxes[-1][0] + w / 2
    ax.annotate("", xy=(last_x, y + h), xytext=(last_x, y + h + 0.6),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="#444"))
    ax.text(last_x, y + h + 0.95, "instance masks\n-> PQ/SQ/RQ\n(E024 final build)",
            ha="center", va="bottom", fontsize=8.5)

    ax.set_xlim(0, last_x + 2.2)
    ax.set_ylim(0, y + h + 2.0)
    ax.set_title("Frozen Spec 06 pipeline: E007 + E015, TTA inference, P5 post-processing refit",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_pipeline_diagram.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIG_DIR / '06_pipeline_diagram.png'}")


def _semantic_union(masks: np.ndarray) -> np.ndarray:
    return masks.any(axis=0) if len(masks) else np.zeros((1, 1), dtype=bool)


def make_overlay_gallery(gallery_dir: Path, images_dir: Path, captions: dict[str, str]) -> None:
    stems = list(captions)
    fig, axes = plt.subplots(1, len(stems), figsize=(5 * len(stems), 5.5))
    if len(stems) == 1:
        axes = [axes]
    for ax, stem in zip(axes, stems):
        d = np.load(gallery_dir / f"{stem}.npz")
        img = np.load(images_dir / f"{stem}.npy")
        pred = _semantic_union(d["pred"])
        gt_keys = [k for k in d.files if k.startswith("gt_")]
        gt = _semantic_union(np.concatenate([d[k] for k in gt_keys], axis=0)) if gt_keys else np.zeros_like(pred)

        ax.imshow(img, cmap="gray")
        overlay = np.zeros((*img.shape, 4))
        overlay[gt & ~pred] = (0.2, 0.6, 1.0, 0.55)   # missed (FN): blue
        overlay[pred & ~gt] = (1.0, 0.25, 0.2, 0.55)  # spurious (FP side): red
        overlay[gt & pred] = (0.15, 0.85, 0.3, 0.55)  # agreement: green
        ax.imshow(overlay)
        ax.set_title(captions[stem], fontsize=10)
        ax.axis("off")

    legend = [mpatches.Patch(color=(0.15, 0.85, 0.3), label="pred & GT agree"),
              mpatches.Patch(color=(0.2, 0.6, 1.0), label="GT missed (FN)"),
              mpatches.Patch(color=(1.0, 0.25, 0.2), label="pred only (FP side)")]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = FIG_DIR / "06_overlay_gallery.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-dir", default="data/interim/E024_quant_gallery")
    ap.add_argument("--images-dir", default="data/interim/E024_quant_gallery_images")
    args = ap.parse_args()

    make_pipeline_diagram()
    captions = {
        "20140509071314Th": "typical",
        "20111215063134Lh": "worst split\n(fragmentation)",
        "20110914063134Lh": "worst merge",
        "20140216230134Lh": "recovered faint filament\n(vs E013)",
    }
    make_overlay_gallery(REPO_ROOT / args.gallery_dir, REPO_ROOT / args.images_dir, captions)
