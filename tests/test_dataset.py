"""CODING_SPEC_07_RESOLUTION_AND_SELF_TRAINING.md E027: train_scale support."""
from __future__ import annotations

import numpy as np

from solar_k import dataset as skds


def _synthetic_disk(h=256, w=256, cx=130, cy=120, radius=90):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = np.where(r <= radius, 150 - 60 * (r / radius) ** 2, 5).astype(np.uint8)
    return img


def _one_image_fixture(tmp_path, monkeypatch, stem="stemX"):
    import cv2

    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path))
    img = _synthetic_disk()
    train_dir = tmp_path / "raw" / "train_images"
    train_dir.mkdir(parents=True)
    raw_path = train_dir / f"{stem}.jpeg"
    cv2.imwrite(str(raw_path), img)
    monkeypatch.setattr(
        "solar_k.io.iter_train_images", lambda: [raw_path],
    )
    images_by_file = {f"{stem}.jpeg": [{"id": f"A-{stem}", "file_name": f"{stem}.jpeg"}]}
    anns_by_image_id = {
        f"A-{stem}": [{"segmentation": [[100, 100, 140, 100, 140, 130, 100, 130]]}],
    }
    return images_by_file, anns_by_image_id


def test_train_scale_default_is_1_and_unchanged_shape(tmp_path, monkeypatch):
    images_by_file, anns_by_image_id = _one_image_fixture(tmp_path, monkeypatch, "stemX")
    ds_default = skds.SolarCropDataset(["stemX"], images_by_file, anns_by_image_id, crop_size=64, length=1, seed=1)
    ds_explicit = skds.SolarCropDataset(["stemX"], images_by_file, anns_by_image_id, crop_size=64, length=1, seed=1, train_scale=1.0)
    img_default, tgt_default, _ = ds_default._load_image_and_target("stemX")
    img_explicit, tgt_explicit, _ = ds_explicit._load_image_and_target("stemX")
    assert np.array_equal(img_default, img_explicit)
    assert np.array_equal(tgt_default, tgt_explicit)
    assert img_default.shape == (256, 256)


def test_train_scale_upsamples_image_and_target(tmp_path, monkeypatch):
    images_by_file, anns_by_image_id = _one_image_fixture(tmp_path, monkeypatch, "stemY")
    ds = skds.SolarCropDataset(["stemY"], images_by_file, anns_by_image_id, crop_size=64, length=1, seed=1, train_scale=1.5)
    img, tgt, _ = ds._load_image_and_target("stemY")
    assert img.shape == (384, 384)
    assert tgt.shape == (384, 384)
    # the annotated square (rows/cols [100,130)*[100,140) at native res) should
    # still show up as positive target mass around its upsampled location,
    # not be blank or mislocated (this is the regression the spec warns
    # about: rasterising polygon coords straight at the scaled canvas size).
    assert tgt[int(115 * 1.5), int(120 * 1.5)] > 0.5
    assert tgt[5, 5] == 0.0


def test_skeleton_recall_with_train_scale_raises(tmp_path, monkeypatch):
    images_by_file, anns_by_image_id = _one_image_fixture(tmp_path, monkeypatch, "stemZ")
    try:
        skds.SolarCropDataset(["stemZ"], images_by_file, anns_by_image_id, skeleton_recall=True, train_scale=1.5)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
