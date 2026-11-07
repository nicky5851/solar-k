import numpy as np

from solar_k import preprocess as pp


def _synthetic_disk(h=256, w=256, cx=130, cy=120, radius=90):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # simple limb-darkening-like profile: bright centre, dim near the limb
    img = np.where(r <= radius, 150 - 60 * (r / radius) ** 2, 5).astype(np.uint8)
    return img, (cx, cy, radius)


def test_fit_disk_recovers_center_and_radius():
    img, (cx, cy, radius) = _synthetic_disk()
    fcx, fcy, fr = pp.fit_disk(img)
    assert abs(fcx - cx) < 3
    assert abs(fcy - cy) < 3
    assert abs(fr - radius) < 5


def test_limb_correction_flattens_radial_profile():
    img, (cx, cy, radius) = _synthetic_disk()
    disk_mask, r = pp.disk_mask_and_r(img.shape, cx, cy, radius)
    corrected = pp.limb_correct(img, disk_mask, r, radius)

    img[disk_mask].astype(np.float64).std()
    corrected_std_normalized = (corrected[disk_mask] / corrected[disk_mask].mean()).std()
    raw_std_normalized = (img[disk_mask].astype(np.float64) / img[disk_mask].mean()).std()
    assert corrected_std_normalized < raw_std_normalized


def test_normalize_disk_is_bounded_and_zero_outside():
    img, (cx, cy, radius) = _synthetic_disk()
    disk_mask, r = pp.disk_mask_and_r(img.shape, cx, cy, radius)
    corrected = pp.limb_correct(img, disk_mask, r, radius)
    normed = pp.normalize_disk(corrected, disk_mask)

    assert normed.min() >= 0.0 and normed.max() <= 1.0
    assert np.array_equal(normed[~disk_mask], np.zeros_like(normed[~disk_mask]))


def test_preprocess_image_roundtrip_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path))
    img, _ = _synthetic_disk()
    result = pp.preprocess_image(img)
    pp.save_processed("test_stem", result["processed"])
    loaded = pp.load_processed("test_stem")
    assert loaded.shape == img.shape
    assert np.allclose(loaded, result["processed"].astype(np.float16).astype(np.float32))


def test_highpass_removes_smooth_component():
    yy, xx = np.mgrid[0:64, 0:64]
    smooth = (xx + yy).astype(np.float32)
    hp = pp.highpass(smooth, sigma=4)
    assert abs(hp.mean()) < 1.0
    assert hp.std() < smooth.std()


def test_load_or_build_caches_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path))
    img, _ = _synthetic_disk()
    raw_path = tmp_path / "raw.png"
    import cv2

    cv2.imwrite(str(raw_path), img)

    calls = []
    orig = pp.preprocess_image
    monkeypatch.setattr(pp, "preprocess_image", lambda im: (calls.append(1), orig(im))[1])

    processed1, disk_info1 = pp.load_or_build("stemA", raw_path)
    assert len(calls) == 1
    processed2, disk_info2 = pp.load_or_build("stemA", raw_path)
    assert len(calls) == 1  # second call is a cache hit, no recompute
    # the cache round-trips through float16, so compare at that precision
    # rather than exact float32 equality (same pattern as
    # test_preprocess_image_roundtrip_cache above)
    assert np.allclose(processed1.astype(np.float16), processed2.astype(np.float16))
    assert disk_info1 == disk_info2


def test_build_cache_skips_already_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path))
    img, _ = _synthetic_disk()
    import cv2

    paths = []
    for i in range(3):
        p = tmp_path / f"img{i}.png"
        cv2.imwrite(str(p), img)
        paths.append((f"stem{i}", p))

    n_built_first = pp.build_cache(paths, workers=1)
    assert n_built_first == 3
    n_built_second = pp.build_cache(paths, workers=1)
    assert n_built_second == 0  # nothing left to build
    for stem, _ in paths:
        assert pp.processed_path(stem).exists()


def test_load_or_build_miss_and_hit_are_bit_identical(tmp_path, monkeypatch):
    """CODING_SPEC_05 determinism: features must not depend on whether the
    preprocessing cache was warm (a miss used to return unrounded float32)."""
    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path))
    img, _ = _synthetic_disk()
    raw_path = tmp_path / "raw.png"
    import cv2

    cv2.imwrite(str(raw_path), img)
    miss, info_miss = pp.load_or_build("stemB", raw_path)
    hit, info_hit = pp.load_or_build("stemB", raw_path)
    assert miss.dtype == hit.dtype == np.float32
    assert np.array_equal(miss, hit)
    assert info_miss == info_hit



def test_load_or_build_scale_1_is_byte_identical_to_default(tmp_path, monkeypatch):
    """CODING_SPEC_07_RESOLUTION_AND_SELF_TRAINING.md E027: the new ``scale``
    kwarg must be a pure no-op at its default -- every pre-E027 call site
    omits it, so scale=1.0 has to reproduce exactly what load_or_build
    returned before this kwarg existed."""
    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path))
    img, _ = _synthetic_disk()
    raw_path = tmp_path / "raw.png"
    import cv2

    cv2.imwrite(str(raw_path), img)
    default, info_default = pp.load_or_build("stemC", raw_path)
    explicit, info_explicit = pp.load_or_build("stemC", raw_path, scale=1.0)
    assert np.array_equal(default, explicit)
    assert info_default == info_explicit
    # and it's cached under the same, unparameterised directory as before
    assert pp.processed_path("stemC") == pp.processed_path("stemC", scale=1.0)


def test_load_or_build_scale_upsamples_image_and_disk_info(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path))
    img, _ = _synthetic_disk()
    raw_path = tmp_path / "raw.png"
    import cv2

    cv2.imwrite(str(raw_path), img)
    base, (cx, cy, r) = pp.load_or_build("stemD", raw_path)
    scaled, (scx, scy, sr) = pp.load_or_build("stemD", raw_path, scale=1.5)
    assert scaled.shape == (round(base.shape[0] * 1.5), round(base.shape[1] * 1.5))
    assert scx == cx * 1.5 and scy == cy * 1.5 and sr == r * 1.5
    # scaled cache lives in its own dir, separate from the scale=1.0 cache
    assert pp.processed_path("stemD", scale=1.5) != pp.processed_path("stemD")


def test_load_or_build_scale_cache_hit_matches_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path))
    img, _ = _synthetic_disk()
    raw_path = tmp_path / "raw.png"
    import cv2

    cv2.imwrite(str(raw_path), img)
    miss, info_miss = pp.load_or_build("stemE", raw_path, scale=1.5)
    hit, info_hit = pp.load_or_build("stemE", raw_path, scale=1.5)
    assert np.array_equal(miss, hit)
    assert info_miss == info_hit



def test_build_cache_scale_warms_separate_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLAR_K_DATA", str(tmp_path))
    img, _ = _synthetic_disk()
    import cv2

    paths = []
    for i in range(2):
        path = tmp_path / f"scaleimg{i}.png"
        cv2.imwrite(str(path), img)
        paths.append((f"scalestem{i}", path))

    n_built_default = pp.build_cache(paths, workers=1)
    assert n_built_default == 2
    n_built_scaled = pp.build_cache(paths, workers=1, scale=1.5)
    assert n_built_scaled == 2  # scale=1.5 cache is separate, still a miss
    for stem, _ in paths:
        assert pp.processed_path(stem, scale=1.5).exists()
        assert pp.processed_path(stem, scale=1.5) != pp.processed_path(stem)
    # re-running at scale=1.5 is now a pure cache hit
    assert pp.build_cache(paths, workers=1, scale=1.5) == 0
