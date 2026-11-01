import pandas as pd

from solar_k import splits


def _make_images(n_days=20, per_day=3):
    rows = []
    sites = ["A", "B", "C"]
    i = 0
    for d in range(n_days):
        date = f"2020-01-{d + 1:02d}"
        for k in range(per_day):
            rows.append(
                {
                    "file_stem": f"img_{i:04d}",
                    "site": sites[i % len(sites)],
                    "date": date,
                    "n_annotators": 1 + (i % 3),
                }
            )
            i += 1
    return pd.DataFrame(rows)


def test_no_image_or_day_appears_in_two_folds():
    images = _make_images()
    folded = splits.assign_folds(images, n_folds=5, seed=0)
    issues = splits.validate_folds(folded, expected_stems=set(images["file_stem"]))
    assert issues == []


def test_every_image_is_assigned():
    images = _make_images()
    folded = splits.assign_folds(images, n_folds=5, seed=0)
    assert folded["fold"].notna().all()
    assert set(folded["file_stem"]) == set(images["file_stem"])


def test_same_day_images_share_a_fold():
    images = _make_images()
    folded = splits.assign_folds(images, n_folds=5, seed=0)
    assert (folded.groupby("date")["fold"].nunique() == 1).all()


def test_fold_sizes_are_reasonably_balanced():
    images = _make_images(n_days=50, per_day=2)
    folded = splits.assign_folds(images, n_folds=5, seed=1)
    counts = folded.groupby("fold").size()
    assert counts.max() - counts.min() <= 4


def test_validate_folds_catches_duplicate_image():
    images = _make_images()
    folded = splits.assign_folds(images, n_folds=5, seed=0)
    dup = pd.concat([folded, folded.iloc[[0]]], ignore_index=True)
    issues = splits.validate_folds(dup)
    assert any("more than once" in issue for issue in issues)


def test_validate_folds_catches_split_day():
    images = _make_images()
    folded = splits.assign_folds(images, n_folds=5, seed=0)
    bad = folded.copy()
    bad.loc[bad.index[0], "fold"] = (bad.loc[bad.index[0], "fold"] + 1) % 5
    issues = splits.validate_folds(bad)
    assert any("span more than one fold" in issue for issue in issues)


def test_validate_folds_catches_missing_image():
    images = _make_images()
    folded = splits.assign_folds(images, n_folds=5, seed=0)
    issues = splits.validate_folds(folded, expected_stems=set(images["file_stem"]) | {"missing_img"})
    assert any("missing from the split" in issue for issue in issues)
