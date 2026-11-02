import numpy as np

from solar_k import tuning as tn
from solar_k import postprocess as skpp


def test_min_area_grid_values():
    grid = tn.min_area_grid(200)
    assert grid == [100, 200, 400]


def test_split_tune_report_partitions_by_day_no_overlap():
    day_by_stem = {}
    for d in range(10):
        for k in range(3):
            day_by_stem[f"img_{d}_{k}"] = f"2020-01-{d + 1:02d}"
    tune, report = tn.split_tune_report(day_by_stem, seed=0)
    assert tune.isdisjoint(report)
    assert tune | report == set(day_by_stem.keys())
    # every stem for a given day lands in exactly one half
    for d in range(10):
        day = f"2020-01-{d + 1:02d}"
        stems = {s for s, dd in day_by_stem.items() if dd == day}
        assert stems <= tune or stems <= report


def test_split_tune_report_deterministic_for_fixed_seed():
    day_by_stem = {f"img_{i}": f"2020-01-{(i % 5) + 1:02d}" for i in range(20)}
    t1, r1 = tn.split_tune_report(day_by_stem, seed=7)
    t2, r2 = tn.split_tune_report(day_by_stem, seed=7)
    assert t1 == t2 and r1 == r2


def test_full_grid_contains_both_modes():
    configs = tn.full_grid(eda_p1_area=200)
    modes = {c.mode for c in configs}
    assert modes == {"threshold", "hysteresis"}
    assert all(isinstance(c, skpp.PostprocessConfig) for c in configs)


def test_grid_search_picks_best_pq_on_tune_only():
    h = w = 10
    gt = {"img1": {"ann1": np.stack([_box(h, w, (2, 5, 2, 5))])}}
    disk_info = {"img1": (5.0, 5.0, 10.0)}
    site_by_stem = {"img1": "A"}

    def compute_probs(stem):
        probs = np.zeros((h, w), dtype=np.float32)
        probs[2:6, 2:6] = 0.9  # exact match to gt at a high threshold
        return probs

    configs = [
        skpp.PostprocessConfig(mode="threshold", t=0.95, min_area=1),  # misses everything
        skpp.PostprocessConfig(mode="threshold", t=0.5, min_area=1),  # exact match
    ]
    best, df = tn.grid_search({"img1"}, compute_probs, gt, disk_info, site_by_stem, configs)
    assert best.t == 0.5
    assert len(df) == 2


def _box(h, w, box):
    m = np.zeros((h, w), dtype=np.uint8)
    r0, r1, c0, c1 = box
    m[r0 : r1 + 1, c0 : c1 + 1] = 1
    return m


def test_grid_search_independent_of_stem_iteration_order():
    """CODING_SPEC_05 determinism: grid_search must not depend on the
    iteration order of the (set-typed) tune_stems it is given."""
    rng = np.random.RandomState(0)
    stems = [f"s{i}" for i in range(8)]
    probs = {s: rng.rand(48, 48).astype(np.float32) for s in stems}
    gt = {}
    for s in stems:
        m = (probs[s] > 0.8).astype(np.uint8)
        gt[s] = {"A": m[None]}
    configs = [skpp.PostprocessConfig(mode="threshold", t=t, closing_radius=1, min_area=2) for t in (0.6, 0.7, 0.8)]
    out = []
    for order in (stems, list(reversed(stems)), stems[3:] + stems[:3]):
        best, df = tn.grid_search(order, lambda s: probs[s], gt, {}, {}, configs)
        out.append((best, df["pq"].tolist()))
    assert out[0] == out[1] == out[2]
