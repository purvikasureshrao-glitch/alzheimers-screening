import numpy as np
from common.metrics import auc_ci, balanced_acc_ci, f1_ci, pick_threshold, wilson_ci


def test_perfect_prediction():
    y = np.array([0, 0, 1, 1] * 10)
    r = balanced_acc_ci(y, y, "test")
    assert r.value == 1.0 and r.lo == 1.0 and r.hi == 1.0 and r.n == 40


def test_group_bootstrap_counts_subjects():
    y = np.array([0, 1] * 10)
    g = np.repeat(np.arange(10), 2)
    assert balanced_acc_ci(y, y, "test", groups=g).n == 10


def test_ci_contains_value_and_auc_ordered():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(0, 1, 200)
    r = auc_ci(y, s, "test")
    assert r.lo <= r.value <= r.hi and 0.5 < r.value < 1


def test_wilson_known_value():
    lo, hi = wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 1e-3 and abs(hi - 0.7634) < 1e-3


def test_pick_threshold_separates():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert 0.3 < pick_threshold(y, s) <= 0.7


def test_f1_runs():
    y = np.array([0, 1, 1, 0, 1])
    assert f1_ci(y, y, "test").value == 1.0
