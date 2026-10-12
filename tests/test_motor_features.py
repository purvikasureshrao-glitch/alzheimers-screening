import numpy as np

from motor.features import FEATURE_NAMES, gap_count, session_features, trial_features
from motor.loader import _cell


def test_feature_count_is_18():
    assert len(FEATURE_NAMES) == 18 == len(set(FEATURE_NAMES))


def test_right_hand_regular_tapping():
    taps = [(0.5 + 0.25 * i, "rshift") for i in range(38)]   # 4 taps/s from 0.5 s
    f = trial_features(taps, "right")
    assert f["irt"] == 0.5
    assert abs(f["tps"] - 3.8) < 1e-9
    assert f["_gaps"] == 0


def test_gap_counts_onset_middle_and_tail():
    assert gap_count(np.array([1.5, 1.6, 1.7, 9.9])) == 2    # onset 1.5 s, 1.7 -> 9.9
    assert gap_count(np.array([])) == 1
    assert gap_count(np.arange(0.2, 10, 0.5)) == 0


def test_dual_pairing_and_unpaired():
    taps = []
    for i in range(10):
        t = 1 + i * 0.5
        taps += [(t, "lshift"), (t + 0.05, "rshift")]
    taps.append((9.9, "lshift"))                              # one unpaired press
    f = trial_features(taps, "dual")
    assert f["n_pairs"] == 10
    assert abs(f["pair_dur"] - 0.05) < 1e-9
    assert abs(f["unpaired_rate"] - 1 / 21) < 1e-9


def test_alternating_errors_and_transitions():
    keys = ["lshift", "rshift", "lshift", "lshift", "rshift"]  # one same-key error
    taps = [(1 + 0.4 * i, k) for i, k in enumerate(keys)]
    f = trial_features(taps, "alt")
    assert abs(f["error_rate"] - 1 / 5) < 1e-9
    assert f["n_trans"] == 3
    assert abs(f["trans_time"] - 0.4) < 1e-9


def test_session_median_and_gap_rule():
    good = [(0.3 + 0.3 * i, "rshift") for i in range(32)]
    late = [(2.0 + 0.3 * i, "rshift") for i in range(26)]      # onset gap
    s = session_features({"right": [good, good, late], "dual": [], "alt": []})
    assert set(s) == set(FEATURE_NAMES)
    assert s["right_irt"] == 0.3                               # median of .3, .3, 2.0
    assert s["gap_count"] == 1.0                               # right had a gap in some trial
    assert np.isnan(s["dual_tps_var"]) or s["dual_tps"] == 0


def test_loader_cells():
    assert _cell("['lshift', 'rshift']") == ["lshift", "rshift"]
    assert _cell("None") == [] and _cell(float("nan")) == []
    assert _cell(1.25) == [1.25]
