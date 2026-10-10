import numpy as np
from eye.features import adaptive_threshold, extract, load_gaze


def _jump_recording():
    t = np.arange(0, 300, 3.0)
    x = np.where(t < 150, 500.0, 900.0) + np.random.default_rng(0).normal(0, 2, len(t))
    y = np.full(len(t), 400.0)
    return t, np.column_stack([x, y, x, y])


def test_one_jump_is_one_saccade():
    f = extract(*_jump_recording())
    assert f["n_sac"] == 1 and f["n_fix"] == 2
    assert 350 < f["mean_sac_amp"] < 450
    assert f["disconj_x"] == 0


def test_threshold_adapts_to_noise():
    rng = np.random.default_rng(0)
    quiet, noisy = rng.normal(0.1, 0.01, 200), rng.normal(0.1, 0.05, 200)
    assert adaptive_threshold(noisy) > adaptive_threshold(quiet)


def test_missing_samples_and_short_files_do_not_crash():
    t, xy = _jump_recording()
    xy[:20] = np.nan
    assert extract(t, xy)["missing_frac"] > 0
    assert np.isnan(extract(t[:2], xy[:2])["n_sac"])


def test_loader_drops_non_increasing_timestamps_and_marks_missing(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("10,1,2,3,4\n12,1,2,3,4\n11,1,2,3,4\n15,-100,-100,5,6\n")
    t, xy = load_gaze(p)
    assert list(t) == [10, 12, 15] and np.isnan(xy[2, 0]) and xy[2, 2] == 5
