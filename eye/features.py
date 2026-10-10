"""Eye features from ADEM_TEST gaze files (PLAN B2, own method [OURS]).

File format (FINDINGS 1): 5 CSV columns timestamp_ms, x_left, y_left, x_right, y_right.
-100 is the tracker's "no data" value. Screen geometry is unpublished, so speeds are px/ms.

Event detection: velocity threshold derived from each recording's own noise, after
Nystrom & Holmqvist (2010): PT = mean + 6 SD of the velocities below PT, iterated to
convergence. [OURS] deviations: iteration starts from the maximum velocity instead of
100 deg/s (no degrees available), and the 6 SD multiplier is the paper's.
"""
import warnings

import numpy as np

TASKS = ("antisaccade1", "visual_attention1", "visual_search1")  # fixed in advance (PLAN B2)
MISSING = -100.0
MIN_FIX_MS = 40.0  # a "fixation" shorter than this is not counted; standard minimum, not tuned
FEATURES = ("n_fix", "mean_fix_ms", "n_sac", "mean_sac_amp", "peak_sac_vel",
            "first_sac_ms", "disconj_x", "disconj_y", "spread_x", "spread_y")
# Reported per recording but NOT given to the classifier: tracker dropout differs by group
# (AD 0.34 vs NC 0.15 on antisaccade1) and is a data-quality signal, not a gaze feature.
DIAGNOSTIC = ("missing_frac",)


def model_columns():
    return [f"{t}.{k}" for t in TASKS for k in FEATURES]


def load_gaze(path):
    """Return t (ms) and an (n,4) array [xl, yl, xr, yr] with NaN for missing.

    Rows whose timestamp is not strictly after the previous kept row are dropped
    (FINDINGS 1: 4 files in NC/220808_zjr_m have one such step).
    """
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    keep = [0]
    for i in range(1, len(a)):
        if a[i, 0] > a[keep[-1], 0]:
            keep.append(i)
    a = a[keep]
    xy = a[:, 1:5].copy()
    xy[xy == MISSING] = np.nan
    return a[:, 0], xy


def adaptive_threshold(v):
    """Nystrom & Holmqvist iterative threshold on a velocity vector (NaNs ignored)."""
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan
    pt = v.max()
    for _ in range(100):
        low = v[v < pt]
        if len(low) < 2:
            break
        new = low.mean() + 6 * low.std()
        if abs(new - pt) < 1e-6:
            break
        pt = new
    return pt


def _runs(mask):
    """(start, end_exclusive) of consecutive True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def extract(t, xy):
    """Feature dict for one recording. NaN where undefined (too little valid data)."""
    out = {k: np.nan for k in FEATURES + DIAGNOSTIC}
    out["missing_frac"] = float(np.mean(np.isnan(xy).any(axis=1))) if len(t) else 1.0
    if len(t) < 4:
        return out
    # cyclopean position: mean of the eyes that are valid at each sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x = np.nanmean(xy[:, [0, 2]], axis=1)
        y = np.nanmean(xy[:, [1, 3]], axis=1)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return out
    t, x, y = t[ok], x[ok], y[ok]
    dt = np.diff(t)
    v = np.hypot(np.diff(x), np.diff(y)) / dt  # px/ms between consecutive valid samples
    pt = adaptive_threshold(v)
    sac = v > pt
    sacs = _runs(sac)
    fixs = [(a, b) for a, b in _runs(~sac) if t[b] - t[a] >= MIN_FIX_MS]
    out["n_sac"] = len(sacs)
    out["n_fix"] = len(fixs)
    if fixs:
        out["mean_fix_ms"] = float(np.mean([t[b] - t[a] for a, b in fixs]))
    if sacs:
        amp = [np.hypot(x[b] - x[a], y[b] - y[a]) for a, b in sacs]
        out["mean_sac_amp"] = float(np.mean(amp))
        out["peak_sac_vel"] = float(max(v[a:b].max() for a, b in sacs))
        out["first_sac_ms"] = float(t[sacs[0][0]] - t[0])  # from recording start, not stimulus onset
    both = np.isfinite(xy).all(axis=1)
    if both.any():
        out["disconj_x"] = float(np.mean(np.abs(xy[both, 0] - xy[both, 2])))
        out["disconj_y"] = float(np.mean(np.abs(xy[both, 1] - xy[both, 3])))
    out["spread_x"], out["spread_y"] = float(x.std()), float(y.std())
    return out


def build_table(root):
    """One row per subject: label (AD=1, NC=0) and TASKS x FEATURES columns."""
    from pathlib import Path
    rows, names = [], []
    for cls, label in (("AD", 1), ("NC", 0)):
        for sub in sorted((Path(root) / cls).iterdir()):
            if not sub.is_dir():
                continue
            row = {"subject": sub.name, "label": label}
            for task in TASKS:
                f = extract(*load_gaze(sub / f"{task}.txt"))
                row.update({f"{task}.{k}": f[k] for k in FEATURES + DIAGNOSTIC})
            rows.append(row)
    return rows
