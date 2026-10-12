"""Finger-tapping features (PLAN B4; Koppelmans et al. 2023, Tables 2-4).

Input per trial: list of (t_seconds, key) with key in {"lshift", "rshift"}, t measured from
the "Start Pressing!" screen, trial length 10 s. Press times only (key-down), as in the
reference task (vnckppl/FingerTappingTask).

The paper names each metric but gives **no formulas**. Every definition below that goes
beyond the paper's one-line description is marked [OURS] and is a choice, not a replication.
Conditions used: right index, simultaneous (dual), alternating. Left index dropped [OURS].
"""
import numpy as np

DURATION = 10.0     # s, per trial (paper)
MIN_GAP = 1.0       # s, "periods of 1 s or longer ... where no key was tapped" (paper)
PAIR_WINDOW = 0.5   # s, max |dt| for two presses to form a dual pair [OURS: paper silent]

CONDITIONS = ("right", "dual", "alt")
KEYS = {"right": "rshift"}

COMMON = ["irt", "tps", "tps_var"]
DUAL_EXTRA = ["unpaired_rate", "pair_dur", "pair_dur_var", "n_pairs"]
ALT_EXTRA = ["error_rate", "trans_time", "trans_time_var", "n_trans"]

FEATURE_NAMES = (
    [f"right_{m}" for m in COMMON]
    + [f"dual_{m}" for m in COMMON + DUAL_EXTRA]
    + [f"alt_{m}" for m in COMMON + ALT_EXTRA]
    + ["gap_count"]
)   # 3 + 7 + 7 + 1 = 18 = the paper's 20 minus 3 left-hand + gap count


def _times(taps, key=None):
    t = np.array(sorted(t for t, k in taps if key is None or k == key), float)
    return t[(t >= 0) & (t <= DURATION)]


def _common(t):
    """irt, taps/s, variance of taps/s."""
    if len(t) == 0:
        return {"irt": np.nan, "tps": 0.0, "tps_var": np.nan}
    per_second = np.histogram(t, bins=np.arange(0, DURATION + 1e-9, 1.0))[0]
    return {
        "irt": float(t[0]),                    # time to first press (paper)
        "tps": len(t) / DURATION,              # [OURS] rate = taps / 10 s
        "tps_var": float(per_second.var()),    # [OURS] variance of the 10 one-second counts
    }


def gap_count(t):
    """Number of >= 1 s periods with no press in the 10 s window.

    Includes the delay before the first press and the silence after the last [OURS]: the
    paper never defines "onset delay".
    """
    edges = np.concatenate([[0.0], t, [DURATION]])
    return int((np.diff(edges) >= MIN_GAP).sum())


def _dual_pairs(l, r):
    """Mutual-nearest-neighbour pairs within PAIR_WINDOW [OURS]."""
    if len(l) == 0 or len(r) == 0:
        return np.array([])
    near_r = np.abs(r[None, :] - l[:, None]).argmin(axis=1)    # for each l, nearest r
    near_l = np.abs(l[None, :] - r[:, None]).argmin(axis=1)
    d = []
    for i, j in enumerate(near_r):
        dt = abs(r[j] - l[i])
        if near_l[j] == i and dt <= PAIR_WINDOW:
            d.append(dt)
    return np.array(d)


def trial_features(taps, condition):
    """Features of one trial of one condition. Missing quantities are NaN."""
    if condition == "right":
        t = _times(taps, KEYS["right"])
        f = _common(t)
        f["_gaps"] = gap_count(t)
        return f
    t = _times(taps)
    f = _common(t)
    f["_gaps"] = gap_count(t)
    n = len(t)
    if condition == "dual":
        d = _dual_pairs(_times(taps, "lshift"), _times(taps, "rshift"))
        f.update({
            "unpaired_rate": (n - 2 * len(d)) / n if n else np.nan,
            "pair_dur": float(d.mean()) if len(d) else np.nan,
            "pair_dur_var": float(d.var()) if len(d) else np.nan,
            "n_pairs": float(len(d)),
        })
    elif condition == "alt":
        seq = sorted((t_, k) for t_, k in taps if 0 <= t_ <= DURATION)
        keys = [k for _, k in seq]
        times = np.array([t_ for t_, _ in seq])
        same = np.array([keys[i] == keys[i + 1] for i in range(len(keys) - 1)], bool)
        dt = np.diff(times)
        good = dt[~same] if len(dt) else np.array([])
        f.update({
            "error_rate": same.sum() / n if n else np.nan,       # paper: errors / total taps
            "trans_time": float(good.mean()) if len(good) else np.nan,
            "trans_time_var": float(good.var()) if len(good) else np.nan,
            "n_trans": float(len(good)),                          # [OURS] count of correct transitions
        })
    else:
        raise ValueError(condition)
    return f


def session_features(trials):
    """One subject -> dict over FEATURE_NAMES.

    `trials` = {"right": [taps, taps, taps], "dual": [...], "alt": [...]}.
    Continuous metrics: median over the completed trials (paper, Table 4 footnote).
    gap_count: for each condition, whether any trial had a gap (max over trials, paper
    Table 3 footnote), summed over the three conditions [OURS] -> 0..3.
    """
    out = {}
    gaps = 0
    for cond in CONDITIONS:
        rows = [trial_features(t, cond) for t in trials.get(cond, [])]
        gaps += int(max((r["_gaps"] for r in rows), default=0) > 0)
        for name in FEATURE_NAMES:
            if name.startswith(cond + "_"):
                m = name[len(cond) + 1:]
                vals = [r[m] for r in rows if m in r and not np.isnan(r[m])]
                out[name] = float(np.median(vals)) if vals else np.nan
    out["gap_count"] = float(gaps)
    return out
