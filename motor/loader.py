"""Read PsychoPy output of the reference task (vnckppl/FingerTappingTask).

UNVERIFIED against real files: the repo ships no sample data. From its script the CSV
(`saveAsWideText`) has one row per repetition (3) with, per condition, a key-name field and
a press-time field: Rk.keys/Rk.rt (right), Bk.keys/Bk.rt (both), Ak.keys/Ak.rt (alternating),
Lk.* (left, unused). Times are seconds from the task screen's first flip. If a real file
differs, fix `_cell` and the column names here, not the feature code.
"""
import ast

import numpy as np
import pandas as pd

COLS = {"right": ("Rk.keys", "Rk.rt"), "dual": ("Bk.keys", "Bk.rt"), "alt": ("Ak.keys", "Ak.rt")}


def _cell(x):
    """A cell holding a list, a scalar, a stringified list, or empty -> Python list."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s in ("", "None"):
            return []
        x = ast.literal_eval(s)
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) else [x]


def trials_from_csv(path, keys_rt=COLS):
    """One subject's CSV -> {"right": [taps, ...], "dual": [...], "alt": [...]}."""
    df = pd.read_csv(path)
    out = {}
    for cond, (kc, tc) in keys_rt.items():
        if kc not in df.columns or tc not in df.columns:
            raise KeyError(f"{path}: columns {kc}/{tc} not found; columns are {list(df.columns)[:12]}...")
        out[cond] = []
        for k, t in zip(df[kc], df[tc]):
            ks, ts = _cell(k), _cell(t)
            out[cond].append(list(zip([float(v) for v in ts], ks)))
    return out
