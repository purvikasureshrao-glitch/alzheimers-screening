"""Metrics with 95% CIs (PLAN section F). Balanced accuracy is the headline.

Bootstrap resamples *subjects* (groups), not rows, so repeated rows from one person
(e.g. NHATS rounds) cannot make the interval too narrow.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


@dataclass
class Result:
    name: str
    value: float
    lo: float
    hi: float
    n: int
    protocol: str

    def __str__(self):
        return f"{self.name} {self.value:.3f} (95% CI {self.lo:.3f}-{self.hi:.3f}), n={self.n}, {self.protocol}"


def _bootstrap(stat, y_true, y_other, groups, n_boot, seed):
    y_true, y_other = np.asarray(y_true), np.asarray(y_other)
    groups = np.arange(len(y_true)) if groups is None else np.asarray(groups)
    ids = np.unique(groups)
    rows = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([rows[g] for g in rng.choice(ids, size=len(ids))])
        if len(np.unique(y_true[pick])) < 2:  # resample lost a class: undefined, skip
            continue
        try:
            vals.append(stat(y_true[pick], y_other[pick]))
        except ValueError:  # e.g. a multi-class resample missing one class
            continue
    if not vals:
        raise ValueError("every bootstrap resample had a single class")
    return np.percentile(vals, [2.5, 97.5])


def balanced_acc_ci(y_true, y_pred, protocol, groups=None, n_boot=2000, seed=0):
    lo, hi = _bootstrap(balanced_accuracy_score, y_true, y_pred, groups, n_boot, seed)
    return Result("balanced accuracy", balanced_accuracy_score(y_true, y_pred), lo, hi,
                  len(np.unique(groups)) if groups is not None else len(y_true), protocol)


def f1_ci(y_true, y_pred, protocol, groups=None, n_boot=2000, seed=0):
    f = lambda t, p: f1_score(t, p, zero_division=0)
    lo, hi = _bootstrap(f, y_true, y_pred, groups, n_boot, seed)
    return Result("F1", f(np.asarray(y_true), np.asarray(y_pred)), lo, hi,
                  len(np.unique(groups)) if groups is not None else len(y_true), protocol)


def auc_ci(y_true, y_score, protocol, groups=None, n_boot=2000, seed=0):
    lo, hi = _bootstrap(roc_auc_score, y_true, y_score, groups, n_boot, seed)
    return Result("AUC", roc_auc_score(y_true, y_score), lo, hi,
                  len(np.unique(groups)) if groups is not None else len(y_true), protocol)


def auc_ovr_ci(y_true, y_prob, protocol, groups=None, n_boot=2000, seed=0):
    """Macro one-vs-rest AUC for >2 classes; `y_prob` is (n, n_classes)."""
    f = lambda t, p: roc_auc_score(t, p, multi_class="ovr", average="macro",
                                   labels=list(range(p.shape[1])))
    y_prob = np.asarray(y_prob)
    lo, hi = _bootstrap(f, y_true, y_prob, groups, n_boot, seed)
    return Result("AUC (macro OvR)", f(np.asarray(y_true), y_prob), lo, hi,
                  len(np.unique(groups)) if groups is not None else len(y_true), protocol)


def wilson_ci(k, n, z=1.96):
    """Wilson interval for a single held-out proportion k/n."""
    if n == 0:
        raise ValueError("n must be > 0")
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return centre - half, centre + half


def pick_threshold(y_true, y_score):
    """Score threshold maximising balanced accuracy. Call on VALIDATION data only,
    then apply unchanged to test (PLAN H7)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    best_t, best = 0.5, -1.0
    for t in np.unique(y_score):
        b = balanced_accuracy_score(y_true, y_score >= t)
        if b > best:
            best_t, best = float(t), b
    return best_t
