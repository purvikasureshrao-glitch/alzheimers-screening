"""3-class (CN / aMCI / AD) evaluation with a Balanced Random Forest (PLAN B4).

Hyper-parameters are the paper's GridSearchCV result (50 trees, depth 5, gini) and are
fixed, so nothing is tuned inside CV. Protocol is ours [OURS]: repeated stratified 5-fold
over subjects, probabilities averaged over repeats. The paper used one 70/30 split with
5-fold CV for tuning, so our numbers are NOT comparable to its 70% accuracy / 0.76 AUC.
"""
import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline

from common.metrics import auc_ovr_ci, balanced_acc_ci

CLASSES = ("CN", "aMCI", "AD")   # label 0, 1, 2


def make_model(seed=0):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        BalancedRandomForestClassifier(n_estimators=50, max_depth=5, criterion="gini",
                                       random_state=seed))


def oof_proba(X, y, n_splits=5, n_repeats=10, seed=0):
    """Out-of-fold class probabilities, averaged over repeats. Rows = subjects."""
    X, y = np.asarray(X, float), np.asarray(y)
    prob = np.zeros((len(y), len(CLASSES)))
    count = np.zeros(len(y))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for tr, te in cv.split(X, y):
        m = make_model(seed).fit(X[tr], y[tr])
        p = np.zeros((len(te), len(CLASSES)))
        p[:, m.classes_.astype(int)] = m.predict_proba(X[te])
        prob[te] += p
        count[te] += 1
    return prob / count[:, None]


def evaluate(X, y, n_splits=5, n_repeats=10):
    protocol = (f"repeated stratified {n_splits}-fold x{n_repeats}, balanced RF "
                f"50 trees depth 5, subject-level, 3 classes")
    prob = oof_proba(X, y, n_splits, n_repeats)
    pred = prob.argmax(axis=1)
    g = np.arange(len(y))
    return [balanced_acc_ci(y, pred, protocol, groups=g),
            auc_ovr_ci(y, prob, protocol, groups=g)]
