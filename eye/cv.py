"""Leave-one-subject-out evaluation, logistic regression, cut-off 0.5 (PLAN B2).

Pilot: 20 subjects. Imputer and scaler are fitted on the 19 training subjects only.
Defaults are sklearn's (L2, C=1): no tuning, so nothing to leak.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.metrics import auc_ci, balanced_acc_ci


def make_model():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), LogisticRegression(max_iter=1000))


def loso(X, y):
    X, y = np.asarray(X, float), np.asarray(y)
    prob = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        prob[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return prob


def evaluate(X, y, protocol="LOSO, logistic regression, cut-off 0.5"):
    prob = loso(X, y)
    pred = (prob >= 0.5).astype(int)
    g = np.arange(len(y))
    return [balanced_acc_ci(y, pred, protocol, groups=g), auc_ci(y, prob, protocol, groups=g)]
