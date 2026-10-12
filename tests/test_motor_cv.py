import numpy as np
import pytest

try:  # sklearn / imblearn native DLLs may be blocked locally; CV runs on Kaggle
    from motor.cv import evaluate
except ImportError as e:
    pytest.skip(f"native import blocked / missing: {e}", allow_module_level=True)


def test_three_class_cv_runs_and_reports_n():
    rng = np.random.default_rng(0)
    y = np.array([0] * 12 + [1] * 8 + [2] * 8)
    X = rng.normal(size=(len(y), 18)) + y[:, None]
    X[2, 3] = np.nan
    bal, auc = evaluate(X, y, n_repeats=2)
    assert bal.n == len(y) and 0 <= bal.value <= 1 and 0 <= auc.value <= 1
