import numpy as np
import pytest

try:  # Smart App Control can block sklearn's native DLLs locally; CV runs on Kaggle anyway
    from eye.cv import evaluate
except ImportError as e:
    pytest.skip(f"sklearn native import blocked: {e}", allow_module_level=True)


def test_loso_runs_and_reports_n():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 10)
    X = rng.normal(size=(20, 5)) + y[:, None]
    X[3, 2] = np.nan
    bal, auc = evaluate(X, y)
    assert bal.n == 20 and 0 <= auc.value <= 1
