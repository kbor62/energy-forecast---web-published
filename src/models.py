"""Model definition: one LightGBM regressor per horizon."""

from __future__ import annotations

import numpy as np
import pandas as pd

import features as F

import lightgbm
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

MODEL_NAME = "lgbm"

LGBM_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=10,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,  # required for subsample to take effect
    colsample_bytree=0.8,
    n_jobs=-1,
    verbose=-1,
)


def build_estimator(seed: int = 42) -> LGBMRegressor:
    """Return an unfitted LightGBM regressor. Missing values (e.g. weather) are handled natively."""
    return LGBMRegressor(random_state=seed, **LGBM_PARAMS)


def prepare_X(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM reads hour and dow as categorical (pandas category dtype)."""
    return F.as_categorical(X)


def fit_estimator(
    estimator: LGBMRegressor,
    X: pd.DataFrame,
    y: np.ndarray,
    min_val_hours: int = 24 * 14,
    val_fraction: float = 0.1,
) -> dict:
    """Fit the estimator, using the last 10 % of rows for early stopping."""
    y = np.asarray(y, dtype=float)
    info: dict = {"n_train_rows": int(len(X))}

    n = len(X)
    n_val = max(min_val_hours, int(val_fraction * n))
    if n_val >= n - 100:
        n_val = max(1, n // 10)
    n_fit = n - n_val

    X_fit, y_fit = X.iloc[:n_fit], y[:n_fit]
    X_stop, y_stop = X.iloc[n_fit:], y[n_fit:]
    estimator.fit(
        X_fit, y_fit, eval_X=X_stop, eval_y=y_stop, eval_metric="l1",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    info["n_early_stopping_rows"] = int(n_val)
    info["best_iteration"] = int(getattr(estimator, "best_iteration_", 0) or 0)
    return info


def library_versions() -> dict[str, str]:
    """Library versions, written to output metadata."""
    import holidays
    import sklearn

    return {
        "python": __import__("platform").python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "holidays": holidays.__version__,
    }
