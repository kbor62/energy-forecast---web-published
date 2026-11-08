"""Model definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import features as F

import lightgbm
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

MODELS = ("lgbm", "ridge", "ridge_ohe")

# Default calendar encoding per model.
DEFAULT_ENCODING = {
    "lgbm": "int",
    "ridge": "trig",
    "ridge_ohe": "both",
}

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


def _one_hot_encoder(cat_cols: list[str]) -> OneHotEncoder:
    """One-hot encoder with fixed category levels."""
    categories = [F.CAT_LEVELS[c] for c in cat_cols]
    return OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=False)


def build_estimator(model: str, encoding: str, alpha="auto", seed: int = 42, extra_cols=()):
    """Return an unfitted estimator for ``model``.

    ``extra_cols``: additional numeric feature columns (e.g. weather ``*_target``);
    without them the ridge models would silently drop those columns.
    """
    cat_cols = F.categorical_columns(encoding)
    num_cols = [c for c in F.feature_columns(encoding) if c not in cat_cols] + list(extra_cols)

    if model == "lgbm":
        # No imputer: LightGBM handles missing values itself.
        return LGBMRegressor(random_state=seed, **LGBM_PARAMS)

    # Ridge: impute and scale the numeric columns only.
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    if model == "ridge_ohe":
        pre = ColumnTransformer(
            [("cat", _one_hot_encoder(cat_cols), cat_cols), ("num", numeric, num_cols)],
            remainder="drop",
        )
    else:
        pre = ColumnTransformer([("num", numeric, num_cols)], remainder="drop")

    if str(alpha).lower() == "auto":
        estimator = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=TimeSeriesSplit(n_splits=3))
    else:
        estimator = Ridge(alpha=float(alpha))

    return Pipeline([("pre", pre), ("model", estimator)])


def prepare_X(model: str, X: pd.DataFrame, encoding: str) -> pd.DataFrame:
    """LightGBM reads calendar columns as category dtype; the others as integers."""
    if model == "lgbm":
        return F.as_categorical(X, F.categorical_columns(encoding))
    return X


def fit_estimator(
    model: str,
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    min_val_hours: int = 24 * 14,
    val_fraction: float = 0.1,
) -> dict:
    """Fit the estimator. LightGBM uses the last 10 % of rows for early stopping."""
    y = np.asarray(y, dtype=float)
    info: dict = {"n_train_rows": int(len(X))}

    if model != "lgbm":
        estimator.fit(X, y)
        inner = getattr(estimator, "named_steps", {}).get("model")
        if hasattr(inner, "alpha_"):
            info["alpha"] = float(inner.alpha_)
        return info

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
    import sklearn

    versions = {
        "python": __import__("platform").python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }
    return versions
