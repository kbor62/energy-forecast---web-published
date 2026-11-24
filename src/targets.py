"""Target transformation: forecast the price level, or the change from a known anchor.

--target_mode (train.py, backtest_rolling.py; stored in the model, used by forecast.py):

  level        y = price(t)                                      (default, unchanged)
  diff_sameh   y = price(t) - price(t - 24*ceil(h/24))           change from the same clock
                                                                 hour on the last known day
  diff_mean24  y = price(t) - mean(price over the 24 h up to and incl. the origin)
  rel_mean24   y = (price(t) - mean24) / scale,  scale = max(mean |price| over the
                                                 168 h up to the origin, SCALE_FLOOR)

t = origin + h. All anchors use prices at or before the origin only. Predictions are
transformed back to EUR/kWh (yhat_price = anchor + scale * yhat) before any metric, so
results stay comparable with the level model. For the non-level modes the anchor (and
for rel_mean24 the scale) is also given to the model as features target_anchor /
target_scale, because a tree cannot otherwise see the value it is predicting from.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MODES = ("level", "diff_sameh", "diff_mean24", "rel_mean24")
SCALE_FLOOR = 0.01  # EUR/kWh: guards against division by prices near zero


def anchor(price: pd.Series, origins, horizon: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(offset, scale) per origin; level -> (0, 1)."""
    origins = pd.DatetimeIndex(origins)
    n = len(origins)
    if mode == "level":
        return np.zeros(n), np.ones(n)
    if mode == "diff_sameh":
        back = pd.Timedelta(hours=24 * math.ceil(horizon / 24))
        offset = price.reindex(origins + pd.Timedelta(hours=horizon) - back).to_numpy(dtype=float)
        return offset, np.ones(n)
    mean24 = price.rolling(24, min_periods=24).mean().reindex(origins).to_numpy(dtype=float)
    if mode == "diff_mean24":
        return mean24, np.ones(n)
    if mode == "rel_mean24":
        scale = price.abs().rolling(168, min_periods=168).mean().reindex(origins).to_numpy(dtype=float)
        return mean24, np.maximum(scale, SCALE_FLOOR)
    raise ValueError(f"unknown target mode {mode!r}; choose from {MODES}")


def to_target(y, offset, scale) -> np.ndarray:
    return (np.asarray(y, dtype=float) - offset) / scale


def to_price(yhat, offset, scale) -> np.ndarray:
    return offset + scale * np.asarray(yhat, dtype=float)


def add_features(X: pd.DataFrame, offset, scale, mode: str) -> pd.DataFrame:
    """Append target_anchor (and target_scale) for the non-level modes."""
    if mode == "level":
        return X
    X = X.copy()
    X["target_anchor"] = offset
    if mode == "rel_mean24":
        X["target_scale"] = scale
    return X


def add_args(ap) -> None:
    ap.add_argument("--target_mode", choices=MODES, default="level",
                    help="level (default): forecast the price. diff_sameh / diff_mean24 / rel_mean24: forecast "
                         "the change from a known anchor (see targets.py); predictions are converted back to prices.")
