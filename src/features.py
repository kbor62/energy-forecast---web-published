"""Feature construction.

History features (lags, rolling statistics, differences) describe the forecast
origin t and use only prices at or before t. Calendar features describe the
target hour t+h. ``assemble`` pairs the two and is used by training,
backtesting and forecasting alike.

Weather (optional): ``assemble(..., weather=df)`` appends ``<col>_target`` columns
holding the weather at the target hour t+h. They are not part of
``feature_columns``; the caller (train / forecast / backtest) supplies them. In
training and the default backtest the values are observed weather, i.e. an
upper bound on what a real forecast delivers (see README section 13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TZ

LAGS_H = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLS_H = [6, 12, 24, 72, 168]
DIFFS_H = [24, 168]
HOUR_HARMONICS = 2  # two harmonics capture the morning and evening peaks
YEAR_HARMONICS = 3

ENCODINGS = ("trig", "int", "both")

# Fixed category levels, independent of the training data.
CAT_LEVELS: dict[str, list[int]] = {
    "hour": list(range(24)),
    "dow": list(range(7)),
    "month": list(range(1, 13)),
    "season": [0, 1, 2, 3],  # 0 winter, 1 spring, 2 summer, 3 fall
    "is_weekday": [0, 1],
}
CAT_COLS: list[str] = list(CAT_LEVELS)

TRIG_COLS: list[str] = (
    [f"sin_hour_{k}" for k in range(1, HOUR_HARMONICS + 1)]
    + [f"cos_hour_{k}" for k in range(1, HOUR_HARMONICS + 1)]
    + ["sin_dow", "cos_dow"]
    + [f"year_sin_{k}" for k in range(1, YEAR_HARMONICS + 1)]
    + [f"year_cos_{k}" for k in range(1, YEAR_HARMONICS + 1)]
)

# lag0h is the price at the forecast origin (the persistence baseline).
HISTORY_COLS: list[str] = (
    ["lag0h"]
    + [f"lag{h}h" for h in LAGS_H]
    + [c for w in ROLLS_H for c in (f"ma{w}h", f"std{w}h")]
    + [f"diff{h}h" for h in DIFFS_H]
)

MAX_HISTORY_HOURS = max(LAGS_H + ROLLS_H + DIFFS_H)


def calendar_columns(encoding: str) -> list[str]:
    if encoding == "trig":
        return [*TRIG_COLS, "is_weekday"]
    if encoding == "int":
        return list(CAT_COLS)
    return [*CAT_COLS, *TRIG_COLS]


def categorical_columns(encoding: str) -> list[str]:
    return [] if encoding == "trig" else list(CAT_COLS)


def history_columns() -> list[str]:
    return list(HISTORY_COLS)


def feature_columns(encoding: str) -> list[str]:
    """Ordered feature list for an encoding."""
    return [*calendar_columns(encoding), *HISTORY_COLS]


def history_features(price: pd.Series) -> pd.DataFrame:
    """History features at the origin; row t uses prices at or before t."""
    feat = pd.DataFrame(index=price.index)
    feat["lag0h"] = price
    for h in LAGS_H:
        feat[f"lag{h}h"] = price.shift(h)

    # Rolling windows exclude the origin hour itself (lag0h already has it).
    history = price.shift(1)
    for w in ROLLS_H:
        min_periods = max(1, w // 2)
        feat[f"ma{w}h"] = history.rolling(w, min_periods=min_periods).mean()
        feat[f"std{w}h"] = history.rolling(w, min_periods=min_periods).std()

    for h in DIFFS_H:
        feat[f"diff{h}h"] = price - price.shift(h)

    return feat.loc[:, HISTORY_COLS]


def calendar_features(index, encoding: str = "trig") -> pd.DataFrame:
    """Calendar features for the given timestamps."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)

    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()

    out = pd.DataFrame(index=idx)

    if encoding in ("int", "both"):
        season = np.zeros(len(idx), dtype="int16")  # winter
        season[np.isin(month, [3, 4, 5])] = 1
        season[np.isin(month, [6, 7, 8])] = 2
        season[np.isin(month, [9, 10, 11])] = 3
        out["hour"] = hour.astype("int16")
        out["dow"] = dow.astype("int16")
        out["month"] = month.astype("int16")
        out["season"] = season

    if encoding in ("trig", "both"):
        for k in range(1, HOUR_HARMONICS + 1):
            out[f"sin_hour_{k}"] = np.sin(2 * np.pi * k * hour / 24)
            out[f"cos_hour_{k}"] = np.cos(2 * np.pi * k * hour / 24)
        out["sin_dow"] = np.sin(2 * np.pi * dow / 7)
        out["cos_dow"] = np.cos(2 * np.pi * dow / 7)
        doy = idx.dayofyear.to_numpy().astype(float)
        for k in range(1, YEAR_HARMONICS + 1):
            ang = 2 * np.pi * k * doy / 365.2425
            out[f"year_sin_{k}"] = np.sin(ang)
            out[f"year_cos_{k}"] = np.cos(ang)

    out["is_weekday"] = (dow < 5).astype("int16")

    return out.loc[:, calendar_columns(encoding)]


def assemble(
    history: pd.DataFrame,
    origins,
    horizon: int,
    encoding: str = "trig",
    weather: pd.DataFrame | None = None,
    exog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Design matrix: history at each origin, calendar at origin + horizon.

    If ``weather`` (hourly, indexed by time) is given, its columns are added as
    ``<col>_target``: the weather at the target hour origin + horizon. In
    training this is observed weather (an upper bound); at inference it must be
    replaced by archived/real forecast values.

    If ``exog`` (ENTSO-E wind/solar forecasts and neighbour prices, see exog.py) is
    given, its features are appended after the weather columns: ``<col>_target`` for
    target-hour forecasts (NaN where not yet published at the decision time) and
    ``<col>_origin`` / ``<col>_origin_lag24h`` for neighbour prices.
    """
    origins = pd.DatetimeIndex(origins)
    targets = origins + pd.Timedelta(hours=horizon)
    calendar = calendar_features(targets, encoding)
    calendar.index = origins

    parts = [calendar, history.loc[origins, HISTORY_COLS]]
    if weather is not None:
        w = weather.reindex(targets)
        w.index = origins
        parts.append(w.add_suffix("_target"))
    cols = feature_columns(encoding)
    if weather is not None:
        cols = cols + [f"{c}_target" for c in weather.columns]
    if exog is not None:
        import exog as _exog

        block = _exog.exog_features(exog, origins, horizon)
        parts.append(block)
        cols = cols + list(block.columns)
    X = pd.concat(parts, axis=1)
    return X.loc[:, cols]


def as_categorical(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Convert calendar columns to pandas category dtype (for LightGBM)."""
    if not cat_cols:
        return X
    X = X.copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    return X


def make_targets(price: pd.Series, horizons) -> pd.DataFrame:
    """One target column per horizon: the price at t + h."""
    return pd.DataFrame(
        {f"target_price_t+{h}": price.shift(-h) for h in horizons},
        index=price.index,
    )


def target_column(horizon: int) -> str:
    return f"target_price_t+{horizon}"


def train_val_masks(
    index,
    val_start: pd.Timestamp,
    horizon: int,
    embargo: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/validation masks. With embargo, training ends h hours before
    val_start so that no training label falls inside the validation window."""
    index = pd.DatetimeIndex(index)
    train_end = val_start - pd.Timedelta(hours=horizon) if embargo else val_start
    return np.asarray(index < train_end), np.asarray(index >= val_start)


def feature_definition(encoding: str) -> dict:
    """Description of the feature set, written to output metadata."""
    return {
        "encoding": encoding,
        "lags_h": list(LAGS_H),
        "rolling_windows_h": list(ROLLS_H),
        "diffs_h": list(DIFFS_H),
        "hour_harmonics": HOUR_HARMONICS,
        "year_harmonics": YEAR_HARMONICS,
        "calendar_evaluated_at": "target hour (t+h)",
        "rolling_computed_on": "price.shift(1)",
        "includes_origin_price": True,
        "n_features": len(feature_columns(encoding)),
        "columns": feature_columns(encoding),
    }
