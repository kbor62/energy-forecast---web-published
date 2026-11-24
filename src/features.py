"""Feature construction.

History features (lags, rolling statistics, differences) describe the forecast
origin t and use only prices at or before t. Calendar features describe the
target hour t+h. ``assemble`` pairs the two and is used by training,
backtesting and forecasting alike.

Weather (optional): ``assemble(..., weather=df)`` appends ``<col>_target`` columns
holding the weather at the target hour t+h. They are not part of
``feature_columns``; the caller (train / forecast / backtest) supplies them. In
training and the default backtest the values are observed weather, i.e. an
upper bound on what a real forecast delivers (see README section 6.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import TZ

LAGS_H = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLS_H = [6, 12, 24, 72, 168]
DIFFS_H = [24, 168]

# Calendar features (evaluated at the target hour t+h):
#   hour, dow   categorical (LightGBM groups any set of levels in one split)
#   month       integer 1-12
#   is_holiday  Norwegian public holiday (holidays package, Sundays excluded)
# Day-of-year harmonics were tested and removed: they let the trees single out
# individual periods and made the backtest about 5 % worse.
# Fixed category levels, independent of the training data.
CAT_LEVELS: dict[str, list[int]] = {
    "hour": list(range(24)),
    "dow": list(range(7)),
}
CAT_COLS: list[str] = list(CAT_LEVELS)

CALENDAR_COLS: list[str] = [*CAT_COLS, "month", "is_holiday"]

HOLIDAY_COUNTRY = "NO"

# lag0h is the price at the forecast origin (the persistence baseline).
HISTORY_COLS: list[str] = (
    ["lag0h"]
    + [f"lag{h}h" for h in LAGS_H]
    + [c for w in ROLLS_H for c in (f"ma{w}h", f"std{w}h")]
    + [f"diff{h}h" for h in DIFFS_H]
)

MAX_HISTORY_HOURS = max(LAGS_H + ROLLS_H + DIFFS_H)


def calendar_columns() -> list[str]:
    return list(CALENDAR_COLS)


def categorical_columns() -> list[str]:
    return list(CAT_COLS)


def history_columns() -> list[str]:
    return list(HISTORY_COLS)


def feature_columns() -> list[str]:
    """Ordered list of the calendar and history features."""
    return [*CALENDAR_COLS, *HISTORY_COLS]


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


def holiday_flags(dates) -> np.ndarray:
    """1 where the local date is a Norwegian public holiday (Sundays not counted), else 0."""
    import holidays

    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return np.zeros(0, dtype="int16")
    days = holidays.country_holidays(
        HOLIDAY_COUNTRY, years=range(dates.year.min(), dates.year.max() + 1), include_sundays=False
    )
    local_days = dates.tz_localize(None).to_numpy().astype("datetime64[D]")
    return np.isin(local_days, np.array(sorted(days), dtype="datetime64[D]")).astype("int16")


def calendar_features(index) -> pd.DataFrame:
    """Calendar features for the given timestamps (local time)."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)

    out = pd.DataFrame(index=idx)
    out["hour"] = idx.hour.to_numpy().astype("int16")
    out["dow"] = idx.dayofweek.to_numpy().astype("int16")
    out["month"] = idx.month.to_numpy().astype("int16")
    out["is_holiday"] = holiday_flags(idx)

    return out.loc[:, CALENDAR_COLS]


def assemble(
    history: pd.DataFrame,
    origins,
    horizon: int,
    weather: pd.DataFrame | None = None,
    exog: pd.DataFrame | None = None,
    nwp: pd.DataFrame | None = None,
    decision_lag_h: int | None = None,
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

    If ``nwp`` (archived weather-model forecasts, see nwp.py) is given, ``<base>_target``
    columns follow: the forecast for t+h from the newest run available at the decision time.

    ``decision_lag_h``: hours between the decision time and the origin (default
    exog.DECISION_LAG_H = 11: origin 23:00, decision 12:00 the same day, forecasting
    tomorrow. 33: origin 23:00 on D+1 (prices known), decision 14:00 on D, forecasting
    D+2 and D+3). Used by the exog availability rule and the NWP lead choice.
    """
    origins = pd.DatetimeIndex(origins)
    targets = origins + pd.Timedelta(hours=horizon)
    calendar = calendar_features(targets)
    calendar.index = origins

    parts = [calendar, history.loc[origins, HISTORY_COLS]]
    if weather is not None:
        w = weather.reindex(targets)
        w.index = origins
        parts.append(w.add_suffix("_target"))
    cols = feature_columns()
    if weather is not None:
        cols = cols + [f"{c}_target" for c in weather.columns]
    import exog as _exog

    lag = _exog.DECISION_LAG_H if decision_lag_h is None else int(decision_lag_h)
    if exog is not None:
        block = _exog.exog_features(exog, origins, horizon, lag)
        parts.append(block)
        cols = cols + list(block.columns)
    if nwp is not None:
        import nwp as _nwp

        block = _nwp.nwp_features(nwp, origins, horizon, lag)
        parts.append(block)
        cols = cols + list(block.columns)
    X = pd.concat(parts, axis=1)
    return X.loc[:, cols]


def as_categorical(X: pd.DataFrame) -> pd.DataFrame:
    """Convert hour and dow to pandas category dtype with fixed levels (for LightGBM)."""
    X = X.copy()
    for c in CAT_COLS:
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


def feature_definition() -> dict:
    """Description of the feature set, written to output metadata."""
    return {
        "lags_h": list(LAGS_H),
        "rolling_windows_h": list(ROLLS_H),
        "diffs_h": list(DIFFS_H),
        "calendar": {
            "categorical": list(CAT_COLS),
            "integer": ["month"],
            "holiday": f"is_holiday ({HOLIDAY_COUNTRY} public holidays, Sundays excluded)",
        },
        "calendar_evaluated_at": "target hour (t+h)",
        "rolling_computed_on": "price.shift(1)",
        "includes_origin_price": True,
        "n_features": len(feature_columns()),
        "columns": feature_columns(),
    }
