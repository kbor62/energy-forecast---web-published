"""ENTSO-E exogenous features: day-ahead wind/solar and load forecasts, neighbour prices.

One hourly parquet (``fetch_entsoe_exog.py``), columns by name prefix:

  wind_<ZONE>_MW, solar_<ZONE>_MW   day-ahead forecast for the TARGET hour -> ``<col>_target``
  load_<ZONE>_MW                    day-ahead total load forecast for the TARGET hour
                                    -> ``<col>_target``
  price_<ZONE>_EUR_MWh              neighbour day-ahead price, known up to the ORIGIN
                                    -> ``<col>_origin`` (at the origin) and
                                    ``<col>_origin_lag24h`` (24 h earlier)

Availability (no look-ahead): day-ahead forecasts for day D+1 are published around
12:00 on D. For an origin t the decision time is taken as t - DECISION_LAG_H (origin
23:00 -> 12:00 the same day), and a target-hour value is only available if its local
date is at most one day after the decision date; later hours are set to NaN, in
training, backtesting and forecasting alike. Neighbour prices are only used at or
before the origin.
"""

from __future__ import annotations

import pandas as pd

from config import TZ, repo_path

ORIGIN_PREFIX = "price_"
DECISION_LAG_H = 11


def split_columns(cols) -> tuple[list[str], list[str]]:
    """(target-hour columns, origin-time columns) by name prefix."""
    cols = list(cols)
    origin = [c for c in cols if c.startswith(ORIGIN_PREFIX)]
    return [c for c in cols if c not in origin], origin


def feature_names(cols) -> list[str]:
    """Feature columns produced for the given exog columns, in assembly order."""
    target, origin = split_columns(cols)
    return [f"{c}_target" for c in target] + [n for c in origin for n in (f"{c}_origin", f"{c}_origin_lag24h")]


def normalise_wind_solar(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    """entsoe-py wind/solar forecast frame -> hourly wind_<zone>_MW / solar_<zone>_MW.

    Wind onshore and offshore are summed; accepts flat or MultiIndex columns.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    wind_cols = [c for c in df.columns if "wind" in str(c).lower()]
    solar_cols = [c for c in df.columns if "solar" in str(c).lower()]
    out = pd.DataFrame(index=df.index)
    if wind_cols:
        out[f"wind_{zone}_MW"] = df[wind_cols].sum(axis=1, min_count=1)
    if solar_cols:
        out[f"solar_{zone}_MW"] = df[solar_cols].sum(axis=1, min_count=1)
    out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index
    return out.tz_convert(TZ).resample("h").mean()


def normalise_load(df, zone: str) -> pd.DataFrame:
    """entsoe-py load forecast (DataFrame 'Forecasted Load' or Series) -> hourly load_<zone>_MW."""
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    s = pd.to_numeric(s, errors="coerce")
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index
    return s.tz_convert(TZ).resample("h").mean().rename(f"load_{zone}_MW").to_frame()


def load_exog(path, cols: list[str] | None = None) -> pd.DataFrame:
    """Exog parquet in the project time zone (duplicates removed, sorted)."""
    df = pd.read_parquet(repo_path(path))
    df.index = df.index.tz_localize(TZ) if df.index.tz is None else df.index.tz_convert(TZ)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if cols is not None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"exog columns not in {path}: {missing}")
        df = df[list(cols)]
    return df


def available_mask(origins: pd.DatetimeIndex, horizon: int, lag_h: int = DECISION_LAG_H) -> pd.Series:
    """True where a day-ahead value for origin + horizon is published by the decision time."""
    origins = pd.DatetimeIndex(origins)
    decision_day = (origins - pd.Timedelta(hours=lag_h)).tz_convert(TZ).normalize().tz_localize(None)
    target_day = (origins + pd.Timedelta(hours=horizon)).tz_convert(TZ).normalize().tz_localize(None)
    return pd.Series((target_day - decision_day) <= pd.Timedelta(days=1), index=origins)


def exog_features(exog: pd.DataFrame, origins, horizon: int, lag_h: int = DECISION_LAG_H) -> pd.DataFrame:
    """Exog feature block for the given origins and horizon (see module docstring)."""
    origins = pd.DatetimeIndex(origins)
    target_cols, origin_cols = split_columns(exog.columns)
    parts = []

    if target_cols:
        at_target = exog[target_cols].reindex(origins + pd.Timedelta(hours=horizon))
        at_target.index = origins
        published = available_mask(origins, horizon, lag_h).to_numpy()
        at_target = at_target.astype(float)
        at_target.loc[~published, :] = float("nan")
        parts.append(at_target.add_suffix("_target"))

    for c in origin_cols:
        now = exog[c].reindex(origins)
        lag = exog[c].reindex(origins - pd.Timedelta(hours=24))
        block = pd.DataFrame({f"{c}_origin": now.to_numpy(), f"{c}_origin_lag24h": lag.to_numpy()}, index=origins)
        parts.append(block)

    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=origins)
