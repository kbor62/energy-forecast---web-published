"""Weather-model forecasts (NWP) as features, with the lead time that was available.

Source: Open-Meteo Previous Runs API (``fetch_openmeteo.py``). For every valid hour t
it stores ``<base>_d<N>`` = the value forecast N x 24 h before t (N = 1..MAX_LEAD_DAYS),
e.g. ``nwp_wind_DE_d2``. Bases (group means over a few points, see fetch_openmeteo.py):

  nwp_wind_DE, nwp_wind_DK, nwp_wind_SE3   wind speed 100 m (km/h), wind-power areas
  nwp_solar_DE                             shortwave radiation (W/m2), solar areas
  nwp_temp_NO1                             2 m temperature (deg C), NO1 demand

Feature for origin t0 and horizon h (target t = t0 + h): ``<base>_target`` = the
value from the newest run that was available at the decision time:

  cutoff = t0 - decision_lag_h - delay_h      (delay_h: run time + publication margin)
  N      = ceil((t - cutoff) / 24 h), at least 1
  value  = <base>_d<N> at t, NaN if N > MAX_LEAD_DAYS or the column is missing.

A run issued N x 24 h before t was therefore issued at or before the cutoff, so no
forecast newer than the decision time is used. Before the forecast archive starts
(early 2024) fetch_openmeteo.py fills the columns with ERA5 reanalysis, so the
features exist in every year and do not act as a "2024 or later" indicator.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from config import TZ, repo_path

MAX_LEAD_DAYS = 4
DEFAULT_DELAY_H = 6
_LEAD = re.compile(r"^(?P<base>.+)_d(?P<n>\d+)$")


def bases(columns) -> list[str]:
    """Feature bases in the file (columns <base>_d<N>), in first-seen order."""
    out: list[str] = []
    for c in columns:
        m = _LEAD.match(str(c))
        if m and m["base"] not in out:
            out.append(m["base"])
    return out


def feature_names(base_cols) -> list[str]:
    return [f"{b}_target" for b in base_cols]


def load_nwp(path, base_cols: list[str] | None = None) -> pd.DataFrame:
    """NWP parquet in the project time zone; keep the lead columns of the given bases."""
    df = pd.read_parquet(repo_path(path))
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
    df.index = df.index.tz_convert(TZ)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if base_cols is not None:
        missing = [b for b in base_cols if b not in bases(df.columns)]
        if missing:
            raise KeyError(f"NWP bases not in {path}: {missing}")
        df = df[[c for c in df.columns if (m := _LEAD.match(c)) and m["base"] in base_cols]]
    return df


def lead_days(origins, horizon: int, decision_lag_h: int, delay_h: int = DEFAULT_DELAY_H) -> np.ndarray:
    """Lead (days) of the newest run available at the decision time, per origin."""
    origins = pd.DatetimeIndex(origins)
    gap_h = horizon + decision_lag_h + delay_h  # target - cutoff, in hours
    return np.full(len(origins), max(1, math.ceil(gap_h / 24)), dtype=int)


def nwp_features(nwp: pd.DataFrame, origins, horizon: int, decision_lag_h: int,
                 delay_h: int = DEFAULT_DELAY_H) -> pd.DataFrame:
    """NWP feature block (<base>_target) for the given origins and horizon."""
    origins = pd.DatetimeIndex(origins)
    targets = origins + pd.Timedelta(hours=horizon)
    n = lead_days(origins, horizon, decision_lag_h, delay_h)
    out = pd.DataFrame(index=origins)
    for b in bases(nwp.columns):
        values = np.full(len(origins), np.nan)
        for lead in np.unique(n):
            col = f"{b}_d{lead}"
            if lead > MAX_LEAD_DAYS or col not in nwp.columns:
                continue
            sel = n == lead
            values[sel] = nwp[col].reindex(targets[sel]).to_numpy(dtype=float)
        out[f"{b}_target"] = values
    return out


def add_args(ap) -> None:
    """--nwp_file / --nwp_cols / --decision_lag_hours for train.py and backtest_rolling.py."""
    import exog as _exog

    ap.add_argument("--nwp_file", default=None,
                    help="Archived weather-model forecasts from fetch_openmeteo.py (e.g. data/nwp_openmeteo.parquet); "
                         "adds <base>_target features with the lead available at the decision time. See nwp.py.")
    ap.add_argument("--nwp_cols", nargs="+", default=None, help="NWP bases to use, e.g. nwp_wind_DE (default: all)")
    ap.add_argument("--decision_lag_hours", type=int, default=_exog.DECISION_LAG_H,
                    help="Hours from the decision time to the origin. 11 (default): origin 23:00, decision 12:00 "
                         "the same day, forecasting tomorrow before the auction. 33: origin 23:00 on D+1 "
                         "(its prices known), decision 14:00 on D, forecasting D+2/D+3.")
