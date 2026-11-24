"""EUR/NOK exchange rate from Norges Bank (daily, business days).

The models work in EUR/kWh. NOK figures in forecast.py use the latest Norges Bank
rate on or before the forecast date; if the rate file is missing or older than
MAX_AGE_DAYS, the fixed config.EUR_TO_NOK is used and a warning is printed.
Backtests keep the fixed rate for their printed metrics (so all documented
comparisons stay on one scale) and store the daily rate per row in column eur_nok.

    python src/fx.py            # fetch / update data/eurnok.parquet (incremental)
    python src/fx.py --full     # re-fetch from --start

Source: Norges Bank open data API, series EXR/B.EUR.NOK.SP (no key required).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from config import EUR_TO_NOK, TZ, repo_path

API_URL = "https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP"
DEFAULT_FILE = "data/eurnok.parquet"
MAX_AGE_DAYS = 7


def parse_sdmx_json(payload: dict) -> pd.Series:
    """Norges Bank sdmx-json -> daily Series (DatetimeIndex, tz-naive dates) named eur_nok."""
    data = payload.get("data", payload)
    obs_dims = data["structure"]["dimensions"]["observation"]
    time_dim = next((d for d in obs_dims if d.get("id") == "TIME_PERIOD"), obs_dims[0])
    dates = [v["id"] for v in time_dim["values"]]
    series = data["dataSets"][0]["series"]
    if not series:
        return pd.Series(dtype=float, name="eur_nok")
    obs = next(iter(series.values()))["observations"]
    idx, vals = [], []
    for k, v in obs.items():
        idx.append(pd.Timestamp(dates[int(k)]))
        vals.append(float(v[0]) if v and v[0] is not None else np.nan)
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name="eur_nok").sort_index()
    return s[~s.index.duplicated(keep="last")].dropna()


def fetch(start: str, end: str | None = None) -> pd.Series:
    import requests

    params = {"format": "sdmx-json", "startPeriod": start, "locale": "en"}
    if end:
        params["endPeriod"] = end
    r = requests.get(API_URL, params=params, timeout=60)
    r.raise_for_status()
    return parse_sdmx_json(r.json())


def load_rates(path=DEFAULT_FILE) -> pd.Series | None:
    p = repo_path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    s = df["eur_nok"].astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.DatetimeIndex(s.index)
    return s.sort_index()


def rate_on(day, rates: pd.Series | None, max_age_days: int = MAX_AGE_DAYS) -> tuple[float, str]:
    """(rate, source) for a date: latest Norges Bank rate on or before it, else the fixed rate."""
    d = pd.Timestamp(day)
    if d.tzinfo is not None:
        d = d.tz_convert(TZ).tz_localize(None)
    d = d.normalize()
    if rates is not None and len(rates):
        known = rates.loc[:d]
        if len(known) and (d - known.index[-1]).days <= max_age_days:
            return float(known.iloc[-1]), f"Norges Bank {known.index[-1].date()}"
    return float(EUR_TO_NOK), "fixed config.EUR_TO_NOK"


def rates_for(timestamps, rates: pd.Series | None) -> np.ndarray:
    """Daily rate (last known on or before each local date) for many timestamps; fixed rate where unknown."""
    ts = pd.DatetimeIndex(timestamps)
    ts = ts.tz_convert(TZ) if ts.tz is not None else ts
    days = ts.tz_localize(None).normalize() if ts.tz is not None else ts.normalize()
    if rates is None or not len(rates):
        return np.full(len(ts), float(EUR_TO_NOK))
    full = rates.reindex(pd.date_range(rates.index.min(), max(rates.index.max(), days.max()), freq="D")).ffill()
    out = full.reindex(days).to_numpy(dtype=float)
    return np.where(np.isfinite(out), out, float(EUR_TO_NOK))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch EUR/NOK from Norges Bank")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--out", default=DEFAULT_FILE)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    out = repo_path(args.out)
    old = None if args.full else load_rates(args.out)
    start = args.start if old is None or not len(old) else (old.index[-1] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    new = fetch(start)
    s = new if old is None else new.combine_first(old)
    s = s.sort_index()
    if not len(s):
        raise RuntimeError("No EUR/NOK observations returned")
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().rename_axis("date").to_parquet(out)
    meta = {"fetched_at": pd.Timestamp.now(tz=TZ).isoformat(), "source": API_URL, "rows": int(len(s)),
            "first": str(s.index.min().date()), "last": str(s.index.max().date()), "last_rate": float(s.iloc[-1])}
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {out}: {len(s):,} days, {s.index.min().date()} -> {s.index.max().date()}, last {s.iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
