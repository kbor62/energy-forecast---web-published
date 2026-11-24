"""Download archived weather-model forecasts (Open-Meteo Previous Runs API).

For a few points per area it fetches, for lead offsets N = 1..4 days, the value that
was forecast N x 24 h before each valid hour (``<variable>_previous_dayN``), averages
the points of each area and saves ONE hourly parquet (default data/nwp_openmeteo.parquet)
with columns ``<base>_d<N>`` (see nwp.py for how they become features):

  nwp_wind_DE   wind_speed_100m     Schleswig-Holstein, Lower Saxony coast, Mecklenburg
  nwp_wind_DK   wind_speed_100m     West Jutland, Zealand
  nwp_wind_SE3  wind_speed_100m     Vastra Gotaland / Lake Vanern area
  nwp_solar_DE  shortwave_radiation Bavaria, Baden-Wurttemberg, central Germany
  nwp_temp_NO1  temperature_2m      Oslo, Hamar

The forecast archive starts in early 2024. Hours before the first archived forecast
are filled with ERA5 reanalysis (Open-Meteo Historical Weather API, the same points and
variables, the same value for every lead): without that, "has weather data" would mark
the 2024+ period and the model would use it as a time indicator (seen in the first
backtest: predictions shifted towards the 2024-25 price level). Reanalysis is the
weather that happened, so those rows are an optimistic stand-in for a forecast; only
rows from 2024 hold real forecasts. --no_reanalysis disables the fill.
Units: wind km/h, radiation W/m2 (mean of the preceding hour), temperature deg C.
No API key; free for non-commercial use (check the Open-Meteo terms otherwise).

    python src/fetch_openmeteo.py
    python src/fetch_openmeteo.py --model ecmwf_ifs025 --start 2024-01-01 --full

Incremental like the other fetchers: with an existing file only the tail is fetched
(last stored day minus --overlap_days) and merged. The end defaults to today + 4 days,
so the forecasts needed for a live forecast are included.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from config import TZ, repo_path
from incremental import merge_new, plan_start

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REANALYSIS_MODEL = "era5"
DEFAULT_MODEL = "ecmwf_ifs025"
LEADS = (1, 2, 3, 4)

# base -> (Open-Meteo variable, [(lat, lon), ...])
GROUPS: dict[str, tuple[str, list[tuple[float, float]]]] = {
    "nwp_wind_DE": ("wind_speed_100m", [(54.5, 9.0), (53.3, 7.6), (53.8, 12.3)]),
    "nwp_wind_DK": ("wind_speed_100m", [(56.2, 8.4), (55.5, 11.6)]),
    "nwp_wind_SE3": ("wind_speed_100m", [(58.6, 13.0)]),
    "nwp_solar_DE": ("shortwave_radiation", [(48.6, 11.5), (48.9, 9.1), (51.3, 11.0)]),
    "nwp_temp_NO1": ("temperature_2m", [(59.91, 10.75), (60.79, 11.07)]),
}


def _points() -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for _, plist in GROUPS.values():
        for p in plist:
            if p not in pts:
                pts.append(p)
    return pts


def _variables() -> list[str]:
    return sorted({v for v, _ in GROUPS.values()})


def request_params(start: pd.Timestamp, end: pd.Timestamp, model: str) -> dict:
    pts = _points()
    return {
        "latitude": ",".join(f"{lat:g}" for lat, _ in pts),
        "longitude": ",".join(f"{lon:g}" for _, lon in pts),
        "hourly": ",".join(f"{v}_previous_day{n}" for v in _variables() for n in LEADS),
        "models": model,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "timezone": "GMT",
        "timeformat": "unixtime",
    }


def _per_point(payload, model: str) -> dict[tuple[float, float], pd.DataFrame]:
    """Open-Meteo JSON (one object, or a list with one object per point, in request order)
    -> {point: hourly frame of the returned variables (UTC index)}."""
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Open-Meteo error: {payload.get('reason', payload)}")
    items = payload if isinstance(payload, list) else [payload]
    pts = _points()
    if len(items) != len(pts):
        raise RuntimeError(f"expected {len(pts)} locations in the response, got {len(items)}")

    per_point: dict[tuple[float, float], pd.DataFrame] = {}
    for p, item in zip(pts, items):
        hourly = item["hourly"]
        idx = pd.to_datetime(hourly["time"], unit="s", utc=True)
        cols = {}
        for k, v in hourly.items():
            if k == "time":
                continue
            name = k[: -len(f"_{model}")] if k.endswith(f"_{model}") else k
            cols[name] = pd.to_numeric(pd.Series(v, index=idx), errors="coerce")
        per_point[p] = pd.DataFrame(cols, index=idx)
    return per_point


def parse_response(payload, model: str = DEFAULT_MODEL) -> pd.DataFrame:
    """Previous Runs response -> hourly frame with columns <base>_d<N>, index Europe/Oslo."""
    per_point = _per_point(payload, model)
    out = {}
    for base, (var, plist) in GROUPS.items():
        for n in LEADS:
            key = f"{var}_previous_day{n}"
            series = [per_point[p][key] for p in plist if key in per_point[p]]
            if series:
                out[f"{base}_d{n}"] = pd.concat(series, axis=1).mean(axis=1, skipna=False)
    df = pd.DataFrame(out)
    df.index = df.index.tz_convert(TZ)
    df.index.name = "timestamp"
    return df.dropna(how="all")


def parse_reanalysis(payload, model: str = REANALYSIS_MODEL) -> pd.DataFrame:
    """Historical Weather (ERA5) response -> hourly frame with columns <base> (group means)."""
    per_point = _per_point(payload, model)
    out = {}
    for base, (var, plist) in GROUPS.items():
        series = [per_point[p][var] for p in plist if var in per_point[p]]
        if series:
            out[base] = pd.concat(series, axis=1).mean(axis=1, skipna=False)
    df = pd.DataFrame(out)
    df.index = df.index.tz_convert(TZ)
    df.index.name = "timestamp"
    return df.dropna(how="all")


def fill_before_archive(df: pd.DataFrame, reanalysis: pd.DataFrame) -> pd.DataFrame:
    """Fill every <base>_d<N> column with the reanalysis <base> for hours BEFORE the
    column's first archived forecast (later gaps are left as they are)."""
    idx = df.index.union(reanalysis.index)
    out = df.reindex(idx)
    for col in out.columns:
        base = col.rsplit("_d", 1)[0]
        if base not in reanalysis.columns:
            continue
        first = df[col].first_valid_index()
        before = idx < first if first is not None else np.ones(len(idx), dtype=bool)
        fill = reanalysis[base].reindex(idx)
        out.loc[before, col] = out.loc[before, col].fillna(fill[before])
    out.index.name = "timestamp"
    return out.dropna(how="all")


def _get(url: str, params: dict, parse, model: str, what: str, retries: int = 4) -> pd.DataFrame:
    import requests

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=180)
            payload = r.json()
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {payload.get('reason', payload) if isinstance(payload, dict) else payload}")
            return parse(payload, model)
        except Exception as exc:  # network errors, rate limits, API errors
            if attempt == retries - 1:
                raise RuntimeError(f"Open-Meteo {what} failed: {exc}") from None
            wait = 5 * 2**attempt
            print(f"    attempt {attempt + 1} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_chunk(start: pd.Timestamp, end: pd.Timestamp, model: str, retries: int = 4) -> pd.DataFrame:
    return _get(API_URL, request_params(start, end, model), parse_response, model,
                f"forecasts {start.date()}..{end.date()}", retries)


def fetch_reanalysis(start: pd.Timestamp, end: pd.Timestamp, model: str = REANALYSIS_MODEL,
                     retries: int = 4) -> pd.DataFrame:
    pts = _points()
    params = {
        "latitude": ",".join(f"{lat:g}" for lat, _ in pts),
        "longitude": ",".join(f"{lon:g}" for _, lon in pts),
        "hourly": ",".join(_variables()),
        "models": model,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "timezone": "GMT",
        "timeformat": "unixtime",
    }
    return _get(ARCHIVE_URL, params, parse_reanalysis, model, f"reanalysis {start.date()}..{end.date()}", retries)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch archived weather-model forecasts from Open-Meteo")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Open-Meteo model id (default ecmwf_ifs025)")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=None, help="Default: today + 4 days")
    ap.add_argument("--out", default="data/nwp_openmeteo.parquet")
    ap.add_argument("--chunk_days", type=int, default=90)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--overlap_days", type=int, default=3)
    ap.add_argument("--full", action="store_true", help="Ignore an existing file and fetch everything")
    ap.add_argument("--reanalysis_start", default="2020-01-01",
                    help="Fill hours from this date up to the first archived forecast with ERA5 reanalysis")
    ap.add_argument("--no_reanalysis", action="store_true", help="Do not fill the years before the forecast archive")
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz=TZ)
    end = (pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=4)) if args.end is None \
        else pd.Timestamp(args.end, tz=TZ)
    out = repo_path(args.out)

    existing = None
    if out.exists() and not args.full:
        existing = pd.read_parquet(out)
        existing.index = existing.index.tz_convert(TZ)
        valid = existing.dropna(how="all")
        first = valid.index.min() if len(valid) else None
        last = valid.index.max() if len(valid) else None
        # the archive itself starts later than --start: do not treat that as a backfill
        fetch_from = plan_start(first, last, start, args.overlap_days, backfill_tolerance_days=10**5)
        print(f"Updating {out} ({len(existing):,} rows, last {last}); fetching from {fetch_from.date()}")
    else:
        fetch_from = start
        print(f"Full fetch {start.date()} -> {end.date()}")

    edges = list(pd.date_range(fetch_from, end, freq=f"{args.chunk_days}D")) + [end]
    frames = []
    for a, b in zip(edges[:-1], edges[1:]):
        b_incl = b - pd.Timedelta(days=1) if b != end else b
        print(f"  {a.date()} .. {b_incl.date()}")
        frames.append(fetch_chunk(a, b_incl, args.model))
        time.sleep(args.sleep)
    new = pd.concat(frames).sort_index()
    new = new[~new.index.duplicated(keep="last")]

    df = merge_new(existing, new)
    df.index.name = "timestamp"

    reanalysis_info = None
    if not args.no_reanalysis:
        r_start = pd.Timestamp(args.reanalysis_start, tz=TZ)
        firsts = [df[c].first_valid_index() for c in df.columns if df[c].first_valid_index() is not None]
        archive_start = max(firsts) if firsts else end
        if min(firsts, default=end) > r_start + pd.Timedelta(days=7):
            print(f"Filling {r_start.date()} .. {archive_start.date()} with {REANALYSIS_MODEL} reanalysis")
            r_edges = list(pd.date_range(r_start, archive_start, freq="365D")) + [archive_start]
            parts = []
            for a, b in zip(r_edges[:-1], r_edges[1:]):
                print(f"  {a.date()} .. {b.date()}")
                parts.append(fetch_reanalysis(a, b))
                time.sleep(args.sleep)
            rean = pd.concat(parts).sort_index()
            rean = rean[~rean.index.duplicated(keep="last")]
            df = fill_before_archive(df, rean)
        else:
            print("Reanalysis fill already present")
        reanalysis_info = {"model": REANALYSIS_MODEL, "from": args.reanalysis_start,
                           "rule": "fills each <base>_d<N> before its first archived forecast"}
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    meta = {
        "fetched_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "model": args.model,
        "api": API_URL,
        "groups": {b: {"variable": v, "points": p} for b, (v, p) in GROUPS.items()},
        "leads_days": list(LEADS),
        "reanalysis_fill": reanalysis_info,
        "units": {"wind_speed_100m": "km/h", "shortwave_radiation": "W/m2", "temperature_2m": "deg C"},
        "rows": int(len(df)),
        "columns": {c: {"non_null": int(df[c].notna().sum()), "first": str(df[c].first_valid_index()),
                        "last": str(df[c].last_valid_index())} for c in df.columns},
    }
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved {out}: {len(df):,} rows, {len(df.columns)} columns, {df.index.min()} -> {df.index.max()}")
    nn = {c: int(df[c].notna().sum()) for c in df.columns}
    if not any(nn.values()):
        raise RuntimeError("No values were returned; check --model and the API response")


if __name__ == "__main__":
    main()
