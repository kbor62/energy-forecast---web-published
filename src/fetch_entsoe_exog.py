"""Download ENTSO-E exogenous data: day-ahead wind/solar forecasts and neighbour prices.

For each wind zone: day-ahead generation forecast for wind (onshore + offshore) and
solar -> wind_<ZONE>_MW, solar_<ZONE>_MW. For each price zone: day-ahead price ->
price_<ZONE>_EUR_MWh. Everything is averaged to hourly values in Europe/Oslo and
saved as ONE parquet (default data/entsoe_exog.parquet) that train.py,
backtest_rolling.py and forecast.py read with --exog_file. How the columns are used
(target-hour forecasts vs. origin-time prices, availability rule): see exog.py.

Requires ENTSOE_TOKEN in .env (same as fetch_prices.py). Zone codes are entsoe-py
areas, e.g. DE_LU, DK_1, DK_2, SE_3, SE_4, NO_2, NL, GB.

    python src/fetch_entsoe_exog.py
    python src/fetch_entsoe_exog.py --wind_zones DE_LU DK_1 --price_zones NO_2 SE_3 --start 2023-01-01

A zone that fails (no data, API error after retries) is skipped with a warning; the
run fails only if nothing at all could be fetched. The fetched end date defaults to
now + 2 days, so tomorrow's forecasts are included once published.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from config import BRUSSELS_TZ, ENTSOE_TOKEN, TZ, repo_path
from exog import normalise_wind_solar
from fetch_prices import build_chunks

DEFAULT_WIND_ZONES = ["DE_LU", "DK_1", "DK_2", "SE_3"]
DEFAULT_PRICE_ZONES = ["NO_2", "SE_3", "DK_1", "DE_LU"]


def _with_retries(fn, what: str, retries: int, sleep: float):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # entsoe-py raises NoMatchingDataError, HTTP errors, ...
            if attempt == retries - 1:
                raise RuntimeError(f"{what} failed after {retries} attempts: {exc}") from exc
            wait = 2**attempt
            print(f"    attempt {attempt + 1} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch_zone(client, kind: str, zone: str, chunks, retries: int = 4, sleep: float = 0.5) -> pd.DataFrame | None:
    """One zone, all chunks -> hourly frame, or None if the zone has no data."""
    frames = []
    for a, b in chunks:
        try:
            if kind == "wind_solar":
                raw = _with_retries(lambda: client.query_wind_and_solar_forecast(zone, start=a, end=b),
                                    f"{kind} {zone} {a.date()}", retries, sleep)
                frames.append(normalise_wind_solar(raw, zone))
            else:
                raw = _with_retries(lambda: client.query_day_ahead_prices(zone, start=a, end=b),
                                    f"price {zone} {a.date()}", retries, sleep)
                s = raw.tz_convert(TZ).resample("h").mean()
                frames.append(s.rename(f"price_{zone}_EUR_MWh").to_frame())
        except RuntimeError as exc:
            print(f"  [skip chunk] {exc}")
        time.sleep(sleep)
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def fetch_all(client, wind_zones, price_zones, chunks, retries: int = 4, sleep: float = 0.5) -> pd.DataFrame:
    parts = []
    for kind, zones in (("wind_solar", wind_zones), ("price", price_zones)):
        for zone in zones:
            print(f"{kind} {zone} ...")
            df = fetch_zone(client, kind, zone, chunks, retries, sleep)
            if df is None:
                print(f"  WARNING: no data for {kind} {zone}; skipped")
                continue
            print(f"  ok: {df.notna().sum().to_dict()}")
            parts.append(df)
    if not parts:
        raise RuntimeError("No ENTSO-E exog data could be fetched")
    return pd.concat(parts, axis=1).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch ENTSO-E wind/solar forecasts and neighbour prices")
    ap.add_argument("--wind_zones", nargs="*", default=DEFAULT_WIND_ZONES, help="Zones for wind/solar day-ahead forecasts")
    ap.add_argument("--price_zones", nargs="*", default=DEFAULT_PRICE_ZONES, help="Zones for day-ahead prices")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None, help="Default: now + 2 days")
    ap.add_argument("--out", default="data/entsoe_exog.parquet")
    ap.add_argument("--chunk", default="6ME")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    token = ENTSOE_TOKEN.strip()
    if not token or token.startswith("paste-your"):
        raise RuntimeError("ENTSOE_TOKEN is missing. Put it in .env (see .env.example).")

    from entsoe import EntsoePandasClient

    start = pd.Timestamp(args.start, tz=BRUSSELS_TZ)
    end = (pd.Timestamp.now(tz=BRUSSELS_TZ) + pd.Timedelta(days=2)).normalize() if args.end is None \
        else pd.Timestamp(args.end, tz=BRUSSELS_TZ)
    chunks = build_chunks(start, end, freq=args.chunk)
    print(f"Fetching {len(chunks)} chunk(s) per zone: {start.date()} -> {end.date()}")

    df = fetch_all(EntsoePandasClient(api_key=token), args.wind_zones, args.price_zones, chunks,
                   args.retries, args.sleep)
    df.index.name = "timestamp"
    out = repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    meta = {
        "fetched_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "start": str(start), "end": str(end),
        "wind_zones": args.wind_zones, "price_zones": args.price_zones,
        "rows": int(len(df)),
        "columns": {c: {"non_null": int(df[c].notna().sum()), "first": str(df[c].first_valid_index()),
                        "last": str(df[c].last_valid_index())} for c in df.columns},
        "source": "ENTSO-E Transparency Platform (A69 wind/solar day-ahead forecast, A44 day-ahead prices)",
    }
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved {out}: {len(df):,} rows, {len(df.columns)} columns, {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()
