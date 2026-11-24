"""Download ENTSO-E exogenous data: day-ahead wind/solar and load forecasts, neighbour prices.

For each wind zone: day-ahead generation forecast for wind (onshore + offshore) and
solar -> wind_<ZONE>_MW, solar_<ZONE>_MW. For each load zone: day-ahead total load
forecast (A65, process type A01) -> load_<ZONE>_MW. For each price zone: day-ahead
price -> price_<ZONE>_EUR_MWh. Everything is averaged to hourly values in Europe/Oslo and
saved as ONE parquet (default data/entsoe_exog.parquet) that train.py,
backtest_rolling.py and forecast.py read with --exog_file. How the columns are used
(target-hour forecasts vs. origin-time prices, availability rule): see exog.py.

Requires ENTSOE_TOKEN in .env (same as fetch_prices.py). Zone codes are entsoe-py
areas, e.g. DE_LU, DK_1, DK_2, SE_3, SE_4, NO_2, NL, GB.

    python src/fetch_entsoe_exog.py
    python src/fetch_entsoe_exog.py --wind_zones DE_LU DK_1 --price_zones NO_2 SE_3 --load_zones NO_1 --start 2023-01-01

A zone that fails (no data, API error after retries) is skipped with a warning; the
run fails only if nothing at all could be fetched. The fetched end date defaults to
now + 2 days, so tomorrow's forecasts are included once published.

Incremental update: if --out already exists, each zone is fetched only from its last
stored value minus --overlap_days (default 3; re-fetches revised/late-published
forecasts) and merged into the file, new values overriding old ones. A zone with no
stored columns, or whose stored data starts well after --start, is fetched in full.
--full ignores the existing file and re-fetches everything.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from config import BRUSSELS_TZ, ENTSOE_TOKEN, TZ, redact, repo_path
from exog import normalise_load, normalise_wind_solar
from fetch_prices import build_chunks
from incremental import merge_new, plan_start

DEFAULT_WIND_ZONES = ["DE_LU", "DK_1", "DK_2", "SE_3"]
DEFAULT_PRICE_ZONES = ["NO_2", "SE_3", "DK_1", "DE_LU"]
DEFAULT_LOAD_ZONES = ["NO_1"]


def _with_retries(fn, what: str, retries: int, sleep: float):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # entsoe-py raises NoMatchingDataError, HTTP errors, ...
            if attempt == retries - 1:
                raise RuntimeError(f"{what} failed after {retries} attempts: {redact(exc)}") from None
            wait = 2**attempt
            print(f"    attempt {attempt + 1} failed ({redact(exc)}); retrying in {wait}s")
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
            elif kind == "load":
                raw = _with_retries(lambda: client.query_load_forecast(zone, start=a, end=b),
                                    f"load {zone} {a.date()}", retries, sleep)
                frames.append(normalise_load(raw, zone))
            else:
                raw = _with_retries(lambda: client.query_day_ahead_prices(zone, start=a, end=b),
                                    f"price {zone} {a.date()}", retries, sleep)
                s = raw.tz_convert(TZ).resample("h").mean()
                frames.append(s.rename(f"price_{zone}_EUR_MWh").to_frame())
        except RuntimeError as exc:
            print(f"  [skip chunk] {redact(exc)}")
        time.sleep(sleep)
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def zone_columns(kind: str, zone: str) -> list[str]:
    if kind == "wind_solar":
        return [f"wind_{zone}_MW", f"solar_{zone}_MW"]
    if kind == "load":
        return [f"load_{zone}_MW"]
    return [f"price_{zone}_EUR_MWh"]


def zone_start(existing: pd.DataFrame | None, kind: str, zone: str, start: pd.Timestamp,
               overlap_days: int = 3, backfill_tolerance_days: int = 7) -> pd.Timestamp:
    """Where to start fetching one zone: full history, or last stored value minus overlap."""
    if existing is None:
        return start
    cols = zone_columns(kind, zone)
    if any(c not in existing.columns or existing[c].first_valid_index() is None for c in cols):
        return start
    # Earliest column, not latest: a zone's columns are fetched together, and one of them can
    # simply have no data early on (solar_SE_3 starts 2021-12). Using the latest would treat
    # that gap as a backfill and re-fetch the full history on every run.
    first = min(existing[c].first_valid_index() for c in cols)
    last = min(existing[c].last_valid_index() for c in cols)
    return plan_start(first, last, start, overlap_days, backfill_tolerance_days)


merge_exog = merge_new


def fetch_all(client, wind_zones, price_zones, chunks, retries: int = 4, sleep: float = 0.5,
              load_zones=()) -> pd.DataFrame:
    """`chunks` is a list of (start, end), or a callable (kind, zone) -> list of (start, end)."""
    parts = []
    for kind, zones in (("wind_solar", wind_zones), ("load", load_zones), ("price", price_zones)):
        for zone in zones:
            print(f"{kind} {zone} ...")
            zc = chunks(kind, zone) if callable(chunks) else chunks
            df = fetch_zone(client, kind, zone, zc, retries, sleep)
            if df is None:
                print(f"  WARNING: no data for {kind} {zone}; skipped")
                continue
            print(f"  ok: {df.notna().sum().to_dict()}")
            parts.append(df)
    if not parts:
        raise RuntimeError("No ENTSO-E exog data could be fetched")
    return pd.concat(parts, axis=1).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch ENTSO-E wind/solar and load forecasts and neighbour prices")
    ap.add_argument("--wind_zones", nargs="*", default=DEFAULT_WIND_ZONES, help="Zones for wind/solar day-ahead forecasts")
    ap.add_argument("--price_zones", nargs="*", default=DEFAULT_PRICE_ZONES, help="Zones for day-ahead prices")
    ap.add_argument("--load_zones", nargs="*", default=DEFAULT_LOAD_ZONES, help="Zones for day-ahead total load forecasts")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None, help="Default: now + 2 days")
    ap.add_argument("--out", default="data/entsoe_exog.parquet")
    ap.add_argument("--chunk", default="6ME")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--overlap_days", type=int, default=3, help="Incremental update: re-fetch this many days before the last stored value")
    ap.add_argument("--full", action="store_true", help="Ignore an existing file and fetch the whole history")
    args = ap.parse_args()

    token = ENTSOE_TOKEN.strip()
    if not token or token.startswith("paste-your"):
        raise RuntimeError("ENTSOE_TOKEN is missing. Put it in .env (see .env.example).")

    from entsoe import EntsoePandasClient

    start = pd.Timestamp(args.start, tz=BRUSSELS_TZ)
    end = (pd.Timestamp.now(tz=BRUSSELS_TZ) + pd.Timedelta(days=2)).normalize() if args.end is None \
        else pd.Timestamp(args.end, tz=BRUSSELS_TZ)
    out = repo_path(args.out)
    existing = None
    if out.exists() and not args.full:
        existing = pd.read_parquet(out)
        print(f"Updating {out} ({len(existing):,} rows, last {existing.index.max()})")
    else:
        print(f"Full fetch {start.date()} -> {end.date()}")

    def chunks_for(kind: str, zone: str):
        zs = zone_start(existing, kind, zone, start, args.overlap_days)
        if existing is not None:
            print(f"  from {zs.date()}" + ("" if zs > start else " (full)"))
        return build_chunks(zs, end, freq=args.chunk)

    new = fetch_all(EntsoePandasClient(api_key=token), args.wind_zones, args.price_zones, chunks_for,
                    args.retries, args.sleep, load_zones=args.load_zones)
    df = merge_exog(existing, new)
    df.index.name = "timestamp"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    meta = {
        "fetched_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "start": str(start), "end": str(end),
        "wind_zones": args.wind_zones, "price_zones": args.price_zones, "load_zones": args.load_zones,
        "rows": int(len(df)),
        "columns": {c: {"non_null": int(df[c].notna().sum()), "first": str(df[c].first_valid_index()),
                        "last": str(df[c].last_valid_index())} for c in df.columns},
        "source": "ENTSO-E Transparency Platform (A69 wind/solar day-ahead forecast, A65/A01 day-ahead load forecast, A44 day-ahead prices)",
    }
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved {out}: {len(df):,} rows, {len(df.columns)} columns, {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()
