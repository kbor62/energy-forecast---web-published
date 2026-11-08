"""Download ENTSO-E day-ahead prices to CSV and Parquet, one year per request.

Requires ENTSOE_TOKEN in a .env file in the repository root (see .env.example).
Available zones: NO_1, NO_2, NO_3, NO_4, NO_5, and others (see ZONES_HINT).

    python src/fetch_prices.py --zone NO_1 --start 2020-01-01 --end 2026-09-23
    python src/fetch_prices.py --zone NO_1 --start 2020-01-01  # Fetch up to today
"""

from __future__ import annotations

import argparse
import json
import time
import pandas as pd

from config import BRUSSELS_TZ, ENTSOE_TOKEN, TZ, repo_path

ZONES_HINT = "NO_1 Oslo, NO_2 Kristiansand, NO_3 Trondheim, NO_4 Tromso, NO_5 Bergen"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch ENTSO-E day-ahead prices.",
        epilog=f"Norwegian bidding zones: {ZONES_HINT}",
    )
    p.add_argument("--zone", default="NO_1", help='Bidding zone, e.g. "NO_1"')
    p.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Default: now + 2 days")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--chunk", default="1YE", help='Chunk size, e.g. "1YE" or "6ME"')
    p.add_argument("--sleep", type=float, default=0.5, help="Seconds between requests")
    p.add_argument("--retries", type=int, default=4, help="Attempts per chunk")
    return p.parse_args()


def build_chunks(
    start_ts: pd.Timestamp, end_ts: pd.Timestamp, freq: str = "1YE"
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split the period into yearly chunks."""
    edges = pd.date_range(start_ts, end_ts, freq=freq, inclusive="left", tz=start_ts.tz)
    if len(edges) == 0 or edges[0] != start_ts:
        edges = pd.DatetimeIndex([start_ts]).append(edges)
    if edges[-1] != end_ts:
        edges = edges.append(pd.DatetimeIndex([end_ts]))
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def main() -> None:
    args = _parse_args()

    token = ENTSOE_TOKEN.strip()
    if not token or token.startswith("paste-your"):
        raise RuntimeError(
            "ENTSOE_TOKEN is missing. Copy .env.example to .env and put your token there "
            "(free from https://transparency.entsoe.eu/)."
        )

    from entsoe import EntsoePandasClient

    zone = args.zone.strip()
    start = pd.Timestamp(args.start, tz=BRUSSELS_TZ)
    if args.end is None:
        # Prices for the next day are published around 13:00, so fetch two days ahead.
        end = (pd.Timestamp.now(tz=BRUSSELS_TZ) + pd.Timedelta(days=2)).normalize()
    else:
        end = pd.Timestamp(args.end, tz=BRUSSELS_TZ)

    csv_path = repo_path(args.out or f"data/entsoe_{zone}_{start.date()}_now.csv")
    parquet_path = csv_path.with_suffix(".parquet")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    client = EntsoePandasClient(api_key=token)
    chunks = build_chunks(start, end, freq=args.chunk)
    print(f"Fetching {zone} in {len(chunks)} chunk(s): {start.date()} -> {end.date()}")

    parts: list[pd.Series] = []
    for i, (a, b) in enumerate(chunks, 1):
        for attempt in range(args.retries):
            try:
                parts.append(client.query_day_ahead_prices(zone, start=a, end=b))
                print(f"  [{i}/{len(chunks)}] {a.date()} -> {b.date()}: ok")
                break
            except Exception as exc:
                if attempt == args.retries - 1:
                    raise RuntimeError(
                        f"ENTSO-E chunk {a.date()} -> {b.date()} failed after "
                        f"{args.retries} attempts: {exc}"
                    ) from exc
                wait = 2**attempt
                print(f"  attempt {attempt + 1} failed: {exc} -- retrying in {wait}s")
                time.sleep(wait)
        time.sleep(float(args.sleep))

    series = pd.concat(parts).sort_index()
    series = series[~series.index.duplicated(keep="last")].tz_convert(TZ)

    df = series.reset_index()
    df.columns = ["timestamp", "price_EUR_per_MWh"]
    df["price_EUR_per_kWh"] = df["price_EUR_per_MWh"] / 1000.0
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    meta = {
        "zone": zone,
        "requested_start": str(start),
        "requested_end": str(end),
        "fetched_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "n_rows": int(len(df)),
        "first_timestamp": str(df["timestamp"].min()),
        "last_timestamp": str(df["timestamp"].max()),
        "source": "ENTSO-E Transparency Platform (day-ahead prices)",
    }
    meta_path = csv_path.with_name(csv_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    span_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
    print(
        f"Rows: {len(df)} | Span: {df['timestamp'].min()} -> {df['timestamp'].max()} "
        f"(~{span_days} days)"
    )


if __name__ == "__main__":
    main()
