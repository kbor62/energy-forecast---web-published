"""
Fetch weather data from MET Norway API (Parquet pipeline)

Description:
    Fetches hourly weather observations from MET Norway's Frost API for Norwegian locations.
    Weather data (temperature, wind speed, cloud coverage) are important features for
    electricity price prediction since:
    - Temperature affects heating demand (cold → high demand → high prices)
    - Wind speed correlates with wind power production (high wind → low prices)
    - Cloud coverage affects solar power production

Usage Examples:
    # Fetch weather for Oslo (default)
    python src/fetch_weather.py --start 2023-01-01 --out data/weather_oslo.parquet

    # Fetch for Bergen
    python src/fetch_weather.py --station SN50540 --start 2023-01-01 --out data/weather_bergen.parquet

    # Custom date range
    python src/fetch_weather.py --start 2023-01-01 --end 2025-12-31 --out data/weather_2023_2025.parquet

Arguments:
    --station    MET station ID (default: SN18700 = Oslo Blindern)
                 Common stations:
                   SN18700 - Oslo (Blindern)
                   SN50540 - Bergen (Florida)
                   SN90450 - Trondheim (Voll)
                   SN99840 - Tromsø (Langnes)
    --start      Start date in YYYY-MM-DD format (default: 2023-01-01)
    --end        End date in YYYY-MM-DD format (default: now)
    --out        Output Parquet file path (default: data/weather_oslo.parquet)
    --client_id  MET Frost API client ID (from environment or .env)
    --overlap_days  Incremental update: re-fetch this many days before the last stored
                 value (default 3)
    --full       Ignore an existing output file and fetch the whole history

Incremental update:
    If --out already exists, only the tail is fetched (last stored hour minus
    --overlap_days) and merged into it, new values overriding old ones. The full
    history is fetched when the file is missing, when the stored data starts later than
    --start, or with --full.

Requirements:
    pip install pandas requests python-dotenv

Configuration:
    Get free API access from https://frost.met.no/auth/requestCredentials.html
    Create a .env file in project root with:
        FROST_CLIENT_ID=your_client_id_here

Output Columns:
    - timestamp: Timezone-aware datetime (Europe/Oslo)
    - temperature_C: Air temperature in Celsius
    - wind_speed_ms: Wind speed in meters per second
    - cloud_coverage_pct: Cloud coverage percentage (0-100, NaN if unavailable)
    - precipitation_mm: Precipitation in mm (NaN if unavailable)

Notes:
    - Hourly resolution data
    - The data/weather_oslo.parquet in this repository has only temperature_C,
      wind_speed_ms and precipitation_mm: Blindern returns no hourly cloud element, so
      that column is simply absent. Training uses those three columns (--weather_cols).
    - Observed weather is what train.py --weather_file uses; it is NOT a forecast
      (see README section 6.4).
    - Missing values are forward-filled (limit 3 hours) for temperature/wind
    - Cloud coverage often has gaps (filled with mean of day)
    - API rate limit: ~10 requests per minute (script includes delays)
    - Data availability: Most stations from 2000+, some longer

API Documentation:
    https://frost.met.no/api.html
"""

import os
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from incremental import merge_new, plan_start


API_BASE = "https://frost.met.no/observations/v0.jsonld"
TZ = "Europe/Oslo"

# Element IDs from MET Frost API
ELEMENTS = {
    "temperature": "air_temperature",
    "wind_speed": "wind_speed",
    "cloud_coverage": "cloud_area_fraction",
    "precipitation": "sum(precipitation_amount PT1H)"  # 1-hour sum
}


def _fetch_chunk(station_id: str, start_date: str, end_date: str, client_id: str) -> pd.DataFrame:
    """Fetch one chunk (must be within 100k observation limit)."""
    params = {
        "sources": station_id,
        "referencetime": f"{start_date}/{end_date}",
        "elements": ",".join(ELEMENTS.values()),
        "timeresolutions": "PT1H",
        "timeoffsets": "PT0H",
    }
    headers = {"Accept": "application/json"}
    auth = (client_id, "")

    response = requests.get(API_BASE, params=params, headers=headers, auth=auth, timeout=60)

    if response.status_code == 401:
        raise RuntimeError("FROST_CLIENT_ID is invalid or not authorized. Get one from https://frost.met.no/")
    elif response.status_code != 200:
        raise RuntimeError(f"Frost API error {response.status_code}: {response.text}")

    data = response.json()
    observations = data.get("data", [])

    if not observations:
        raise RuntimeError(f"No data returned for station {station_id} in period {start_date}/{end_date}")

    records = []
    for obs in observations:
        timestamp = obs["referenceTime"]
        observation_dict = {"timestamp": timestamp}
        for measurement in obs.get("observations", []):
            element_id = measurement.get("elementId", "")
            value = measurement.get("value")
            if "air_temperature" in element_id:
                observation_dict["temperature_C"] = value
            elif "wind_speed" in element_id:
                observation_dict["wind_speed_ms"] = value
            elif "cloud_area_fraction" in element_id:
                observation_dict["cloud_coverage_pct"] = value
            elif "precipitation_amount" in element_id:
                observation_dict["precipitation_mm"] = value
        records.append(observation_dict)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
    df = df.set_index("timestamp").sort_index()
    return df


def fetch_frost_data(station_id: str, start_date: str, end_date: str, client_id: str) -> pd.DataFrame:
    """Fetch weather observations, splitting into yearly chunks to stay within 100k limit."""
    from dateutil.relativedelta import relativedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Build yearly chunks
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + relativedelta(years=1), end)
        s = chunk_start.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"  Fetching {s} -> {e}...")
        chunk_df = _fetch_chunk(station_id, s, e, client_id)
        chunks.append(chunk_df)
        chunk_start = chunk_end

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.asfreq("h")

    if "temperature_C" in df.columns:
        df["temperature_C"] = df["temperature_C"].ffill(limit=3)
    if "wind_speed_ms" in df.columns:
        df["wind_speed_ms"] = df["wind_speed_ms"].ffill(limit=3)
    if "cloud_coverage_pct" in df.columns:
        daily_mean = df.groupby(df.index.date)["cloud_coverage_pct"].transform("mean")
        df["cloud_coverage_pct"] = df["cloud_coverage_pct"].fillna(daily_mean)

    return df


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Fetch weather data from MET Norway Frost API")
    ap.add_argument("--station", default="SN18700", help="MET station ID (default: SN18700 = Oslo Blindern)")
    ap.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--out", default="data/weather_oslo.parquet", help="Output Parquet path")
    ap.add_argument("--overlap_days", type=int, default=3, help="Incremental update: re-fetch this many days before the last stored value")
    ap.add_argument("--full", action="store_true", help="Ignore an existing file and fetch the whole history")
    ap.add_argument("--client_id", default=None, help="Frost API client ID (default: from FROST_CLIENT_ID env)")
    args = ap.parse_args()

    client_id = args.client_id or os.getenv("FROST_CLIENT_ID")
    if not client_id:
        raise RuntimeError(
            "FROST_CLIENT_ID not set. Get credentials from https://frost.met.no/auth/requestCredentials.html "
            "and add to .env file: FROST_CLIENT_ID=your_id"
        )

    end_date = args.end if args.end else datetime.now().strftime("%Y-%m-%d")

    old = None
    fetch_start = args.start
    out_path = Path(args.out)
    if out_path.exists() and not args.full:
        old = pd.read_parquet(out_path)
        start_ts = pd.Timestamp(args.start, tz=TZ)
        inc = plan_start(old.index.min(), old.index.max(), start_ts, args.overlap_days)
        if inc > start_ts:
            fetch_start = min(inc.strftime("%Y-%m-%d"), (pd.Timestamp(end_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
            print(f"Existing {out_path.name}: {len(old):,} rows, last {old.index.max()}")
        else:
            old = None  # stored data starts too late -> full fetch

    mode = "Updating" if old is not None else "Fetching"
    print(f"{mode} weather data {fetch_start} -> {end_date} for station {args.station} (yearly chunks)...")
    df = fetch_frost_data(args.station, fetch_start, end_date, client_id)
    if old is not None:
        df = merge_new(old, df).asfreq("h")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=True)

    print(f"[OK] Saved weather data -> {args.out}")
    print(f"Rows: {len(df):,} | Span: {df.index.min()} -> {df.index.max()}")
    print(f"Columns: {list(df.columns)}")

    print("\nData completeness:")
    for col in df.columns:
        pct = (df[col].notna().sum() / len(df)) * 100
        print(f"  {col}: {pct:.1f}%")


if __name__ == "__main__":
    main()
