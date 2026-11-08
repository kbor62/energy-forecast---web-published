"""
Fetch weather forecast from MET Norway LocationForecast API.
Provides 10-day forecasts for temperature, wind, precipitation, and cloud cover.

Key locations:
- Oslo (Norway) - domestic demand indicator
- Hamburg (Germany) - wind power proxy for DE
- Copenhagen (Denmark) - wind power proxy for DK

API: https://api.met.no/weatherapi/locationforecast/2.0/compact
No authentication required, but User-Agent header is mandatory.

Archiving: every run also stores the forecast exactly as issued in
data/forecast_archive/metno_<YYYYmmddTHHMMZ>.parquet (index = valid time in UTC,
one column group per location, plus issued_at). The stamp is MET's own
updated_at, so running twice within one model run stores nothing new. The --out
file is overwritten on every run; the archive is not. Archived snapshots are what
forecast.py and backtest_rolling.py --weather_mode archive read, and the only way
to evaluate weather forecasts honestly. Run this every few hours (README 13.5).
Options: --archive_dir, --no_archive.

Usage:
    python src/fetch_weather_forecast.py --out data/weather_forecast.parquet
    python src/fetch_weather_forecast.py --locations oslo hamburg --format csv
"""

import argparse
import pandas as pd
import requests
from datetime import datetime, timezone
from pathlib import Path

# Key locations for electricity market modeling
LOCATIONS = {
    'oslo': {'lat': 59.9139, 'lon': 10.7522, 'name': 'Oslo, Norway'},
    'hamburg': {'lat': 53.5511, 'lon': 9.9937, 'name': 'Hamburg, Germany'},  # DE wind proxy
    'copenhagen': {'lat': 55.6761, 'lon': 12.5683, 'name': 'Copenhagen, Denmark'},  # DK wind proxy
}

BASE_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
USER_AGENT = "energy-forecast/1.0 github.com/kbor62/energy-forecast"

def fetch_forecast(lat, lon, location_name):
    """Fetch 10-day weather forecast from MET Norway."""
    params = {'lat': lat, 'lon': lon}
    headers = {'User-Agent': USER_AGENT}

    print(f"Fetching forecast for {location_name} (lat={lat}, lon={lon})...")

    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Model-run time (MET 'updated_at'); falls back to the fetch time.
        updated = data.get('properties', {}).get('meta', {}).get('updated_at')
        issued_at = pd.Timestamp(updated) if updated else pd.Timestamp.now(tz='UTC')
        issued_at = issued_at.tz_convert('UTC') if issued_at.tzinfo else issued_at.tz_localize('UTC')

        # Parse timeseries
        records = []
        for entry in data['properties']['timeseries']:
            ts = pd.to_datetime(entry['time'])
            instant = entry['data']['instant']['details']

            # Extract key variables
            record = {
                'timestamp': ts,
                'temperature_C': instant.get('air_temperature'),
                'wind_speed_ms': instant.get('wind_speed'),
                'wind_from_direction': instant.get('wind_from_direction'),
                'cloud_area_fraction': instant.get('cloud_area_fraction'),
            }

            # Add precipitation if available (next_1_hours)
            if 'next_1_hours' in entry['data']:
                precip = entry['data']['next_1_hours']['details'].get('precipitation_amount', 0.0)
                record['precipitation_mm'] = precip
            else:
                record['precipitation_mm'] = None

            records.append(record)

        df = pd.DataFrame(records)
        df = df.set_index('timestamp').sort_index()

        # Resample to hourly (forecast is often 1-hour resolution but can have gaps)
        df = df.resample('h').mean()
        raw = df.copy()  # unfilled values, as issued: this is what gets archived

        # Forward-fill small gaps (max 3 hours)
        df = df.ffill(limit=3)

        print(f"  [OK] Retrieved {len(df)} hours of forecast data (issued {issued_at:%Y-%m-%d %H:%M}Z)")
        return df, raw, issued_at

    except requests.RequestException as e:
        print(f"  [ERROR] API error: {e}")
        return None

def combine_locations(forecasts):
    """Combine forecasts from multiple locations with prefixes."""
    combined = pd.DataFrame()

    for loc_key, df in forecasts.items():
        if df is None or df.empty:
            continue

        # Add location prefix to column names
        df_prefixed = df.add_prefix(f'{loc_key}_')

        if combined.empty:
            combined = df_prefixed
        else:
            combined = combined.join(df_prefixed, how='outer')

    return combined

def archive_snapshot(raws, issued_at, archive_dir):
    """Save the unmodified forecast as issued, one file per MET model run.

    Training on these (instead of observations) lets the model learn from the
    forecast error it will face at inference. Skips runs already archived.
    """
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = archive_dir / f"metno_{issued_at:%Y%m%dT%H%MZ}.parquet"
    if path.exists():
        print(f"[ARCHIVE] already have {path.name} - skipped")
        return None
    snap = combine_locations(raws)
    snap['issued_at'] = issued_at
    snap.to_parquet(path)
    print(f"[ARCHIVE] saved {path} ({len(snap)} rows)")
    return path

def main():
    ap = argparse.ArgumentParser(description='Fetch weather forecasts for energy price modeling')
    ap.add_argument('--out', default='data/weather_forecast.parquet', help='Output file path')
    ap.add_argument('--locations', nargs='+', default=['oslo', 'hamburg', 'copenhagen'],
                    choices=list(LOCATIONS.keys()), help='Locations to fetch')
    ap.add_argument('--format', choices=['parquet', 'csv'], default='parquet')
    ap.add_argument('--archive_dir', default='data/forecast_archive',
                    help='Directory for per-run snapshots (kept; the --out file is overwritten each run)')
    ap.add_argument('--no_archive', action='store_true')
    args = ap.parse_args()

    # Fetch forecasts for each location
    forecasts, raws, issued = {}, {}, []
    for loc_key in args.locations:
        loc = LOCATIONS[loc_key]
        res = fetch_forecast(loc['lat'], loc['lon'], loc['name'])
        if res is not None:
            forecasts[loc_key], raws[loc_key], issued_at = res
            issued.append(issued_at)

    if not forecasts:
        print("[ERROR] No forecast data retrieved")
        return

    if not args.no_archive:
        # One file per run, stamped with the earliest model-run time across locations.
        archive_snapshot(raws, min(issued), args.archive_dir)

    # Combine all locations
    combined = combine_locations(forecasts)

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    if args.format == 'parquet':
        combined.to_parquet(args.out)
    else:
        combined.to_csv(args.out)

    print(f"\n[OK] Saved weather forecast -> {args.out}")
    print(f"Rows: {len(combined)} | Span: {combined.index.min()} -> {combined.index.max()}")
    print(f"Columns ({len(combined.columns)}): {list(combined.columns)[:10]}...")

    # Show sample
    print("\nSample data (first 5 rows):")
    print(combined.head())

    # Data completeness
    print("\nData completeness:")
    for col in combined.columns:
        pct = (combined[col].notna().sum() / len(combined)) * 100
        print(f"  {col}: {pct:.1f}%")

if __name__ == "__main__":
    main()
