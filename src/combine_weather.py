"""Combine historical weather with forecast data.

NOTE: only used by build_features.py --add-weather-forecast (legacy, see README
section 12). Training, forecasting and backtesting do not read the combined file.

Merges:
  - Historical weather observations (from fetch_weather.py) — columns without prefix
  - Weather forecasts (from fetch_weather_forecast.py) — columns with location prefix

The combined file uses a consistent column schema. Forecast columns are mapped
so that oslo_temperature_C → temperature_C (primary signal), and regional
wind/cloud columns are kept for market-wide proxy features.

Usage:
    python src/combine_weather.py
    python src/combine_weather.py --historical data/weather_oslo.parquet --forecast data/weather_forecast.parquet
"""

import argparse
import pandas as pd
from pathlib import Path

TZ = "Europe/Oslo"

# Map forecast columns to canonical names used in build_features.py
# Only oslo columns are mapped to the primary (unprefixed) schema
FORECAST_COLUMN_MAP = {
    "oslo_temperature_C": "temperature_C",
    "oslo_wind_speed_ms": "wind_speed_ms",
    "oslo_precipitation_mm": "precipitation_mm",
    "oslo_cloud_area_fraction": "cloud_area_fraction",
}


def _load_tz(path: Path, tz: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(tz)
    else:
        df.index = df.index.tz_convert(tz)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine historical and forecast weather data")
    ap.add_argument("--historical", default="data/weather_oslo.parquet")
    ap.add_argument("--forecast", default="data/weather_forecast.parquet")
    ap.add_argument("--out", default="data/weather_combined.parquet")
    args = ap.parse_args()

    hist_path = Path(args.historical)
    fcst_path = Path(args.forecast)

    hist_df = _load_tz(hist_path, TZ) if hist_path.exists() else pd.DataFrame()
    fcst_df = _load_tz(fcst_path, TZ) if fcst_path.exists() else pd.DataFrame()

    if hist_df.empty and fcst_df.empty:
        print("[ERROR] No weather data available")
        return

    if not hist_df.empty:
        print(f"[OK] Historical: {len(hist_df)} rows  {hist_df.index.min()} -> {hist_df.index.max()}")

    if not fcst_df.empty:
        # Rename oslo_* columns to canonical names so historical and forecast align
        fcst_df = fcst_df.rename(columns=FORECAST_COLUMN_MAP)
        # Keep regional columns (hamburg_*, copenhagen_*) as-is for additional features
        print(f"[OK] Forecast: {len(fcst_df)} rows  {fcst_df.index.min()} -> {fcst_df.index.max()}")

    # Splice: historical up to cutoff, forecast after
    if not hist_df.empty and not fcst_df.empty:
        cutoff = hist_df.index.max()
        fcst_df = fcst_df[fcst_df.index > cutoff]
        print(f"Cutoff: {cutoff} — {len(fcst_df)} forecast rows kept")

    combined = pd.concat([hist_df, fcst_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out)

    print(f"\n[OK] Saved combined weather -> {args.out}")
    print(f"Rows: {len(combined)} | Span: {combined.index.min()} -> {combined.index.max()}")
    print(f"Columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()
