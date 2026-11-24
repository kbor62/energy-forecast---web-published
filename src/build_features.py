"""Build the feature file: price, history features and targets.

Calendar features are not stored; they depend on the horizon and are built
at training time by ``features.assemble``.

    python src/build_features.py
    python src/build_features.py --csv
    python src/build_features.py --add-weather-forecast --out data/features_weather_forecast.parquet

Weather forecast mode (--add-weather-forecast) - LEGACY:
    The extra <col>_at_t{h}h columns are NOT read by train.py (it selects only the
    price-history columns), so a model trained from this file ignores them. Weather
    is added at training time with train.py --weather_file instead (README 6.4).
    Kept for reference.

    For each horizon h (1..48), adds the weather VALUE at t+h as a feature.
    During training this uses historical observations — the same data a
    real forecast would have provided. At inference time the pipeline
    substitutes actual forecast values for the same columns.

    This is the correct way to model weather forecast value: the model
    sees "what will the weather be at the target hour?" rather than
    "what was the weather some hours ago?".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import features as F
from config import PRICE_COL, TZ, repo_path
from price_io import describe_series, load_price_series


DEFAULT_INPUT = "data/entsoe_NO_1_2020-01-01_now.parquet"
DEFAULT_OUTPUT = "data/features_price_only.parquet"
WEATHER_INPUT = "data/weather_combined.parquet"

# Canonical weather variables used as forecast features
WEATHER_FORECAST_COLS = ["temperature_C", "wind_speed_ms", "precipitation_mm"]

# Horizons at which weather forecast values are added as features
WEATHER_FORECAST_HORIZONS = [1, 6, 12, 24, 36, 48]


def load_weather(weather_path: Path) -> pd.DataFrame | None:
    if not weather_path.exists():
        print(f"Weather file not found: {weather_path}")
        return None
    df = pd.read_parquet(weather_path)
    df.index = df.index.tz_localize(TZ) if df.index.tz is None else df.index.tz_convert(TZ)
    return df


def add_weather_forecast_features(
    df: pd.DataFrame,
    weather: pd.DataFrame,
    horizons: list[int] = WEATHER_FORECAST_HORIZONS,
) -> pd.DataFrame:
    """Add weather-at-target-horizon features.

    For each horizon h and weather variable v, adds column v_at_t{h}h
    whose value at row t is the observed (or forecasted) weather at t+h.

    During training: uses historical observations (acts as oracle).
    During inference: replace these columns with actual forecast values.
    """
    weather_aligned = weather.reindex(df.index, method="nearest")

    for col in WEATHER_FORECAST_COLS:
        if col not in weather_aligned.columns:
            print(f"  Warning: {col} not in weather data — skipping")
            continue
        for h in horizons:
            df[f"{col}_at_t{h}h"] = weather_aligned[col].shift(-h)

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the price feature file")
    ap.add_argument("--inp", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=DEFAULT_OUTPUT)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--horizon", type=int, default=48, help="Max target horizon in hours")
    ap.add_argument("--csv", action="store_true", help="Also write CSV copy")
    ap.add_argument(
        "--add-weather-forecast",
        action="store_true",
        help=(
            "Add weather-at-horizon features. Requires data/weather_combined.parquet. "
            "Defaults --out to data/features_weather_forecast.parquet."
        ),
    )
    args = ap.parse_args()

    # If weather forecast mode and user didn't specify --out, use a dedicated output file
    if args.add_weather_forecast and args.out == DEFAULT_OUTPUT:
        args.out = "data/features_weather_forecast.parquet"

    inp_path = repo_path(args.inp)
    out_path = repo_path(args.out)

    price = load_price_series(inp_path, tz=TZ)
    if args.start:
        price = price.loc[pd.Timestamp(args.start, tz=TZ):]
    if args.end:
        price = price.loc[: pd.Timestamp(args.end, tz=TZ)]

    horizons = list(range(1, args.horizon + 1))
    history = F.history_features(price)
    targets = F.make_targets(price, horizons)
    out = pd.concat([price.rename(PRICE_COL), history, targets], axis=1)

    weather_features_added = False
    if args.add_weather_forecast:
        weather_path = repo_path(WEATHER_INPUT)
        weather = load_weather(weather_path)
        if weather is not None:
            out = add_weather_forecast_features(out, weather)
            weather_features_added = True
            n_added = len(WEATHER_FORECAST_COLS) * len(WEATHER_FORECAST_HORIZONS)
            print(f"Added {n_added} weather forecast features ({len(WEATHER_FORECAST_COLS)} vars x {len(WEATHER_FORECAST_HORIZONS)} horizons)")
        else:
            print("Run fetch_weather.py and combine_weather.py first")

    out.index.name = "timestamp"
    out = out.dropna(subset=[F.target_column(1)], how="all")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=True)
    if args.csv:
        out.to_csv(out_path.with_suffix(".csv"), index=True)

    n_weather_cols = len(WEATHER_FORECAST_COLS) * len(WEATHER_FORECAST_HORIZONS) if weather_features_added else 0
    print(f"Rows: {len(out):,}   history columns: {len(F.history_columns()) + n_weather_cols}")
    print(f"Span: {out.index.min()} -> {out.index.max()}")
    print(f"Output: {out_path}")

    meta = {
        "built_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "source_file": str(inp_path),
        "series": describe_series(price),
        "horizons": [1, args.horizon],
        "history_columns": F.history_columns(),
        "n_history_columns": len(F.history_columns()),
        "calendar_features": "built per horizon at t+h by features.assemble",
        "weather_forecast_features": weather_features_added,
        "weather_forecast_cols": WEATHER_FORECAST_COLS if weather_features_added else [],
        "weather_forecast_horizons": WEATHER_FORECAST_HORIZONS if weather_features_added else [],
        "rows": int(len(out)),
    }
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
