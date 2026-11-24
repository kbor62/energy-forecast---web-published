"""Test whether weather features improve the model.

Trains two models with identical settings:
  1. Baseline: price history only (data/features_price_only.parquet)
  2. Weather:  the same plus observed weather at t+h (train.py --weather_file
     data/weather_oslo.parquet)

The weather is OBSERVED, so the result is an upper bound on the value of a real
forecast. Honest evaluation needs archived forecasts (README section 6.4).

If the weather model's MAE doesn't improve by at least THRESHOLD_PCT, the test
concludes that weather forecast features add no value.

Usage:
    python test_weather_forecast_value.py
"""

import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime

THRESHOLD_PCT = 0.5

BASELINE_FEATURES = "data/features_price_only.parquet"
WEATHER_FEATURES = BASELINE_FEATURES  # same price features; weather added in train.py
WEATHER_FILE = "data/weather_oslo.parquet"

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
BASELINE_OUT = f"models/lgbm_baseline_test_{_ts}"
WEATHER_OUT = f"models/lgbm_weather_test_{_ts}"

TRAIN_CMD = [
    sys.executable, "src/train.py",
    "--horizon", "48",
    "--val_days", "30",
]


def run_train(features_file: str, out_dir: str, weather_file: str | None = None) -> float:
    if not Path(features_file).exists():
        print(f"[ERROR] Features file not found: {features_file}")
        print("  Run: python src/build_features.py --add-weather-forecast")
        sys.exit(1)

    cmd = TRAIN_CMD + ["--inp", features_file, "--out_dir", out_dir]
    if weather_file:
        cmd += ["--weather_file", weather_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    match = re.search(r"Mean validation MAE across horizons:\s+([\d.]+)", output)
    if not match:
        print(f"[ERROR] Could not parse MAE from output:\n{output[-500:]}")
        sys.exit(1)

    return float(match.group(1))


def main() -> None:
    print("=" * 60)
    print("WEATHER FORECAST VALUE TEST")
    print("=" * 60)
    print()

    print("1. Baseline (price history only)...")
    baseline_mae = run_train(BASELINE_FEATURES, BASELINE_OUT)
    print(f"   MAE: {baseline_mae:.4f} EUR/kWh")
    print()

    print("2. With weather forecast features...")
    weather_mae = run_train(WEATHER_FEATURES, WEATHER_OUT, WEATHER_FILE)
    print(f"   MAE: {weather_mae:.4f} EUR/kWh")
    print()

    improvement_pct = (baseline_mae - weather_mae) / baseline_mae * 100

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline MAE:  {baseline_mae:.4f} EUR/kWh")
    print(f"Weather MAE:   {weather_mae:.4f} EUR/kWh")
    print(f"Improvement:   {improvement_pct:+.2f}%")
    print()

    if improvement_pct >= THRESHOLD_PCT:
        print(f"[WEATHER HELPS] >{THRESHOLD_PCT}% improvement — weather forecast features add value.")
    else:
        print(f"[NO IMPROVEMENT] <{THRESHOLD_PCT}% improvement — weather forecasts don't help beyond price history.")
    print("=" * 60)


if __name__ == "__main__":
    main()
