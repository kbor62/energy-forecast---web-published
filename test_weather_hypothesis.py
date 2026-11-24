#!/usr/bin/env python3
"""DEPRECATED - do not use. Replaced by test_weather_forecast_value.py.

This script copies a temperature feature file over data/features_price_only.parquet
(destroying the baseline file), and train.py never read the temperature columns, so
both runs used identical features and its conclusion ("price history already
captures the weather signal") is not valid. It now refuses to run without --force.

Original description: compare backtest results with and without temperature features.

This script runs the same model (lgbm) against two feature sets:
  1. Price history only
  2. Price history + temperature lags

And compares validation MAE to test the hypothesis.
"""

import subprocess
import sys
import json
from pathlib import Path
import pandas as pd
import shutil


def run_test(features_file, output_dir, label):
    """Run training with given features file and output directory."""
    print(f"\n{'='*70}")
    print(f"TEST: {label}")
    print(f"{'='*70}")
    print(f"Features: {features_file}")
    print(f"Model output: {output_dir}")
    print("Model: lgbm\n")

    # Copy features file
    src = Path("data") / features_file
    dst = Path("data/features_price_only.parquet")

    if src != dst:
        shutil.copy(src, dst)

    # Clean output directory to avoid locking issues
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
        print(f"Cleaned {output_dir}")

    # Train with specific output directory
    train_cmd = [
        "python", "src/train.py",
        "--horizon", "48",
        "--val_days", "30",
        "--out_dir", output_dir
    ]

    print("Running: " + " ".join(train_cmd))
    result = subprocess.run(train_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Training failed")
        print(result.stderr[:500])
        return None

    print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)

    # Get validation MAE from the output
    for line in result.stdout.split('\n'):
        if 'Mean validation MAE' in line:
            mae_str = line.split(':')[1].strip().split()[0]
            mae = float(mae_str)
            return mae

    return None


def main():
    if "--force" not in sys.argv:
        sys.exit("Deprecated and destructive (overwrites data/features_price_only.parquet). "
                 "Use test_weather_forecast_value.py, or pass --force.")
    print("\n" + "="*70)
    print("WEATHER FEATURE HYPOTHESIS TEST")
    print("="*70)

    # Test 1: Without temperature
    mae_without = run_test(
        "features_price_only.parquet",
        "models/lgbm_baseline",
        "BASELINE: Price history only"
    )

    # Test 2: With temperature
    mae_with = run_test(
        "features_price_temp.parquet",
        "models/lgbm_with_temp",
        "WEATHER TEST: Price history + Temperature lags"
    )

    # Results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    if mae_without and mae_with:
        improvement = ((mae_without - mae_with) / mae_without) * 100
        print(f"\nMAE without temp:   {mae_without:.4f} EUR/kWh")
        print(f"MAE with temp:      {mae_with:.4f} EUR/kWh")
        print(f"Improvement:        {improvement:+.2f}%")

        print(f"\n{'='*70}")
        if improvement < 0.5:
            print("[HYPOTHESIS CONFIRMED]")
            print("Temperature features add < 0.5% improvement.")
            print("Conclusion: Price history already captures weather signal.")
            print("Recommendation: Focus on weather FORECASTS, not historical data.")
        elif improvement < 2:
            print("[MARGINAL IMPROVEMENT]")
            print(f"Temperature features add {improvement:.2f}% improvement.")
            print("Signal is weak - price history dominates.")
        else:
            print("[HYPOTHESIS REJECTED]")
            print(f"Temperature features add {improvement:.2f}% improvement.")
            print("Weather data has meaningful independent predictive value!")
    else:
        print("ERROR: Could not extract MAE values")
        if mae_without:
            print(f"  Test 1 (baseline): {mae_without:.4f}")
        if mae_with:
            print(f"  Test 2 (with temp): {mae_with:.4f}")


if __name__ == "__main__":
    main()

