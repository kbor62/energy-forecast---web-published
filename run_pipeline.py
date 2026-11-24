#!/usr/bin/env python3
"""
Convenience wrapper for the energy forecast pipeline.

Runs: fetch data -> build features -> train -> forecast -> (optional) backtest.
Every run retrains unless --skip-train is given. The model directory is named
from the options: models/lgbm[_wx][_ex][_nwp][_t<mode>][_dl<lag>]. Each step can
also be run by hand; see README.md.

Production (D+2/D+3, run after ~13:00 when the D+1 auction prices are published):
  python run_pipeline.py --exog --nwp --decision-lag-hours 33 --target-mode diff_sameh
      -> models/lgbm_ex_nwp_tdiff_sameh_dl33, data/forecast_lgbm_ex_nwp_tdiff_sameh_dl33.csv

  Same, fresh data but no retraining (e.g. daily, with weekly retraining):
  python run_pipeline.py --exog --nwp --decision-lag-hours 33 --target-mode diff_sameh --skip-train

  Backtest of the production setup on existing data (no downloads; one command):
  python run_pipeline.py --exog --nwp --decision-lag-hours 33 --target-mode diff_sameh --skip-fetch --backtest --backtest-start "2026-01-01 00:00" --backtest-end "2026-08-31 00:00" --backtest-origin-hour 23

Other setups:
  python run_pipeline.py                  price only, D+1 before the auction (fallback, models/lgbm)
  python run_pipeline.py --exog           D+1 before the auction (run before ~12:00) with ENTSO-E
                                          wind/solar/load forecasts and neighbour prices
  python run_pipeline.py --weather        observed Oslo weather (Frost) for training, latest MET
                                          forecast snapshot for forecasting

Downloads:
  --skip-fetch          no downloads at all (existing files)
  --skip-price-fetch    skip only the prices
  --skip-exog-fetch     skip only the ENTSO-E exog file
  --skip-nwp-fetch      skip only the Open-Meteo file
  --skip-weather-fetch  skip only the Frost history and MET snapshot
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, name):
    """Execute a command. Exits with error if it fails."""
    print(f"\n{'='*70}")
    print(f"Step: {name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        print(f"\n{'='*70}")
        print(f"FAILED: {name} (exit code {result.returncode})")
        print(f"{'='*70}")
        sys.exit(1)

    print(f"✓ {name} completed")
    return result


def check_weather_freshness(price_parquet, weather_path, max_gap_hours=24):
    """Warn if the weather history ends well before the price data (those rows train with NaN weather)."""
    import pandas as pd
    sys.path.insert(0, str(repo_root / "src"))
    from price_io import load_price_series
    from config import TZ
    price_end = load_price_series(price_parquet, tz=TZ).dropna().index[-1]
    w = pd.read_parquet(weather_path)
    w_end = w.index.tz_localize(TZ) if w.index.tz is None else w.index.tz_convert(TZ)
    gap = (price_end - w_end.max()) / pd.Timedelta(hours=1)
    print(f"Weather history ends {w_end.max()}; prices end {price_end}")
    if gap > max_gap_hours:
        print(f"WARNING: weather history is {gap:.0f}h older than the prices; rows after it train with NaN weather")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--zone",
        default="NO_1",
        help="Price zone (default: NO_1)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=48,
        help="Forecast horizon in hours (default: 48)"
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=30,
        help="Validation window in days (default: 30)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip all downloads: prices and, with --weather, weather history and forecast snapshot "
             "(use the existing files)"
    )
    parser.add_argument(
        "--skip-price-fetch",
        action="store_true",
        help="Skip only the price download (use existing data/entsoe_*.parquet); weather is still fetched"
    )
    parser.add_argument(
        "--skip-weather-fetch",
        action="store_true",
        help="Skip only the weather downloads (history and forecast snapshot); prices are still fetched"
    )
    parser.add_argument(
        "--weather",
        action="store_true",
        help="Add weather: fetch Frost history + MET forecast snapshot, train with weather, "
             "forecast from the snapshot. Models go to models/lgbm_wx. "
             "Needs FROST_CLIENT_ID in .env. --skip-fetch / --skip-weather-fetch skip the weather downloads."
    )
    parser.add_argument(
        "--exog",
        action="store_true",
        help="Add ENTSO-E exog features: day-ahead wind/solar forecasts of neighbouring zones and neighbour prices "
             "(fetch_entsoe_exog.py, needs ENTSOE_TOKEN). Combines with --weather. Models go to "
             "models/lgbm[_wx]_ex. --skip-fetch / --skip-exog-fetch skip the download."
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip feature build and training; forecast with the existing models in models/lgbm[_wx][_ex]. "
             "Fetching still runs, so this gives a fresh forecast from updated prices/weather/exog"
    )
    parser.add_argument(
        "--nwp",
        action="store_true",
        help="Add weather-model forecasts (fetch_openmeteo.py -> data/nwp_openmeteo.parquet). Models go to "
             "models/lgbm[_wx][_ex]_nwp. --skip-fetch / --skip-nwp-fetch skip the download."
    )
    parser.add_argument(
        "--skip-nwp-fetch",
        action="store_true",
        help="Skip only the Open-Meteo download (use the existing --nwp-file)"
    )
    parser.add_argument(
        "--nwp-file",
        default="data/nwp_openmeteo.parquet",
        help="Open-Meteo NWP parquet (default: data/nwp_openmeteo.parquet)"
    )
    parser.add_argument(
        "--decision-lag-hours",
        type=int,
        default=11,
        help="Hours from decision time to origin. 11 (default): forecast tomorrow before the auction. "
             "33: forecast D+2/D+3 at ~14:00 when tomorrow's prices are known (model dir suffix _dl33)"
    )
    parser.add_argument(
        "--target-mode",
        default="level",
        choices=["level", "diff_sameh", "diff_mean24", "rel_mean24"],
        help="Forecast the price level (default) or the change from a known anchor (see src/targets.py). "
             "Non-default modes add the model dir suffix _t<mode>"
    )
    parser.add_argument(
        "--skip-exog-fetch",
        action="store_true",
        help="Skip only the ENTSO-E exog download (use the existing --exog-file)"
    )
    parser.add_argument(
        "--exog-file",
        default="data/entsoe_exog.parquet",
        help="ENTSO-E exog parquet (default: data/entsoe_exog.parquet)"
    )
    parser.add_argument(
        "--weather-file",
        default="data/weather_oslo.parquet",
        help="Observed weather parquet used for training (default: data/weather_oslo.parquet)"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run rolling-origin backtest after forecast"
    )
    parser.add_argument(
        "--backtest-weather-mode",
        choices=["observed", "archive"],
        default="observed",
        help="With --weather: predict with observed weather (upper bound) or archived MET snapshots (realistic)"
    )
    parser.add_argument(
        "--backtest-origin-hour",
        type=int,
        default=None,
        help="Local clock hour of each backtest origin (= last known price hour). 23 = forecast made ~12:00 "
             "the same day, before tomorrow's prices are published. Default: 48 absolute hours apart"
    )
    parser.add_argument(
        "--backtest-snapshot-lag",
        type=int,
        default=0,
        help="With --backtest-weather-mode archive: snapshots must be issued this many hours before the origin "
             "(11 for --backtest-origin-hour 23)"
    )
    parser.add_argument(
        "--backtest-start",
        help="Backtest start date (required if --backtest; e.g., '2026-01-01 00:00')"
    )
    parser.add_argument(
        "--backtest-end",
        help="Backtest end date (required if --backtest; e.g., '2026-08-31 00:00')"
    )

    args = parser.parse_args()

    # Validate backtest arguments
    if args.backtest:
        if not args.backtest_start or not args.backtest_end:
            parser.error("--backtest requires both --backtest-start and --backtest-end")

    # Paths
    global repo_root
    repo_root = Path(__file__).parent

    price_csv = repo_root / f"data/entsoe_{args.zone}_2020-01-01_now.csv"
    price_parquet = repo_root / f"data/entsoe_{args.zone}_2020-01-01_now.parquet"
    features_file = repo_root / "data/features_price_only.parquet"

    print(f"\n{'='*70}")
    print(f"Energy Forecast Pipeline")
    print(f"{'='*70}")
    print("Model: lgbm")
    print(f"Zone: {args.zone}")
    print(f"Horizon: {args.horizon}h")
    print(f"Validation: {args.val_days} days")
    print(f"Weather: {'ON (' + args.weather_file + ')' if args.weather else 'off'}")
    print(f"ENTSO-E exog: {'ON (' + args.exog_file + ')' if args.exog else 'off'}")
    if args.backtest:
        print(f"Backtest: {args.backtest_start} to {args.backtest_end}")
    print()

    try:
        # Step 1: Fetch prices
        skip_prices = args.skip_fetch or args.skip_price_fetch
        skip_weather = args.skip_fetch or args.skip_weather_fetch
        if skip_prices:
            print("Step 1/4: Skipping price fetch (--skip-fetch / --skip-price-fetch)")
            if not price_parquet.exists():
                raise FileNotFoundError(f"Skip requested but file not found: {price_parquet}")
        else:
            run_command(
                [
                    sys.executable, "src/fetch_prices.py",
                    "--zone", args.zone,
                    "--start", "2020-01-01",
                    "--out", str(price_csv)
                ],
                f"1/4: Fetch prices ({args.zone})"
            )
            if not price_parquet.exists():
                raise FileNotFoundError(f"fetch_prices.py succeeded but {price_parquet} not found")

        # Step 1b: Weather (optional)
        weather_path = repo_root / args.weather_file
        if args.weather:
            if skip_weather:
                print("Step 1b: Skipping weather downloads (--skip-fetch / --skip-weather-fetch)")
                if not weather_path.exists():
                    raise FileNotFoundError(f"--weather needs {weather_path}; run without --skip-fetch / --skip-weather-fetch first")
            else:
                run_command(
                    [sys.executable, "src/fetch_weather.py", "--out", str(weather_path)],
                    "1b: Fetch weather history (Frost)"
                )
                # Snapshot is best-effort: forecast.py falls back to observations (with a warning).
                snap = subprocess.run([sys.executable, "src/fetch_weather_forecast.py"], cwd=repo_root)
                if snap.returncode != 0:
                    print("WARNING: forecast snapshot failed; forecast will use the latest earlier snapshot, if any")
            if not args.skip_train:
                check_weather_freshness(price_parquet, weather_path)

        # Step 1c: ENTSO-E exog (optional)
        exog_path = repo_root / args.exog_file
        if args.exog:
            if args.skip_fetch or args.skip_exog_fetch:
                print("Step 1c: Skipping ENTSO-E exog download (--skip-fetch / --skip-exog-fetch)")
                if not exog_path.exists():
                    raise FileNotFoundError(f"--exog needs {exog_path}; run without --skip-fetch / --skip-exog-fetch first")
            else:
                run_command(
                    [sys.executable, "src/fetch_entsoe_exog.py", "--out", str(exog_path)],
                    "1c: Fetch ENTSO-E wind/solar forecasts and neighbour prices"
                )

        # Step 1e: EUR/NOK from Norges Bank (best effort; forecast.py falls back to the fixed rate)
        if not args.skip_fetch:
            fxr = subprocess.run([sys.executable, "src/fx.py"], cwd=repo_root)
            if fxr.returncode != 0:
                print("WARNING: EUR/NOK fetch failed; the forecast uses the last stored or the fixed rate")

        # Step 1d: Open-Meteo weather-model forecasts (optional)
        nwp_path = repo_root / args.nwp_file
        if args.nwp:
            if args.skip_fetch or args.skip_nwp_fetch:
                print("Step 1d: Skipping Open-Meteo download (--skip-fetch / --skip-nwp-fetch)")
                if not nwp_path.exists():
                    raise FileNotFoundError(f"--nwp needs {nwp_path}; run without --skip-fetch / --skip-nwp-fetch first")
            else:
                run_command(
                    [sys.executable, "src/fetch_openmeteo.py", "--out", str(nwp_path)],
                    "1d: Fetch weather-model forecasts (Open-Meteo)"
                )

        # Step 2: Build features
        if args.skip_train:
            print("Steps 2-3/4: Skipping feature build and training (--skip-train)")
        else:
            run_command(
                [
                    sys.executable, "src/build_features.py",
                    "--inp", str(price_parquet),
                    "--out", str(features_file),
                    "--horizon", str(args.horizon)
                ],
                "2/4: Build features"
            )

        # Step 3: Train
        train_cmd = [
            sys.executable, "src/train.py",
            "--inp", str(features_file),
            "--horizon", str(args.horizon),
            "--val_days", str(args.val_days)
        ]

        # Models with extra features are kept apart from the price-only models.
        suffix = ("_wx" if args.weather else "") + ("_ex" if args.exog else "") + ("_nwp" if args.nwp else "")
        if args.target_mode != "level":
            suffix += f"_t{args.target_mode}"
        if args.decision_lag_hours != 11:
            suffix += f"_dl{args.decision_lag_hours}"
        model_dir = repo_root / f"models/lgbm{suffix}"
        train_cmd.extend(["--out_dir", str(model_dir), "--decision_lag_hours", str(args.decision_lag_hours),
                          "--target_mode", args.target_mode])
        if args.weather:
            train_cmd.extend(["--weather_file", args.weather_file])
        if args.exog:
            train_cmd.extend(["--exog_file", args.exog_file])
        if args.nwp:
            train_cmd.extend(["--nwp_file", args.nwp_file])

        if not args.skip_train:
            run_command(train_cmd, f"3/4: Train ({model_dir.name})")

        if args.skip_train and not (model_dir / "manifest.json").exists():
            raise FileNotFoundError(f"--skip-train needs trained models in {model_dir}; run once without --skip-train")

        # Step 4: Forecast
        run_command(
            [
                sys.executable, "src/forecast.py",
                "--price", str(price_parquet),
                "--model_dir", str(model_dir),
                "--horizon", str(args.horizon),
                "--save_png"
            ],
            f"4/4: Forecast ({args.horizon}h)"
        )

        # Step 5: Backtest (optional)
        if args.backtest:
            backtest_cmd = [
                sys.executable, "src/backtest_rolling.py",
                "--inp", str(price_parquet),
                "--start", args.backtest_start,
                "--end", args.backtest_end,
                "--step", str(args.horizon),
                "--horizon", str(args.horizon),
                "--decision_lag_hours", str(args.decision_lag_hours),
                "--target_mode", args.target_mode
            ]

            if args.backtest_origin_hour is not None:
                backtest_cmd.extend(["--origin_hour", str(args.backtest_origin_hour),
                                     "--snapshot_lag_hours", str(args.backtest_snapshot_lag)])
            backtest_name = "lgbm"
            if args.weather:
                backtest_cmd.extend(["--weather_file", args.weather_file,
                                     "--weather_mode", args.backtest_weather_mode])
                backtest_name += "_wx" if args.backtest_weather_mode == "observed" else "_wxfc"
            if args.exog:
                backtest_cmd.extend(["--exog_file", args.exog_file])
                backtest_name += "_ex"
            if args.nwp:
                backtest_cmd.extend(["--nwp_file", args.nwp_file])
                backtest_name += "_nwp"
            if args.target_mode != "level":
                backtest_name += f"_t{args.target_mode}"
            if args.decision_lag_hours != 11:
                backtest_name += f"_dl{args.decision_lag_hours}"
            if args.backtest_origin_hour is not None:
                backtest_name += f"_o{args.backtest_origin_hour}"

            run_command(backtest_cmd, "5: Backtest (rolling-origin)")

        # Summary
        print(f"\n{'='*70}")
        print("✓ PIPELINE COMPLETED")
        print(f"{'='*70}")
        print(f"\nOutputs:")
        print(f"  Features:   {features_file}")
        print(f"  Model:      {model_dir}/")
        if args.weather:
            print(f"  Weather:    {weather_path} (+ snapshots in data/forecast_archive/)")
        if args.exog:
            print(f"  ENTSO-E:    {exog_path}")
        if args.nwp:
            print(f"  NWP:        {nwp_path}")
        print(f"  Forecast:   data/forecast_{model_dir.name}.csv")
        print(f"  Figures:    figs/")
        if args.backtest:
            print(f"  Backtest:   data/backtest_rolling_{backtest_name}.csv")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ PIPELINE FAILED")
        print(f"{'='*70}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
