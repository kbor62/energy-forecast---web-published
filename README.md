===============================================================================
 ENERGY FORECAST - ELECTRICITY PRICE FORECAST FOR NO1 (NORWAY)
 Setup, operation and maintenance
===============================================================================

This file describes how to set up the solution, run it, schedule it and keep it
working over time. All commands are run from the repository root with the
virtual environment activated. "python" means the venv's python
(.\.venv\Scripts\python.exe on Windows).

Contents
  1. Overview
  2. Setup
  3. Running the forecast
  4. Scheduled operation
  5. Data sources and downloads
  6. Model configuration
  7. Evaluating changes (backtest and analysis tools)
  8. Automated tests
  9. Maintenance tasks
 10. Troubleshooting
 11. Known limits
 12. Repository layout and legacy files


-------------------------------------------------------------------------------
1. OVERVIEW
-------------------------------------------------------------------------------

The solution forecasts hourly day-ahead electricity prices for bidding zone
NO1 with LightGBM (one model per forecast horizon, 48 horizons).

Day-ahead prices for D+1 are published by the auction around 12:45-13:00 on
day D, so D+1 is known once the forecast runs in the afternoon. The production
setup therefore forecasts D+2 and D+3 (horizons 1-24 and 25-48 after the last
known price hour), using:
  * price history of NO1 (ENTSO-E)
  * calendar features at the target hour (hour, weekday, month, holiday)
  * weather-model forecasts for D+2/D+3 (Open-Meteo, ECMWF IFS)
  * neighbour-zone prices up to the last known hour (ENTSO-E)
and predicts the change from the price at the same clock hour on the last
known day (target mode diff_sameh), converted back to EUR/kWh and NOK/kWh.

Pipeline steps (all wrapped by run_pipeline.py):
  fetch data -> build features -> train -> forecast -> (optional) backtest

Generated files (data\, models\, figs\) are not versioned; only source code is
in git.


-------------------------------------------------------------------------------
2. SETUP
-------------------------------------------------------------------------------

2.1 Python environment
    Python 3.14 or newer. Create the venv and install dependencies:

      python -m venv .venv
      .\.venv\Scripts\python.exe -m pip install -r requirements.txt

    requirements.txt pins pandas, numpy, scikit-learn and lightgbm. The other
    packages are not version-sensitive. Saved .joblib models can only be loaded
    with the scikit-learn version they were trained with (see 9.2).

2.2 API keys (.env)
    Keys are read from .env in the repository root, never from source code.

      copy .env.example .env
      (then edit .env)

      ENTSOE_TOKEN=<token>        required: prices, neighbour prices, wind/solar/load
                                  (free, https://transparency.entsoe.eu/ ->
                                  account settings -> Web API security token)
      FROST_CLIENT_ID=<id>        only for the optional observed-weather variant
                                  (https://frost.met.no/auth/requestCredentials.html)

    Open-Meteo, MET (api.met.no) and Norges Bank need no key.
    .env is git-ignored. Never commit it.

2.3 Configuration (src\config.py)
      EUR_TO_NOK  fallback EUR/NOK rate. Used for NOK figures when no Norges
                  Bank rate newer than 7 days is stored, and as the fixed rate
                  for backtest/EDA/feature-importance figures (see 5.6).
      TZ          "Europe/Oslo"

2.4 First run
    The first run downloads the full history from 2020 onwards for every
    source and takes a while. Later runs only fetch the tail (section 5.7).

      python run_pipeline.py --exog --nwp --decision-lag-hours 33 --target-mode diff_sameh

    Then check that all tests pass (section 8).


-------------------------------------------------------------------------------
3. RUNNING THE FORECAST
-------------------------------------------------------------------------------

3.1 Production setup (D+2/D+3, run after ~13:00)

      python run_pipeline.py --exog --nwp --decision-lag-hours 33 --target-mode diff_sameh

    This fetches prices, EUR/NOK, ENTSO-E exog data and Open-Meteo forecasts,
    rebuilds features, retrains and writes the forecast.

      Model:     models\lgbm_ex_nwp_tdiff_sameh_dl33\
      Forecast:  data\forecast_lgbm_ex_nwp_tdiff_sameh_dl33.csv / .json
      Plot:      figs\forecast_lgbm_ex_nwp_tdiff_sameh_dl33_<origin>.png

    Run it after the D+1 auction results are published (about 13:00), so the
    last known price hour is 23:00 on D+1.

3.2 Forecast without retraining
    Fetches fresh data and forecasts with the existing models:

      python run_pipeline.py --exog --nwp --decision-lag-hours 33 --target-mode diff_sameh --skip-train

    The options must match the ones the models were trained with (they select
    the model directory). It fails if the model directory has no manifest.

3.3 Other setups
      python run_pipeline.py
          price-only model (models\lgbm), forecast before the auction (D+1).
          Simplest fallback; needs only ENTSOE_TOKEN.
      python run_pipeline.py --exog
          D+1 before the auction (run before ~12:00) with ENTSO-E wind/solar/
          load forecasts and neighbour prices.
      python run_pipeline.py --weather
          adds observed Oslo weather (Frost) for training and MET forecast
          snapshots for forecasting (section 6.4).

    Download switches: --skip-fetch (no downloads), --skip-price-fetch,
    --skip-weather-fetch, --skip-exog-fetch, --skip-nwp-fetch.
    Run python run_pipeline.py --help for all options.

3.4 Running the steps by hand
    Each step of the wrapper can be run on its own (here: price-only model):

      python src/fetch_prices.py --zone NO_1 --start 2020-01-01
      python src/build_features.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --out data/features_price_only.parquet --horizon 48
      python src/train.py --inp data/features_price_only.parquet --horizon 48 --val_days 30
      python src/forecast.py --price data/entsoe_NO_1_2020-01-01_now.parquet --model_dir models/lgbm --horizon 48 --save_png

    Extra inputs are added with --exog_file, --nwp_file, --weather_file,
    --decision_lag_hours and --target_mode on train.py (and backtest_rolling.py).
    forecast.py reads these settings from the model manifest.

3.5 Output
    forecast_<model>.csv   one row per target hour: price in EUR/kWh and NOK/kWh
                           with prediction interval
    forecast_<model>.json  the same plus metadata: origin, decision time,
                           EUR/NOK rate and its source, weather/exog coverage
    The plot shows the last known prices and the forecast, with the EUR/NOK rate
    used noted under the plot.


-------------------------------------------------------------------------------
4. SCHEDULED OPERATION
-------------------------------------------------------------------------------

Nothing runs by itself. Use Windows Task Scheduler (or cron/GitHub Actions).

4.1 Daily forecast
      Program:    <repo>\.venv\Scripts\python.exe
      Arguments:  run_pipeline.py --exog --nwp --decision-lag-hours 33 --target-mode diff_sameh
      Start in:   <repo>
      Trigger:    daily, about 13:30 (after the D+1 auction results)

    Each run retrains on all data (a few minutes). To save time, retrain
    weekly and add --skip-train to the daily runs.

4.2 MET forecast snapshots (only for the --weather variant)
    The MET API serves only the current forecast; past forecasts cannot be
    fetched later. To build an archive for realistic weather backtests,
    schedule every 6 hours:

      Arguments:  src\fetch_weather_forecast.py

    Each run stores data\forecast_archive\metno_<issued>.parquet (about 20 KB),
    named by MET's model-run time; the same run is never stored twice.

4.3 Check after each run
    * The wrapper exits with an error if a required step fails; a failed
      optional download (EUR/NOK, MET snapshot) only warns.
    * forecast_<model>.json shows the origin; it should be 23:00 of the
      latest published day.
    * data\*_meta.json files record the last timestamp of every download.


-------------------------------------------------------------------------------
5. DATA SOURCES AND DOWNLOADS
-------------------------------------------------------------------------------

  script                        source                 output (data\)              key
  src\fetch_prices.py           ENTSO-E A44, NO1       entsoe_NO_1_2020-01-01_now  ENTSOE_TOKEN
  src\fetch_entsoe_exog.py      ENTSO-E A44/A65/A69    entsoe_exog.parquet         ENTSOE_TOKEN
  src\fetch_openmeteo.py        Open-Meteo             nwp_openmeteo.parquet       -
  src\fx.py                     Norges Bank            eurnok.parquet              -
  src\fetch_weather.py          MET Frost (Blindern)   weather_oslo.parquet        FROST_CLIENT_ID
  src\fetch_weather_forecast.py MET Locationforecast   weather_forecast.parquet    -
                                                       + forecast_archive\

Every download also writes a _meta.json with the range, row count and fetch
time.

5.1 Prices (fetch_prices.py)
    NO1 day-ahead prices from 2020-01-01 to now + 2 days, in EUR/MWh,
    aggregated to hourly (15-minute prices since Oct 2025 are averaged per
    hour; up to a few missing hours are filled). Stored as .parquet and .csv.

5.2 ENTSO-E exogenous data (fetch_entsoe_exog.py)
    Column prefixes:
      wind_<ZONE>_MW, solar_<ZONE>_MW  day-ahead wind/solar forecast (A69)
                                       default zones DE_LU, DK_1, DK_2, SE_3
      load_<ZONE>_MW                   day-ahead load forecast (A65), NO_1
      price_<ZONE>_EUR_MWh             neighbour day-ahead prices (A44)
                                       default zones NO_2, SE_3, DK_1, DE_LU
    Change zones with --wind_zones, --load_zones, --price_zones (entsoe-py area
    codes). A new zone is fetched in full; existing zones incrementally. A zone
    that fails is skipped with a warning; the run fails only if nothing arrives.

5.3 Weather-model forecasts (fetch_openmeteo.py)
    Open-Meteo Previous Runs API, model ecmwf_ifs025 (--model). For each valid
    hour and lead N = 1..4 days: the value forecast N x 24 h earlier. Points are
    averaged per area, columns <base>_d<N>:
      nwp_wind_DE   wind speed 100 m   Schleswig-Holstein, Lower Saxony coast, Mecklenburg
      nwp_wind_DK   wind speed 100 m   West Jutland, Zealand
      nwp_wind_SE3  wind speed 100 m   Lake Vanern area
      nwp_solar_DE  shortwave rad.     Bavaria, Baden-Wurttemberg, central Germany
      nwp_temp_NO1  temperature 2 m    Oslo, Hamar
    The archive starts in 2024. Earlier hours are filled with ERA5 reanalysis
    (same points, same value for every lead); without the fill the model learns
    "weather present" as a time indicator. --no_reanalysis disables the fill.
    The end defaults to today + 4 days so the live forecast has its leads.
    Free for non-commercial use.

5.4 Observed weather (fetch_weather.py, optional)
    Hourly temperature, wind speed and precipitation at Oslo-Blindern
    (SN18700) from Frost. Keep it as recent as the price file; target hours
    after its end train with NaN weather.

5.5 MET forecast snapshots (fetch_weather_forecast.py, optional)
    Current MET forecast for Oslo, Hamburg and Copenhagen. Overwrites
    weather_forecast.parquet and adds a file to data\forecast_archive\.
    Options: --archive_dir, --no_archive.

5.6 EUR/NOK (fx.py)
    Daily rate, series EXR/B.EUR.NOK.SP, business days. run_pipeline.py fetches
    it on every run unless --skip-fetch; a failed fetch only warns.
      * forecast.py uses the latest rate on or before the forecast date and
        records it in the JSON (eur_nok, eur_nok_source). If no rate newer than
        7 days is stored it falls back to config.EUR_TO_NOK with a warning.
      * backtest_rolling.py, eda.py and feature_importance.py report NOK at the
        fixed config.EUR_TO_NOK, so evaluation figures stay on one scale.
        Backtest rows also carry the daily rate in column eur_nok.
      * Models work in EUR; the rate never affects training.
    python src/fx.py --full re-fetches from --start (default 2020-01-01).

5.7 Incremental updates (src\incremental.py)
    fetch_prices.py, fetch_weather.py, fetch_entsoe_exog.py, fetch_openmeteo.py
    and fx.py behave the same way:
      * Output file missing: the full history from --start is fetched.
      * Output file present: only the tail is fetched, from the last stored
        timestamp minus --overlap_days (default 3), and merged. Values in the
        overlap are replaced (late or revised data); older rows are kept.
      * Full re-fetch automatically when the stored data starts more than
        7 days after --start, and always with --full (for the exog file this is
        decided per zone).
    Use --full after changing zone, station or start date with an existing
    output file, otherwise old and new data are mixed.
    fetch_weather_forecast.py is not incremental: each run is one snapshot.


-------------------------------------------------------------------------------
6. MODEL CONFIGURATION
-------------------------------------------------------------------------------

6.1 Model
    LightGBM, one model per horizon (1..48), trained in EUR/kWh. Validation =
    the last 30 days (--val_days) with an h-hour embargo; early stopping after
    100 rounds on a 10 % time-ordered tail; seed 42. Each model directory has a
    manifest.json and per-horizon *_meta.json (features, settings, library
    versions) and training_summary.csv.

6.2 Feature set (src\features.py)
    Calendar, at the target hour t+h (Europe/Oslo):
      hour, dow     categorical with fixed levels 0-23 / 0-6
      month         integer 1-12
      is_holiday    Norwegian public holidays (holidays package, Sundays not
                    counted; Christmas Eve, New Year's Eve and bridge days are
                    not flagged)
    Price history at the origin t: lag0h, lags 1-168 h, rolling mean/std over
    6-168 h, differences over 24 and 168 h.
    Changing the feature list requires retraining: forecast.py stops with an
    error if the features it builds differ from the model's saved list.

6.3 Optional inputs (train.py / backtest_rolling.py; run_pipeline.py flag)
      --exog_file   (--exog)     ENTSO-E data (5.2). Wind/solar/load at the
                                 target hour; neighbour prices only at the
                                 origin and 24 h before (a neighbour price at
                                 the target hour is set in the same auction and
                                 is unknown). --exog_cols selects columns.
      --nwp_file    (--nwp)      Open-Meteo forecasts (5.3). For target t the
                                 newest run available at decision time is used:
                                 cutoff = origin - decision_lag - 6 h,
                                 lead N = ceil((t - cutoff) / 24 h), NaN if N > 4.
      --weather_file (--weather) observed Oslo weather (6.4).
    Inputs can be combined. LightGBM handles NaN.

6.4 Observed weather variant (--weather)
    Trains on OBSERVED weather at t+h and forecasts with the latest MET
    snapshot (falls back to observations for hours the snapshot does not cover).
    Training on observations and serving forecasts is a mismatch, so backtests
    with observed weather overstate the benefit. Realistic backtests need
    --weather_mode archive (only origins with an archived snapshot).
    --weather_cols selects columns (default temperature_C wind_speed_ms
    precipitation_mm).

6.5 Decision time (--decision_lag_hours; --decision-lag-hours)
    Hours between the decision time and the origin (= last known price hour).
      11 (default)  origin 23:00 on D, decision 12:00 on D: forecast D+1 before
                    the auction.
      33            origin 23:00 on D+1, decision 14:00 on D: horizons 1-24 = D+2,
                    25-48 = D+3 (production setup).
    It controls which ENTSO-E values count as published (wind/solar/load for a
    target date are assumed available from 12:00 the day before; later hours
    are NaN) and which NWP lead is used. Stored in the manifest; forecast.py
    reads it.

6.6 Target mode (--target_mode; --target-mode; src\targets.py)
      level        price(t)                                   (default)
      diff_sameh   price(t) - price at the same clock hour on the last known day
                   (production setup)
      diff_mean24  price(t) - mean of the 24 h up to the origin
      rel_mean24   (price(t) - mean24) / max(mean |price| over 168 h, 0.01)
    Anchors use only prices at or before the origin. Predictions are converted
    back to EUR/kWh before metrics, intervals and output.

6.7 Model directory names
    run_pipeline.py builds the directory name from the options, in this order:
      models\lgbm[_wx][_ex][_nwp][_t<mode>][_dl<lag>]
    _t<mode> is omitted for level, _dl<lag> for 11. Backtest outputs use the
    same suffixes plus _o<hour> for --origin_hour.


-------------------------------------------------------------------------------
7. EVALUATING CHANGES (BACKTEST AND ANALYSIS TOOLS)
-------------------------------------------------------------------------------

Use these tools before changing the production setup (new features, zones,
parameters, library upgrades). Always compare variants on the same price file
and the same origins; do not download new data between the runs being compared.

7.1 Rolling-origin backtest (src\backtest_rolling.py)
    Refits every 720 h on data before each origin (minimum 8,760 training
    hours, h-hour embargo) and forecasts 48 h ahead. Example matching the
    production setup:

      python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48 --origin_hour 23 --decision_lag_hours 33 --exog_file data/entsoe_exog.parquet --exog_cols price_NO_2_EUR_MWh price_SE_3_EUR_MWh price_DK_1_EUR_MWh price_DE_LU_EUR_MWh --nwp_file data/nwp_openmeteo.parquet --target_mode diff_sameh

    --origin_hour fixes the local clock hour of every origin (--step must then
    be a multiple of 24; DST-ambiguous hours are skipped). Without it, origins
    are --step absolute hours apart and drift with DST.
    --weather_mode observed|archive (with --weather_file); in archive mode
    --snapshot_lag_hours keeps snapshots issued after the decision time out.

    Output: data\backtest_rolling_<model>.csv, _by_horizon.csv, _meta.json and
    figs\backtest_rolling_<model>.png. The console prints MAE/RMSE/nMAE for the
    model and two baselines (persistence and weekly t-168).

    Via the wrapper: run_pipeline.py ... --backtest --backtest-start ...
    --backtest-end ... [--backtest-origin-hour 23] [--backtest-weather-mode ...]
    [--backtest-snapshot-lag N].

7.2 Comparing backtests
      python src/backtest_rolling.py --compare data/backtest_rolling_A.csv data/backtest_rolling_B.csv
          MAE by horizon, figs\compare_by_horizon.png (common origins only)
      python src/stat_tests.py A.csv B.csv [C.csv ...] --baselines [--horizons 25-48] --out_csv data/stat_tests.csv
          Friedman / Iman-Davenport and pairwise Wilcoxon with Holm correction
      python src/weather_block_test.py base.csv extended.csv --out_csv data/block_test.csv
          paired per-origin difference by clock-hour block, block-bootstrap CI;
          works for any two backtests (base and extended)

7.3 Feature importance (src\feature_importance.py)
      python src/feature_importance.py --model_dir models/lgbm
    Permutation importance on each model's validation window, horizons
    1 6 12 24 36 48, 10 repeats. Output data\feature_importance_*.csv and
    figs\importance\. Price-only models only.

7.4 Exploratory data analysis (src\eda.py)
      python src/eda.py
    Time series, distribution, hour/month profiles, heatmap, ACF and spikes
    (|z| > 6 robust SD from a 168 h rolling median). Output figs\eda\ and
    data\eda_*.csv/.json.

7.5 Weather scripts in the repository root
      test_weather_forecast_value.py  quick A/B of observed weather on the last
                                      30 days (slow)
      test_weather_shifts.py          weather benefit split by size of weather
                                      change, from existing backtests
    Both use observed weather, i.e. an upper bound.


-------------------------------------------------------------------------------
8. AUTOMATED TESTS
-------------------------------------------------------------------------------

      python tests/test_pipeline.py        (or: pytest tests/)

All tests must pass. They cover feature alignment and the holiday flag,
no look-ahead (perturbing unpublished values or later prices leaves features
unchanged) for weather, ENTSO-E and NWP inputs, snapshot choice, origin clock
time, the incremental merge, NWP lead choice and reanalysis fill, target-mode
round trips and the EUR/NOK lookup. They use synthetic data and need no API
keys or network. Run them after every code change and every library upgrade.


-------------------------------------------------------------------------------
9. MAINTENANCE TASKS
-------------------------------------------------------------------------------

9.1 Git
    Only source code is versioned. data\, models\, figs\ and .env are
    git-ignored. Do not commit generated files or keys.

9.2 Upgrading libraries
    1. Change the pins in requirements.txt and reinstall.
    2. Run the tests (section 8).
    3. Retrain all models in use: .joblib files cannot be loaded across
       scikit-learn versions, and LightGBM/pandas changes can shift results.
    4. Optionally rerun a backtest (7.1) on the same price file before and
       after to check that accuracy is unchanged.

9.3 Retraining
    run_pipeline.py retrains on every run unless --skip-train is given.
    Retrain at least weekly so recent price levels are in the training data,
    and always after changing features, options or libraries.

9.4 Backups
    data\, models\ and figs\ exist only on this machine. Everything except the
    MET snapshot archive (data\forecast_archive\) can be downloaded and
    rebuilt. Back up the archive if the --weather variant is used.

9.5 API keys
    If a download fails with an authentication error, renew the key and update
    .env (ENTSO-E token: account settings on transparency.entsoe.eu; Frost:
    frost.met.no).

9.6 Changing zones, stations or areas
    * Price zone: run_pipeline.py --zone <ZONE> (entsoe-py area code). Use a
      separate output file or --full (5.7).
    * ENTSO-E zones: fetch_entsoe_exog.py options (5.2); retrain afterwards.
    * Open-Meteo points/areas: defined in src\fetch_openmeteo.py; re-fetch with
      --full and retrain.

9.7 Changing the forecast logic
    Add or change features in src\features.py (calendar, price history),
    src\exog.py (ENTSO-E), src\nwp.py (Open-Meteo) or src\weather_io.py
    (observed weather and snapshots). Keep the no-look-ahead rules, add a test,
    evaluate with a backtest (section 7) and retrain.


-------------------------------------------------------------------------------
10. TROUBLESHOOTING
-------------------------------------------------------------------------------

  "ENTSOE_TOKEN is missing"
      .env is missing or has no token (2.2).
  forecast.py: feature list differs from the model
      The code or options changed after training. Retrain, or run with the same
      options the model was trained with.
  --skip-train needs trained models in models\...
      The option combination points to a directory that has not been trained.
      Run once without --skip-train.
  Error loading .joblib model
      scikit-learn version differs from training. Reinstall the pinned version
      or retrain (9.2).
  Forecast origin is a day older than expected
      The D+1 auction results were not yet published or not yet fetched. Run
      after ~13:00, or rerun the price fetch.
  Many NaN inputs reported in the forecast JSON
      A source is stale. Check the last timestamp in the corresponding
      data\*_meta.json and rerun its fetch (with --full if gaps are suspected).
  EUR/NOK fallback warning
      Norges Bank fetch failed or is older than 7 days. Run python src/fx.py
      and check data\eurnok_meta.json.


-------------------------------------------------------------------------------
11. KNOWN LIMITS
-------------------------------------------------------------------------------

  * Open-Meteo forecasts exist from 2024; earlier years use ERA5 reanalysis
    (actual weather, i.e. a perfect forecast), so the weather effect learned
    from those years is optimistic. One weather model (ECMWF IFS 0.25), a few
    points per area, wind speed rather than wind power.
  * Open-Meteo's "N x 24 h before valid time" is taken at face value; a 6 h
    margin covers run time and publication delay.
  * ENTSO-E publication time (about 12:00 the day before) is assumed, not
    given per value; stored wind/solar forecasts are the latest values, not an
    archive of vintages.
  * Prices known at the origin are assumed known up to 23:00 of the last
    published day; the exact publication minute is not modelled.
  * diff_sameh anchors on the last known day's profile and reacts slowly to
    sudden price rises.
  * Not included: reservoir filling (NVE), cross-border capacities and flows,
    generation outages.
  * Observed-weather variant: one station (Oslo) stands in for NO1; MET
    forecasts are hourly for about 60 h, then 6-hourly without precipitation.


-------------------------------------------------------------------------------
12. REPOSITORY LAYOUT AND LEGACY FILES
-------------------------------------------------------------------------------

  run_pipeline.py              wrapper for the whole pipeline
  src\config.py                EUR_TO_NOK fallback, time zone
  src\fetch_*.py, src\fx.py    downloads (section 5)
  src\incremental.py           shared incremental download logic
  src\price_io.py              price file loading and hourly aggregation
  src\build_features.py        price history features -> data\features_price_only.parquet
  src\features.py              feature assembly at the target hour
  src\exog.py, src\nwp.py,
  src\weather_io.py            optional inputs and their availability rules
  src\targets.py               target modes
  src\models.py, src\train.py  model definition and training
  src\forecast.py              forecast from a trained model directory
  src\backtest_rolling.py      rolling-origin backtest and comparison
  src\stat_tests.py,
  src\weather_block_test.py    statistical comparison of backtests
  src\feature_importance.py, src\eda.py, src\plotting.py, src\metrics.py
  tests\test_pipeline.py       automated tests

  Legacy, not used by the pipeline:
    build_features.py --add-weather-forecast and src\combine_weather.py: the
      weather columns they add are never read by train.py. Use
      train.py --weather_file instead.
    test_weather_hypothesis.py: deprecated; refuses to run without --force
      because it overwrites data\features_price_only.parquet.
    data\weather_combined.parquet, temperature_oslo.parquet,
      features_price_temp*, features_*weather* files.
