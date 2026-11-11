===============================================================================
 REPRODUCING ALL DATA, MODELS, FORECASTS AND FIGURES
 Electricity price forecasting for NO1 (Norway) - Kjetil Borg
 Price-only results as of 22 September 2026; weather extension added in
 November 2026 (sections 13-14)
===============================================================================

This file lists, step by step, how every data file, model, forecast, figure and
table of the project is produced, and how to produce them again. All commands
are run from the repository root in a terminal with the virtual environment
activated. Commands shown with "python" assume the venv's python
(.\.venv\Scripts\python.exe on Windows).

Contents
  0. What "reproduce" means here (two modes)
  0a. Convenience: wrapper script (optional)
  1. Environment
  2. Configuration
  3. Step 1 - Download prices
  4. Steps 2-5 - Features, training, forecast and backtest for each model
  5. Step 6 - Exploratory data analysis
  6. Step 7 - Model comparison across horizons
  7. Step 8 - Statistical tests
  8. Step 9 - Feature importance
  9. Step 10 - Automated tests
 10. Outputs and the step that produces them
 11. Reference values for checking a reproduction
 12. Numbers computed from the backtest CSVs
 13. Weather extension: observed weather, forecast snapshots, pipeline flag
 14. Weather backtest results (observed weather) and the per-block test
 15. Backtest origin clock time (--origin_hour)
 16. ENTSO-E exogenous features: wind/solar forecasts and neighbour prices

-------------------------------------------------------------------------------
0. WHAT "REPRODUCE" MEANS HERE (TWO MODES)
-------------------------------------------------------------------------------

Mode A - exact reproduction (recommended for checking a reproduction)
  Use the price file from 21 Sep 2026 (84,666 rows, 2020-01-01 00:00 to
  2026-09-22 23:45, fetched 21 Sep 2026 13:39). Results are not versioned
  (data\, models\ and figs\ are git-ignored), but this one file is still in the
  git history, commit 928e127. Restore it (this overwrites the current file; copy
  it away first if you want to keep it):

    git show 928e127:data/entsoe_NO_1_2020-01-01_now.parquet > data/entsoe_NO_1_2020-01-01_now.parquet

  Then skip step 1. Every later step is deterministic (fixed seed 42, fixed
  exchange rate), so the numbers in section 11 should be reproduced to the last
  printed decimal with the same library versions.

Mode B - fresh reproduction from the API
  Download the prices again (step 1). If you download without an end date you
  get newer data: the forecast origin moves, the 30-day validation window
  moves, and the models are trained on more hours, so forecasts and validation
  figures will differ. The backtest (Jan-Aug 2026) is still comparable but not
  bit-identical. To come as close as possible to Mode A, fetch with
  --end 2026-09-23 (see step 1). ENTSO-E can revise published data, so even
  then exact equality is not guaranteed.

Note (6 Nov 2026): the price file in the working tree was refreshed on 30 Oct
  2026 (88,414 rows to 2026-10-31 23:45). With that file, or any later download,
  the numbers in section 11 will differ (Mode B). Use the restored 21 Sep file
  (Mode A) for checking section 11.


-------------------------------------------------------------------------------

For a hands-off run of the complete pipeline (steps 1-5), use the Python
convenience wrapper:

  python run_pipeline.py                              # Run with defaults
  python run_pipeline.py --model ridge_ohe --encoding both
  python run_pipeline.py --skip-fetch --model lgbm
  python run_pipeline.py --skip-price-fetch --weather    # new weather, existing prices
  python run_pipeline.py --skip-weather-fetch --weather  # new prices, existing weather
  python run_pipeline.py --backtest --backtest-start "2026-01-01 00:00" --backtest-end "2026-08-31 00:00"
  python run_pipeline.py --weather                    # add weather (section 13)

With --weather the wrapper also fetches weather, trains with it and forecasts
from the latest forecast snapshot (section 13.6). Run `python run_pipeline.py
--help` for all options. This script is optional;
you can always run the steps individually as described below.


-------------------------------------------------------------------------------
1. ENVIRONMENT
-------------------------------------------------------------------------------

1.1 Python and libraries used for the results in section 11 (recorded in every
    model's *_meta.json):

      python        3.14.2
      pandas        3.0.6
      numpy         2.5.3
      scikit-learn  1.9.1
      lightgbm      4.7.0
      (plus pyarrow, matplotlib, joblib, scipy, python-dotenv, entsoe-py)

    requirements.txt pins these four versions; the other packages are not
    version-sensitive. Install with:

      python -m venv .venv
      .\.venv\Scripts\python.exe -m pip install -r requirements.txt

    Saved .joblib models can only be loaded with the same scikit-learn
    version they were trained with (1.9.x).

1.2 scipy is required by src/stat_tests.py and src/eda.py. It is installed
    automatically with scikit-learn.


-------------------------------------------------------------------------------
2. CONFIGURATION  (src\config.py)
-------------------------------------------------------------------------------

  EUR_TO_NOK   = 10.83     display rate of 18 Sep 2026. Only scales NOK
                           figures; models train on EUR/kWh.
  TZ           = "Europe/Oslo"

  Keep EUR_TO_NOK at 10.83 to reproduce the NOK numbers in section 11.

  The ENTSO-E API token is NOT stored in config.py. It is read from a .env
  file in the repository root (only needed for step 1):

      copy .env.example .env
      (then edit .env)   ENTSOE_TOKEN=<your own ENTSO-E API token>

  For the weather extension also set FROST_CLIENT_ID in .env (section 13.3).

  .env is git-ignored. Never commit it.

-------------------------------------------------------------------------------
3. STEP 1 - DOWNLOAD PRICES   (skip in Mode A)
-------------------------------------------------------------------------------

  python src/fetch_prices.py --zone NO_1 --start 2020-01-01 --end 2026-09-23

  Output:
    data\entsoe_NO_1_2020-01-01_now.csv
    data\entsoe_NO_1_2020-01-01_now.parquet
    data\entsoe_NO_1_2020-01-01_now_meta.json   (zone, range, fetch time, rows)

  Check: n_rows 84666, last_timestamp 2026-09-22 23:45:00+02:00.
  (Without --end the script fetches up to "now + 2 days".)

-------------------------------------------------------------------------------
4. STEPS 2-5 - FEATURES, TRAINING, FORECAST AND BACKTEST PER MODEL
-------------------------------------------------------------------------------

Run the steps in this order. Step 2 is run once; steps 3-5 are run once per
model variant.

  2. src/build_features.py  -> data\features_price_only.parquet (+ _meta.json)
  3. src/train.py           -> models\<model>_<encoding>\  48 models + manifest
                               (validation = last 30 days, h-hour embargo)
  4. src/forecast.py        -> data\forecast_<model>_<encoding>.csv/.json
                               figs\forecast_<model>_<encoding>_<origin>.png
  5. src/backtest_rolling.py
                            -> data\backtest_rolling_<model>_<encoding>.csv
                               data\backtest_rolling_<...>_by_horizon.csv
                               data\backtest_rolling_<...>_meta.json
                               figs\backtest_rolling_<model>_<encoding>.png

Step 2 - features (once):

  python src/build_features.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --out data/features_price_only.parquet --horizon 48

Steps 3-5 for the four variants compared in section 11:

  lgbm_int:
    python src/train.py --inp data/features_price_only.parquet --model lgbm --encoding int --horizon 48 --val_days 30
    python src/forecast.py --price data/entsoe_NO_1_2020-01-01_now.parquet --model_dir models/lgbm_int --horizon 48 --save_png
    python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model lgbm --encoding int --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48

  lgbm_trig:
    python src/train.py --inp data/features_price_only.parquet --model lgbm --encoding trig --horizon 48 --val_days 30
    python src/forecast.py --price data/entsoe_NO_1_2020-01-01_now.parquet --model_dir models/lgbm_trig --horizon 48 --save_png
    python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model lgbm --encoding trig --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48

  ridge_trig:
    python src/train.py --inp data/features_price_only.parquet --model ridge --encoding trig --horizon 48 --val_days 30
    python src/forecast.py --price data/entsoe_NO_1_2020-01-01_now.parquet --model_dir models/ridge_trig --horizon 48 --save_png
    python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model ridge --encoding trig --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48

  ridge_ohe_both:
    python src/train.py --inp data/features_price_only.parquet --model ridge_ohe --encoding both --horizon 48 --val_days 30
    python src/forecast.py --price data/entsoe_NO_1_2020-01-01_now.parquet --model_dir models/ridge_ohe_both --horizon 48 --save_png
    python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model ridge_ohe --encoding both --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48

Every variant must use the same price file. In Mode B, run step 1 first and do
not download again between variants.

The same steps can be run with the wrapper script run_pipeline.py.

Backtest settings used:
  origins 2026-01-01 00:00 .. 2026-08-29 01:00, step 48 h -> 121 origins
  horizons 1..48 -> 5,808 forecasts per model
  refit every 720 h, minimum 8,760 training hours, embargo h hours

Training settings:
  horizon 48, val_days 30 (validation starts 2026-08-23 22:00 in Mode A),
  Ridge alpha chosen by RidgeCV (logspace -3..3, TimeSeriesSplit 3),
  LightGBM early stopping 100 rounds on a 10 % time-ordered tail.


-------------------------------------------------------------------------------
5. STEP 6 - EXPLORATORY DATA ANALYSIS
-------------------------------------------------------------------------------

  python src/eda.py

  Defaults: whole period, spike rule |z| > 6 robust SD from a centred 168 h
  rolling median, ACF to 336 h.

  Output:
    figs\eda\eda_timeseries.png, eda_distribution.png, eda_by_hour.png,
            eda_by_month.png, eda_heatmap.png, eda_acf.png, eda_outliers.png
    data\eda_summary_by_year.csv   (annual statistics)
    data\eda_outliers.csv          (flagged hours)
    data\eda_meta.json             (settings and headline numbers)

  Only reads the price file; can be run any time after step 1.


-------------------------------------------------------------------------------
6. STEP 7 - MODEL COMPARISON ACROSS HORIZONS
-------------------------------------------------------------------------------

  python src/backtest_rolling.py --compare data/backtest_rolling_lgbm_int.csv data/backtest_rolling_ridge_ohe_both.csv data/backtest_rolling_ridge_trig.csv

  Output: figs\compare_by_horizon.png
  Prints aggregate MAE / nMAE per model and both baselines.


-------------------------------------------------------------------------------
7. STEP 8 - STATISTICAL TESTS (Friedman / Iman-Davenport, Wilcoxon + Holm)
-------------------------------------------------------------------------------

  All four variants plus both baselines, all horizons and per half:

  python src/stat_tests.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_trig.csv data/backtest_rolling_ridge_trig.csv data/backtest_rolling_ridge_ohe_both.csv --baselines --out_csv data/stat_tests_all.csv
  python src/stat_tests.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_trig.csv data/backtest_rolling_ridge_trig.csv data/backtest_rolling_ridge_ohe_both.csv --baselines --horizons 1-24 --out_csv data/stat_tests_1-24.csv
  python src/stat_tests.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_trig.csv data/backtest_rolling_ridge_trig.csv data/backtest_rolling_ridge_ohe_both.csv --baselines --horizons 25-48 --out_csv data/stat_tests_25-48.csv

  Calendar-encoding ablation (single planned test):

  python src/stat_tests.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_trig.csv --out_csv data/stat_tests_encoding.csv

  Ridge encodings:

  python src/stat_tests.py data/backtest_rolling_ridge_trig.csv data/backtest_rolling_ridge_ohe_both.csv

  Output: pairwise tables (CSV); Friedman ranks and p-values are printed to
  the console.


-------------------------------------------------------------------------------
8. STEP 9 - FEATURE IMPORTANCE
-------------------------------------------------------------------------------

  python src/feature_importance.py --model_dir models/lgbm_int models/ridge_ohe_both

  Defaults: horizons 1 6 12 24 36 48, 10 repeats, seed 42,
  evaluation window = each model's own validation window (val_start in the
  model metadata, 2026-08-23 22:00 in Mode A).

  Output:
    data\feature_importance_<model>.csv          per feature and horizon
    data\feature_importance_groups_<model>.csv   per feature group
    figs\importance\feature_importance_groups_<model>.png

  Requires the trained models from step 3.


-------------------------------------------------------------------------------
9. STEP 10 - AUTOMATED TESTS
-------------------------------------------------------------------------------

  python tests/test_pipeline.py        (or: pytest tests/)

  Expected: 27/27 passed (13 pipeline tests, 6 weather, 2 origin-time and 6 ENTSO-E exog tests; sections 13.8, 15 and 16.7).

-------------------------------------------------------------------------------
10. OUTPUTS AND THE STEP THAT PRODUCES THEM
-------------------------------------------------------------------------------

  Figures
    figs\eda\eda_timeseries.png, eda_heatmap.png, eda_acf.png,
        eda_distribution.png (also eda_by_hour.png, eda_by_month.png,
        eda_outliers.png)                                            step 6
    figs\compare_by_horizon.png                                      step 7
    figs\importance\feature_importance_groups_lgbm_int.png          step 9

  Tables
    data\entsoe_..._meta.json, features_price_only_meta.json   data and feature description
    data\eda_summary_by_year.csv                                annual statistics, step 6
    backtest console output / backtest_rolling_*.csv           overall results, step 5
    backtest_rolling_*_by_horizon.csv                          results by horizon, step 5
    models\<variant>\training_summary.csv                      mean over 48 horizons, step 3
    stat_tests.py console output + stat_tests_*.csv            statistical tests, step 8
    computed from backtest_rolling_*.csv (section 12)          decision test, step 5

-------------------------------------------------------------------------------
11. REFERENCE VALUES FOR CHECKING A REPRODUCTION (Mode A)
-------------------------------------------------------------------------------

Price data
  hourly observations after aggregation      58,967
  sub-hourly rows aggregated                  25,704
  missing hours filled                        4
  usable training rows (after 168 h warm-up)  58,798

Rolling backtest, 121 origins x 48 horizons, NOK/kWh (printed by each run)
  variant          MAE      RMSE     nMAE
  lgbm_int         0.1747   0.2617   15.8 %
  lgbm_trig        0.1798   0.2681   16.2 %
  ridge_trig       0.1777   0.2582   16.1 %
  ridge_ohe_both   0.1765   0.2555   15.9 %
  persistence      0.2253   0.3480   20.4 %
  weekly t-168     0.2469   0.3539   22.3 %

Statistical tests (all horizons, 4 variants + 2 baselines)
  Friedman chi2 = 42.19, Iman-Davenport F = 13.25
  mean ranks: lgbm_int 2.47, ridge_ohe_both 2.62, lgbm_trig 2.77,
              ridge_trig 2.88, weekly 4.63, persistence 5.64
  lgbm_int vs ridge_trig Holm p = 0.077; lgbm_int vs lgbm_trig unadjusted p = 0.221

LightGBM (int) early stopping, final models: 972 trees at 1 h, 552 at 6 h,
336 at 24 h, 230 at 48 h.

EDA headline: ACF 0.987 (1 h), 0.927 (24 h), 0.835 (168 h); 692 negative
hours; 805 upward and 675 downward spikes (k = 6).

Forecast origin in Mode A: 2026-09-22 23:00 (price 0.1474 EUR/kWh).


-------------------------------------------------------------------------------
12. NUMBERS COMPUTED FROM THE BACKTEST CSVs (not printed by a script)
-------------------------------------------------------------------------------

  - Decision test: for each origin, the 4 hours in horizons 25-48
    with the lowest predicted price; mean realised (actual) price of those
    hours, x10.83. Oracle = 4 lowest actual; weekly = 4 lowest
    baseline_weekly_t168; "no information" = mean of all 24 hours.
  - Spike analysis: hours with actual >= 95th percentile of actual.
  - 15-minute effect: |price at hh:00 - mean of hh:00..hh:45|, Oct 2025
    onwards, from the raw parquet.


-------------------------------------------------------------------------------
13. WEATHER EXTENSION (added 6 November 2026)
-------------------------------------------------------------------------------

Sections 1-12 describe the price-only models and are unchanged: without any of
the weather options everything below is inactive and the numbers
in section 11 are reproduced exactly as before. Weather is an optional add-on.

13.1 Idea and what the numbers mean
  A weather model gets, for target hour t+h, the temperature, wind speed and
  precipitation at t+h as extra features (columns <name>_target, e.g.
  temperature_C_target), next to the price features. Station: Oslo (Blindern,
  SN18700) for observations; Oslo coordinates for the MET forecast.

  Observed vs forecast weather - read this before quoting any result:
    * Training (train.py) and the default backtest use OBSERVED weather at t+h.
      That is what happened, not what was forecast, so the validation/backtest
      scores are an UPPER BOUND on the gain from weather. In a quick check
      (last 30 days, LightGBM) validation MAE improved 1-6 % at h = 1, 12, 24
      and 48 with observed weather (not a full backtest result).
    * In live use forecast.py has only a MET forecast. The model was trained on
      perfect weather and is served imperfect weather (train/serve mismatch).
    * The only honest measurement is to predict with the forecast that existed
      at each origin. That requires archived forecasts (13.5), so it can only
      cover the period since archiving started.

13.2 Files and directories
  src\weather_io.py          snapshot lookup and weather-at-target assembly
  src\fetch_weather.py       observed hourly weather (Frost) -> data\weather_oslo.parquet
  src\fetch_weather_forecast.py   MET forecast -> data\weather_forecast.parquet (latest,
                             overwritten) + data\forecast_archive\ (every run, kept)
  data\weather_oslo.parquet  observed: temperature_C, wind_speed_ms, precipitation_mm
                             (hourly, Europe/Oslo)
  data\forecast_archive\metno_<YYYYmmddTHHMMZ>.parquet
                             one MET model run as issued: index = valid time (UTC),
                             columns oslo_/hamburg_/copenhagen_ x {temperature_C,
                             wind_speed_ms, wind_from_direction, cloud_area_fraction,
                             precipitation_mm}, plus issued_at
  models\<model>_<enc>_wx\   weather-trained models (manifest has weather_file and
                             weather_cols)
  Not used by the pipeline: data\weather_combined.parquet, data\temperature_oslo.parquet,
  data\features_price_temp*, features_price_weather*, features_weather_forecast*,
  features_future_weather* (see 13.9).

13.3 Configuration
  Weather history needs a free Frost client id in .env:
      FROST_CLIENT_ID=<your id>      (https://frost.met.no/auth/requestCredentials.html)
  The MET forecast API needs no key (a User-Agent is set in the script). Extra
  packages: requests, python-dateutil (listed in requirements.txt).

13.4 Step by step
  a) Observed weather history (re-run to extend it; fetches 2020 onwards):
       python src/fetch_weather.py --out data/weather_oslo.parquet
     Keep it as new as the price file; rows after its end train with NaN weather.
  b) Forecast snapshot (see 13.5 for scheduling):
       python src/fetch_weather_forecast.py
  c) Train with weather (separate model directory):
       python src/train.py --inp data/features_price_only.parquet --model lgbm --encoding int --horizon 48 --val_days 30 --weather_file data/weather_oslo.parquet --out_dir models/lgbm_int_wx
     --weather_cols selects columns (default temperature_C wind_speed_ms
     precipitation_mm). LightGBM accepts NaN; ridge imputes the median.
  d) Forecast:
       python src/forecast.py --price data/entsoe_NO_1_2020-01-01_now.parquet --model_dir models/lgbm_int_wx --horizon 48 --save_png
     Weather at origin+h = latest snapshot issued at or before --weather_as_of
     (default now; the --origin if you give one), observed weather for hours the
     snapshot does not cover (--weather_obs, default the training file). The run
     prints the number of values from snapshot / observed / missing; the same is
     stored under "weather" in data\forecast_<model>_<enc>_wx.json. Missing values
     are passed on as NaN with a warning. Outputs: data\forecast_<model>_<enc>_wx.*
  e) Backtest (observed weather, upper bound):
       python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model lgbm --encoding int --weather_file data/weather_oslo.parquet --weather_mode observed --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48
     Realistic variant, only for origins that have an archived snapshot:
       ... --weather_mode archive [--weather_archive data/forecast_archive]
     In archive mode training still uses observed weather, predictions use only
     the snapshot (no fallback to observations, so nothing from the future leaks
     in), origins with no snapshot issued at or before them are skipped, and the
     number skipped is stored in the _meta.json. Outputs get the suffix _wx
     (observed) or _wxfc (archive), e.g. data\backtest_rolling_lgbm_int_wx.csv.
  f) Compare with the price-only run on the same origins:
       python src/backtest_rolling.py --compare data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_int_wx.csv
  g) Quick A/B on the last 30 days (observed weather, all 48 horizons, slow):
       python test_weather_forecast_value.py

13.5 Archiving forecast snapshots (do this from now on)
  MET updates the forecast roughly every hour, and the API gives only the
  current forecast. Past forecasts cannot be fetched later, so each run must be
  stored when it is available. fetch_weather_forecast.py does this automatically:
  every run writes data\forecast_archive\metno_<issued>.parquet, named by MET's
  own model-run time; the same run is never stored twice. Nothing runs by
  itself - schedule it, e.g. every 6 hours with Windows Task Scheduler:
      Program:    <repo>\.venv\Scripts\python.exe
      Arguments:  src\fetch_weather_forecast.py
      Start in:   <repo>
  Each file is about 20 KB. Results are not versioned: .gitignore excludes data/,
  models/ and figs/, so the archive lives only on this machine - back up
  data\forecast_archive\ yourself, since past forecasts cannot be fetched again.
  Options: --archive_dir, --no_archive.
  After a few weeks the archive supports --weather_mode archive backtests.

13.6 One-command pipeline
  python run_pipeline.py --weather
    1b) fetch_weather.py -> data\weather_oslo.parquet; fetch_weather_forecast.py
        (a failed snapshot only warns); warns if the weather history ends >24 h
        before the prices
    3)  train.py ... --weather_file ... --out_dir models/<model>_<enc>_wx
    4)  forecast.py --model_dir models/<model>_<enc>_wx
  Examples with lgbm_both and weather (model goes to models\lgbm_both_wx):

    Everything fresh: prices, weather history, forecast snapshot, train, forecast
      python run_pipeline.py --model lgbm --encoding both --weather

    Only new weather (existing prices):
      python run_pipeline.py --skip-price-fetch --model lgbm --encoding both --weather

    Only new prices (existing weather and latest snapshot):
      python run_pipeline.py --skip-weather-fetch --model lgbm --encoding both --weather

    No downloads at all (existing files and latest snapshot):
      python run_pipeline.py --skip-fetch --model lgbm --encoding both --weather

    With rolling backtest (observed weather; use --backtest-weather-mode archive for
    archived forecasts):
      python run_pipeline.py --skip-fetch --model lgbm --encoding both --weather --backtest --backtest-start "2026-01-01 00:00" --backtest-end "2026-08-31 00:00"

    Quick forecast refresh without retraining (new snapshot, then forecast):
      python src/fetch_weather_forecast.py
      python src/forecast.py --model_dir models/lgbm_both_wx --horizon 48 --save_png

  Every run of the pipeline retrains; prices are always fetched from 2020-01-01.
  --weather-file changes the observed weather file. --skip-fetch skips all
  downloads (the existing files and the latest archived snapshot are used);
  --skip-price-fetch skips only the prices; --skip-weather-fetch skips only the
  weather history and the forecast snapshot. With --backtest, --backtest-weather-mode observed|archive picks the
  prediction mode (13.4 e). Without --weather the pipeline is unchanged.

13.7 How it is built (for review)
  features.assemble(history, origins, h, encoding, weather=None) appends
    <col>_target = weather at origin + h; no argument -> price-only features.
  models.build_estimator(..., extra_cols=...) must list the weather columns,
    otherwise the ridge models silently drop them; train.py and
    backtest_rolling.py pass them. LightGBM takes all columns.
  weather_io.pick_snapshot / weather_for_forecast: latest snapshot issued at or
    before as_of; observed_path=None disables the fallback to observations.
  forecast.py refuses to run if the built feature list differs from the
    model's saved "features" (model/manifest mismatch).

13.8 Tests  (python tests/test_pipeline.py  or  pytest tests/ ; expected 27/27)
  The 13 tests of section 9, two origin-time tests (section 15), six ENTSO-E exog
  tests (section 16.7) and six weather tests: target-hour alignment of the
  weather columns, price-only features unchanged without weather, ridge keeps the
  weather columns, snapshot choice never looks ahead, snapshot-only mode has no
  fallback to observations, archiving is idempotent.

13.9 Known limits and legacy files
  * One station (Oslo) stands in for NO1. Hamburg and Copenhagen forecasts are
    archived but not used as features (no history to train on).
  * The MET forecast is hourly for about 60 hours; beyond that it is 6-hourly and
    precipitation is missing. The 48 h horizon is covered; the tail of the
    archived snapshots has gaps.
  * Reservoir filling (NVE) and cross-border flows are not implemented.
  * Results with observed weather overstate the benefit (13.1); archive-mode
    backtests are small until enough snapshots exist.
  * backtest_rolling.py and forecast.py use one weather source per run; the
    weather columns are the same set as in training.
  * build_features.py --add-weather-forecast, combine_weather.py and the
    features_*weather*/temperature files are LEGACY: train.py reads only the
    price-history columns of a feature file, so those weather columns were never
    used. A comparison based on them (including the earlier "weather does not
    help" result of test_weather_hypothesis.py) is invalid. That script is
    deprecated and refuses to run without --force because it overwrites
    data\features_price_only.parquet.


-------------------------------------------------------------------------------
14. WEATHER BACKTEST RESULTS (OBSERVED WEATHER) - run 8 November 2026
-------------------------------------------------------------------------------

Same backtest settings as section 4 (origins 2026-01-01 00:00 .. 2026-08-29 01:00,
step 48 h, 121 origins x 48 horizons = 5,808 forecasts, refit every 720 h, price
file of 30 Oct 2026, seed 42). Weather = OBSERVED Oslo temperature, wind speed and
precipitation at t+h (--weather_mode observed), so every gain below is an UPPER
BOUND on what real forecasts would give (13.1). Run in a cloud container with the
pinned versions (scikit-learn 1.9.1, lightgbm 4.7.0, pandas 3.0.6, numpy 2.5.3).
The price-only lgbm_int run reproduced the section 11 values exactly (MAE 0.1747,
RMSE 0.2617), so the setup matches. Origins fall at 00:00 local time in winter and
01:00 in summer (48 absolute hours apart), i.e. a midnight origin, not the ~13:00
day-ahead publication time.

14.1 Commands
  python src/train.py --inp data/features_price_only.parquet --model lgbm --encoding int --horizon 48 --val_days 30 --weather_file data/weather_oslo.parquet --out_dir models/lgbm_int_wx
  python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model lgbm --encoding int --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48 --weather_file data/weather_oslo.parquet --weather_mode observed
  (and the same with --encoding both; the price-only lgbm_both run is new and not
  part of section 11)
  python src/weather_block_test.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_int_wx.csv --out_csv data/weather_block_test_lgbm_int.csv

14.2 Overall (NOK/kWh, 5,808 forecasts)
  variant               MAE      RMSE     nMAE    mean validation MAE (EUR/kWh)
  lgbm_int              0.1747   0.2617   15.8 %  0.0229
  lgbm_int + weather    0.1696   0.2453   15.3 %  0.0225
  lgbm_both             0.1842   0.2730   16.6 %  0.0247
  lgbm_both + weather   0.1754   0.2567   15.8 %  0.0242
  persistence           0.2253   0.3480   20.4 %
  weekly t-168          0.2469   0.3539   22.3 %
  lgbm_int is better than lgbm_both with and without weather (compared on totals;
  no paired test between the variants). Best: lgbm_int + weather.

14.3 Weather vs no weather, paired per origin (src\weather_block_test.py)
  Per origin, the MAE inside each block of TARGET hours (Europe/Oslo clock) is
  computed; the paired difference weather minus no weather is summarised with a
  block-bootstrap 95 % CI (seed 42), the share of origins where weather is better and
  a Wilcoxon p-value. Primary tests, fixed in advance: 06-20 (variable-price hours)
  and 21-05 (flat hours), Holm over the two. The five blocks 00-05, 06-09, 10-15,
  16-20, 21-23 are exploratory (Holm over the five).

  lgbm_int        change   95 % CI of difference     better on   Holm p
    all hours     -2.9 %   [-0.0107, +0.0006]        60 % origins 0.045*
    06-20         -4.4 %   [-0.0175, -0.0010]        55 %         0.052
    21-05         +1.6 %   [-0.0022, +0.0059]        49 %         0.71
    10-15         -4.8 %   [-0.0194, -0.0029]        60 %         0.021
    other blocks  -4.3..+2.6 %  CIs include 0                      0.19-0.84
  lgbm_both
    all hours     -4.8 %   [-0.0139, -0.0041]        69 %         <0.001
    06-20         -5.7 %   [-0.0196, -0.0060]        66 %         <0.001
    21-05         -2.1 %   [-0.0066, +0.0015]        54 %         0.13
    06-09 / 10-15 / 16-20 / 21-23: -4.3 / -4.6 / -8.2 / -5.2 %, all Holm p < 0.03
    00-05         -0.1 %   CI includes 0                           0.90
  (* Wilcoxon p; the bootstrap CI of the overall lgbm_int difference includes 0.)
  Pattern: the gain is concentrated in the hours with large price variation
  (about 06-20) and is absent at night. For lgbm_int the overall gain is not
  clearly separable from chance; for lgbm_both it is.

14.4 Reading the results
  * Observed weather overstates the benefit; real forecasts will give less. Only an
    archive-mode backtest (13.5) measures that, and the archive is still empty.
  * The two primary blocks and the overall figure are the planned tests; the five
    smaller blocks are exploratory, and origins are not fully independent (prices
    are autocorrelated from day to day).
  * Daily averages hide the structure: night and late-evening hours have low error
    because prices are flat there. Report per-block figures next to the total.
  * A midnight origin differs from practical use (prices for the next day are
    known from about 13:00). Testing an origin around 12:00 is not implemented.
  * Files (git-ignored): models\lgbm_int_wx, models\lgbm_both_wx,
    data\backtest_rolling_lgbm_{int,both}[_wx]*.csv and _meta.json,
    data\weather_block_test_lgbm_{int,both}.csv, figs\backtest_rolling_*_wx.png.


-------------------------------------------------------------------------------
15. BACKTEST ORIGIN CLOCK TIME (--origin_hour)
-------------------------------------------------------------------------------

Origin = the last hour whose price is known when the forecast is made. History
features are computed up to the origin; horizons 1..48 are the hours after it.

15.1 Default (unchanged)
  Origins are --step absolute hours apart from --start. Local clock time then
  drifts by an hour at each DST change (00:00 in winter, 01:00 in summer in the
  2026 backtest). A midnight origin is not how the forecast is used in practice.

15.2 Practical case: --origin_hour 23
  Day-ahead prices for day D are published on D-1; prices for D+1 are published
  around 13:00 on D. A forecast made at about 12:00 on D therefore knows prices up
  to D 23:00, so the origin is D 23:00 and horizons 1-24 are tomorrow's auction
  prices, 25-48 the day after. Weather forecasts must not be newer than the
  decision time; in archive mode use --snapshot_lag_hours 11 (origin 23:00 minus
  12:00) so snapshots issued after about 12:00 are not used.

    python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model lgbm --encoding int --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48 --origin_hour 23
    ... --weather_file data/weather_oslo.parquet --weather_mode archive --origin_hour 23 --snapshot_lag_hours 11

  --origin_hour (0-23) fixes the local clock hour of every origin, independent of
  DST; --step must be a multiple of 24 (48 = every second day). A local hour that
  does not exist or is ambiguous on a DST day (02:00-03:00) is skipped. Outputs get
  the suffix _o<hour> (e.g. backtest_rolling_lgbm_int_o23.csv), so they do not
  overwrite the default runs; --compare works across them only on common origins,
  so compare runs with the same --origin_hour.

15.3 Pipeline
  python run_pipeline.py --skip-fetch --model lgbm --encoding int --weather --backtest --backtest-start "2026-01-01 00:00" --backtest-end "2026-08-31 00:00" --backtest-origin-hour 23
  --backtest-snapshot-lag N is passed on as --snapshot_lag_hours (archive mode).

15.4 Caveats
  * Only the scoring changes: the backtest still trains on every hour as a
    forecast origin and scores the chosen origins. train.py's validation window
    (last 30 days) uses every hour as origin and is unaffected.
  * Prices known at the origin are assumed to be known to the last hour of D; this
    matches the day-ahead schedule for NO1 but the publication time (about 13:00)
    is not modelled to the minute.


-------------------------------------------------------------------------------
16. ENTSO-E EXOGENOUS FEATURES: WIND/SOLAR FORECASTS AND NEIGHBOUR PRICES
-------------------------------------------------------------------------------

Optional add-on (added November 2026). Without --exog_file nothing changes. It
complements the Oslo weather of section 13: Oslo weather describes NO1 demand,
while neighbouring zones' wind and solar generation (Germany, Denmark, Sweden)
sets the supply side that NO1 trades with through the interconnectors.

STATUS: implemented and unit-tested, but NOT yet run on real ENTSO-E data (the
development environment cannot reach the ENTSO-E API). No results exist yet; run
the fetch (16.3) and the backtests (16.4) yourself.

16.1 Features (module src\exog.py)
  One hourly parquet, columns by name prefix:
    wind_<ZONE>_MW, solar_<ZONE>_MW  day-ahead forecast (A69) for the TARGET hour
                                     -> <col>_target
    price_<ZONE>_EUR_MWh             neighbour day-ahead price (A44), used only
                                     up to the ORIGIN -> <col>_origin and
                                     <col>_origin_lag24h
  Default zones: wind/solar DE_LU, DK_1, DK_2, SE_3; prices NO_2, SE_3, DK_1, DE_LU.
  A neighbour price at the target hour is NOT a feature: it is set in the same
  auction as the NO1 price and is unknown at forecast time.

16.2 Availability rule (no look-ahead)
  Day-ahead wind/solar forecasts for day D+1 are published around 12:00 on D. For
  an origin t the decision time is t - 11 h (origin 23:00 -> 12:00 the same day). A
  target-hour value is available only if its local date is at most one day after
  the decision date; later hours are NaN - in training, backtesting and forecasting
  alike. With origin 23:00 horizons 1-24 (tomorrow) have wind/solar values and
  25-48 do not. LightGBM handles NaN; ridge imputes the median. The publication
  time (about 12:00) is an ASSUMPTION; ENTSO-E does not give the publication time
  per value, and some TSOs publish later. If so, results are slightly optimistic.

16.3 Fetch
  python src/fetch_entsoe_exog.py
  python src/fetch_entsoe_exog.py --wind_zones DE_LU DK_1 --price_zones NO_2 SE_3 --start 2023-01-01
  First run fetches the full history (slow). Later runs update incrementally: each
  zone is fetched from its last stored value minus --overlap_days (default 3, covers
  late/revised forecasts) and merged into the existing file. A zone with no stored
  data, or stored data starting later than --start, is fetched in full; --full
  forces a complete re-fetch (use it after changing --start to an earlier date
  for all zones or if the file looks wrong).
  Needs ENTSOE_TOKEN in .env. Output: data\entsoe_exog.parquet (+ _meta.json with
  rows and first/last time per column). Fetches until now + 2 days, so tomorrow's
  forecasts are included once published. A zone that fails is skipped with a
  warning; the run fails only if nothing arrives. Re-run to refresh before
  forecasting. Zone codes are entsoe-py areas (DE_LU, DK_1, DK_2, SE_3, SE_4, NO_2, ...).

16.4 Train, forecast, backtest
  python src/train.py --inp data/features_price_only.parquet --model lgbm --encoding both --horizon 48 --val_days 30 --exog_file data/entsoe_exog.parquet --out_dir models/lgbm_both_ex
  python src/forecast.py --model_dir models/lgbm_both_ex --horizon 48 --save_png
  python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model lgbm --encoding both --exog_file data/entsoe_exog.parquet --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48 --origin_hour 23
  --exog_cols picks columns (default: all in the file). Weather and exog combine:
  pass both --weather_file and --exog_file (model directory suffix _wx_ex; backtest
  output suffixes _wx, _ex, _o23, in that order). forecast.py reads the file path
  and columns from the model manifest; --exog_file overrides the path.
  Compare variants on the same origins (same --origin_hour), see section 15.
  Suggested comparison, all with --origin_hour 23: price only; + Oslo weather;
  + ENTSO-E; both. Use src\weather_block_test.py for paired tests (it takes any two
  backtest CSVs, base and extended).

16.5 Pipeline
  python run_pipeline.py --model lgbm --encoding both --weather --exog
  --exog fetches the ENTSO-E file first (needs ENTSOE_TOKEN), trains with it and
  forecasts with it; --skip-fetch or --skip-exog-fetch use the existing file.
  Models go to models\<model>_<enc>[_wx]_ex. With --backtest the exog file is
  passed on (and --backtest-origin-hour works as before).

16.6 Limits
  * Forecast hours after the last published day are NaN. For a clean live
    forecast run it before about 13:00 with prices known through the end of the
    same day (origin 23:00); after 13:00 the origin moves a day on and the
    required forecasts are not yet published.
  * Neighbour-price features are lags of the origin only (origin and 24 h before).
  * Wind/solar forecasts are as stored by ENTSO-E (not an archive of vintages);
    revised values may differ slightly from what was known at the time.
  * Load forecasts, net transfer capacities and flows are not included.

16.7 Tests (tests\test_pipeline.py)
  Normalisation of entsoe-py wind/solar frames (flat and MultiIndex), the
  availability rule, no look-ahead (perturbing unpublished values or later prices
  leaves the features unchanged), column order of assemble(), and the fetch logic
  with a fake client (failing zones skipped, error when nothing arrives), and the
  incremental update (per-zone start date, backfill, merge).
