===============================================================================
 REPRODUCING ALL DATA, MODELS, FORECASTS AND FIGURES
 Electricity Price Forecasting Using Machine Learning (NO1) - Kjetil Borg
 State as of 22 September 2026
===============================================================================

This file lists, step by step, how every data file, model, forecast, figure and
table that the project report relies on was produced, and how to produce them
again. All commands are run from the repository root in a terminal with the
virtual environment activated. Commands shown with "python" assume the venv's
python (.\.venv\Scripts\python.exe on Windows).

Contents
  0. What "reproduce" means here (two modes)
  1. Environment
  2. Configuration
  3. Step 1 - Download prices
  4. Steps 2-5 - Features, training, forecast and backtest for each model
  5. Step 6 - Exploratory data analysis
  6. Step 7 - Model comparison across horizons
  7. Step 8 - Statistical tests
  8. Step 9 - Feature importance
  9. Step 10 - Automated tests
 10. Where each figure and table in the report comes from
 11. Reference values for checking a reproduction
 12. Numbers computed from the backtest CSVs

-------------------------------------------------------------------------------
0. WHAT "REPRODUCE" MEANS HERE (TWO MODES)
-------------------------------------------------------------------------------

Mode A - exact reproduction (recommended for checking the report)
  Use the saved price file data\entsoe_NO_1_2020-01-01_now.parquet as it is
  (84,666 rows, 2020-01-01 00:00 to 2026-09-22 23:45, fetched 21 Sep 2026
  13:39). Skip step 1. Every later step is
  deterministic (fixed seed 42, fixed exchange rate), so the numbers in
  section 11 should be reproduced to the last printed decimal with the same
  library versions.

Mode B - fresh reproduction from the API
  Download the prices again (step 1). If you download without an end date you
  get newer data: the forecast origin moves, the 30-day validation window
  moves, and the models are trained on more hours, so forecasts and validation
  figures will differ. The backtest (Jan-Aug 2026) is still comparable but not
  bit-identical. To come as close as possible to Mode A, fetch with
  --end 2026-09-23 (see step 1). ENTSO-E can revise published data, so even
  then exact equality is not guaranteed.


-------------------------------------------------------------------------------
1. ENVIRONMENT
-------------------------------------------------------------------------------

1.1 Python and libraries used for the reported results (recorded in every
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

  EUR_TO_NOK   = 10.83     reporting rate of 18 Sep 2026. Only scales NOK
                           figures; models train on EUR/kWh.
  TZ           = "Europe/Oslo"

  Keep EUR_TO_NOK at 10.83 to reproduce the NOK numbers in the report.

  The ENTSO-E API token is NOT stored in config.py. It is read from a .env
  file in the repository root (only needed for step 1):

      copy .env.example .env
      (then edit .env)   ENTSOE_TOKEN=<your own ENTSO-E API token>

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

Steps 3-5 for the five variants used in the report, in the order they were
run (random forest last, it is slow):

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

  rf_trig:
    python src/train.py --inp data/features_price_only.parquet --model rf --encoding trig --horizon 48 --val_days 30
    python src/forecast.py --price data/entsoe_NO_1_2020-01-01_now.parquet --model_dir models/rf_trig --horizon 48 --save_png
    python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model rf --encoding trig --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48

Every variant must use the same price file. In Mode B, run step 1 first and do
not download again between variants.

The reported results were produced by these same commands, run through a
PowerShell wrapper script that is not part of the delivery.

Backtest settings used:
  origins 2026-01-01 00:00 .. 2026-08-29 01:00, step 48 h -> 121 origins
  horizons 1..48 -> 5,808 forecasts per model
  refit every 720 h, minimum 8,760 training hours, embargo h hours

Training settings:
  horizon 48, val_days 30 (validation starts 2026-08-23 22:00 in Mode A),
  Ridge alpha chosen by RidgeCV (logspace -3..3, TimeSeriesSplit 3),
  LightGBM early stopping 100 rounds on a 10 % time-ordered tail,
  Random forest 300 trees, max_depth 15, min_samples_split 10.

Run times on the machine used: under 2 minutes of training per variant for
LightGBM and ridge, about 31 minutes for the random forest. The random forest
backtest refits nine times and takes several hours; it is not hanging.


-------------------------------------------------------------------------------
5. STEP 6 - EXPLORATORY DATA ANALYSIS
-------------------------------------------------------------------------------

  python src/eda.py

  Defaults: whole period, spike rule |z| > 6 robust SD from a centred 168 h
  rolling median, ACF to 336 h.

  Output:
    figs\eda\eda_timeseries.png, eda_distribution.png, eda_by_hour.png,
            eda_by_month.png, eda_heatmap.png, eda_acf.png, eda_outliers.png
    data\eda_summary_by_year.csv   (annual statistics, report Table 2)
    data\eda_outliers.csv          (flagged hours)
    data\eda_meta.json             (settings and headline numbers)

  Only reads the price file; can be run any time after step 1.


-------------------------------------------------------------------------------
6. STEP 7 - MODEL COMPARISON ACROSS HORIZONS
-------------------------------------------------------------------------------

  python src/backtest_rolling.py --compare data/backtest_rolling_lgbm_int.csv data/backtest_rolling_ridge_ohe_both.csv data/backtest_rolling_ridge_trig.csv data/backtest_rolling_rf_trig.csv

  Output: figs\compare_by_horizon.png  (report Figure 5)
  Prints aggregate MAE / nMAE per model and both baselines.
  The plot accepts at most four runs; lgbm_trig is left out of the figure.


-------------------------------------------------------------------------------
7. STEP 8 - STATISTICAL TESTS (Friedman / Iman-Davenport, Wilcoxon + Holm)
-------------------------------------------------------------------------------

  All five variants plus both baselines, all horizons and per half:

  python src/stat_tests.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_trig.csv data/backtest_rolling_ridge_trig.csv data/backtest_rolling_ridge_ohe_both.csv data/backtest_rolling_rf_trig.csv --baselines --out_csv data/stat_tests_all.csv
  python src/stat_tests.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_trig.csv data/backtest_rolling_ridge_trig.csv data/backtest_rolling_ridge_ohe_both.csv data/backtest_rolling_rf_trig.csv --baselines --horizons 1-24 --out_csv data/stat_tests_1-24.csv
  python src/stat_tests.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_trig.csv data/backtest_rolling_ridge_trig.csv data/backtest_rolling_ridge_ohe_both.csv data/backtest_rolling_rf_trig.csv --baselines --horizons 25-48 --out_csv data/stat_tests_25-48.csv

  Calendar-encoding ablation (single planned test):

  python src/stat_tests.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_trig.csv --out_csv data/stat_tests_encoding.csv

  Ridge encodings:

  python src/stat_tests.py data/backtest_rolling_ridge_trig.csv data/backtest_rolling_ridge_ohe_both.csv

  Output: pairwise tables (CSV); Friedman ranks and p-values are printed to
  the console (report Table 6).


-------------------------------------------------------------------------------
8. STEP 9 - FEATURE IMPORTANCE
-------------------------------------------------------------------------------

  python src/feature_importance.py --model_dir models/lgbm_int models/ridge_ohe_both models/rf_trig

  Defaults: horizons 1 6 12 24 36 48, 10 repeats, seed 42,
  evaluation window = each model's own validation window (val_start in the
  model metadata, 2026-08-23 22:00 in Mode A).

  Output:
    data\feature_importance_<model>.csv          per feature and horizon
    data\feature_importance_groups_<model>.csv   per feature group
    figs\importance\feature_importance_groups_<model>.png   (Figure 6: lgbm_int)

  Requires the trained models from step 3.


-------------------------------------------------------------------------------
9. STEP 10 - AUTOMATED TESTS
-------------------------------------------------------------------------------

  python tests/test_pipeline.py        (or: pytest tests/)

  Expected: 13/13 passed.

-------------------------------------------------------------------------------
10. WHERE EACH FIGURE AND TABLE IN THE REPORT COMES FROM
-------------------------------------------------------------------------------

  Figure 1   figs\eda\eda_timeseries.png                           step 6
  Figure 2   figs\eda\eda_heatmap.png                              step 6
  Figure 3   figs\eda\eda_acf.png                                  step 6
  Figure 4   figs\eda\eda_distribution.png                         step 6
  Figure 5   figs\compare_by_horizon.png                           step 7
  Figure 6   figs\importance\feature_importance_groups_lgbm_int.png step 9

  Table 1    data\entsoe_..._meta.json, features_price_only_meta.json
  Table 2    data\eda_summary_by_year.csv                           step 6
  Table 3    backtest console output / backtest_rolling_*.csv       step 5
  Table 4    backtest_rolling_*_by_horizon.csv                      step 5
  Table 5    models\<variant>\training_summary.csv (mean over 48 horizons) step 3
  Table 6    stat_tests.py console output + stat_tests_*.csv        step 8
  Table 7    computed from backtest_rolling_*.csv (section 12)       step 5

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
  rf_trig          0.1871   0.2752   16.9 %
  persistence      0.2253   0.3480   20.4 %
  weekly t-168     0.2469   0.3539   22.3 %

Statistical tests (all horizons, 5 variants + 2 baselines)
  Friedman chi2 = 114.63, Iman-Davenport F = 22.50
  mean ranks: lgbm_int 3.11, ridge_ohe_both 3.37, lgbm_trig 3.56,
              ridge_trig 3.62, rf_trig 3.98, weekly 4.96, persistence 5.40
  lgbm_int vs rf_trig Holm p = 0.040; lgbm_int vs lgbm_trig unadjusted p = 0.019

LightGBM (int) early stopping, final models: 972 trees at 1 h, 552 at 6 h,
336 at 24 h, 230 at 48 h.

EDA headline: ACF 0.987 (1 h), 0.927 (24 h), 0.835 (168 h); 692 negative
hours; 805 upward and 675 downward spikes (k = 6).

Forecast origin in Mode A: 2026-09-22 23:00 (price 0.1474 EUR/kWh).


-------------------------------------------------------------------------------
12. NUMBERS COMPUTED FROM THE BACKTEST CSVs (not printed by a script)
-------------------------------------------------------------------------------

  - Decision test (Table 7): for each origin, the 4 hours in horizons 25-48
    with the lowest predicted price; mean realised (actual) price of those
    hours, x10.83. Oracle = 4 lowest actual; weekly = 4 lowest
    baseline_weekly_t168; "no information" = mean of all 24 hours.
  - Spike analysis (10.1): hours with actual >= 95th percentile of actual.
  - 15-minute effect (7.1): |price at hh:00 - mean of hh:00..hh:45|, Oct 2025
    onwards, from the raw parquet.
