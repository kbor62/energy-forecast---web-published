"""Paired test of weather vs no weather per time-of-day block.

Compares two rolling backtests of the same model on the same origins (e.g.
backtest_rolling_lgbm_int.csv and backtest_rolling_lgbm_int_wx.csv). For each block
of target hours the MAE is computed per forecast origin, and the paired difference
(weather minus no weather) is summarised with a block-bootstrap 95 % CI, the share of
origins where weather is better, and a Wilcoxon signed-rank p-value.

Blocks (fixed in advance, Europe/Oslo clock hours of the TARGET hour):
  primary    06-20 (variable-price hours) and 21-05 (flat hours); Holm over these two
  secondary  00-05, 06-09, 10-15, 16-20, 21-23; exploratory, Holm over these five
  all        all 24 hours

    python src/weather_block_test.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_lgbm_int_wx.csv
    python src/weather_block_test.py BASE.csv WEATHER.csv --out_csv data/weather_block_test_lgbm_int.csv

Note: with --weather_mode observed the weather is OBSERVED, so the gain is an upper bound.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import EUR_TO_NOK, TZ, repo_path
from metrics import block_bootstrap_ci

PRIMARY = {"06-20 (variable)": range(6, 21), "21-05 (flat)": [*range(21, 24), *range(0, 6)]}
SECONDARY = {
    "00-05": range(0, 6),
    "06-09": range(6, 10),
    "10-15": range(10, 16),
    "16-20": range(16, 21),
    "21-23": range(21, 24),
}


def load_pair(base_csv, weather_csv) -> pd.DataFrame:
    a = pd.read_csv(repo_path(base_csv))
    b = pd.read_csv(repo_path(weather_csv))
    m = a.merge(b, on=["origin", "horizon_hours"], suffixes=("_0", "_w"))
    if not (len(m) == len(a) == len(b)):
        raise ValueError(f"Runs must cover the same (origin, horizon) pairs: {len(a)}, {len(b)}, shared {len(m)}")
    m["hour"] = pd.to_datetime(m["timestamp_0"], utc=True).dt.tz_convert(TZ).dt.hour
    m["e0"] = (m["predicted_0"] - m["actual_0"]).abs() * EUR_TO_NOK
    m["ew"] = (m["predicted_w"] - m["actual_w"]).abs() * EUR_TO_NOK
    return m


def test_block(m: pd.DataFrame, name: str, hours, seed: int = 42) -> dict:
    per_origin = m[m["hour"].isin(list(hours))].groupby("origin")[["e0", "ew"]].mean()
    d = per_origin["ew"] - per_origin["e0"]
    lo, hi = block_bootstrap_ci(d, block_hours=1, seed=seed)
    return {
        "block": name, "n_origins": len(d),
        "mae_base": per_origin["e0"].mean(), "mae_weather": per_origin["ew"].mean(),
        "pct_change": (per_origin["ew"].mean() / per_origin["e0"].mean() - 1) * 100,
        "diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
        "weather_better_pct": (d < 0).mean() * 100, "p_wilcoxon": wilcoxon(d).pvalue,
    }


def holm(rows: list[dict]) -> None:
    order = np.argsort([r["p_wilcoxon"] for r in rows])
    k, running = len(rows), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * rows[i]["p_wilcoxon"]))
        rows[i]["p_holm"] = running


def main() -> None:
    ap = argparse.ArgumentParser(description="Weather vs no weather, paired per time-of-day block")
    ap.add_argument("base_csv", help="Backtest CSV without weather")
    ap.add_argument("weather_csv", help="Backtest CSV with weather (same origins)")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    m = load_pair(args.base_csv, args.weather_csv)
    overall = [test_block(m, "ALL hours", range(24), args.seed)]
    primary = [test_block(m, n, h, args.seed) for n, h in PRIMARY.items()]
    secondary = [test_block(m, n, h, args.seed) for n, h in SECONDARY.items()]
    overall[0]["p_holm"] = overall[0]["p_wilcoxon"]
    holm(primary)
    holm(secondary)

    table = pd.concat(
        [pd.DataFrame(r).assign(group=g) for g, r in
         (("all", overall), ("primary", primary), ("secondary", secondary))],
        ignore_index=True,
    )
    pd.set_option("display.width", 250)
    print(f"Weather minus no weather, MAE per origin (NOK/kWh); negative = weather better")
    print(table.round(4).to_string(index=False))
    if args.out_csv:
        out = repo_path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)


if __name__ == "__main__":
    main()
