"""Does weather help more when the weather changes abruptly?

Uses the existing rolling backtests (same origins, same model) with and without
weather, and splits forecast errors by how much the weather at the target hour
differs from the weather the price history has already "seen" at the origin.

Shift measure per forecast row (origin t, target t+h):
    d_temp = mean temp over 24 h ending at t+h  -  mean temp over 24 h ending at t
    d_wind = same for wind speed
Rows are grouped into quantile bins of |d_temp| and |d_wind|. For each bin:
MAE without / with weather, relative improvement, and a block-bootstrap 95 % CI
(resampling whole origins, since errors within one origin are correlated).

Weather is OBSERVED, so this is an upper bound on what a real forecast gives.

Usage:
    python test_weather_shifts.py [--model lgbm] [--bins 4] [--n_boot 2000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load(model):
    base = pd.read_csv(DATA / f"backtest_rolling_{model}.csv")
    wx = pd.read_csv(DATA / f"backtest_rolling_{model}_wx.csv")
    key = ["origin", "timestamp", "horizon_hours"]
    df = base.merge(wx[key + ["predicted"]], on=key, suffixes=("_base", "_wx"))
    assert len(df) == len(base) == len(wx), "backtests do not line up"
    for c in ("origin", "timestamp"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["ae_base"] = (df.predicted_base - df.actual).abs()
    df["ae_wx"] = (df.predicted_wx - df.actual).abs()
    return df


def add_shifts(df):
    w = pd.read_parquet(DATA / "weather_oslo.parquet")
    w.index = pd.to_datetime(w.index, utc=True)
    roll = w[["temperature_C", "wind_speed_ms"]].rolling("24h").mean()
    at_t = roll.reindex(df.origin).to_numpy()
    at_th = roll.reindex(df.timestamp).to_numpy()
    df["d_temp"] = at_th[:, 0] - at_t[:, 0]
    df["d_wind"] = at_th[:, 1] - at_t[:, 1]
    return df.dropna(subset=["d_temp", "d_wind"])


def boot_ci(sub, n_boot, rng):
    """95 % CI of relative MAE improvement, resampling origins."""
    g = sub.groupby("origin")[["ae_base", "ae_wx"]].agg(["sum", "count"])
    sb, sw, n = g[("ae_base", "sum")].to_numpy(), g[("ae_wx", "sum")].to_numpy(), g[("ae_base", "count")].to_numpy()
    idx = rng.integers(0, len(sb), size=(n_boot, len(sb)))
    mb = sb[idx].sum(1) / n[idx].sum(1)
    mw = sw[idx].sum(1) / n[idx].sum(1)
    imp = 100 * (mb - mw) / mb
    return np.percentile(imp, [2.5, 97.5])


def table(df, col, bins, n_boot, rng):
    df = df.copy()
    df["bin"] = pd.qcut(df[col].abs(), bins, labels=False)
    rows = []
    for b, sub in df.groupby("bin"):
        mb, mw = sub.ae_base.mean(), sub.ae_wx.mean()
        lo, hi = boot_ci(sub, n_boot, rng)
        rows.append({
            "variable": col, "bin": f"Q{b + 1}",
            "abs_shift_range": f"{sub[col].abs().min():.1f}-{sub[col].abs().max():.1f}",
            "n_rows": len(sub), "n_origins": sub.origin.nunique(),
            "mae_base_ct": 100 * mb, "mae_wx_ct": 100 * mw,
            "improvement_pct": 100 * (mb - mw) / mb, "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(42)

    df = add_shifts(load(a.model))
    out = [table(df, c, a.bins, a.n_boot, rng) for c in ("d_temp", "d_wind")]

    # Horizon split: is the effect just "long horizon"? Repeat temp bins per half.
    for lo, hi in ((1, 24), (25, 48)):
        t = table(df[df.horizon_hours.between(lo, hi)], "d_temp", a.bins, a.n_boot, rng)
        t["variable"] = f"d_temp h{lo}-{hi}"
        out.append(t)

    res = pd.concat(out, ignore_index=True)
    mb, mw = df.ae_base.mean(), df.ae_wx.mean()
    print(f"{a.model}: {df.origin.nunique()} origins, {len(df)} rows, "
          f"{df.origin.min():%Y-%m-%d} to {df.origin.max():%Y-%m-%d}")
    print(f"Overall MAE  base {100*mb:.2f}  wx {100*mw:.2f} EUR-cent/kWh  "
          f"improvement {100*(mb-mw)/mb:.1f} %\n")
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print(res.to_string(index=False))
    path = DATA / f"weather_shift_test_{a.model}.csv"
    res.to_csv(path, index=False)
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
