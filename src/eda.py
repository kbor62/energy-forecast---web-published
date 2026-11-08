"""Exploratory data analysis of the price series: figures in figs/eda/ and
tables in data/ (eda_summary_by_year.csv, eda_outliers.csv, eda_meta.json).

Outputs:
  - figs/eda/*.png: timeseries, distribution, by-hour, by-month, heatmap, ACF, outliers
  - data/eda_summary_by_year.csv: annual statistics
  - data/eda_outliers.csv: flagged anomalies
  - data/eda_meta.json: settings and headline statistics

Examples:
    python src/eda.py
    python src/eda.py --k_spike 5  # Spike threshold (default 6)
    python src/eda.py --acf_max_lag 240  # Maximum ACF lag (default 336)
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from config import EUR_TO_NOK, ROOT, TZ, repo_path
from plotting import (
    INK,
    INK_SOFT,
    OBSERVED,
    SERIES_COLORS,
    format_time_axis,
    save_figure,
    style_axes,
    style_legend,
)
from price_io import describe_series, load_price_series

DEFAULT_PRICE = "data/entsoe_NO_1_2020-01-01_now.parquet"
UNIT = "NOK/kWh"
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Colour ramp for the heatmap.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

BOX_STYLE = dict(
    patch_artist=True,
    showfliers=False,
    widths=0.6,
    boxprops=dict(facecolor=SERIES_COLORS[0], edgecolor=SERIES_COLORS[0], alpha=0.35),
    medianprops=dict(color=INK, linewidth=1.6),
    whiskerprops=dict(color=INK_SOFT, linewidth=1.0),
    capprops=dict(color=INK_SOFT, linewidth=1.0),
)


def describe(values: pd.Series) -> dict:
    v = values.dropna().to_numpy()
    q1, q3 = np.percentile(v, [25, 75])
    return {
        "n_hours": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)),
        "min": float(v.min()),
        "q25": float(q1),
        "q75": float(q3),
        "max": float(v.max()),
        "skewness": float(stats.skew(v)),
        "excess_kurtosis": float(stats.kurtosis(v)),  # Fisher: normal = 0
        "n_negative": int((v < 0).sum()),
        "share_negative": float((v < 0).mean()),
    }


def summary_by_year(price: pd.Series) -> pd.DataFrame:
    rows = [{"period": str(y), **describe(g)} for y, g in price.groupby(price.index.year)]
    rows.append({"period": "all", **describe(price)})
    return pd.DataFrame(rows)


def autocorrelation(x: pd.Series, max_lag: int) -> np.ndarray:
    """Autocorrelation for lags 0..max_lag."""
    v = x.dropna().to_numpy(dtype=float)
    v = v - v.mean()
    n = v.size
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(v, size)
    acov = np.fft.irfft(f * np.conjugate(f), size)[: max_lag + 1]
    return acov / acov[0]


def flag_outliers(price: pd.Series, window_h: int, k: float) -> pd.DataFrame:
    """Flag spikes (more than k robust SDs from a centred rolling median) and negative prices."""
    med = price.rolling(window_h, center=True, min_periods=window_h // 2).median()
    mad = (price - med).abs().rolling(window_h, center=True, min_periods=window_h // 2).median()
    robust_sd = (1.4826 * mad).replace(0, np.nan)
    z = (price - med) / robust_sd

    out = pd.DataFrame({"price": price, "rolling_median": med, "robust_z": z})
    out["spike_high"] = z > k
    out["spike_low"] = z < -k
    out["negative"] = price < 0

    q1, q3 = price.quantile([0.25, 0.75])
    out["iqr_extreme"] = (price > q3 + 3 * (q3 - q1)) | (price < q1 - 3 * (q3 - q1))
    return out


def new_axes(width=11, height=4.8):
    fig, ax = plt.subplots(figsize=(width, height))
    style_axes(ax, ylabel=UNIT)
    return fig, ax


def title(ax, text: str, sub: str | None = None) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=13, pad=18 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK_SOFT, fontsize=9, va="bottom")


def plot_timeseries(price: pd.Series, out) -> None:
    daily = price.resample("D").mean()
    rolling = daily.rolling(30, min_periods=15).mean()
    fig, ax = new_axes()
    ax.plot(daily.index, daily, color=SERIES_COLORS[0], linewidth=0.8, alpha=0.7, label="Daily mean")
    ax.plot(rolling.index, rolling, color=OBSERVED, linewidth=2.0, label="30-day rolling mean")
    ax.axhline(0, color=INK_SOFT, linewidth=0.8)
    format_time_axis(ax)
    title(ax, "Day-ahead price, NO1", f"{daily.index[0]:%b %Y} - {daily.index[-1]:%b %Y}")
    style_legend(ax.legend(loc="upper right", frameon=True))
    save_figure(fig, out)


def plot_distribution(price: pd.Series, s: dict, out) -> None:
    fig, ax = new_axes()
    style_axes(ax, ylabel="Hours")
    ax.hist(price, bins=120, color=SERIES_COLORS[0], alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color=INK_SOFT, linewidth=1.0)
    ax.axvline(s["median"], color=INK, linewidth=1.4, linestyle=(0, (4, 2)), label=f"Median {s['median']:.2f}")
    ax.axvline(s["mean"], color=SERIES_COLORS[1], linewidth=1.4, linestyle=(0, (1, 1.5)), label=f"Mean {s['mean']:.2f}")
    ax.set_xlabel(UNIT, color=INK_SOFT)
    title(
        ax, "Distribution of hourly prices",
        f"skewness {s['skewness']:.2f}   excess kurtosis {s['excess_kurtosis']:.1f}   "
        f"negative hours {s['n_negative']:,} ({s['share_negative']:.1%})",
    )
    style_legend(ax.legend(loc="upper right", frameon=True))
    save_figure(fig, out)


def plot_box(price: pd.Series, key: np.ndarray, levels, labels, xlabel: str, heading: str, out) -> None:
    groups = [price[key == lv].dropna().to_numpy() for lv in levels]
    fig, ax = new_axes()
    ax.boxplot(groups, **BOX_STYLE)
    ax.set_xticks(range(1, len(groups) + 1), labels)
    ax.set_xlabel(xlabel, color=INK_SOFT)
    title(ax, heading, "Box = interquartile range, line = median, whiskers = 1.5 x IQR; points beyond hidden")
    save_figure(fig, out)


def plot_heatmap(price: pd.Series, out) -> pd.DataFrame:
    table = (
        price.to_frame("p")
        .assign(hour=price.index.hour, dow=price.index.dayofweek)
        .pivot_table(index="hour", columns="dow", values="p", aggfunc="mean")
    )
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap=SEQ_BLUE)
    ax.set_xticks(range(7), DOW_LABELS)
    ax.set_yticks(range(24), [f"{h:02d}" for h in range(24)])
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_ylabel("Hour of day", color=INK_SOFT)
    mid = np.nanmean(table.to_numpy())
    for i in range(24):
        for j in range(7):
            v = table.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > mid else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"Mean price ({UNIT})", color=INK_SOFT)
    cbar.outline.set_visible(False)
    title(ax, "Mean price by hour and weekday")
    save_figure(fig, out)
    return table


def plot_acf(acf: np.ndarray, n: int, out) -> None:
    lags = np.arange(acf.size)
    band = 1.96 / np.sqrt(n)
    fig, ax = new_axes()
    style_axes(ax, ylabel="Autocorrelation")
    ax.fill_between(lags, -band, band, color=INK_SOFT, alpha=0.15, linewidth=0, label="95% band (white noise)")
    ax.plot(lags, acf, color=SERIES_COLORS[0], linewidth=2.0, label="ACF")
    for lag, name in ((24, "24 h"), (48, "48 h"), (168, "1 week"), (336, "2 weeks")):
        if lag < acf.size:
            ax.axvline(lag, color=INK_SOFT, linewidth=0.8, linestyle=(0, (2, 2)))
            ax.text(lag, 1.0, f" {name}\n {acf[lag]:.2f}", color=INK_SOFT, fontsize=8, va="top")
    ax.set_xlabel("Lag (hours)", color=INK_SOFT)
    ax.set_ylim(min(-0.1, acf.min() - 0.05), 1.05)
    title(ax, "Autocorrelation of hourly price", "Justifies the lag set: 1-3 h, 24 h, 48 h and 168 h")
    style_legend(ax.legend(loc="lower left", frameon=True))
    save_figure(fig, out)


def plot_outliers(flags: pd.DataFrame, out) -> None:
    p = flags["price"]
    spikes = flags[flags["spike_high"] | flags["spike_low"]]
    neg = flags[flags["negative"]]
    fig, ax = new_axes()
    ax.plot(p.index, p, color=OBSERVED, linewidth=0.4, alpha=0.6, label="Hourly price")
    ax.scatter(spikes.index, spikes["price"], s=14, color=SERIES_COLORS[1], edgecolor="white",
               linewidth=0.5, zorder=3, label=f"Spike ({len(spikes):,} h)")
    ax.scatter(neg.index, neg["price"], s=10, color=SERIES_COLORS[2], marker="v", edgecolor="white",
               linewidth=0.4, zorder=3, label=f"Negative ({len(neg):,} h)")
    ax.axhline(0, color=INK_SOFT, linewidth=0.8)
    format_time_axis(ax)
    title(ax, "Spikes and negative prices", "Spike = more than k robust SDs from a 7-day rolling median")
    style_legend(ax.legend(loc="upper right", frameon=True, markerscale=1.5))
    save_figure(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exploratory analysis of the price series")
    ap.add_argument("--price", default=DEFAULT_PRICE)
    ap.add_argument("--start", default=None, help="Restrict to prices from this date")
    ap.add_argument("--end", default=None, help="Restrict to prices up to this date")
    ap.add_argument("--acf_lags", type=int, default=336)
    ap.add_argument("--spike_window", type=int, default=168, help="Rolling window (hours) for spike detection")
    ap.add_argument("--spike_k", type=float, default=6.0, help="Robust z threshold for a spike")
    ap.add_argument("--out_dir", default="figs/eda")
    args = ap.parse_args()

    price_eur = load_price_series(repo_path(args.price), tz=TZ)
    price_eur = price_eur.loc[args.start : args.end].dropna()
    price = price_eur * EUR_TO_NOK
    price.name = "price_NOK_per_kWh"

    fig_dir = repo_path(args.out_dir)
    data_dir = ROOT / "data"

    summary = summary_by_year(price)
    summary.to_csv(data_dir / "eda_summary_by_year.csv", index=False)
    overall = summary.iloc[-1].to_dict()

    flags = flag_outliers(price, args.spike_window, args.spike_k)
    flagged = flags[flags[["spike_high", "spike_low", "negative", "iqr_extreme"]].any(axis=1)]
    flagged.to_csv(data_dir / "eda_outliers.csv", index_label="timestamp")

    acf = autocorrelation(price, args.acf_lags)

    plot_timeseries(price, fig_dir / "eda_timeseries.png")
    plot_distribution(price, overall, fig_dir / "eda_distribution.png")
    plot_box(price, price.index.hour, range(24), [f"{h:02d}" for h in range(24)],
             "Hour of day", "Price by hour of day", fig_dir / "eda_by_hour.png")
    plot_box(price, price.index.month, range(1, 13), MONTH_LABELS,
             "Month", "Price by month", fig_dir / "eda_by_month.png")
    heat = plot_heatmap(price, fig_dir / "eda_heatmap.png")
    plot_acf(acf, len(price), fig_dir / "eda_acf.png")
    plot_outliers(flags, fig_dir / "eda_outliers.png")

    headline = {
        "acf_lag_1": float(acf[1]),
        "acf_lag_24": float(acf[24]),
        "acf_lag_168": float(acf[168]) if acf.size > 168 else None,
        "n_spike_high": int(flags["spike_high"].sum()),
        "n_spike_low": int(flags["spike_low"].sum()),
        "n_negative": int(flags["negative"].sum()),
        "n_iqr_extreme": int(flags["iqr_extreme"].sum()),
        "peak_hour_weekday": int(heat.iloc[:, :5].mean(axis=1).idxmax()),
        "trough_hour_weekday": int(heat.iloc[:, :5].mean(axis=1).idxmin()),
    }
    meta = {
        "series": describe_series(price_eur),
        "unit": UNIT,
        "eur_to_nok": EUR_TO_NOK,
        "spike_rule": f"|price - rolling median| > {args.spike_k} x 1.4826 x rolling MAD, "
                      f"centred window {args.spike_window} h",
        "iqr_rule": "price outside [Q1 - 3 IQR, Q3 + 3 IQR] over the whole period",
        "headline": headline,
    }
    (data_dir / "eda_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nPeriod: {price.index[0]} .. {price.index[-1]}   hours: {len(price):,}   unit: {UNIT} (EUR/NOK {EUR_TO_NOK})")
    print(
        f"\nACF  lag 1: {headline['acf_lag_1']:.3f}   lag 24: {headline['acf_lag_24']:.3f}   "
        f"lag 168: {headline['acf_lag_168']:.3f}"
    )
    print(
        f"Spikes: {headline['n_spike_high']} high, {headline['n_spike_low']} low   "
        f"negative hours: {headline['n_negative']}   IQR extremes: {headline['n_iqr_extreme']}"
    )


if __name__ == "__main__":
    main()
