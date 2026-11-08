"""Rolling-origin backtest.

For each origin, every horizon is forecast with a model trained only on labels
known at the last refit time. Models are refit every --refit_every hours.

Weather (optional, --weather_file): adds the weather at t+h as features. Training
always uses observed weather. --weather_mode controls prediction:
  observed  observed weather (oracle / upper bound; default)
  archive   latest archived MET snapshot issued at or before each origin, with no
            fallback to observations. Origins without a snapshot are skipped.
Outputs get the suffix _wx (observed) or _wxfc (archive); compare them with the
price-only run using --compare.

    python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --model lgbm --encoding int --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48
    python src/backtest_rolling.py --model ridge_ohe --encoding both --start "2026-01-01 00:00" --end "2026-08-31 00:00"
    python src/backtest_rolling.py --model lgbm --encoding int --weather_file data/weather_oslo.parquet --weather_mode observed --start "2026-01-01 00:00" --end "2026-08-31 00:00"
    python src/backtest_rolling.py --compare data/backtest_rolling_lgbm_int.csv data/backtest_rolling_ridge_ohe_both.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import features as F
import models as M
from config import EUR_TO_NOK, ROOT, TZ, repo_path
from metrics import block_bootstrap_ci, skill, summarize
from plotting import (
    INK,
    INK_SOFT,
    MAX_SERIES,
    SERIES_COLORS,
    SERIES_DASHES,
    save_figure,
    style_axes,
    style_legend,
)
from price_io import load_price_series
from weather_io import ARCHIVE_DIR, load_observed, pick_snapshot, weather_for_forecast

DEFAULT_INPUT = "data/entsoe_NO_1_2020-01-01_now.parquet"


def to_tz(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize(TZ) if ts.tz is None else ts.tz_convert(TZ)
    return ts.floor("h")


def run_backtest(args) -> tuple[pd.DataFrame, dict]:
    price = load_price_series(repo_path(args.inp), tz=TZ)
    history = F.history_features(price)
    horizons = list(range(1, args.horizon + 1))
    targets = F.make_targets(price, horizons)

    encoding = args.encoding or M.DEFAULT_ENCODING[args.model]
    complete = ~history.isna().any(axis=1)

    # Weather at the target hour. Training always uses observed weather.
    #   observed: predictions also use observed weather (oracle, upper bound)
    #   archive:  predictions use the latest archived MET snapshot issued <= origin
    #             (realistic; only origins with a snapshot are evaluated)
    weather_obs = None
    extra_cols: list[str] = []
    if args.weather_file:
        weather_obs = load_observed(args.weather_file, args.weather_cols)
        extra_cols = [f"{c}_target" for c in args.weather_cols]
        print(f"Weather: {args.weather_cols} from {args.weather_file}   prediction mode: {args.weather_mode}")

    origins = pd.date_range(to_tz(args.start), to_tz(args.end), freq=f"{args.step}h", tz=TZ)
    usable = [
        o
        for o in origins
        if o in history.index
        and complete.loc[o]
        and o + pd.Timedelta(hours=args.horizon) <= price.index.max()
    ]
    n_skipped_no_snapshot = 0
    if weather_obs is not None and args.weather_mode == "archive":
        with_snap = [o for o in usable if pick_snapshot(o, args.weather_archive) is not None]
        n_skipped_no_snapshot = len(usable) - len(with_snap)
        if n_skipped_no_snapshot:
            print(f"Skipping {n_skipped_no_snapshot} origins with no forecast snapshot issued at/before them")
        usable = with_snap
    if not usable:
        raise ValueError("No usable origins (check --start/--end, and the snapshot archive for --weather_mode archive)")
    print(f"Model: {args.model}   encoding: {encoding}")
    print(f"Origins: {len(usable)}  ({usable[0]} .. {usable[-1]}, step {args.step}h)")
    print(f"Horizons: 1..{args.horizon}   refit every {args.refit_every}h")

    refit_step = pd.Timedelta(hours=args.refit_every)
    cache: dict[int, object] = {}
    anchor: pd.Timestamp | None = None
    rows: list[dict] = []
    fit_notes: list[dict] = []

    for i, origin in enumerate(usable, start=1):
        if anchor is None or origin - anchor >= refit_step:
            anchor = origin
            cache = {}
            print(f"  [{i}/{len(usable)}] refit at {origin}")

        origin_price = float(price.loc[origin])

        weather_at_origin = weather_obs
        if weather_obs is not None and args.weather_mode == "archive":
            # Snapshot only: no observed fallback, so no future observations leak in.
            weather_at_origin, _ = weather_for_forecast(
                args.weather_cols,
                origin + pd.to_timedelta(horizons, unit="h"),
                origin,
                args.weather_archive,
                observed_path=None,
            )

        for h in horizons:
            estimator = cache.get(h)
            if estimator is None:
                label_cutoff = anchor - pd.Timedelta(hours=h)
                train_index = history.index[
                    (history.index <= label_cutoff)
                    & complete
                    & targets[F.target_column(h)].notna()
                ]
                if len(train_index) < args.min_train_hours:
                    raise ValueError(
                        f"Only {len(train_index)} training rows for h={h} at {anchor}; "
                        "move --start later or lower --min_train_hours."
                    )

                X = M.prepare_X(
                    args.model, F.assemble(history, train_index, h, encoding, weather=weather_obs), encoding
                )
                y = targets.loc[train_index, F.target_column(h)].to_numpy(dtype=float)

                estimator = M.build_estimator(
                    args.model, encoding, alpha=args.alpha, seed=args.seed, extra_cols=extra_cols
                )
                info = M.fit_estimator(args.model, estimator, X, y)
                cache[h] = estimator
                if h == 1:
                    fit_notes.append({"anchor": str(anchor), **info})

            x_row = M.prepare_X(
                args.model, F.assemble(history, [origin], h, encoding, weather=weather_at_origin), encoding
            )
            pred = float(estimator.predict(x_row)[0])

            target_ts = origin + pd.Timedelta(hours=h)
            weekly_ts = target_ts - pd.Timedelta(hours=168)
            rows.append(
                {
                    "origin": origin,
                    "timestamp": target_ts,
                    "horizon_hours": h,
                    "predicted": pred,
                    "actual": float(price.get(target_ts, np.nan)),
                    "baseline_persistence": origin_price,
                    "baseline_weekly_t168": float(price.get(weekly_ts, np.nan)),
                }
            )

    out = pd.DataFrame(rows)
    out["error"] = out["predicted"] - out["actual"]

    meta = {
        "run_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "model": args.model,
        "encoding": encoding,
        "feature_definition": F.feature_definition(encoding),
        "weather_file": args.weather_file,
        "weather_cols": args.weather_cols if args.weather_file else [],
        "weather_mode": args.weather_mode if args.weather_file else None,
        "origins_skipped_no_snapshot": n_skipped_no_snapshot,
        "n_origins": len(usable),
        "first_origin": str(usable[0]),
        "last_origin": str(usable[-1]),
        "step_hours": args.step,
        "horizon_hours": args.horizon,
        "refit_every_hours": args.refit_every,
        "min_train_hours": args.min_train_hours,
        "alpha": args.alpha,
        "seed": args.seed,
        "library_versions": M.library_versions(),
        "fit_notes": fit_notes[:20],
    }
    return out, meta


def per_horizon_summary(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    records = []
    for h, grp in df.groupby("horizon_hours"):
        y = grp["actual"].to_numpy(dtype=float) * scale
        record = {"horizon_hours": int(h), "n_origins": int(grp["origin"].nunique())}
        for name, col in (
            ("model", "predicted"),
            ("persistence", "baseline_persistence"),
            ("weekly", "baseline_weekly_t168"),
        ):
            m = summarize(y, grp[col].to_numpy(dtype=float) * scale)
            record[f"{name}_mae"] = m["mae"]
            record[f"{name}_rmse"] = m["rmse"]
        records.append(record)

    summary = pd.DataFrame(records).sort_values("horizon_hours").reset_index(drop=True)
    summary["skill_vs_persistence"] = [
        skill(m, b) for m, b in zip(summary["model_mae"], summary["persistence_mae"])
    ]
    summary["skill_vs_weekly"] = [
        skill(m, b) for m, b in zip(summary["model_mae"], summary["weekly_mae"])
    ]
    return summary


def per_origin_mae(df: pd.DataFrame, column: str, scale: float) -> pd.Series:
    """Mean absolute error per forecast origin."""
    return (
        df.assign(abs_err=(df[column] - df["actual"]).abs() * scale)
        .groupby("origin")["abs_err"]
        .mean()
    )


def print_report(df: pd.DataFrame, scale: float, args, model_name: str) -> pd.DataFrame:
    y = df["actual"].to_numpy(dtype=float) * scale
    print("\n" + "=" * 68)
    print(f"Rolling-origin backtest: {model_name}")
    print(f"Origins: {df['origin'].nunique()}   forecasts: {len(df)}   unit: NOK/kWh")
    print("=" * 68)

    for label, col in (
        ("model", "predicted"),
        ("persistence", "baseline_persistence"),
        ("weekly t-168", "baseline_weekly_t168"),
    ):
        m = summarize(y, df[col].to_numpy(dtype=float) * scale)
        print(
            f"{label:<14} MAE {m['mae']:.4f}   RMSE {m['rmse']:.4f}   "
            f"nMAE {m['nmae']:.1%}   n={m['n']}"
        )

    origin_mae = per_origin_mae(df, "predicted", scale)
    lo, hi = np.percentile(origin_mae.dropna(), [5, 95])
    print(
        f"\nPer-origin MAE: mean {origin_mae.mean():.4f}   median "
        f"{origin_mae.median():.4f}   5-95% [{lo:.4f}, {hi:.4f}]"
    )

    # Paired comparison against each baseline, one MAE per origin.
    block = max(1, int(np.ceil(args.horizon / max(1, args.step))))
    for label, col in (
        ("persistence", "baseline_persistence"),
        ("weekly t-168", "baseline_weekly_t168"),
    ):
        baseline_mae = per_origin_mae(df, col, scale)
        diff = (origin_mae - baseline_mae).dropna()
        if len(diff) < 2:
            continue
        ci_lo, ci_hi = block_bootstrap_ci(diff, block_hours=block, seed=args.seed)
        wins = float((diff < 0).mean())
        verdict = "better" if ci_hi < 0 else ("worse" if ci_lo > 0 else "not distinguishable")
        print(
            f"Model minus {label}: {diff.mean():+.4f} NOK/kWh   "
            f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}] (block bootstrap, block={block} origins)   "
            f"model better on {wins:.0%} of origins -> {verdict}"
        )

    return per_horizon_summary(df, scale)


def plot_mae_by_horizon(summary: pd.DataFrame, model_name: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(summary["horizon_hours"], summary["model_mae"],
            label=model_name, lw=2, color=SERIES_COLORS[0])
    ax.plot(summary["horizon_hours"], summary["persistence_mae"],
            label="persistence", lw=1.4, ls="--", color=INK_SOFT)
    ax.plot(summary["horizon_hours"], summary["weekly_mae"],
            label="weekly t-168", lw=1.4, ls=":", color=INK)
    ax.set_xlabel("Forecast horizon (hours ahead)", color=INK_SOFT, fontsize=10)
    ax.set_title(
        f"Error by horizon, averaged over {int(summary['n_origins'].max())} origins",
        color=INK, fontsize=13, loc="left", pad=12,
    )
    ax.set_ylim(bottom=0)
    style_axes(ax, "MAE (NOK/kWh)")
    style_legend(ax.legend(loc="upper left", frameon=True, fontsize=9))
    fig.tight_layout()
    save_figure(fig, out_png)


def compare_runs(paths: list[Path], scale: float, out_png: Path) -> None:
    """Compare prediction CSVs on the (origin, horizon) pairs they all share."""
    frames: dict[str, pd.DataFrame] = {}
    for p in paths:
        df = pd.read_csv(p)
        df["origin"] = pd.to_datetime(df["origin"], utc=True).dt.tz_convert(TZ)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
        frames[p.stem.replace("backtest_rolling_", "")] = df

    # Use only the (origin, horizon) pairs present in every run.
    common: set | None = None
    for df in frames.values():
        keys = set(zip(df["origin"], df["horizon_hours"]))
        common = keys if common is None else common & keys
    print("\n" + "=" * 68)
    print(f"Comparison across runs (common origins only): {len(common)} pairs")
    print("=" * 68)

    fig, ax = plt.subplots(figsize=(11, 6))
    baseline_drawn = False
    for i, (name, df) in enumerate(frames.items()):
        mask = [k in common for k in zip(df["origin"], df["horizon_hours"])]
        df = df.loc[mask].sort_values(["origin", "horizon_hours"])
        y = df["actual"].to_numpy(dtype=float) * scale
        m = summarize(y, df["predicted"].to_numpy(dtype=float) * scale)
        print(f"{name:<34} MAE {m['mae']:.4f}   nMAE {m['nmae']:.1%}   n={m['n']}")
        s = per_horizon_summary(df, scale)
        ax.plot(
            s["horizon_hours"], s["model_mae"], lw=2, label=name,
            color=SERIES_COLORS[i % MAX_SERIES],
            linestyle=SERIES_DASHES[i % MAX_SERIES],
        )

        if not baseline_drawn:
            for label, col, style in (
                ("weekly t-168", "baseline_weekly_t168", ":"),
                ("persistence", "baseline_persistence", "--"),
            ):
                if col not in df.columns:
                    continue
                b = summarize(y, df[col].to_numpy(dtype=float) * scale)
                print(f"{label + ' baseline':<34} MAE {b['mae']:.4f}   nMAE {b['nmae']:.1%}")
                ax.plot(
                    s["horizon_hours"],
                    s["weekly_mae" if col.endswith("t168") else "persistence_mae"],
                    lw=1.4, ls=style, label=label,
                    color=INK if col.endswith("t168") else INK_SOFT,
                )
            baseline_drawn = True

    ax.set_xlabel("Forecast horizon (hours ahead)", color=INK_SOFT, fontsize=10)
    ax.set_title("Error by horizon: model comparison", color=INK, fontsize=13,
                 loc="left", pad=12)
    ax.set_ylim(bottom=0)
    style_axes(ax, "MAE (NOK/kWh)")
    style_legend(ax.legend(loc="upper left", frameon=True, fontsize=9))
    fig.tight_layout()
    save_figure(fig, out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling-origin backtest for price forecasting")
    ap.add_argument("--inp", default=DEFAULT_INPUT)
    ap.add_argument("--model", choices=M.MODELS, default="ridge_ohe")
    ap.add_argument("--encoding", choices=F.ENCODINGS, default=None)
    ap.add_argument("--start", help="First forecast origin, e.g. '2026-01-01 00:00'")
    ap.add_argument("--end", help="Last forecast origin")
    ap.add_argument(
        "--step", type=int, default=48,
        help="Hours between forecast origins",
    )
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--refit_every", type=int, default=720, help="Refit every N hours (default 30 days)")
    ap.add_argument("--min_train_hours", type=int, default=8760)
    ap.add_argument("--alpha", default="auto", help="Ridge alpha, or 'auto' for RidgeCV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weather_file", default=None, help="Observed weather parquet; adds weather at t+h as features")
    ap.add_argument("--weather_cols", nargs="+", default=["temperature_C", "wind_speed_ms", "precipitation_mm"])
    ap.add_argument(
        "--weather_mode", choices=["observed", "archive"], default="observed",
        help="observed: predict with observed weather (upper bound). "
        "archive: predict with MET snapshots issued before each origin (realistic)",
    )
    ap.add_argument("--weather_archive", default=ARCHIVE_DIR)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_summary", default=None)
    ap.add_argument("--out_png", default=None)
    ap.add_argument("--compare", nargs="+", default=None, help="Compare existing prediction CSVs instead")
    args = ap.parse_args()

    if args.compare:
        out_png = repo_path(args.out_png) if args.out_png else ROOT / "figs" / "compare_by_horizon.png"
        compare_runs([repo_path(p) for p in args.compare], EUR_TO_NOK, out_png)
        return

    if not args.start or not args.end:
        ap.error("--start and --end are required unless --compare is used")

    df, meta = run_backtest(args)
    meta["eur_nok"] = EUR_TO_NOK

    tag = f"{args.model}_{meta['encoding']}"
    if args.weather_file:
        tag += "_wx" if args.weather_mode == "observed" else "_wxfc"
    out_csv = repo_path(args.out_csv or f"data/backtest_rolling_{tag}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_csv.with_name(out_csv.stem + "_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    summary = print_report(df, EUR_TO_NOK, args, tag)

    out_summary = repo_path(args.out_summary or f"data/backtest_rolling_{tag}_by_horizon.csv")
    summary.to_csv(out_summary, index=False)

    plot_mae_by_horizon(summary, tag, repo_path(args.out_png or f"figs/backtest_rolling_{tag}.png"))


if __name__ == "__main__":
    sys.exit(main())
