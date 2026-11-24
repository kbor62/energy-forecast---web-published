"""Rolling-origin backtest.

For each origin, every horizon is forecast with a model trained only on labels
known at the last refit time. Models are refit every --refit_every hours.

Weather (optional, --weather_file): adds the weather at t+h as features. Training
always uses observed weather. --weather_mode controls prediction:
  observed  observed weather (oracle / upper bound; default)
  archive   latest archived MET snapshot issued at or before each origin (minus
            --snapshot_lag_hours), with no fallback to observations. Origins
            without a snapshot are skipped.
Outputs get the suffix _wx (observed) or _wxfc (archive); compare them with the
price-only run using --compare.

ENTSO-E exog (--exog_file): neighbouring zones' day-ahead wind/solar forecasts at the
target hour and neighbour prices at the origin (exog.py); outputs get the suffix _ex.
Weather-model forecasts (--nwp_file, nwp.py): suffix _nwp. --decision_lag_hours other
than 11 (e.g. 33 for D+2/D+3 forecasts made at 14:00 on D): suffix _dl<lag>.
Wind/solar values not yet published at the decision time (origin minus 11 h) are NaN.

Origin time: by default origins are --step absolute hours apart from --start. With
--origin_hour H every origin is at local clock hour H (DST-proof; outputs get _o<H>).
The origin is the last KNOWN price hour. Use --origin_hour 23 for the practical case:
forecast made about 12:00 on day D, when D's prices are known and D+1's are not yet
published, so horizons 1-24 are tomorrow's auction prices. Add --snapshot_lag_hours 11
so archived weather forecasts cannot be newer than that decision time.

    python src/backtest_rolling.py --inp data/entsoe_NO_1_2020-01-01_now.parquet --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48
    python src/backtest_rolling.py --weather_file data/weather_oslo.parquet --weather_mode observed --start "2026-01-01 00:00" --end "2026-08-31 00:00"
    python src/backtest_rolling.py --compare data/backtest_rolling_lgbm.csv data/backtest_rolling_lgbm_wx.csv


    # Sammenlignbar med gamle lgbm_int (origin ved midnatt)
python src/backtest_rolling.py --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48

# Praktisk oppsett, origin 23:00
python src/backtest_rolling.py --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48 --origin_hour 23
python src/backtest_rolling.py --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48 --origin_hour 23 --exog_file data/entsoe_exog.parquet
python src/backtest_rolling.py --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48 --origin_hour 23 --weather_file data/weather_oslo.parquet --weather_cols temperature_C
python src/backtest_rolling.py --start "2026-01-01 00:00" --end "2026-08-31 00:00" --step 48 --horizon 48 --origin_hour 23 --weather_file data/weather_oslo.parquet --weather_cols temperature_C --exog_file data/entsoe_exog.parquet

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
import exog as X_
import nwp as N_
import targets as T_
import fx as FX_
import models as M
from config import EUR_TO_NOK, ROOT, TZ, repo_path
from metrics import block_bootstrap_ci, skill, summarize
from plotting import (
    INK,
    INK_SOFT,
    MAX_SERIES,
    SERIES_COLORS,
    SERIES_DASHES,
    fixed_rate_note,
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


def build_origins(start, end, step: int, origin_hour: int | None = None) -> pd.DatetimeIndex:
    """Forecast origins between start and end.

    Default: every ``step`` absolute hours from start (the local clock time then
    drifts by one hour at each DST change). With ``origin_hour``: one origin per
    ``step // 24`` days at that local clock hour, DST-proof (``step`` must be a
    multiple of 24). The origin is the last known price hour: origin_hour=23 means
    the forecast is made around 12:00 the same day, when that day's prices are
    known and tomorrow's are not yet published.
    """
    start, end = to_tz(start), to_tz(end)
    if origin_hour is None:
        return pd.date_range(start, end, freq=f"{step}h", tz=TZ)
    if step % 24:
        raise ValueError("--origin_hour requires --step to be a multiple of 24")
    if not 0 <= origin_hour <= 23:
        raise ValueError("--origin_hour must be 0-23")
    days = pd.date_range(start.tz_localize(None).normalize(), end.tz_localize(None), freq=f"{step // 24}D")
    local = pd.DatetimeIndex(days) + pd.Timedelta(hours=origin_hour)
    origins = local.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT").dropna()
    return origins[(origins >= start) & (origins <= end)]


def run_backtest(args) -> tuple[pd.DataFrame, dict]:
    price = load_price_series(repo_path(args.inp), tz=TZ)
    history = F.history_features(price)
    horizons = list(range(1, args.horizon + 1))
    targets = F.make_targets(price, horizons)

    complete = ~history.isna().any(axis=1)

    # Weather at the target hour. Training always uses observed weather.
    #   observed: predictions also use observed weather (oracle, upper bound)
    #   archive:  predictions use the latest archived MET snapshot issued <= origin
    #             (realistic; only origins with a snapshot are evaluated)
    weather_obs = None
    if args.weather_file:
        weather_obs = load_observed(args.weather_file, args.weather_cols)
        print(f"Weather: {args.weather_cols} from {args.weather_file}   prediction mode: {args.weather_mode}")

    # ENTSO-E exog (wind/solar forecasts at the target hour, neighbour prices at the origin).
    # Values are as published, so no oracle problem; hours not yet published at the decision
    # time (origin - 11 h) are NaN, see exog.py. Used in training and prediction alike.
    exog = None
    if args.exog_file:
        exog = X_.load_exog(args.exog_file, args.exog_cols)
        print(f"Exog: {list(exog.columns)} from {args.exog_file}")

    # Archived weather-model forecasts (Open-Meteo), lead chosen per horizon so that only
    # runs available at the decision time are used (nwp.py).
    nwp = None
    if args.nwp_file:
        nwp = N_.load_nwp(args.nwp_file, args.nwp_cols)
        print(f"NWP: {N_.bases(nwp.columns)} from {args.nwp_file}")
    lag = args.decision_lag_hours
    print(f"Decision lag: {lag} h before the origin")

    origins = build_origins(args.start, args.end, args.step, args.origin_hour)
    snapshot_lag = pd.Timedelta(hours=args.snapshot_lag_hours)
    usable = [
        o
        for o in origins
        if o in history.index
        and complete.loc[o]
        and o + pd.Timedelta(hours=args.horizon) <= price.index.max()
    ]
    n_skipped_no_snapshot = 0
    if weather_obs is not None and args.weather_mode == "archive":
        with_snap = [o for o in usable if pick_snapshot(o - snapshot_lag, args.weather_archive) is not None]
        n_skipped_no_snapshot = len(usable) - len(with_snap)
        if n_skipped_no_snapshot:
            print(f"Skipping {n_skipped_no_snapshot} origins with no forecast snapshot issued at/before them")
        usable = with_snap
    if not usable:
        raise ValueError("No usable origins (check --start/--end, and the snapshot archive for --weather_mode archive)")
    print(f"Model: {M.MODEL_NAME}")
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
                origin - snapshot_lag,
                args.weather_archive,
                observed_path=None,
            )

        for h in horizons:
            estimator = cache.get(h)
            if estimator is None:
                label_cutoff = anchor - pd.Timedelta(hours=h)
                off_all, sc_all = T_.anchor(price, history.index, h, args.target_mode)
                train_index = history.index[
                    (history.index <= label_cutoff)
                    & complete
                    & targets[F.target_column(h)].notna()
                    & np.isfinite(off_all) & np.isfinite(sc_all)
                ]
                if len(train_index) < args.min_train_hours:
                    raise ValueError(
                        f"Only {len(train_index)} training rows for h={h} at {anchor}; "
                        "move --start later or lower --min_train_hours."
                    )

                X = M.prepare_X(F.assemble(history, train_index, h, weather=weather_obs, exog=exog, nwp=nwp,
                                           decision_lag_h=lag))
                off_tr, sc_tr = T_.anchor(price, train_index, h, args.target_mode)
                X = T_.add_features(X, off_tr, sc_tr, args.target_mode)
                y = T_.to_target(targets.loc[train_index, F.target_column(h)].to_numpy(dtype=float), off_tr, sc_tr)

                estimator = M.build_estimator(seed=args.seed)
                info = M.fit_estimator(estimator, X, y)
                cache[h] = estimator
                if h == 1:
                    fit_notes.append({"anchor": str(anchor), **info})

            x_row = M.prepare_X(F.assemble(history, [origin], h, weather=weather_at_origin, exog=exog, nwp=nwp,
                                           decision_lag_h=lag))
            off_o, sc_o = T_.anchor(price, [origin], h, args.target_mode)
            x_row = T_.add_features(x_row, off_o, sc_o, args.target_mode)
            pred = float(T_.to_price(estimator.predict(x_row), off_o, sc_o)[0])

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
    # Daily Norges Bank EUR/NOK at the target date (fixed rate where unknown). The printed
    # NOK metrics keep the fixed config rate so that runs stay comparable with earlier ones.
    out["eur_nok"] = FX_.rates_for(out["timestamp"], FX_.load_rates())

    meta = {
        "run_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "model": M.MODEL_NAME,
        "feature_definition": F.feature_definition(),
        "weather_file": args.weather_file,
        "weather_cols": args.weather_cols if args.weather_file else [],
        "weather_mode": args.weather_mode if args.weather_file else None,
        "exog_file": args.exog_file,
        "exog_cols": list(exog.columns) if exog is not None else [],
        "nwp_file": args.nwp_file,
        "nwp_cols": N_.bases(nwp.columns) if nwp is not None else [],
        "decision_lag_hours": lag,
        "target_mode": args.target_mode,
        "origin_hour": args.origin_hour,
        "snapshot_lag_hours": args.snapshot_lag_hours,
        "origins_skipped_no_snapshot": n_skipped_no_snapshot,
        "n_origins": len(usable),
        "first_origin": str(usable[0]),
        "last_origin": str(usable[-1]),
        "step_hours": args.step,
        "horizon_hours": args.horizon,
        "refit_every_hours": args.refit_every,
        "min_train_hours": args.min_train_hours,
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
    save_figure(fig, out_png, note=fixed_rate_note())


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
    save_figure(fig, out_png, note=fixed_rate_note())


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling-origin backtest for price forecasting")
    ap.add_argument("--inp", default=DEFAULT_INPUT)
    ap.add_argument("--start", help="First forecast origin, e.g. '2026-01-01 00:00'")
    ap.add_argument("--end", help="Last forecast origin")
    ap.add_argument(
        "--step", type=int, default=48,
        help="Hours between forecast origins",
    )
    ap.add_argument(
        "--origin_hour", type=int, default=None,
        help="Local clock hour (0-23) of every origin (= last known price hour); needs --step multiple of 24. "
        "23 = forecast made ~12:00 the same day, before tomorrow's prices are published. Output tag gets _o<hour>.",
    )
    ap.add_argument(
        "--snapshot_lag_hours", type=int, default=0,
        help="archive mode: snapshots must be issued at least this many hours before the origin "
        "(the decision is made earlier than the last known price hour; 11 for --origin_hour 23)",
    )
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--refit_every", type=int, default=720, help="Refit every N hours (default 30 days)")
    ap.add_argument("--min_train_hours", type=int, default=8760)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weather_file", default=None, help="Observed weather parquet; adds weather at t+h as features")
    ap.add_argument("--weather_cols", nargs="+", default=["temperature_C", "wind_speed_ms", "precipitation_mm"])
    ap.add_argument(
        "--weather_mode", choices=["observed", "archive"], default="observed",
        help="observed: predict with observed weather (upper bound). "
        "archive: predict with MET snapshots issued before each origin (realistic)",
    )
    ap.add_argument("--weather_archive", default=ARCHIVE_DIR)
    ap.add_argument("--exog_file", default=None, help="ENTSO-E exog parquet (wind/solar forecasts, neighbour prices); see exog.py")
    ap.add_argument("--exog_cols", nargs="+", default=None, help="Exog columns (default: all in the file)")
    N_.add_args(ap)
    T_.add_args(ap)
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

    tag = M.MODEL_NAME
    if args.weather_file:
        tag += "_wx" if args.weather_mode == "observed" else "_wxfc"
    if args.exog_file:
        tag += "_ex"
    if args.nwp_file:
        tag += "_nwp"
    if args.target_mode != "level":
        tag += f"_t{args.target_mode}"
    if args.decision_lag_hours != X_.DECISION_LAG_H:
        tag += f"_dl{args.decision_lag_hours}"
    if args.origin_hour is not None:
        tag += f"_o{args.origin_hour}"
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
