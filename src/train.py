"""Train direct multi-step models, one per horizon.

The last --val_days days are held out for validation. Training for horizon h
stops h hours before the validation window (the embargo), so that no training
label falls inside it.

    python src/train.py --inp data/features_price_only.parquet --model lgbm --encoding int --horizon 48 --val_days 30
    python src/train.py --model ridge --encoding trig
    python src/train.py --model ridge_ohe --encoding both

Weather (optional): --weather_file adds the weather at the target hour t+h
(<col>_target) as features, using OBSERVED values, so validation scores are an
upper bound on the value of a weather forecast. Use a separate --out_dir
(e.g. models/lgbm_int_wx) to keep the price-only models. The manifest records
weather_file and weather_cols; forecast.py reads them to know what to supply.

    python src/train.py --model lgbm --encoding int --weather_file data/weather_oslo.parquet --out_dir models/lgbm_int_wx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

import features as F
import models as M
from config import TZ, repo_path
from metrics import summarize

DEFAULT_INPUT = "data/features_price_only.parquet"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train direct multi-step price models")
    ap.add_argument("--inp", default=DEFAULT_INPUT, help="Feature parquet written by build_features.py")
    ap.add_argument("--out_dir", default=None, help="Model output dir (default: models/<model>_<encoding>)")
    ap.add_argument("--model", choices=M.MODELS, default="lgbm")
    ap.add_argument(
        "--encoding",
        choices=F.ENCODINGS,
        default=None,
        help="Calendar encoding (default depends on the model)",
    )
    ap.add_argument("--horizon", type=int, default=48, help="Train horizons 1..N")
    ap.add_argument("--val_start", default=None, help="YYYY-MM-DD start of the validation window")
    ap.add_argument("--val_days", type=int, default=30, help="If no --val_start, last N days")
    ap.add_argument("--alpha", default="auto", help="Ridge alpha, or 'auto' for RidgeCV")
    ap.add_argument(
        "--weather_file",
        default=None,
        help="Hourly weather parquet (e.g. data/weather_oslo.parquet). Adds weather at the "
        "target hour t+h as features, using OBSERVED values (upper bound on forecast value).",
    )
    ap.add_argument(
        "--weather_cols",
        nargs="+",
        default=["temperature_C", "wind_speed_ms", "precipitation_mm"],
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no_embargo",
        action="store_true",
        help="Disable the embargo between training and validation",
    )
    ap.add_argument(
        "--keep_warmup",
        action="store_true",
        help="Keep the first hours where long lags are still NaN",
    )
    return ap.parse_args()


def clear_stale_models(out_dir: Path, horizons: list[int]) -> None:
    """Remove models from an earlier run for horizons outside this run."""
    planned = {out_dir / f"price_model_t{h}h.joblib" for h in horizons}
    stale = sorted(p for p in out_dir.glob("price_model_t*h.joblib") if p not in planned)
    if not stale:
        return

    for model_path in stale:
        model_path.unlink()
        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        if meta_path.exists():
            meta_path.unlink()
    print(f"Removed {len(stale)} stale model(s) from a previous run: {[p.name for p in stale]}")


def main() -> None:
    args = parse_args()
    encoding = args.encoding or M.DEFAULT_ENCODING[args.model]
    out_dir = repo_path(args.out_dir or f"models/{args.model}_{encoding}")
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_path = repo_path(args.inp)

    df = pd.read_parquet(feature_path)
    df.index.name = "timestamp"
    df.index = df.index.tz_localize(TZ) if df.index.tz is None else df.index.tz_convert(TZ)
    df = df.sort_index()

    history = df[F.history_columns()]
    weather = None
    if args.weather_file:
        weather = pd.read_parquet(repo_path(args.weather_file))
        weather.index = (
            weather.index.tz_localize(TZ) if weather.index.tz is None else weather.index.tz_convert(TZ)
        )
        weather = weather[~weather.index.duplicated()].sort_index()[args.weather_cols]
        print(f"Weather features (observed, at t+h): {list(weather.columns)}")
    if not args.keep_warmup:
        history = history.iloc[F.MAX_HISTORY_HOURS :]
        df = df.iloc[F.MAX_HISTORY_HOURS :]

    if args.val_start:
        val_start = pd.Timestamp(args.val_start, tz=TZ)
    else:
        val_start = df.index[-1] - pd.Timedelta(days=args.val_days)

    horizons = list(range(1, args.horizon + 1))
    embargo = not args.no_embargo

    clear_stale_models(out_dir, horizons)

    print(f"Model: {args.model}   encoding: {encoding}   features: {len(F.feature_columns(encoding)) + (len(weather.columns) if weather is not None else 0)}")
    print(f"Rows: {len(df):,}   span {df.index[0]} .. {df.index[-1]}")
    print(f"Validation starts: {val_start}   embargo: {'h hours' if embargo else 'DISABLED'}")

    versions = M.library_versions()
    results: list[dict] = []

    for horizon in horizons:
        target_col = F.target_column(horizon)
        y_all = df[target_col]
        labelled = y_all.notna()
        X_all = F.assemble(history, df.index[labelled], horizon, encoding, weather=weather)
        y_all = y_all[labelled]

        train_end = val_start - pd.Timedelta(hours=horizon) if embargo else val_start
        train_mask, val_mask = F.train_val_masks(X_all.index, val_start, horizon, embargo)

        X_train = M.prepare_X(args.model, X_all[train_mask], encoding)
        X_val = M.prepare_X(args.model, X_all[val_mask], encoding)
        y_train = y_all[train_mask].to_numpy(dtype=float)
        y_val = y_all[val_mask].to_numpy(dtype=float)

        estimator = M.build_estimator(
            args.model, encoding, alpha=args.alpha, seed=args.seed,
            extra_cols=[c for c in X_all.columns if c.endswith("_target")],
        )
        fit_info = M.fit_estimator(args.model, estimator, X_train, y_train)

        y_train_pred = estimator.predict(X_train)
        y_val_pred = estimator.predict(X_val)

        train_metrics = summarize(y_train, y_train_pred)
        val_metrics = summarize(y_val, y_val_pred)

        # 80 % prediction interval from the quantiles of the validation residuals.
        residuals = y_val - y_val_pred
        prediction_intervals = {
            "pi_80": {
                "lower": float(np.percentile(residuals, 10)),
                "upper": float(np.percentile(residuals, 90)),
            },
            "method": "empirical quantiles of validation residuals (y - yhat)",
            "n_residuals": int(residuals.size),
        }

        payload = {
            "estimator": estimator,
            "model": args.model,
            "encoding": encoding,
            "features": list(X_all.columns),
            "categorical_features": F.categorical_columns(encoding),
            "horizon": horizon,
            "target": target_col,
        }
        model_path = out_dir / f"price_model_t{horizon}h.joblib"
        dump(payload, model_path)

        meta = {
            "horizon_hours": horizon,
            "target": target_col,
            "model": args.model,
            "encoding": encoding,
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "val_start": str(val_start),
            "train_end": str(train_end),
            "embargo_hours": horizon if embargo else 0,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "prediction_intervals": prediction_intervals,
            "fit_info": fit_info,
            "seed": args.seed,
            "library_versions": versions,
            "units": "EUR/kWh",
        }
        (out_dir / f"price_model_t{horizon}h_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        results.append(
            {
                "horizon_hours": horizon,
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_nmae": val_metrics["nmae"],
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "best_iteration": fit_info.get("best_iteration"),
            }
        )

        print(
            f"  t+{horizon:>2}h  train MAE {train_metrics['mae']:.4f}   "
            f"val MAE {val_metrics['mae']:.4f}   val RMSE {val_metrics['rmse']:.4f}"
            + (f"   trees {fit_info['best_iteration']}" if fit_info.get("best_iteration") else "")
        )

    summary = pd.DataFrame(results)
    summary_path = out_dir / "training_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "trained_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "model": args.model,
        "encoding": encoding,
        "horizons": [int(r["horizon_hours"]) for r in results],
        "feature_file": str(feature_path),
        "weather_file": args.weather_file,
        "weather_cols": list(weather.columns) if weather is not None else [],
        "feature_definition": F.feature_definition(encoding),
        "val_start": str(val_start),
        "embargo": embargo,
        "seed": args.seed,
        "library_versions": versions,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print(f"Mean validation MAE across horizons: {summary['val_mae'].mean():.4f} EUR/kWh")


if __name__ == "__main__":
    main()
