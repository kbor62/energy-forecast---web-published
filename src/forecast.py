"""Forecast the next 48 hours from the last observed price.

Loads a trained model and generates forecasts with prediction intervals.

Weather models (manifest lists weather_cols) also need weather at origin+h. It is
taken from the latest archived MET snapshot issued at or before --weather_as_of
(default: now, or the --origin if one is given); hours the snapshot does not
cover fall back to observed weather (--weather_obs, default: the file the model
was trained with). The run prints how many values came from each source and
stores the details under "weather" in the JSON output. Price-only models ignore
all of this.

Models trained with --exog_file also need ENTSO-E exog data (day-ahead wind/solar
forecasts of neighbouring zones, neighbour prices; see exog.py): the file in the
manifest, or --exog_file. Refresh it with fetch_entsoe_exog.py first. Forecast hours
whose wind/solar forecast is not published at the decision time are passed as NaN.

Models trained with --nwp_file need the archived weather-model forecasts refreshed with
fetch_openmeteo.py (the file in the manifest, or --nwp_file). The decision lag the model
was trained for (manifest decision_lag_hours: 11 = before the auction, 33 = D+2/D+3 after
the auction) decides which forecast run is used per hour; run the forecast accordingly
(lag 33: after about 13:00, when tomorrow's prices are known).

    python src/forecast.py --price data/entsoe_NO_1_2020-01-01_now.parquet --model_dir models/lgbm --horizon 48 --save_png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import load

import features as F
import exog as X_
import nwp as N_
import targets as T_
import fx as FX_
import models as M
from config import ROOT, TZ, repo_path
from plotting import save_figure
from price_io import load_price_series
from weather_io import ARCHIVE_DIR, weather_for_forecast

DEFAULT_PRICE = "data/entsoe_NO_1_2020-01-01_now.parquet"


def load_manifest(model_dir: Path) -> dict:
    manifest_path = model_dir / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_models(model_dir: Path, horizon: int) -> dict[int, dict]:
    """Load the model and metadata for each horizon."""
    models: dict[int, dict] = {}

    for h in range(1, horizon + 1):
        payload = load(model_dir / f"price_model_t{h}h.joblib")
        meta_path = model_dir / f"price_model_t{h}h_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        models[h] = {"payload": payload, "meta": meta, "intervals": meta["prediction_intervals"]}

    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="Forecast the next N hours with a trained model set")
    ap.add_argument("--price", default=DEFAULT_PRICE, help="Raw price CSV/parquet")
    ap.add_argument("--model_dir", default="models/lgbm", help="Directory of trained models")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--origin", default=None, help="Override the forecast origin timestamp")
    ap.add_argument("--weather_archive", default=ARCHIVE_DIR, help="MET forecast snapshots (weather models only)")
    ap.add_argument("--weather_obs", default=None, help="Observed weather parquet (default: the file the model was trained with)")
    ap.add_argument(
        "--weather_as_of",
        default=None,
        help="Use the latest snapshot issued at/before this time (default: now, or --origin if given)",
    )
    ap.add_argument("--exog_file", default=None, help="ENTSO-E exog parquet (default: the file the model was trained with)")
    ap.add_argument("--nwp_file", default=None, help="Open-Meteo NWP parquet (default: the file the model was trained with)")
    ap.add_argument("--fx_file", default=FX_.DEFAULT_FILE, help="EUR/NOK rates from fx.py (fixed rate if missing)")
    ap.add_argument("--save_png", action="store_true")
    ap.add_argument("--history_hours", type=int, default=72, help="History shown in the plot")
    args = ap.parse_args()

    model_dir = repo_path(args.model_dir)
    tag = model_dir.name
    out_csv = repo_path(args.out_csv or f"data/forecast_{tag}.csv")
    out_json = repo_path(args.out_json or f"data/forecast_{tag}.json")

    price = load_price_series(repo_path(args.price), tz=TZ)
    manifest = load_manifest(model_dir)
    models = load_models(model_dir, args.horizon)
    model_name = models[1]["payload"]["model"]

    observed = price.dropna()

    if args.origin:
        origin = pd.Timestamp(args.origin)
        origin = origin.tz_localize(TZ) if origin.tz is None else origin.tz_convert(TZ)
        origin = origin.floor("h")
    else:
        origin = observed.index[-1]

    history = F.history_features(price)

    # Models trained with ENTSO-E exog features (wind/solar forecasts, neighbour prices) need the
    # same file, refreshed with fetch_entsoe_exog.py. Wind/solar hours not yet published are NaN.
    exog_cols = manifest.get("exog_cols") or []
    exog = None
    if exog_cols:
        exog_path = args.exog_file or manifest.get("exog_file")
        exog = X_.load_exog(exog_path, exog_cols)
        n_exog = len(X_.feature_names(exog_cols))
        last = exog.dropna(how="all").index.max()
        print(f"Exog: {exog_path}   {len(exog_cols)} columns -> {n_exog} features   data until {last}")
        if last < origin:
            print("  WARNING: exog data ends before the origin; run fetch_entsoe_exog.py")
        elif int(manifest.get("decision_lag_hours", X_.DECISION_LAG_H)) <= X_.DECISION_LAG_H \
                and last < origin + pd.Timedelta(hours=24):
            print("  WARNING: tomorrow's wind/solar/load forecasts are missing; run fetch_entsoe_exog.py")

    lag = int(manifest.get("decision_lag_hours", X_.DECISION_LAG_H))
    nwp_cols = manifest.get("nwp_cols") or []
    nwp = None
    if nwp_cols:
        nwp_path = args.nwp_file or manifest.get("nwp_file")
        nwp = N_.load_nwp(nwp_path, nwp_cols)
        print(f"NWP: {nwp_path}   {nwp_cols}   data until {nwp.dropna(how='all').index.max()}")
    print(f"Decision lag: {lag} h (decision time for this origin: {origin - pd.Timedelta(hours=lag)})")

    # Weather-trained models need weather at origin+h: archived forecast where it
    # covers the hour, observations otherwise. Models without weather skip this.
    weather_cols = manifest.get("weather_cols") or []
    weather, weather_info = None, None
    if weather_cols:
        if args.weather_as_of:
            as_of = pd.Timestamp(args.weather_as_of)
            as_of = as_of.tz_localize(TZ) if as_of.tzinfo is None else as_of
        else:
            as_of = origin if args.origin else pd.Timestamp.now(tz=TZ)
        targets = origin + pd.to_timedelta(range(1, args.horizon + 1), unit="h")
        weather, weather_info = weather_for_forecast(
            weather_cols, targets, as_of, args.weather_archive, args.weather_obs or manifest.get("weather_file"),
        )
        print(
            f"Weather: snapshot {weather_info['snapshot_issued_at']}   "
            f"snapshot cells {weather_info['cells_from_snapshot']}, observed {weather_info['cells_from_observed']}, "
            f"missing {weather_info['cells_missing']} of {weather_info['cells_total']}"
        )
        if weather_info["snapshot_issued_at"] is None:
            print("  WARNING: no forecast snapshot found - using observations only; run fetch_weather_forecast.py")
        if weather_info["cells_missing"]:
            print("  WARNING: missing weather values are passed to the model as NaN")
    print(
        f"Model: {model_name}   horizons: {len(models)}   "
        f"trained {manifest.get('trained_at', '?')[:16]}"
    )
    print(f"Forecast origin: {origin}  (price {float(price.loc[origin]):.4f} EUR/kWh)")

    rows = []
    nwp_missing_hours: list[int] = []
    for h in sorted(models):
        payload = models[h]["payload"]
        X = F.assemble(history, [origin], h, weather=weather, exog=exog, nwp=nwp, decision_lag_h=lag)
        mode = payload.get("target_mode", "level")
        off, sc = T_.anchor(price, [origin], h, mode)
        X = T_.add_features(X, off, sc, mode)
        if list(X.columns) != list(payload["features"]):
            raise RuntimeError(
                f"t+{h}h: model expects {len(payload['features'])} features, built {X.shape[1]}. "
                "Model/manifest mismatch (weather/exog/NWP columns), or a model trained with an older "
                "feature set: retrain with train.py."
            )
        if nwp is not None:
            missing_nwp = [c for c in N_.feature_names(nwp_cols) if pd.isna(X[c].iloc[0])]
            if missing_nwp:
                nwp_missing_hours.append(h)

        X = M.prepare_X(X)
        pred = float(T_.to_price(payload["estimator"].predict(X), off, sc)[0])

        pi80 = models[h]["intervals"]["pi_80"]
        rows.append(
            {
                "timestamp": origin + pd.Timedelta(hours=h),
                "horizon_hours": h,
                "predicted_EUR_per_kWh": pred,
                "pi_80_lower_EUR": pred + pi80["lower"],
                "pi_80_upper_EUR": pred + pi80["upper"],
            }
        )

    if nwp_missing_hours:
        print(f"  WARNING: NWP values missing for {len(nwp_missing_hours)} horizons (passed as NaN); "
              "run fetch_openmeteo.py")
    forecast = pd.DataFrame(rows).set_index("timestamp").sort_index()
    eur_nok, fx_source = FX_.rate_on(pd.Timestamp.now(tz=TZ), FX_.load_rates(args.fx_file))
    print(f"EUR/NOK {eur_nok:.4f} ({fx_source})")
    if fx_source.startswith("fixed"):
        print("  WARNING: no recent Norges Bank rate; run python src/fx.py")
    for col in [c for c in forecast.columns if c.endswith("_EUR") or c.endswith("_EUR_per_kWh")]:
        forecast[col.replace("EUR", "NOK")] = forecast[col] * eur_nok

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(out_csv)

    document = {
        "generated_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "forecast_origin": origin.isoformat(),
        "origin_price_EUR_per_kWh": float(price.loc[origin]),
        "model": model_name,
        "decision_lag_hours": lag,
        "target_mode": manifest.get("target_mode", "level"),
        "model_dir": str(model_dir),
        "model_trained_at": manifest.get("trained_at"),
        "weather": weather_info,
        "eur_nok": eur_nok,
        "eur_nok_source": fx_source,
        "interval_method": models[min(models)]["intervals"].get("method"),
        "horizons": [
            {
                "timestamp": ts.isoformat(),
                "hours_ahead": int(row["horizon_hours"]),
                "price_EUR_per_kWh": float(row["predicted_EUR_per_kWh"]),
                "price_NOK_per_kWh": float(row["predicted_NOK_per_kWh"]),
                "pi_80": {
                    "lower_EUR": float(row["pi_80_lower_EUR"]),
                    "upper_EUR": float(row["pi_80_upper_EUR"]),
                },
            }
            for ts, row in forecast.iterrows()
        ],
    }
    out_json.write_text(json.dumps(document, indent=2), encoding="utf-8")

    if args.save_png:
        hist = price.loc[origin - pd.Timedelta(hours=max(1, args.history_hours)) : origin]

        plt.figure(figsize=(14, 7))
        plt.plot(
            hist.index, hist.to_numpy() * eur_nok,
            label=f"Observed (last {args.history_hours}h)", color="steelblue", linewidth=1.6,
        )
        bridge = pd.concat(
            [pd.Series([float(price.loc[origin])], index=[origin]),
             forecast["predicted_EUR_per_kWh"]]
        )
        plt.plot(
            bridge.index, bridge.to_numpy() * eur_nok,
            label=f"Forecast ({model_dir.name})", color="crimson",
            linewidth=2, linestyle="--",
        )
        plt.fill_between(
            forecast.index,
            forecast["pi_80_lower_NOK"],
            forecast["pi_80_upper_NOK"],
            alpha=0.18, color="crimson", label="80% prediction interval",
        )
        plt.axvline(origin, color="gray", linestyle=":", linewidth=1.2, label="Forecast origin")
        plt.title(f"Price forecast from {origin:%Y-%m-%d %H:%M} ({model_dir.name})")
        plt.xlabel("Time (Europe/Oslo)")
        plt.ylabel("NOK/kWh")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_png = ROOT / "figs" / f"forecast_{tag}_{origin:%Y%m%d_%H%M}.png"
        save_figure(plt.gcf(), out_png, note=f"EUR/NOK {eur_nok:.4f} ({fx_source}); models work in EUR")


if __name__ == "__main__":
    main()
