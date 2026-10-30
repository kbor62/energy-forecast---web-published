"""Forecast the next 48 hours from the last observed price.

    python src/forecast.py --model_dir models/lgbm_int --save_png
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
import models as M
from config import EUR_TO_NOK, ROOT, TZ, repo_path
from plotting import save_figure
from price_io import load_price_series

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
    ap.add_argument("--model_dir", default="models/lgbm_int", help="Directory of trained models")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--origin", default=None, help="Override the forecast origin timestamp")
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
    encoding = models[1]["payload"]["encoding"]
    model_name = models[1]["payload"]["model"]

    observed = price.dropna()

    if args.origin:
        origin = pd.Timestamp(args.origin)
        origin = origin.tz_localize(TZ) if origin.tz is None else origin.tz_convert(TZ)
        origin = origin.floor("h")
    else:
        origin = observed.index[-1]

    history = F.history_features(price)
    print(
        f"Model: {model_name}   encoding: {encoding}   horizons: {len(models)}   "
        f"trained {manifest.get('trained_at', '?')[:16]}"
    )
    print(f"Forecast origin: {origin}  (price {float(price.loc[origin]):.4f} EUR/kWh)")

    rows = []
    for h in sorted(models):
        payload = models[h]["payload"]
        X = F.assemble(history, [origin], h, encoding)

        X = M.prepare_X(payload["model"], X, encoding)
        pred = float(payload["estimator"].predict(X)[0])

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

    forecast = pd.DataFrame(rows).set_index("timestamp").sort_index()
    for col in [c for c in forecast.columns if c.endswith("_EUR") or c.endswith("_EUR_per_kWh")]:
        forecast[col.replace("EUR", "NOK")] = forecast[col] * EUR_TO_NOK

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(out_csv)

    document = {
        "generated_at": pd.Timestamp.now(tz=TZ).isoformat(),
        "forecast_origin": origin.isoformat(),
        "origin_price_EUR_per_kWh": float(price.loc[origin]),
        "model": model_name,
        "encoding": encoding,
        "model_dir": str(model_dir),
        "model_trained_at": manifest.get("trained_at"),
        "eur_nok": EUR_TO_NOK,
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
            hist.index, hist.to_numpy() * EUR_TO_NOK,
            label=f"Observed (last {args.history_hours}h)", color="steelblue", linewidth=1.6,
        )
        bridge = pd.concat(
            [pd.Series([float(price.loc[origin])], index=[origin]),
             forecast["predicted_EUR_per_kWh"]]
        )
        plt.plot(
            bridge.index, bridge.to_numpy() * EUR_TO_NOK,
            label=f"Forecast ({model_name}/{encoding})", color="crimson",
            linewidth=2, linestyle="--",
        )
        plt.fill_between(
            forecast.index,
            forecast["pi_80_lower_NOK"],
            forecast["pi_80_upper_NOK"],
            alpha=0.18, color="crimson", label="80% prediction interval",
        )
        plt.axvline(origin, color="gray", linestyle=":", linewidth=1.2, label="Forecast origin")
        plt.title(f"Price forecast from {origin:%Y-%m-%d %H:%M} ({model_name}, {encoding})")
        plt.xlabel("Time (Europe/Oslo)")
        plt.ylabel("NOK/kWh")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_png = ROOT / "figs" / f"forecast_{tag}_{origin:%Y%m%d_%H%M}.png"
        save_figure(plt.gcf(), out_png)


if __name__ == "__main__":
    main()
