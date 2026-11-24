"""Permutation feature importance on each model's validation window.

A feature (or a group of related features) is shuffled and the increase in MAE
is recorded, averaged over several repeats.

    python src/feature_importance.py --model_dir models/lgbm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

import features as F
import models as M
from config import EUR_TO_NOK, ROOT, TZ, repo_path
from plotting import INK, INK_SOFT, fixed_rate_note, save_figure

DEFAULT_INPUT = "data/features_price_only.parquet"
UNIT = "NOK/kWh"

SEQ_BLUE = matplotlib.colors.LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    """Groups of related features."""
    cal = set(F.calendar_columns())
    groups = {
        "Origin price (lag0h)": ["lag0h"],
        "Short lags (1-12 h)": [f"lag{h}h" for h in F.LAGS_H if h < 24],
        "Daily/weekly lags (24-168 h)": [f"lag{h}h" for h in F.LAGS_H if h >= 24],
        "Rolling means": [f"ma{w}h" for w in F.ROLLS_H],
        "Rolling std": [f"std{w}h" for w in F.ROLLS_H],
        "Differences": [f"diff{h}h" for h in F.DIFFS_H],
        "Calendar (target hour)": [c for c in columns if c in cal],
    }
    return {k: [c for c in v if c in columns] for k, v in groups.items() if any(c in columns for c in v)}


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = df.index.tz_localize(TZ) if df.index.tz is None else df.index.tz_convert(TZ)
    return df.sort_index().iloc[F.MAX_HISTORY_HOURS :]


def load_model(model_dir: Path, horizon: int) -> tuple[dict, dict]:
    model_path = model_dir / f"price_model_t{horizon}h.joblib"
    meta_path = model_dir / f"price_model_t{horizon}h_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return load(model_path), meta


def evaluation_set(df: pd.DataFrame, payload: dict, meta: dict, eval_start: str | None):
    h = payload["horizon"]
    target = F.target_column(h)
    if eval_start:
        start = pd.Timestamp(eval_start, tz=TZ)
    else:
        start = pd.Timestamp(meta["val_start"]).tz_convert(TZ)

    rows = df.index[(df.index >= start) & df[target].notna()]
    X = F.assemble(df[F.history_columns()], rows, h)
    X = X.loc[:, payload["features"]]
    X = M.prepare_X(X)
    y = df.loc[rows, target].to_numpy(dtype=float)
    return X, y, start


def mae(estimator, X, y) -> float:
    return float(np.mean(np.abs(estimator.predict(X) - y)))


def permuted(X: pd.DataFrame, cols: list[str], idx: np.ndarray) -> pd.DataFrame:
    """Shuffle the given columns with one shared permutation."""
    Xp = X.copy()
    for c in cols:
        values = X[c].to_numpy()[idx]
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            values = pd.Categorical(values, categories=X[c].cat.categories)
        Xp[c] = values
    return Xp


def permutation_importance(estimator, X, y, units: dict[str, list[str]], repeats: int, seed: int):
    rng = np.random.default_rng(seed)
    base = mae(estimator, X, y)
    out = {}
    for name, cols in units.items():
        scores = [mae(estimator, permuted(X, cols, rng.permutation(len(X))), y) - base for _ in range(repeats)]
        out[name] = (float(np.mean(scores)), float(np.std(scores, ddof=1)) if repeats > 1 else 0.0)
    return base, out


def plot_group_heatmap(groups: pd.DataFrame, tag: str, out: Path) -> None:
    pivot = groups.pivot(index="group", columns="horizon_hours", values="perm_mae_increase")
    order = list(dict.fromkeys(groups["group"]))
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(1.1 * pivot.shape[1] + 4, 0.55 * pivot.shape[0] + 1.8))
    im = ax.imshow(pivot.clip(lower=0).to_numpy(), aspect="auto", cmap=SEQ_BLUE)
    ax.set_xticks(range(pivot.shape[1]), [f"{h} h" for h in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    mid = np.nanmax(pivot.to_numpy()) / 2
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > mid else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"MAE increase ({UNIT})", color=INK_SOFT)
    cbar.outline.set_visible(False)
    ax.set_xlabel("Forecast horizon", color=INK_SOFT)
    ax.set_title(f"Group permutation importance, {tag}", loc="left", color=INK, fontsize=12, pad=10)
    save_figure(fig, out, note=fixed_rate_note())


def run_model(model_dir: Path, df: pd.DataFrame, args) -> None:
    tag = model_dir.name
    rows, group_rows = [], []
    print(f"\n=== {tag}")
    for h in args.horizons:
        payload, meta = load_model(model_dir, h)
        X, y, start = evaluation_set(df, payload, meta, args.eval_start)
        cols = list(X.columns)

        base, single = permutation_importance(
            payload["estimator"], X, y, {c: [c] for c in cols}, args.repeats, args.seed + h
        )
        _, grouped = permutation_importance(
            payload["estimator"], X, y, feature_groups(cols), args.repeats, args.seed + h
        )

        for c in cols:
            m, s = single[c]
            rows.append({
                "model": tag, "horizon_hours": h, "feature": c,
                "perm_mae_increase": m * EUR_TO_NOK, "perm_std": s * EUR_TO_NOK,
            })
        for g, (m, s) in grouped.items():
            group_rows.append({
                "model": tag, "horizon_hours": h, "group": g,
                "perm_mae_increase": m * EUR_TO_NOK, "perm_std": s * EUR_TO_NOK,
                "share_of_baseline_mae": m / base if base else np.nan,
            })
        top = sorted(single.items(), key=lambda kv: -kv[1][0])[:3]
        print(f"  h={h:>2}  eval from {start:%Y-%m-%d %H:%M}  n={len(y)}  "
              f"MAE {base * EUR_TO_NOK:.4f} {UNIT}   top: " + ", ".join(k for k, _ in top))

    table = pd.DataFrame(rows)
    table["rank"] = table.groupby("horizon_hours")["perm_mae_increase"].rank(ascending=False, method="min")
    groups = pd.DataFrame(group_rows)

    data_dir, fig_dir = ROOT / "data", repo_path(args.out_dir)
    table.to_csv(data_dir / f"feature_importance_{tag}.csv", index=False)
    groups.to_csv(data_dir / f"feature_importance_groups_{tag}.csv", index=False)

    plot_group_heatmap(groups, tag, fig_dir / f"feature_importance_groups_{tag}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Permutation feature importance for trained models")
    ap.add_argument("--model_dir", nargs="+", required=True, help="One or more model dirs, e.g. models/lgbm")
    ap.add_argument("--inp", default=DEFAULT_INPUT, help="Feature parquet written by build_features.py")
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 6, 12, 24, 36, 48])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--eval_start", default=None, help="Override the evaluation window start (default: model's val_start)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="figs/importance")
    args = ap.parse_args()

    df = load_features(repo_path(args.inp))
    for d in args.model_dir:
        run_model(repo_path(d), df, args)


if __name__ == "__main__":
    main()
