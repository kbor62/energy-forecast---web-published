"""Significance tests on backtest results, with one MAE per forecast origin.

Two methods: Wilcoxon signed-rank test. Three or more: Friedman test with the
Iman-Davenport correction, then pairwise Wilcoxon tests with Holm correction.

    python src/stat_tests.py data/backtest_rolling_lgbm_int.csv data/backtest_rolling_rf_trig.csv --baselines
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import repo_path

BASELINES = {
    "persistence": "baseline_persistence",
    "weekly_t168": "baseline_weekly_t168",
}


def parse_horizons(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    lo, _, hi = spec.partition("-")
    return int(lo), int(hi or lo)


def label_for(path: Path) -> str:
    return path.stem.replace("backtest_rolling_", "")


def per_origin_mae(df: pd.DataFrame, column: str) -> pd.Series:
    return (df[column] - df["actual"]).abs().groupby(df["origin"]).mean()


def load_scores(paths: list[Path], baselines: bool, horizons) -> pd.DataFrame:
    """One column per method, one row per origin."""
    columns: dict[str, pd.Series] = {}
    for i, path in enumerate(paths):
        df = pd.read_csv(path, parse_dates=["origin"])
        if horizons:
            df = df[df["horizon_hours"].between(*horizons)]
        df = df.dropna(subset=["actual"])
        columns[label_for(path)] = per_origin_mae(df, "predicted")
        if baselines and i == 0:  # baselines are identical in every file
            for name, col in BASELINES.items():
                columns[name] = per_origin_mae(df.dropna(subset=[col]), col)

    return pd.DataFrame(columns).dropna()


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def wilcoxon_pair(a: pd.Series, b: pd.Series) -> dict:
    diff = a - b
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {
        "median_diff": float(diff.median()),
        "a_better_share": float((diff < 0).mean()),
        "W": float(res.statistic),
        "p": float(res.pvalue),
    }


def friedman(scores: pd.DataFrame) -> dict:
    n, k = scores.shape
    chi2, p = stats.friedmanchisquare(*[scores[c] for c in scores])
    # Iman-Davenport F statistic.
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
    p_id = float(stats.f.sf(ff, k - 1, (k - 1) * (n - 1)))
    ranks = scores.rank(axis=1, method="average").mean().sort_values()
    return {"chi2": chi2, "p": p, "F_ID": ff, "p_ID": p_id, "mean_ranks": ranks}


def main() -> None:
    ap = argparse.ArgumentParser(description="Wilcoxon / Friedman on backtest results")
    ap.add_argument("files", nargs="+", help="backtest_rolling_*.csv files")
    ap.add_argument("--baselines", action="store_true", help="Include persistence and weekly t-168")
    ap.add_argument("--horizons", default=None, help="Restrict to a horizon range, e.g. 1-24 or 25-48")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out_csv", default=None, help="Write the pairwise table here")
    args = ap.parse_args()

    paths = [repo_path(f) for f in args.files]
    horizons = parse_horizons(args.horizons)
    scores = load_scores(paths, args.baselines, horizons)
    n, k = scores.shape

    print(f"Methods: {k}   common origins: {n}   horizons: {args.horizons or 'all'}")
    print("\nMean per-origin MAE (EUR/kWh):")
    print(scores.mean().sort_values().to_string(float_format=lambda v: f"{v:.5f}"))

    if k >= 3:
        fr = friedman(scores)
        print(f"\nFriedman: chi2={fr['chi2']:.2f}  p={fr['p']:.2e}")
        print(f"Iman-Davenport: F={fr['F_ID']:.2f}  p={fr['p_ID']:.2e}")
        print("Mean rank (1 = best):")
        print(fr["mean_ranks"].to_string(float_format=lambda v: f"{v:.2f}"))
        if fr["p_ID"] >= args.alpha:
            print(f"\nNo significant difference at alpha={args.alpha}; post-hoc tests not warranted.")
            return

    rows = []
    for a, b in combinations(scores.columns, 2):
        rows.append({"A": a, "B": b, **wilcoxon_pair(scores[a], scores[b])})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist()) if len(table) > 1 else table["p"]
    table["significant"] = table["p_holm"] < args.alpha

    print("\nPairwise Wilcoxon signed-rank (median_diff = A - B; negative -> A better):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    if args.out_csv:
        out = repo_path(args.out_csv)
        table.to_csv(out, index=False)


if __name__ == "__main__":
    main()
