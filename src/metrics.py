"""Error metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def summarize(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and nMAE over the rows where both values are finite."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    score_mae = mae(y_true, y_pred)
    denom = float(np.mean(np.abs(y_true)))
    return {
        "n": int(y_true.size),
        "mae": score_mae,
        "rmse": rmse(y_true, y_pred),
        "nmae": float(score_mae / denom),
    }


def skill(model_mae: float, baseline_mae: float) -> float:
    """Fraction of baseline error removed."""
    return 1.0 - model_mae / baseline_mae


def block_bootstrap_ci(
    diff: pd.Series,
    block_hours: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Moving-block bootstrap confidence interval for the mean of ``diff``."""
    values = pd.Series(diff).dropna().to_numpy(dtype=float)
    n = values.size
    block = max(1, min(int(block_hours), n))
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    rng = np.random.default_rng(seed)

    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([values[s : s + block] for s in starts])[:n]
        means[b] = sample.mean()

    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
