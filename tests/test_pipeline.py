"""Leakage and consistency tests: features are causal, training and forecasting
build the same feature vector, training labels stay out of the validation
window, and 15-minute prices are averaged to hourly means.

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import features as F  # noqa: E402
from config import TZ  # noqa: E402

HOURS = 24 * 400


def synthetic_price(seed: int = 0) -> pd.Series:
    """A price series with daily, weekly and yearly structure plus noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=HOURS, freq="h", tz=TZ)
    t = np.arange(len(idx))
    level = (
        0.08
        + 0.02 * np.sin(2 * np.pi * t / 24)
        + 0.01 * np.sin(2 * np.pi * t / (24 * 7))
        + 0.03 * np.sin(2 * np.pi * t / (24 * 365))
    )
    noise = rng.normal(0, 0.004, size=len(idx)).cumsum() * 0.02
    return pd.Series(level + noise, index=idx, name="price_EUR_per_kWh")


# --------------------------------------------------------------------------- #
def test_history_features_are_causal():
    """Truncating the series must not change any earlier feature value."""
    price = synthetic_price()
    cut = len(price) // 2

    full = F.history_features(price).iloc[:cut]
    truncated = F.history_features(price.iloc[:cut])

    pd.testing.assert_frame_equal(full, truncated)


def test_future_prices_do_not_leak_into_past_features():
    """Perturbing prices after T leaves every feature at or before T untouched."""
    price = synthetic_price()
    cut = len(price) // 2

    tampered = price.copy()
    tampered.iloc[cut + 1 :] = tampered.iloc[cut + 1 :] * 5.0 + 1.0

    before = F.history_features(price).iloc[: cut + 1]
    after = F.history_features(tampered).iloc[: cut + 1]

    pd.testing.assert_frame_equal(before, after)


def test_rolling_windows_exclude_the_current_hour():
    """Rolling statistics exclude the origin hour."""
    price = synthetic_price()
    feat = F.history_features(price)
    w = F.ROLLS_H[0]
    expected = price.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
    pd.testing.assert_series_equal(feat[f"ma{w}h"], expected, check_names=False)


def test_targets_point_at_the_right_hour():
    price = synthetic_price()
    targets = F.make_targets(price, [1, 24, 48])
    for h in (1, 24, 48):
        col = targets[F.target_column(h)].dropna()
        shifted = price.reindex(col.index + pd.Timedelta(hours=h))
        assert np.allclose(col.to_numpy(), shifted.to_numpy())


def test_calendar_block_describes_the_target_hour():
    """For horizon h, hour/dow/month must be those of t+h, not of t."""
    price = synthetic_price()
    history = F.history_features(price)
    origins = history.index[300:320]

    for h in (1, 13, 48):
        X = F.assemble(history, origins, h, encoding="int")
        target_idx = origins + pd.Timedelta(hours=h)
        assert np.array_equal(X["hour"].to_numpy(), target_idx.hour.to_numpy())
        assert np.array_equal(X["dow"].to_numpy(), target_idx.dayofweek.to_numpy())
        assert np.array_equal(X["month"].to_numpy(), target_idx.month.to_numpy())


def test_history_block_describes_the_origin_hour():
    price = synthetic_price()
    history = F.history_features(price)
    origins = history.index[300:320]
    X = F.assemble(history, origins, 48, encoding="int")
    assert np.allclose(X["lag0h"].to_numpy(), price.loc[origins].to_numpy())


def test_forecast_row_matches_the_training_row():
    """A single-origin assemble equals that origin's row in a bulk assemble."""
    price = synthetic_price()
    history = F.history_features(price)
    origins = history.index[500:540]
    origin = origins[17]

    for encoding in F.ENCODINGS:
        bulk = F.assemble(history, origins, 48, encoding)
        single = F.assemble(history, [origin], 48, encoding)
        assert list(bulk.columns) == list(single.columns)
        pd.testing.assert_frame_equal(bulk.loc[[origin]], single)


def test_feature_columns_are_deterministic_and_target_free():
    for encoding in F.ENCODINGS:
        cols = F.feature_columns(encoding)
        assert cols == F.feature_columns(encoding)
        assert len(cols) == len(set(cols)), "duplicate feature names"
        assert not any(c.startswith("target") for c in cols)
        assert not any(c.startswith("price_") for c in cols)


def test_categorical_levels_are_fixed_not_data_dependent():
    """A slice missing whole seasons still produces the full category domain."""
    price = synthetic_price()
    january = price.loc["2024-01-01":"2024-01-20"]
    X = F.as_categorical(
        F.calendar_features(january.index, "int"), F.categorical_columns("int")
    )
    assert list(X["month"].cat.categories) == F.CAT_LEVELS["month"]
    assert list(X["season"].cat.categories) == F.CAT_LEVELS["season"]


def test_embargo_keeps_training_labels_out_of_validation():
    price = synthetic_price()
    index = price.index
    val_start = index[-24 * 30]

    for horizon in (1, 24, 48):
        train_mask, val_mask = F.train_val_masks(index, val_start, horizon, embargo=True)
        last_label = index[train_mask].max() + pd.Timedelta(hours=horizon)
        assert last_label < val_start, (
            f"horizon {horizon}h: training label at {last_label} lands inside the "
            f"validation window starting {val_start}"
        )
        assert index[val_mask].min() >= val_start
        assert not (train_mask & val_mask).any(), "train and validation overlap"


def test_no_embargo_reproduces_the_old_overlap():
    """Without the embargo, training labels overlap the validation window."""
    price = synthetic_price()
    index = price.index
    val_start = index[-24 * 30]
    train_mask, _ = F.train_val_masks(index, val_start, 48, embargo=False)
    last_label = index[train_mask].max() + pd.Timedelta(hours=48)
    assert last_label >= val_start


def test_retraining_removes_superseded_horizon_models():
    """A shorter --horizon must not leave the previous run's later models behind."""
    import tempfile

    from train import clear_stale_models

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for h in (1, 2, 3):
            (out_dir / f"price_model_t{h}h.joblib").write_bytes(b"stub")
            (out_dir / f"price_model_t{h}h_meta.json").write_text("{}")

        clear_stale_models(out_dir, [1, 2])

        assert (out_dir / "price_model_t1h.joblib").exists()
        assert (out_dir / "price_model_t2h.joblib").exists()
        assert not (out_dir / "price_model_t3h.joblib").exists()
        assert not (out_dir / "price_model_t3h_meta.json").exists()


# --------------------------------------------------------------------------- #
def test_quarter_hour_prices_average_to_hourly_means():
    """15-minute prices are averaged to hourly means."""
    import tempfile

    from price_io import load_price_series

    hourly = pd.date_range("2025-09-30 20:00", periods=4, freq="h", tz=TZ)
    quarters = pd.date_range("2025-10-01 00:00", periods=8, freq="15min", tz=TZ)
    q_prices = [0.10, 0.20, 0.30, 0.40, 0.05, 0.05, 0.05, 0.25]
    frame = pd.DataFrame({
        "timestamp": list(hourly) + list(quarters),
        "price_EUR_per_kWh": [0.1, 0.2, 0.3, 0.4] + q_prices,
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prices.csv"
        frame.to_csv(path, index=False)
        series = load_price_series(path, verbose=False)

    assert np.isclose(series[hourly[0]], 0.1), "hourly data must pass through unchanged"
    assert np.isclose(series[pd.Timestamp("2025-10-01 00:00", tz=TZ)], 0.25)
    assert np.isclose(series[pd.Timestamp("2025-10-01 01:00", tz=TZ)], 0.10)
    assert series.index.freqstr in ("h", "H"), "result must be on an hourly grid"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
