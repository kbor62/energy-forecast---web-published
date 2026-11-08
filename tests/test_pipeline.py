"""Leakage and consistency tests: features are causal, training and forecasting
build the same feature vector, training labels stay out of the validation
window, 15-minute prices are averaged to hourly means, and the weather
features / forecast snapshots behave (target-hour alignment, ridge keeps the
weather columns, snapshot choice never looks ahead, snapshot-only mode has no
fallback to observations, archiving is idempotent).

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

# --------------------------------------------------------------------------- #
# Weather features and forecast snapshots
# --------------------------------------------------------------------------- #
def _weather_frame(index) -> pd.DataFrame:
    """Deterministic hourly weather: temperature = position, wind = 2 * position."""
    pos = np.arange(len(index), dtype=float)
    return pd.DataFrame({"temperature_C": pos, "wind_speed_ms": 2 * pos}, index=index)


def test_weather_features_describe_the_target_hour():
    """<col>_target at origin t holds the weather at t + h, not at t."""
    price = synthetic_price()
    history = F.history_features(price)
    weather = _weather_frame(price.index)
    origins = price.index[400:405]

    for h in (1, 24):
        X = F.assemble(history, origins, h, "int", weather=weather)
        expected = weather["temperature_C"].reindex(origins + pd.Timedelta(hours=h)).to_numpy()
        assert np.allclose(X["temperature_C_target"].to_numpy(), expected)
        assert list(X.columns[-2:]) == ["temperature_C_target", "wind_speed_ms_target"]


def test_assemble_without_weather_is_unchanged():
    """No weather argument -> exactly the price-only feature set."""
    price = synthetic_price()
    history = F.history_features(price)
    X = F.assemble(history, price.index[400:403], 6, "int")
    assert list(X.columns) == F.feature_columns("int")


def test_ridge_uses_the_weather_columns():
    """Ridge must not silently drop weather columns (remainder='drop')."""
    import models as M

    price = synthetic_price()
    history = F.history_features(price).iloc[F.MAX_HISTORY_HOURS:]
    weather = _weather_frame(price.index)
    origins = history.index[:-6]
    y = price.shift(-6).loc[origins].to_numpy()

    X = F.assemble(history, origins, 6, "trig", weather=weather)
    cols = ["temperature_C_target", "wind_speed_ms_target"]
    est = M.build_estimator("ridge", "trig", alpha="1.0", extra_cols=cols)
    est.fit(X, y)
    assert est.named_steps["pre"].transform(X.iloc[:2]).shape[1] == X.shape[1]

    dropped = M.build_estimator("ridge", "trig", alpha="1.0")
    dropped.fit(X, y)
    assert dropped.named_steps["pre"].transform(X.iloc[:2]).shape[1] == X.shape[1] - len(cols)


def _write_snapshot(archive: Path, issued: str, start: str, hours: int) -> None:
    """Snapshot with columns as written by fetch_weather_forecast.py."""
    issued_ts = pd.Timestamp(issued, tz="UTC")
    idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    snap = pd.DataFrame(
        {"oslo_temperature_C": 100.0, "oslo_wind_speed_ms": 50.0, "oslo_precipitation_mm": np.nan},
        index=idx,
    )
    snap["issued_at"] = issued_ts
    archive.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(archive / f"metno_{issued_ts:%Y%m%dT%H%MZ}.parquet")


def test_snapshot_choice_never_looks_ahead():
    """pick_snapshot returns the latest snapshot issued at or before as_of."""
    import tempfile

    import weather_io as W

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp)
        _write_snapshot(archive, "2026-01-01T06:00", "2026-01-01 06:00", 60)
        _write_snapshot(archive, "2026-01-01T12:00", "2026-01-01 12:00", 60)

        assert W.pick_snapshot(pd.Timestamp("2026-01-01 06:59", tz="UTC"), archive)[0] == pd.Timestamp(
            "2026-01-01T06:00", tz="UTC")
        assert W.pick_snapshot(pd.Timestamp("2026-01-01 12:00", tz="UTC"), archive)[0] == pd.Timestamp(
            "2026-01-01T12:00", tz="UTC")
        assert W.pick_snapshot(pd.Timestamp("2026-01-01 05:59", tz="UTC"), archive) is None


def test_snapshot_only_mode_does_not_fall_back_to_observations():
    """With observed_path=None, cells the snapshot lacks stay NaN (no future leak)."""
    import tempfile

    import weather_io as W

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp)
        _write_snapshot(archive, "2026-01-01T06:00", "2026-01-01 06:00", 10)
        origin = pd.Timestamp("2026-01-01 07:00", tz="UTC").tz_convert(TZ)
        targets = origin + pd.to_timedelta(range(1, 13), unit="h")  # runs past the 10 h snapshot

        obs = pd.DataFrame({"temperature_C": -1.0, "wind_speed_ms": -1.0, "precipitation_mm": -1.0},
                           index=pd.date_range("2025-12-01", "2026-02-01", freq="h", tz=TZ))
        obs_path = Path(tmp) / "obs.parquet"
        obs.to_parquet(obs_path)

        cols = ["temperature_C", "wind_speed_ms", "precipitation_mm"]
        only, info = W.weather_for_forecast(cols, targets, origin, archive, observed_path=None)
        assert info["cells_from_observed"] == 0
        assert only["temperature_C"].iloc[:2].eq(100.0).all()          # inside the snapshot
        assert only["temperature_C"].iloc[-1:].isna().all()           # beyond it: NaN, not observed
        assert only["precipitation_mm"].isna().all()                   # MET gives no precipitation here

        filled, info = W.weather_for_forecast(cols, targets, origin, archive, observed_path=obs_path)
        assert filled["temperature_C"].iloc[-1] == -1.0                # live mode falls back
        assert info["cells_from_observed"] > 0


def test_snapshot_archive_skips_an_already_stored_run():
    """The same MET model run is archived once and unchanged on the second call."""
    import tempfile

    import fetch_weather_forecast as FW

    idx = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    raw = {"oslo": pd.DataFrame({"temperature_C": 1.0}, index=idx)}
    issued = pd.Timestamp("2026-01-01T00:00", tz="UTC")
    with tempfile.TemporaryDirectory() as tmp:
        first = FW.archive_snapshot(raw, issued, tmp)
        assert first is not None and first.name == "metno_20260101T0000Z.parquet"
        assert FW.archive_snapshot(raw, issued, tmp) is None
        saved = pd.read_parquet(first)
        assert "issued_at" in saved.columns and "oslo_temperature_C" in saved.columns


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
