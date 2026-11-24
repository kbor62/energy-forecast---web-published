"""Leakage and consistency tests: features are causal, training and forecasting
build the same feature vector, training labels stay out of the validation
window, 15-minute prices are averaged to hourly means, and the weather
features / forecast snapshots behave (target-hour alignment, holiday flag,
snapshot choice never looks ahead, snapshot-only mode has no
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
        X = F.assemble(history, origins, h)
        target_idx = origins + pd.Timedelta(hours=h)
        assert np.array_equal(X["hour"].to_numpy(), target_idx.hour.to_numpy())
        assert np.array_equal(X["dow"].to_numpy(), target_idx.dayofweek.to_numpy())
        assert np.array_equal(X["month"].to_numpy(), target_idx.month.to_numpy())


def test_history_block_describes_the_origin_hour():
    price = synthetic_price()
    history = F.history_features(price)
    origins = history.index[300:320]
    X = F.assemble(history, origins, 48)
    assert np.allclose(X["lag0h"].to_numpy(), price.loc[origins].to_numpy())


def test_forecast_row_matches_the_training_row():
    """A single-origin assemble equals that origin's row in a bulk assemble."""
    price = synthetic_price()
    history = F.history_features(price)
    origins = history.index[500:540]
    origin = origins[17]

    bulk = F.assemble(history, origins, 48)
    single = F.assemble(history, [origin], 48)
    assert list(bulk.columns) == list(single.columns)
    pd.testing.assert_frame_equal(bulk.loc[[origin]], single)


def test_feature_columns_are_deterministic_and_target_free():
    cols = F.feature_columns()
    assert cols == F.feature_columns()
    assert len(cols) == len(set(cols)), "duplicate feature names"
    assert not any(c.startswith("target") for c in cols)
    assert not any(c.startswith("price_") for c in cols)


def test_categorical_levels_are_fixed_not_data_dependent():
    """A slice missing hours/weekdays still produces the full category domain;
    only hour and dow are categorical, month stays an integer."""
    price = synthetic_price()
    morning = price.loc["2024-01-01 06:00":"2024-01-02 09:00"]
    morning = morning[morning.index.hour < 10]
    X = F.as_categorical(F.calendar_features(morning.index))
    assert list(X["hour"].cat.categories) == F.CAT_LEVELS["hour"]
    assert list(X["dow"].cat.categories) == F.CAT_LEVELS["dow"]
    assert not isinstance(X["month"].dtype, pd.CategoricalDtype)
    assert not isinstance(X["is_holiday"].dtype, pd.CategoricalDtype)


def test_holiday_flag_uses_the_local_date_of_the_target_hour():
    """17 May and Christmas Day are holidays, 24 Dec and ordinary Sundays are not;
    the flag follows the target hour's local (Europe/Oslo) date."""
    idx = pd.DatetimeIndex(
        ["2024-05-16 23:00", "2024-05-17 00:00", "2024-05-17 23:00", "2024-05-18 00:00",
         "2024-12-24 12:00", "2024-12-25 12:00", "2024-06-02 12:00"]
    ).tz_localize(TZ)
    assert F.calendar_features(idx)["is_holiday"].tolist() == [0, 1, 1, 0, 0, 1, 0]

    # Origin 23:00 on 16 May: h=1 is 00:00 on 17 May (holiday), h=25 is 00:00 on 18 May.
    price = synthetic_price()
    history = F.history_features(price)
    origin = pd.Timestamp("2024-05-16 23:00", tz=TZ)
    assert int(F.assemble(history, [origin], 1)["is_holiday"].iloc[0]) == 1
    assert int(F.assemble(history, [origin], 25)["is_holiday"].iloc[0]) == 0


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
        X = F.assemble(history, origins, h, weather=weather)
        expected = weather["temperature_C"].reindex(origins + pd.Timedelta(hours=h)).to_numpy()
        assert np.allclose(X["temperature_C_target"].to_numpy(), expected)
        assert list(X.columns[-2:]) == ["temperature_C_target", "wind_speed_ms_target"]


def test_assemble_without_weather_is_unchanged():
    """No weather argument -> exactly the price-only feature set."""
    price = synthetic_price()
    history = F.history_features(price)
    X = F.assemble(history, price.index[400:403], 6)
    assert list(X.columns) == F.feature_columns()


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


# --------------------------------------------------------------------------- #
# Backtest origin clock time
# --------------------------------------------------------------------------- #
def test_default_origins_drift_with_dst_but_origin_hour_does_not():
    """Absolute 48 h steps change local clock time at DST; --origin_hour keeps it fixed."""
    import backtest_rolling as B

    drifting = B.build_origins("2026-01-01 00:00", "2026-08-31 00:00", 48)
    assert set(drifting.hour) == {0, 1}

    fixed = B.build_origins("2026-01-01 00:00", "2026-08-31 00:00", 48, origin_hour=23)
    assert set(fixed.hour) == {23}
    assert fixed[0] == pd.Timestamp("2026-01-01 23:00", tz=TZ)
    assert (fixed[1:] - fixed[:-1] != pd.Timedelta(hours=48)).any(), "spacing is 2 calendar days, not 48 h"
    days = pd.DatetimeIndex(fixed.tz_localize(None)).normalize()
    assert ((days[1:] - days[:-1]) == pd.Timedelta(days=2)).all()


def test_origin_hour_validation_and_dst_gaps():
    """step must be whole days; a nonexistent local hour (DST) is dropped, not an error."""
    import backtest_rolling as B

    for bad in ({"step": 36, "origin_hour": 12}, {"step": 48, "origin_hour": 24}):
        try:
            B.build_origins("2026-01-01 00:00", "2026-02-01 00:00", **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")

    # 02:00 does not exist on 2026-03-29 (spring forward); step 1 day hits that date
    o = B.build_origins("2026-03-27 00:00", "2026-03-31 00:00", 24, origin_hour=2)
    assert pd.Timestamp("2026-03-29 02:00", tz="UTC").tz_convert(TZ) not in o
    assert len(o) == 3  # 27, 28 and 30 March; the 29th is dropped, 31st 02:00 is after the end


# --------------------------------------------------------------------------- #
# ENTSO-E exogenous features (wind/solar forecasts, neighbour prices)
# --------------------------------------------------------------------------- #
def _exog_frame(n_hours: int = 24 * 12) -> pd.DataFrame:
    idx = pd.date_range("2026-03-01", periods=n_hours, freq="h", tz=TZ)
    pos = np.arange(n_hours, dtype=float)
    return pd.DataFrame({"wind_DE_LU_MW": pos, "solar_DE_LU_MW": 2 * pos, "load_NO_1_MW": 4 * pos,
                         "price_SE_3_EUR_MWh": 3 * pos}, index=idx)


def test_wind_solar_frames_are_normalised_to_hourly_zone_columns():
    """Onshore + offshore are summed, solar kept; flat and MultiIndex columns both work."""
    import exog as X

    idx = pd.date_range("2026-01-01", periods=8, freq="15min", tz="Europe/Berlin")
    flat = pd.DataFrame({"Solar": 10.0, "Wind Offshore": 100.0, "Wind Onshore": 50.0}, index=idx)
    multi = flat.copy()
    multi.columns = pd.MultiIndex.from_tuples([(c, "Actual Aggregated") for c in flat.columns])

    for frame in (flat, multi):
        out = X.normalise_wind_solar(frame, "DE_LU")
        assert list(out.columns) == ["wind_DE_LU_MW", "solar_DE_LU_MW"]
        assert len(out) == 2 and (out["wind_DE_LU_MW"] == 150.0).all() and (out["solar_DE_LU_MW"] == 10.0).all()
        assert str(out.index.tz) == TZ


def test_load_forecast_frames_are_normalised_to_hourly_load_column():
    """entsoe-py load forecast (DataFrame or Series, 15-min) -> hourly load_<zone>_MW in Europe/Oslo."""
    import exog as X

    idx = pd.date_range("2026-01-01", periods=8, freq="15min", tz="Europe/Oslo")
    values = [100.0, 200.0, 300.0, 400.0, 10.0, 20.0, 30.0, 40.0]
    for raw in (pd.DataFrame({"Forecasted Load": values}, index=idx), pd.Series(values, index=idx)):
        out = X.normalise_load(raw, "NO_1")
        assert list(out.columns) == ["load_NO_1_MW"] and str(out.index.tz) == TZ
        assert out["load_NO_1_MW"].tolist() == [250.0, 25.0]
    assert X.split_columns(["load_NO_1_MW", "price_SE_3_EUR_MWh"]) == (["load_NO_1_MW"], ["price_SE_3_EUR_MWh"])


def test_exog_availability_follows_the_day_ahead_schedule():
    """Origin 23:00 on D: tomorrow (h 1-24) is published, the day after (h 25-48) is not."""
    import exog as X

    o = pd.DatetimeIndex([pd.Timestamp("2026-03-10 23:00", tz=TZ)])
    assert all(X.available_mask(o, h).iloc[0] for h in range(1, 25))
    assert not any(X.available_mask(o, h).iloc[0] for h in range(25, 49))

    # Origin 00:00 on D: decision ~13:00 on D-1, so only hours of D itself are published.
    o0 = pd.DatetimeIndex([pd.Timestamp("2026-03-10 00:00", tz=TZ)])
    assert X.available_mask(o0, 23).iloc[0] and not X.available_mask(o0, 24).iloc[0]


def test_exog_features_have_no_look_ahead():
    """Unpublished target hours (wind/solar/load) are NaN; neighbour prices come from the origin and 24 h before only."""
    import exog as X

    ex = _exog_frame()
    origin = pd.DatetimeIndex([pd.Timestamp("2026-03-05 23:00", tz=TZ)])
    f24 = X.exog_features(ex, origin, 24)
    f30 = X.exog_features(ex, origin, 30)

    assert f24["wind_DE_LU_MW_target"].iloc[0] == ex["wind_DE_LU_MW"].loc[origin[0] + pd.Timedelta(hours=24)]
    assert f24["load_NO_1_MW_target"].iloc[0] == ex["load_NO_1_MW"].loc[origin[0] + pd.Timedelta(hours=24)]
    assert np.isnan(f30["load_NO_1_MW_target"].iloc[0])
    assert np.isnan(f30["wind_DE_LU_MW_target"].iloc[0]) and np.isnan(f30["solar_DE_LU_MW_target"].iloc[0])
    assert f30["price_SE_3_EUR_MWh_origin"].iloc[0] == ex["price_SE_3_EUR_MWh"].loc[origin[0]]
    assert f30["price_SE_3_EUR_MWh_origin_lag24h"].iloc[0] == ex["price_SE_3_EUR_MWh"].loc[origin[0] - pd.Timedelta(hours=24)]

    tampered = ex.copy()  # change prices after the origin and wind in the unpublished hours
    tampered.loc[tampered.index > origin[0], "price_SE_3_EUR_MWh"] = -999.0
    tampered.loc[tampered.index > origin[0] + pd.Timedelta(hours=24), ["wind_DE_LU_MW", "load_NO_1_MW"]] = -999.0
    pd.testing.assert_frame_equal(X.exog_features(tampered, origin, 30), f30)


def test_assemble_appends_exog_features_after_price_and_weather():
    """Column order: price/calendar, weather, exog; feature_names matches what is built."""
    import exog as X

    price = synthetic_price()
    history = F.history_features(price)
    ex = _exog_frame()
    ex.index = price.index[: len(ex)]
    weather = _weather_frame(price.index)
    origins = price.index[100:103]

    X0 = F.assemble(history, origins, 6)
    X1 = F.assemble(history, origins, 6, weather=weather, exog=ex)
    assert list(X1.columns) == (list(X0.columns) + ["temperature_C_target", "wind_speed_ms_target"]
                                + X.feature_names(ex.columns))
    assert list(F.assemble(history, origins, 6, exog=ex).columns)[len(X0.columns):] == X.feature_names(ex.columns)


def test_exog_fetch_skips_failing_zones_and_fails_when_nothing_arrives():
    """Offline check of fetch_entsoe_exog with a fake client."""
    import fetch_entsoe_exog as FX

    idx = pd.date_range("2026-01-01", periods=8, freq="15min", tz="Europe/Brussels")

    class Fake:
        def query_wind_and_solar_forecast(self, zone, start, end):
            if zone == "BAD":
                raise ValueError("no data")
            return pd.DataFrame({"Solar": 1.0, "Wind Onshore": 2.0}, index=idx)

        def query_day_ahead_prices(self, zone, start, end):
            return pd.Series(50.0, index=idx)

        def query_load_forecast(self, zone, start, end):
            return pd.DataFrame({"Forecasted Load": 4000.0}, index=idx)

    chunks = [(pd.Timestamp("2026-01-01", tz="Europe/Brussels"), pd.Timestamp("2026-01-02", tz="Europe/Brussels"))]
    df = FX.fetch_all(Fake(), ["DE_LU", "BAD"], ["NO_2"], chunks, retries=1, sleep=0, load_zones=["NO_1"])
    assert sorted(df.columns) == ["load_NO_1_MW", "price_NO_2_EUR_MWh", "solar_DE_LU_MW", "wind_DE_LU_MW"]
    assert (df["load_NO_1_MW"].dropna() == 4000.0).all() and str(df.index.tz) == TZ
    assert FX.zone_columns("load", "NO_1") == ["load_NO_1_MW"]

    class Empty(Fake):
        def query_wind_and_solar_forecast(self, zone, start, end):
            raise ValueError("no data")

        def query_day_ahead_prices(self, zone, start, end):
            raise ValueError("no data")

    try:
        FX.fetch_all(Empty(), ["DE_LU"], ["NO_2"], chunks, retries=1, sleep=0)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when no zone returns data")


def _nwp_frame(n_hours: int = 24 * 12) -> pd.DataFrame:
    """NWP columns <base>_d<N> whose value encodes the lead N (N*1000 + hour index)."""
    idx = pd.date_range("2026-03-01", periods=n_hours, freq="h", tz=TZ)
    pos = np.arange(n_hours, dtype=float)
    return pd.DataFrame({f"nwp_wind_DE_d{n}": n * 1000 + pos for n in (1, 2, 3, 4)}, index=idx)


def test_nwp_lead_follows_the_decision_time():
    """Decision 14:00 on D, origin 23:00 on D+1 (lag 33), 6 h margin: D+2 00-08 use the
    2-day-old run, D+2 09:00 onward the 3-day-old run, D+3 late hours the 4-day-old run."""
    import nwp as N

    o = pd.DatetimeIndex([pd.Timestamp("2026-03-05 23:00", tz=TZ)])
    lead = {h: int(N.lead_days(o, h, 33)[0]) for h in (1, 9, 10, 33, 34, 48)}
    assert lead == {1: 2, 9: 2, 10: 3, 33: 3, 34: 4, 48: 4}
    assert int(N.lead_days(o, 1, 11)[0]) == 1 and int(N.lead_days(o, 24, 11)[0]) == 2


def test_nwp_features_use_only_runs_available_at_the_decision_time():
    import nwp as N

    nwp = _nwp_frame()
    o = pd.DatetimeIndex([pd.Timestamp("2026-03-05 23:00", tz=TZ)])
    pos = int((o[0] - nwp.index[0]) / pd.Timedelta(hours=1))
    f10 = N.nwp_features(nwp, o, 10, 33)
    assert list(f10.columns) == ["nwp_wind_DE_target"]
    assert f10.iloc[0, 0] == 3000 + pos + 10                        # 3-day-old run at the target hour

    tampered = nwp.copy()                                            # newer runs must not matter
    tampered[["nwp_wind_DE_d1", "nwp_wind_DE_d2"]] = -999.0
    pd.testing.assert_frame_equal(N.nwp_features(tampered, o, 10, 33), f10)
    assert np.isnan(N.nwp_features(nwp[["nwp_wind_DE_d1"]], o, 48, 33).iloc[0, 0])  # lead 4 missing -> NaN
    assert N.bases(nwp.columns) == ["nwp_wind_DE"]


def test_openmeteo_response_is_parsed_into_group_means():
    """One JSON object per point (list, request order) -> <base>_d<N> group means; errors raise."""
    import fetch_openmeteo as FO

    t = [1772323200, 1772326800]                                     # two hours, unix seconds
    items = []
    for i, (lat, lon) in enumerate(FO._points()):
        hourly = {"time": t}
        for v in FO._variables():
            for n in FO.LEADS:
                hourly[f"{v}_previous_day{n}"] = [float(i), float(i) + n]
        items.append({"latitude": lat, "longitude": lon, "hourly": hourly})
    df = FO.parse_response(items)
    assert len(df) == 2 and str(df.index.tz) == TZ
    assert set(df.columns) == {f"{b}_d{n}" for b in FO.GROUPS for n in FO.LEADS}
    pts = FO._points()
    de = [pts.index(p) for p in FO.GROUPS["nwp_wind_DE"][1]]
    assert df["nwp_wind_DE_d2"].iloc[1] == np.mean(de) + 2
    try:
        FO.parse_response({"error": True, "reason": "bad model"})
    except RuntimeError as exc:
        assert "bad model" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for an API error")


def test_reanalysis_fills_only_the_years_before_the_forecast_archive():
    """ERA5 values fill each lead column before its first archived forecast; forecasts and
    later gaps are left untouched."""
    import fetch_openmeteo as FO

    t = [1700000000, 1700003600, 1700007200]
    items = [{"hourly": {"time": t, **{v: [1.0, 2.0, 3.0] for v in FO._variables()}}} for _ in FO._points()]
    rean = FO.parse_reanalysis(items)
    assert set(rean.columns) == set(FO.GROUPS) and len(rean) == 3

    idx = rean.index
    fc = pd.DataFrame({"nwp_wind_DE_d1": [np.nan, 20.0, np.nan], "nwp_wind_DE_d2": [np.nan, np.nan, 30.0]}, index=idx)
    out = FO.fill_before_archive(fc, rean)
    assert np.isnan(out["nwp_wind_DE_d1"].iloc[2])                     # gap after the archive start stays
    assert out["nwp_wind_DE_d1"].iloc[:2].tolist() == [1.0, 20.0]
    assert out["nwp_wind_DE_d2"].tolist() == [1.0, 2.0, 30.0]


def test_decision_lag_moves_exog_availability_and_nwp_columns_follow_exog():
    """Lag 33 (forecast D+2/D+3 after the auction): ENTSO-E day-ahead values for D+2 are not
    published at 14:00 on D, neighbour prices at the origin are; NWP columns come last."""
    import exog as X
    import nwp as N

    price = synthetic_price()
    history = F.history_features(price)
    ex = _exog_frame()
    ex.index = price.index[: len(ex)]
    nwp = _nwp_frame()
    nwp.index = price.index[: len(nwp)]
    origins = price.index[100:101]
    X33 = F.assemble(history, origins, 1, exog=ex, nwp=nwp, decision_lag_h=33)
    assert np.isnan(X33["wind_DE_LU_MW_target"].iloc[0]) and not np.isnan(X33["price_SE_3_EUR_MWh_origin"].iloc[0])
    X11 = F.assemble(history, origins, 1, exog=ex, nwp=nwp)
    assert not np.isnan(X11["wind_DE_LU_MW_target"].iloc[0])
    assert list(X33.columns)[-1:] == N.feature_names(["nwp_wind_DE"])
    assert list(X33.columns) == list(X11.columns)


def test_target_anchors_use_only_known_prices_and_round_trip():
    """Every anchor uses prices at or before the origin; to_price(to_target(y)) == y."""
    import targets as T

    price = synthetic_price()
    origins = price.index[400:405]
    for mode in T.MODES:
        for h in (1, 24, 25, 48):
            off, sc = T.anchor(price, origins, h, mode)
            tampered = price.copy()
            tampered.loc[tampered.index > origins[-1]] = 999.0       # prices after the last origin
            off2, sc2 = T.anchor(tampered, origins, h, mode)
            assert np.allclose(off, off2) and np.allclose(sc, sc2), (mode, h)
            y = price.reindex(origins + pd.Timedelta(hours=h)).to_numpy()
            assert np.allclose(T.to_price(T.to_target(y, off, sc), off, sc), y)
    # same clock hour on the last known day: h=5 -> t-24, h=30 -> t-48
    o = origins[:1]
    assert T.anchor(price, o, 5, "diff_sameh")[0][0] == price.loc[o[0] + pd.Timedelta(hours=5 - 24)]
    assert T.anchor(price, o, 30, "diff_sameh")[0][0] == price.loc[o[0] + pd.Timedelta(hours=30 - 48)]
    assert np.isclose(T.anchor(price, o, 5, "diff_mean24")[0][0], price.loc[:o[0]].iloc[-24:].mean())
    X = T.add_features(pd.DataFrame(index=o), *T.anchor(price, o, 5, "rel_mean24"), "rel_mean24")
    assert list(X.columns) == ["target_anchor", "target_scale"]
    assert list(T.add_features(pd.DataFrame(index=o), *T.anchor(price, o, 5, "level"), "level").columns) == []


def test_eurnok_rates_parse_and_fall_back_to_the_fixed_rate():
    """Norges Bank sdmx-json is parsed; weekends use the last business day; stale or
    missing data falls back to config.EUR_TO_NOK."""
    import fx as FXR
    from config import EUR_TO_NOK

    payload = {"data": {
        "dataSets": [{"series": {"0:0:0:0": {"observations": {"0": ["11.50"], "1": ["11.60"]}}}}],
        "structure": {"dimensions": {"series": [], "observation": [
            {"id": "TIME_PERIOD", "values": [{"id": "2026-11-19"}, {"id": "2026-11-20"}]}]}},
    }}
    r = FXR.parse_sdmx_json(payload)
    assert r.tolist() == [11.5, 11.6] and str(r.index[1].date()) == "2026-11-20"
    assert FXR.rate_on(pd.Timestamp("2026-11-22 14:00", tz=TZ), r) == (11.6, "Norges Bank 2026-11-20")
    assert FXR.rate_on("2026-12-15", r)[0] == EUR_TO_NOK                  # older than MAX_AGE_DAYS
    assert FXR.rate_on("2026-11-22", None)[0] == EUR_TO_NOK                # no file
    ts = pd.DatetimeIndex(["2026-11-19 23:00", "2026-11-21 12:00", "2026-11-01 00:00"]).tz_localize(TZ)
    assert FXR.rates_for(ts, r).tolist() == [11.5, 11.6, EUR_TO_NOK]


def test_exog_incremental_start_and_merge():
    import fetch_entsoe_exog as FX

    tz = "Europe/Oslo"
    start = pd.Timestamp("2020-01-01", tz="Europe/Brussels")
    idx = pd.date_range("2020-01-01", "2026-11-10 23:00", freq="h", tz=tz)
    old = pd.DataFrame({"wind_DE_LU_MW": 1.0, "solar_DE_LU_MW": 1.0, "price_NO_2_EUR_MWh": 5.0}, index=idx)
    assert FX.zone_start(None, "price", "NO_2", start) == start
    assert FX.zone_start(old, "price", "SE_3", start) == start                       # new zone -> full
    inc = FX.zone_start(old, "wind_solar", "DE_LU", start, overlap_days=3)
    assert inc == pd.Timestamp("2026-11-07", tz="Europe/Brussels")                   # last - 3 days
    late = old.loc["2024-01-01":]
    assert FX.zone_start(late, "price", "NO_2", start) == start                      # backfill needed
    gap = old.copy()
    gap.loc[:"2021-12-01", "solar_DE_LU_MW"] = float("nan")                          # solar starts late at the source
    assert FX.zone_start(gap, "wind_solar", "DE_LU", start, overlap_days=3) == inc   # still incremental
    new = pd.DataFrame({"price_NO_2_EUR_MWh": 9.0},
                       index=pd.date_range("2026-11-10", "2026-11-12 23:00", freq="h", tz=tz))
    m = FX.merge_exog(old, new)
    assert m.loc["2026-11-10 12:00", "price_NO_2_EUR_MWh"] == 9.0                     # overlap: new wins
    assert m.loc["2026-11-12 12:00", "price_NO_2_EUR_MWh"] == 9.0                     # extended
    assert m.loc["2026-11-05 12:00", "price_NO_2_EUR_MWh"] == 5.0                    # old kept
    assert m.loc["2026-11-10 12:00", "wind_DE_LU_MW"] == 1.0                         # other columns kept


def test_incremental_plan_and_merge():
    import incremental as INC

    tz = "Europe/Oslo"
    start = pd.Timestamp("2020-01-01", tz=tz)
    first, last = pd.Timestamp("2020-01-01", tz=tz), pd.Timestamp("2026-11-10 23:00", tz=tz)
    assert INC.plan_start(None, None, start) == start
    assert INC.plan_start(first, last, start, 3) == pd.Timestamp("2026-11-07", tz=tz)
    assert INC.plan_start(pd.Timestamp("2024-01-01", tz=tz), last, start) == start   # backfill
    assert INC.plan_start(first, last, start, 100000) == start                       # never before start
    old = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.date_range("2026-11-01", periods=3, freq="D", tz=tz))
    new = pd.DataFrame({"a": [9.0, 8.0]}, index=pd.date_range("2026-11-03", periods=2, freq="D", tz=tz))
    m = INC.merge_new(old, new)
    assert list(m["a"]) == [1.0, 2.0, 9.0, 8.0]


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
