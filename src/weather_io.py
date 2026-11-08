"""Weather inputs for forecasting: archived MET forecast snapshots + observations.

``fetch_weather_forecast.py`` writes one snapshot per MET model run to
``data/forecast_archive/metno_<issued>.parquet`` (index = valid time in UTC,
column ``issued_at``). ``weather_for_forecast`` builds the weather series a
weather-trained model needs at the target hours: snapshot values where the
snapshot covers the hour, observations otherwise.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from config import TZ, repo_path

ARCHIVE_DIR = "data/forecast_archive"
LOCATION_PREFIX = "oslo_"  # snapshot columns are prefixed per location


def list_snapshots(archive_dir: str | Path = ARCHIVE_DIR) -> list[tuple[pd.Timestamp, Path]]:
    """(issued_at UTC, path) for each archived snapshot, oldest first."""
    out = []
    for p in repo_path(archive_dir).glob("metno_*.parquet"):
        try:
            issued = pd.to_datetime(p.stem.removeprefix("metno_"), format="%Y%m%dT%H%MZ", utc=True)
        except ValueError:
            continue
        out.append((issued, p))
    return sorted(out)


def pick_snapshot(as_of: pd.Timestamp, archive_dir: str | Path = ARCHIVE_DIR):
    """Latest snapshot issued at or before ``as_of`` (None if there is none)."""
    as_of = as_of.tz_convert("UTC") if as_of.tzinfo else as_of.tz_localize(TZ).tz_convert("UTC")
    ok = [(i, p) for i, p in list_snapshots(archive_dir) if i <= as_of]
    return ok[-1] if ok else None


@lru_cache(maxsize=64)
def _read_snapshot(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_observed(path: str | Path, cols: list[str]) -> pd.DataFrame:
    """Observed hourly weather in the project time zone, restricted to ``cols``."""
    return _canonical(pd.read_parquet(repo_path(path)), cols)


def _canonical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.removeprefix(LOCATION_PREFIX))
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
    df.index = df.index.tz_convert(TZ)
    df = df[~df.index.duplicated()].sort_index()
    return df.reindex(columns=cols)


def weather_for_forecast(
    cols: list[str],
    targets: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    archive_dir: str | Path = ARCHIVE_DIR,
    observed_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Weather at ``targets`` as it was known at ``as_of``.

    Uses the latest snapshot issued <= as_of; hours the snapshot does not
    cover (before its first valid time) fall back to observations.
    """
    snap_hit = pick_snapshot(as_of, archive_dir)
    parts, info = [], {"as_of": str(as_of), "snapshot_issued_at": None, "observed_file": None}

    if snap_hit is not None:
        issued, path = snap_hit
        parts.append(_canonical(_read_snapshot(str(path)), cols))
        info["snapshot_issued_at"] = issued.isoformat()
        info["snapshot_file"] = path.name
    if observed_path is not None and repo_path(observed_path).exists():
        parts.append(_canonical(pd.read_parquet(repo_path(observed_path)), cols))
        info["observed_file"] = str(observed_path)

    if not parts:
        raise RuntimeError(
            f"No weather available as of {as_of}: no snapshot in {archive_dir} and no observed file. "
            "Run src/fetch_weather_forecast.py (and fetch_weather.py)."
        )

    W = parts[0]
    for extra in parts[1:]:
        W = W.combine_first(extra)
    W = W.reindex(targets)

    from_snap = parts[0].reindex(targets).notna() if snap_hit is not None else None
    info["cells_total"] = int(W.size)
    info["cells_missing"] = int(W.isna().sum().sum())
    info["cells_from_snapshot"] = int(from_snap.sum().sum()) if from_snap is not None else 0
    info["cells_from_observed"] = info["cells_total"] - info["cells_missing"] - info["cells_from_snapshot"]
    return W, info
