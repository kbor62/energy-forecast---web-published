"""Load the raw price series as a gap-free hourly series."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import PRICE_COL, TIMESTAMP_COL, TZ

INTERPOLATE_LIMIT = 6
INTERPOLATE_DIRECTION = "forward"


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
            df = df.set_index(TIMESTAMP_COL)
    else:
        df = pd.read_parquet(path)
        if TIMESTAMP_COL in df.columns:
            df = df.set_index(TIMESTAMP_COL)
    return df


def load_price_series(
    path: str | Path,
    tz: str = TZ,
    price_col: str = PRICE_COL,
    interpolate: bool = True,
    verbose: bool = True,
) -> pd.Series:
    """Hourly price series in EUR/kWh: duplicates removed, 15-minute prices
    averaged to hourly means, gaps of up to 6 hours interpolated."""
    path = Path(path)
    df = _read_any(path)
    df.index = df.index.tz_localize(tz) if df.index.tz is None else df.index.tz_convert(tz)
    df = df.sort_index()

    if df.index.has_duplicates:
        n_dup = int(df.index.duplicated(keep="last").sum())
        if verbose:
            print(f"Found {n_dup} duplicate timestamps; keeping the last value per timestamp.")
        df = df[~df.index.duplicated(keep="last")]

    raw = pd.to_numeric(df[price_col], errors="coerce")

    # From October 2025 prices are 15-minute; the hourly price is the mean of the quarters.
    n_subhourly = int((raw.index.minute != 0).sum())
    series = raw.resample("h").mean()
    if verbose and n_subhourly:
        print(
            f"Aggregated {n_subhourly:,} sub-hourly rows to hourly means "
            f"({len(raw):,} rows -> {len(series):,} hours)."
        )
    series.name = price_col

    if interpolate:
        n_missing = int(series.isna().sum())
        series = series.interpolate(
            "time", limit=INTERPOLATE_LIMIT, limit_direction=INTERPOLATE_DIRECTION
        )
        n_filled = n_missing - int(series.isna().sum())
        if verbose and n_filled:
            print(
                f"Filled {n_filled} of {n_missing} missing hours "
                f"(interpolation, limit={INTERPOLATE_LIMIT}h)."
            )

    return series


def describe_series(series: pd.Series) -> dict:
    """Summary of the series, written to output metadata."""
    return {
        "n_hours": int(len(series)),
        "n_missing": int(series.isna().sum()),
        "start": str(series.index.min()),
        "end": str(series.index.max()),
        "timezone": str(series.index.tz),
        "interpolation": f"time, limit={INTERPOLATE_LIMIT}, direction={INTERPOLATE_DIRECTION}",
    }
