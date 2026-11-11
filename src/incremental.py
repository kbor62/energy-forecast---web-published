"""Shared logic for incremental updates of downloaded history.

A fetch script that finds its output file already present only needs the tail of the
history: from the last stored timestamp minus a few days of overlap (late-published
or revised values are re-fetched and override the old ones). It falls back to the
full history when nothing is stored or when the stored data starts later than the
requested start (backfill).
"""

from __future__ import annotations

import pandas as pd


def plan_start(first: pd.Timestamp | None, last: pd.Timestamp | None, start: pd.Timestamp,
               overlap_days: int = 3, backfill_tolerance_days: int = 7) -> pd.Timestamp:
    """Where to start fetching: `start` (full) or last stored value - overlap (incremental).

    `first`/`last` are the first/last stored timestamps (None if nothing is stored).
    The result is midnight in the timezone of `start` and never earlier than `start`.
    """
    if first is None or last is None:
        return start
    if first > start + pd.Timedelta(days=backfill_tolerance_days):
        return start
    inc = (last - pd.Timedelta(days=overlap_days)).tz_convert(start.tz).normalize()
    return max(inc, start)


def merge_new(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Union of old and new; where both have a value the new one wins."""
    if old is None:
        return new
    return new.combine_first(old).sort_index()
