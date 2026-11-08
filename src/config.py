"""Project constants."""

from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:  # secrets live in .env (git-ignored), never in code
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv is only needed for step 1 (fetch_prices.py)
    pass

TZ = "Europe/Oslo"
BRUSSELS_TZ = "Europe/Brussels"  # time zone used by the ENTSO-E API

PRICE_COL = "price_EUR_per_kWh"
TIMESTAMP_COL = "timestamp"

# Display rate (18 Sep 2026). Models train on EUR/kWh; NOK is for display only.
EUR_TO_NOK = 10.83

# ENTSO-E API token (free from https://transparency.entsoe.eu/), read from .env.
# Used by fetch_prices.py only.
ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN", "")

# MET Norway Frost API (https://frost.met.no/), read from .env.
# Used by fetch_weather.py.
FROST_CLIENT_ID = os.environ.get("FROST_CLIENT_ID", "")


def repo_path(value: str | Path) -> Path:
    """Path relative to the repository root."""
    p = Path(value)
    return p if p.is_absolute() else ROOT / p
