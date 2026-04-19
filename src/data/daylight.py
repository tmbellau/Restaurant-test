"""Astronomical daylight features for London.

Sunrise, sunset, and day-length are purely deterministic (from location +
date), free, and strongly drive cycling demand (more riders when more
daylight). Perfect-history, zero-cost features.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache

import pandas as pd
from astral import LocationInfo
from astral.sun import sun

_LONDON = LocationInfo(
    name="London", region="UK", timezone="Europe/London",
    latitude=51.5131, longitude=-0.1318,
)


@lru_cache(maxsize=4096)
def _sun_for(d: date) -> dict:
    return sun(_LONDON.observer, date=d, tzinfo=_LONDON.timezone)


def daylight_features_for_date(d: date) -> dict[str, float]:
    """Return daylight features for a single date (local London time)."""
    s = _sun_for(d)
    sunrise: datetime = s["sunrise"]
    sunset: datetime = s["sunset"]
    daylight_hours = (sunset - sunrise).total_seconds() / 3600.0
    sunset_hour = sunset.hour + sunset.minute / 60.0
    sunrise_hour = sunrise.hour + sunrise.minute / 60.0
    return {
        "daylight_hours": daylight_hours,
        "sunrise_hour": sunrise_hour,
        "sunset_hour": sunset_hour,
    }


def add_daylight_features(daily: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Append daylight_hours / sunrise_hour / sunset_hour columns."""
    out = daily.copy()
    dates = pd.to_datetime(out[date_col]).dt.date
    feats = [daylight_features_for_date(d) for d in dates]
    for k in ["daylight_hours", "sunrise_hour", "sunset_hour"]:
        out[k] = [f[k] for f in feats]
    return out
