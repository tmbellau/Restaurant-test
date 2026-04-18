"""Fetch real historical weather from Open-Meteo for a single location.

Open-Meteo Archive API: free, no key, 80+ years of hourly data globally.
For this London-only pipeline we need one (lat, lng) = Wagamama Soho.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = (
    "temperature_2m,apparent_temperature,precipitation,"
    "wind_speed_10m,weather_code"
)
WEATHER_COLS = [
    "temperature_c",
    "apparent_temperature_c",
    "precipitation_mm",
    "wind_speed_kmh",
    "weather_code",
]


def fetch_weather(
    lat: float,
    lng: float,
    start_date: str,
    end_date: str,
    restaurant_id: str,
    cache_dir: Path | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo archive for one (lat, lng).

    Returns DataFrame with: restaurant_id, timestamp_utc, + weather columns.
    """
    if cache_dir:
        cache_path = cache_dir / f"weather_{restaurant_id}.parquet"
        if cache_path.exists():
            log.info("weather.cache_hit", path=str(cache_path))
            return pd.read_parquet(cache_path)

    log.info(
        "weather.fetch.start",
        lat=lat,
        lng=lng,
        date_range=f"{start_date} to {end_date}",
    )

    params = {
        "latitude": lat,
        "longitude": lng,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
        "start_date": start_date,
        "end_date": end_date,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(OPEN_METEO_ARCHIVE, params=params)
        resp.raise_for_status()
        data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise RuntimeError("Open-Meteo returned no hourly data for request")

    df = pd.DataFrame(
        {
            "restaurant_id": restaurant_id,
            "timestamp_utc": pd.to_datetime(times, utc=True),
            "temperature_c": hourly.get("temperature_2m", []),
            "apparent_temperature_c": hourly.get("apparent_temperature", []),
            "precipitation_mm": hourly.get("precipitation", []),
            "wind_speed_kmh": hourly.get("wind_speed_10m", []),
            "weather_code": hourly.get("weather_code", []),
        }
    )

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_dir / f"weather_{restaurant_id}.parquet", index=False)

    log.info("weather.fetch.done", rows=len(df))
    return df
