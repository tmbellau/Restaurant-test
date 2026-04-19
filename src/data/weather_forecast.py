"""Open-Meteo weather FORECAST API client (for inference-time predictions).

Complements historical_weather.py (which uses the Archive API). At
inference time we don't have actual future weather — we have weather
forecasts that degrade with horizon. Forecast accuracy:
    1-2 days ahead: very good
    3-5 days ahead: good
    6-7 days ahead: useful but noticeably worse
    8+ days ahead: largely unreliable

Free, no API key required.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import httpx
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"


def fetch_forecast(
    lat: float,
    lng: float,
    days: int = 7,
) -> pd.DataFrame:
    """Fetch 7-day weather forecast for the given location.

    Returns daily-aggregated weather aligned with the training feature set:
    temp_mean_c, temp_min_c, temp_max_c, precipitation_total_mm,
    wind_speed_mean_kmh. Index is `date` (local Europe/London dates).
    """
    if not (1 <= days <= 14):
        raise ValueError("days must be 1-14")

    today = date.today()
    end = today + timedelta(days=days - 1)

    params = {
        "latitude": lat,
        "longitude": lng,
        "timezone": "Europe/London",
        "start_date": today.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum",
            "wind_speed_10m_max",
            "weather_code",
        ]),
    }

    with httpx.Client(timeout=20) as client:
        resp = client.get(FORECAST_ENDPOINT, params=params)
        resp.raise_for_status()
        data = resp.json()

    daily = data.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_mean_c": daily["temperature_2m_mean"],
        "temp_min_c": daily["temperature_2m_min"],
        "temp_max_c": daily["temperature_2m_max"],
        "precipitation_total_mm": daily["precipitation_sum"],
        "wind_speed_mean_kmh": daily["wind_speed_10m_max"],
        "weather_code": daily["weather_code"],
    })
    log.info("weather_forecast.done", n_days=len(df), lat=lat, lng=lng)
    return df
