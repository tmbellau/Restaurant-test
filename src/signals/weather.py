"""Open-Meteo weather source.

Proves the signal-hub pattern end-to-end with the simplest possible external
source: no API key, generous rate limits, clean JSON, and the UK Met Office
2km model via open-meteo's UKMO endpoint for high-accuracy UK forecasts.

Returns hourly weather observations and forecasts with the columns used by
Layer 2 feature engineering:
    - temperature_c
    - apparent_temperature_c (feels_like)
    - precipitation_mm
    - precipitation_probability (forecast only)
    - wind_speed_kmh
    - weather_code
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd
import structlog

from src.config import settings
from src.signals.base import BaseSignalSource

log = structlog.get_logger(__name__)


# Column layout emitted by this source. Feature engineering joins on these.
FEATURE_SCHEMA: dict[str, str] = {
    "temperature_c": "float",
    "apparent_temperature_c": "float",
    "precipitation_mm": "float",
    "precipitation_probability": "float",
    "wind_speed_kmh": "float",
    "weather_code": "int",
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "precipitation_probability",
    "wind_speed_10m",
    "weather_code",
]


class OpenMeteoWeatherSource(BaseSignalSource):
    """Fetches hourly weather (historical + forecast) from Open-Meteo.

    This implementation takes a (lat, lng) per restaurant. For Layer 1 we
    resolve lat/lng from the restaurants table in the caller; the method
    signature accepts restaurant_id for protocol compatibility and expects
    the caller to pre-resolve coordinates via the DB before calling.

    For ad-hoc / test use, `fetch_by_coordinates` is the direct entrypoint.
    """

    source_name = "open_meteo_weather"
    refresh_interval_seconds = 6 * 3600  # 6 hours
    feature_columns = list(FEATURE_SCHEMA.keys())

    def __init__(self, http_timeout: float = 15.0) -> None:
        super().__init__()
        self._timeout = http_timeout

    # ---- public coordinate-based entrypoint ----

    def fetch_by_coordinates(
        self,
        lat: float,
        lng: float,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly weather for a (lat, lng) between start_utc and end_utc.

        Automatically chooses the forecast or historical API based on the
        request window relative to now.
        """
        now = datetime.now(timezone.utc)
        use_historical = end_utc < now
        base = (
            settings.open_meteo_historical_url
            if use_historical
            else settings.open_meteo_base_url
        )
        endpoint = f"{base}/archive" if use_historical else f"{base}/forecast"

        params: dict[str, str | float] = {
            "latitude": lat,
            "longitude": lng,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
            "start_date": start_utc.date().isoformat(),
            "end_date": end_utc.date().isoformat(),
        }

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(endpoint, params=params)
            resp.raise_for_status()
            data = resp.json()

        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            log.warning("weather.empty_response", lat=lat, lng=lng)
            return pd.DataFrame(columns=["timestamp_utc", *FEATURE_SCHEMA.keys()])

        df = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(times, utc=True),
                "temperature_c": hourly.get("temperature_2m", []),
                "apparent_temperature_c": hourly.get("apparent_temperature", []),
                "precipitation_mm": hourly.get("precipitation", []),
                "precipitation_probability": hourly.get("precipitation_probability", [None] * len(times)),
                "wind_speed_kmh": hourly.get("wind_speed_10m", []),
                "weather_code": hourly.get("weather_code", []),
            }
        )
        return df

    # ---- BaseSignalSource contract ----

    def _fetch_raw_impl(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        """Resolve restaurant coordinates from DB, then call fetch_by_coordinates."""
        from src.db.engine import SessionLocal
        from src.db.models import Restaurant

        with SessionLocal() as session:
            loc = session.get(Restaurant, restaurant_id)
            if loc is None:
                raise ValueError(f"Unknown restaurant: {restaurant_id}")
            df = self.fetch_by_coordinates(loc.lat, loc.lng, start_utc, end_utc)
        df.insert(0, "restaurant_id", restaurant_id)
        return df

    def transform_to_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Raw Open-Meteo output already matches feature_columns; return as-is.

        Layer 2 feature engineering adds derived features (temp_anomaly,
        warm_evening, precip_intensity_cat) downstream; this source emits
        only the primitives.
        """
        return raw_df

    def get_feature_schema(self) -> dict[str, str]:
        return dict(FEATURE_SCHEMA)
