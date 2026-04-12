"""Weather feature engineering.

Per the spec: use anomalies (vs 30-year seasonal norm), not raw values.
Also compute feels_like, is_raining, precip_intensity_cat, warm_evening.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate monthly normal temperatures for London (30-year avg, °C).
_LONDON_MONTHLY_NORM = {
    1: 5.2, 2: 5.4, 3: 7.5, 4: 9.9, 5: 13.2, 6: 16.2,
    7: 18.4, 8: 18.1, 9: 15.3, 10: 11.8, 11: 8.1, 12: 5.6,
}


def compute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived weather features.

    Expects columns: temperature_c, apparent_temperature_c, precipitation_mm,
    wind_speed_kmh, weather_code, and local_ts (local datetime).
    """
    out = df.copy()

    # Temperature anomaly vs seasonal norm
    month = pd.to_datetime(out["local_ts"]).dt.month
    seasonal_norm = month.map(_LONDON_MONTHLY_NORM).fillna(12.0)
    out["temp_anomaly_c"] = out["temperature_c"] - seasonal_norm

    # Feels-like (already from Open-Meteo, but ensure column exists)
    if "apparent_temperature_c" in out.columns:
        out["feels_like_c"] = out["apparent_temperature_c"]
    else:
        out["feels_like_c"] = out["temperature_c"]

    # Rain features
    out["is_raining"] = (out["precipitation_mm"].fillna(0) > 0.1).astype(int)
    precip = out["precipitation_mm"].fillna(0)
    out["precip_intensity_cat"] = pd.cut(
        precip,
        bins=[-np.inf, 0.1, 2.5, 7.5, np.inf],
        labels=["none", "light", "moderate", "heavy"],
    ).astype(str)

    # Warm evening: feels_like > 18 AND local hour > 17
    local_hour = pd.to_datetime(out["local_ts"]).dt.hour
    out["warm_evening"] = (
        (out["feels_like_c"] > 18) & (local_hour >= 17)
    ).astype(int)

    return out
