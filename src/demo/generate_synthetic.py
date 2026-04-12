"""Demo data generator: real external signals + synthetic POS.

Per the spec: fully synthetic data produces unrealistically clean patterns.
This generator uses real weather from Open-Meteo for 5 London postcodes,
then generates synthetic POS covers with published elasticities and
NegBinomial noise.

Usage:
    python -m src.demo.generate_synthetic
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import structlog

from src.signals.weather import OpenMeteoWeatherSource

log = structlog.get_logger(__name__)

# 5 demo locations per the spec
DEMO_LOCATIONS = [
    {"id": "LON-EC1", "name": "Wagamama Old Street", "postcode": "EC1V 9QQ",
     "lat": 51.5252, "lng": -0.0879, "city_tier": "tier1",
     "footfall_zone_class": "high", "seating_capacity": 120,
     "base_hourly_covers": 45},
    {"id": "LON-W1", "name": "Wagamama Soho", "postcode": "W1D 4DL",
     "lat": 51.5131, "lng": -0.1318, "city_tier": "tier1",
     "footfall_zone_class": "very_high", "seating_capacity": 100,
     "base_hourly_covers": 40},
    {"id": "LON-SE1", "name": "Wagamama Southwark", "postcode": "SE1 9SG",
     "lat": 51.5047, "lng": -0.0886, "city_tier": "tier1",
     "footfall_zone_class": "high", "seating_capacity": 90,
     "base_hourly_covers": 35},
    {"id": "LON-SW19", "name": "Wagamama Wimbledon", "postcode": "SW19 1QB",
     "lat": 51.4214, "lng": -0.2064, "city_tier": "tier2",
     "footfall_zone_class": "medium", "seating_capacity": 80,
     "base_hourly_covers": 25},
    {"id": "OX-OX1", "name": "Wagamama Oxford", "postcode": "OX1 1EP",
     "lat": 51.7520, "lng": -1.2577, "city_tier": "tier2",
     "footfall_zone_class": "high", "seating_capacity": 85,
     "base_hourly_covers": 30},
]

# Opening hours (local): 11:00 - 23:00
OPEN_HOUR = 11
CLOSE_HOUR = 23

# Hourly shape: lunch peak 12-14, dinner peak 18-21
HOUR_MULTIPLIER = {
    11: 0.3, 12: 0.9, 13: 1.0, 14: 0.7, 15: 0.4, 16: 0.35,
    17: 0.5, 18: 0.85, 19: 1.0, 20: 0.95, 21: 0.7, 22: 0.4,
}

DOW_MULTIPLIER = {0: 0.8, 1: 0.8, 2: 0.85, 3: 0.9, 4: 1.15, 5: 1.3, 6: 1.0}


def _monthly_seasonality(month: int) -> float:
    """Seasonal multiplier: Jan dip, summer boost, Dec spike."""
    return {
        1: 0.75, 2: 0.82, 3: 0.90, 4: 0.95, 5: 1.0, 6: 1.05,
        7: 1.08, 8: 1.05, 9: 0.98, 10: 0.95, 11: 0.92, 12: 1.10,
    }.get(month, 1.0)


def _weather_effect(row: dict) -> float:
    """Rain and temperature effects on walk-in demand."""
    mult = 1.0
    temp = row.get("temperature_c", 15)
    precip = row.get("precipitation_mm", 0)

    # Rain penalty
    if precip > 5:
        mult *= 0.80
    elif precip > 1:
        mult *= 0.90

    # Temperature effect (warm evenings boost, extreme cold hurts)
    if temp > 22:
        mult *= 1.08
    elif temp < 3:
        mult *= 0.82
    return mult


def generate_covers(
    location: dict,
    weather_df: pd.DataFrame,
    days: int = 365,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic hourly cover data for one location.

    Uses NegBinomial noise (overdispersed, not Gaussian) per the spec.
    """
    if rng is None:
        rng = np.random.default_rng(hash(location["id"]) % 2**31)

    base = location["base_hourly_covers"]
    capacity = location["seating_capacity"]
    turnover = 1.75
    effective_cap = capacity * turnover

    start = datetime(2024, 4, 1, tzinfo=timezone.utc)
    rows = []

    # Index weather by date+hour for fast lookup
    weather_lookup: dict[str, dict] = {}
    if not weather_df.empty:
        for _, wr in weather_df.iterrows():
            ts = pd.Timestamp(wr["timestamp_utc"])
            key = f"{ts.date().isoformat()}T{ts.hour:02d}"
            weather_lookup[key] = wr.to_dict()

    for d in range(days):
        dt = start + timedelta(days=d)
        dow = dt.weekday()
        month = dt.month
        day_of_month = dt.day

        # Payday effect
        payday_mult = 1.12 if (day_of_month == 25 or (dow == 4 and day_of_month > 24)) else 1.0

        for h in range(OPEN_HOUR, CLOSE_HOUR):
            hour_mult = HOUR_MULTIPLIER.get(h, 0.5)
            dow_mult = DOW_MULTIPLIER.get(dow, 1.0)
            season_mult = _monthly_seasonality(month)

            # Weather effect
            w_key = f"{dt.date().isoformat()}T{h:02d}"
            w_row = weather_lookup.get(w_key, {})
            weather_mult = _weather_effect(w_row)

            # Expected covers
            mu = base * hour_mult * dow_mult * season_mult * weather_mult * payday_mult

            # NegBinomial noise (overdispersed)
            r = 8  # dispersion parameter
            p = r / (r + mu)
            cover = rng.negative_binomial(r, p)

            # Cap at effective capacity (censoring)
            cover = min(cover, int(effective_cap))

            ts = datetime(dt.year, dt.month, dt.day, h, 0, tzinfo=timezone.utc)
            rows.append({
                "restaurant_id": location["id"],
                "timestamp_utc": ts,
                "channel": "dine_in",
                "cover_count": cover,
                "transaction_id": f"SYN-{location['id']}-{d:04d}-{h:02d}",
                "item_id": None,
                "item_qty": cover,
                "item_price": 13.50,
                "voided": False,
                "comped": False,
                "record_version": 1,
            })

    df = pd.DataFrame(rows)

    # Add 2% missing hours (random gaps)
    n_drop = int(len(df) * 0.02)
    drop_idx = rng.choice(df.index, size=n_drop, replace=False)
    df = df.drop(drop_idx).reset_index(drop=True)

    return df


def fetch_real_weather(location: dict, days: int = 365) -> pd.DataFrame:
    """Fetch real historical weather from Open-Meteo."""
    src = OpenMeteoWeatherSource()
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=days)
    try:
        return src.fetch_by_coordinates(location["lat"], location["lng"], start, end)
    except Exception as e:
        log.warning("demo.weather_fetch_failed", location=location["id"], error=str(e))
        return pd.DataFrame()


def generate_all(days: int = 365) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Generate demo data for all 5 locations.

    Returns (list_of_cover_dfs, list_of_weather_dfs).
    """
    cover_dfs = []
    weather_dfs = []
    for loc in DEMO_LOCATIONS:
        log.info("demo.generating", location=loc["id"])
        weather = fetch_real_weather(loc, days)
        weather_dfs.append(weather)
        covers = generate_covers(loc, weather, days)
        cover_dfs.append(covers)
        log.info("demo.done", location=loc["id"], rows=len(covers))
    return cover_dfs, weather_dfs


if __name__ == "__main__":
    cover_dfs, weather_dfs = generate_all()
    all_covers = pd.concat(cover_dfs)
    print(f"Generated {len(all_covers)} cover rows across {len(DEMO_LOCATIONS)} locations")
    # Save to CSV for ingestion via the POS pipeline
    out_path = "data/pos_inbox/demo_synthetic.csv"
    all_covers.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
