"""End-to-end smoke test for the London Santander pipeline.

Runs without network by generating tiny synthetic Santander trip records,
verifies the full pipeline (aggregate -> features -> assembler) produces
valid output.

Usage:
    python scripts/smoke_london_pipeline.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def make_synthetic_trips(n_days: int = 60) -> pd.DataFrame:
    """Synthesize per-trip records like Santander would produce."""
    from src.data.santander import SOHO_STATIONS

    station_ids = list(SOHO_STATIONS.keys())
    rng = np.random.default_rng(42)
    rows = []
    start = datetime(2024, 1, 1)
    for d in range(n_days):
        day = start + timedelta(days=d)
        dow = day.weekday()
        for hour in range(24):
            hour_mult = 0.2 + 0.8 * np.sin(np.pi * hour / 24) ** 2
            n_trips = int(rng.poisson(15 * hour_mult))
            for _ in range(n_trips):
                ts = day.replace(hour=hour, minute=int(rng.integers(0, 60)))
                start_stn = str(rng.choice(station_ids))
                end_stn = str(rng.choice(station_ids))
                rows.append({
                    "start_ts": ts,
                    "end_ts": ts + timedelta(minutes=int(rng.integers(5, 30))),
                    "start_station": start_stn,
                    "end_station": end_stn,
                })
    return pd.DataFrame(rows)


def run_smoke_test() -> None:
    print("=" * 70)
    print("London Santander pipeline smoke test")
    print("=" * 70)

    from src.data.feature_channels import get_active_channels, get_cold_channels
    from src.data.santander import SOHO_STATIONS, aggregate_hourly, restaurant_meta
    from src.data.uk_cultural_calendar import build_uk_cultural_calendar
    from src.features.assembler import assemble_features

    print("\n[1/5] Synthesizing trip records")
    trips = make_synthetic_trips(n_days=60)
    print(f"      {len(trips)} synthetic trips")

    print("\n[2/5] Aggregating to hourly counts")
    station_ids = set(SOHO_STATIONS.keys())
    hourly = aggregate_hourly(trips, station_ids)
    assert "cover_count" in hourly.columns
    assert len(hourly) > 0
    print(f"      {len(hourly)} hourly rows, avg hourly trips: {hourly['cover_count'].mean():.1f}")

    print("\n[3/5] UK cultural calendar")
    cultural = build_uk_cultural_calendar(
        hourly["timestamp_utc"].min().date(),
        hourly["timestamp_utc"].max().date(),
        "soho_cycles",
    )
    print(f"      {int(cultural['is_cultural_period'].sum())} cultural days")

    print("\n[4/5] Channel registry")
    active = get_active_channels()
    cold = get_cold_channels()
    print(f"      Active ({len(active)}): {active}")
    print(f"      Cold   ({len(cold)}): {cold}")

    print("\n[5/5] Assembler with empty signal frames")
    meta = restaurant_meta()
    features = assemble_features(
        covers_hourly=hourly,
        weather_df=pd.DataFrame(),
        holiday_df=pd.DataFrame(),
        event_df=pd.DataFrame(),
        transport_df=pd.DataFrame(),
        closures_df=pd.DataFrame(),
        restaurant_meta=meta[[
            "restaurant_id", "seating_capacity", "turnover_rate_per_hour",
            "city_tier", "footfall_zone_class", "country_code",
        ]],
        tz_map={"soho_cycles": "Europe/London"},
    )
    print(f"      {len(features)} rows, {len(features.columns)} cols")
    assert "cover_count" in features.columns

    print("\n" + "=" * 70)
    print("Smoke test PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
