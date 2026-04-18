"""Hybrid hourly target: TfL daily volume x BestTime hourly shape.

The model's demand target is built by combining:
    1. TfL daily entries+exits at Tottenham Court Road (REAL daily volume,
       captures weather/holiday/event effects day-to-day)
    2. BestTime typical hourly pattern for Wagamama Soho (REAL venue-specific
       hourly shape from Google Popular Times)

For each historical day:
    hourly_count[h] = tfl_daily_total × besttime_proportion[day_of_week][h]

This is semi-synthetic at the hourly level (the hourly distribution within
a day uses a typical pattern, not per-day observations) but anchored in
real data on both axes.
"""

from __future__ import annotations

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def build_hybrid_hourly(
    tfl_daily: pd.DataFrame,
    besttime_pattern: dict[int, list[float]],
    restaurant_id: str,
) -> pd.DataFrame:
    """Combine TfL daily counts with BestTime hourly proportions.

    Args:
        tfl_daily: DataFrame with date, total_footfall columns.
        besttime_pattern: {day_of_week (0=Mon): [24 hourly proportions]}.
        restaurant_id: ID string for output DataFrame.

    Returns:
        DataFrame with: timestamp_utc, restaurant_id, cover_count, channel.
    """
    rows: list[dict] = []

    for _, day_row in tfl_daily.iterrows():
        d = day_row["date"]
        daily_total = day_row["total_footfall"]
        dow = d.weekday()  # 0=Monday, matches BestTime's day_int

        proportions = besttime_pattern.get(dow)
        if proportions is None:
            log.warning("hybrid.missing_dow", date=str(d), dow=dow)
            proportions = [1.0 / 24] * 24

        for hour, prop in enumerate(proportions):
            ts = pd.Timestamp(
                year=d.year, month=d.month, day=d.day,
                hour=hour, tz="Europe/London",
            ).tz_convert("UTC")

            hourly_count = int(round(daily_total * prop))
            rows.append(
                {
                    "timestamp_utc": ts,
                    "restaurant_id": restaurant_id,
                    "cover_count": hourly_count,
                    "channel": "dine_in",
                }
            )

    result = pd.DataFrame(rows)
    result = result.sort_values("timestamp_utc").reset_index(drop=True)

    log.info(
        "hybrid.built",
        n_days=len(tfl_daily),
        n_hours=len(result),
        total_volume=int(result["cover_count"].sum()),
        avg_hourly=int(result["cover_count"].mean()),
    )
    return result
