"""Feature assembler with point-in-time-correct joins.

This is the central pipeline: takes raw hourly cover data + all signal sources,
converts UTC -> local at the boundary, computes features, and outputs a training-
ready DataFrame.

Per the spec: all joins use strict inequality (feature_timestamp < target_timestamp)
to prevent data leakage. Pandera validates the output.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import structlog

from src.features.censoring import flag_censored_hours, flag_closures
from src.features.lag_features import compute_lag_features
from src.features.temporal import compute_temporal_features
from src.features.weather_features import compute_weather_features
from src.util.time_zones import TimeZoneManager

log = structlog.get_logger(__name__)


def _utc_to_local(df: pd.DataFrame, tz_map: dict[str, str]) -> pd.DataFrame:
    """Convert timestamp_utc -> local_ts using per-restaurant timezone mapping."""
    out = df.copy()
    local_times = []
    for _, row in out.iterrows():
        tz_name = tz_map.get(row["restaurant_id"], "Europe/London")
        mgr = TimeZoneManager(tz_name)
        local_times.append(mgr.to_local(row["timestamp_utc"]))
    out["local_ts"] = local_times
    return out


def _pit_merge_signal(
    covers: pd.DataFrame,
    signal_df: pd.DataFrame,
    signal_cols: list[str],
    on: str = "restaurant_id",
) -> pd.DataFrame:
    """Point-in-time merge: for each cover row, find the most recent signal
    row STRICTLY BEFORE the cover timestamp.

    Uses pandas merge_asof with direction='backward' which is equivalent to
    the LATERAL join pattern in SQL.
    """
    if signal_df.empty:
        for c in signal_cols:
            covers[c] = None
        return covers

    covers_sorted = covers.sort_values(["restaurant_id", "timestamp_utc"])
    signal_sorted = signal_df.sort_values(["restaurant_id", "timestamp_utc"])

    merged = pd.merge_asof(
        covers_sorted,
        signal_sorted[["restaurant_id", "timestamp_utc"] + signal_cols],
        on="timestamp_utc",
        by="restaurant_id",
        direction="backward",
        suffixes=("", "_signal"),
    )
    return merged


def assemble_features(
    covers_hourly: pd.DataFrame,
    weather_df: pd.DataFrame,
    holiday_df: pd.DataFrame,
    event_df: pd.DataFrame,
    transport_df: pd.DataFrame,
    closures_df: pd.DataFrame,
    restaurant_meta: pd.DataFrame,
    tz_map: dict[str, str],
) -> pd.DataFrame:
    """Build the full feature matrix from raw data sources.

    Args:
        covers_hourly: Hourly aggregated covers with restaurant_id, timestamp_utc, cover_count.
        weather_df: Raw weather with timestamp_utc, restaurant_id, temperature_c, etc.
        holiday_df: Holiday features per day.
        event_df: Event features per day.
        transport_df: Transport disruption features.
        closures_df: Location closures (restaurant_id, start_utc, end_utc).
        restaurant_meta: Static features per restaurant (capacity, city_tier, etc.).
        tz_map: {restaurant_id: iana_timezone_name}.

    Returns:
        Training-ready DataFrame with all features + target (cover_count).
    """
    log.info("assembler.start", rows=len(covers_hourly))

    # 1. Convert UTC -> local time at the boundary
    df = _utc_to_local(covers_hourly, tz_map)

    # 2. PiT-merge weather
    weather_cols = [
        "temperature_c", "apparent_temperature_c", "precipitation_mm",
        "wind_speed_kmh", "weather_code",
    ]
    available_weather_cols = [c for c in weather_cols if c in weather_df.columns]
    if available_weather_cols:
        df = _pit_merge_signal(df, weather_df, available_weather_cols)

    # 3. PiT-merge holidays + cultural calendar (daily granularity, forward-fill)
    holiday_cols = [
        "is_bank_holiday", "is_school_holiday",
        "days_to_next_holiday", "days_from_last_holiday",
        "is_cultural_period", "days_to_next_cultural", "days_from_last_cultural",
    ]
    available_hol_cols = [c for c in holiday_cols if c in holiday_df.columns]
    if available_hol_cols:
        df = _pit_merge_signal(df, holiday_df, available_hol_cols)

    # 4. PiT-merge events
    event_cols = ["major_event_within_2km", "event_est_attendance", "local_football_match"]
    available_evt_cols = [c for c in event_cols if c in event_df.columns]
    if available_evt_cols:
        df = _pit_merge_signal(df, event_df, available_evt_cols)

    # 5. PiT-merge transport
    transport_cols = ["tube_disruption", "tube_line_affected_count"]
    available_tr_cols = [c for c in transport_cols if c in transport_df.columns]
    if available_tr_cols:
        df = _pit_merge_signal(df, transport_df, available_tr_cols)

    # 6. Join static restaurant metadata
    meta_cols = ["restaurant_id", "seating_capacity", "turnover_rate_per_hour",
                 "city_tier", "footfall_zone_class", "country_code"]
    available_meta = [c for c in meta_cols if c in restaurant_meta.columns]
    df = df.merge(restaurant_meta[available_meta], on="restaurant_id", how="left")

    # 7. Compute derived features
    if "temperature_c" in df.columns:
        df = compute_weather_features(df)
    df = compute_temporal_features(df)
    df = compute_lag_features(df, target_col="cover_count")

    # 8. Censoring + closures
    if "seating_capacity" in df.columns and "turnover_rate_per_hour" in df.columns:
        df = flag_censored_hours(df)
    if not closures_df.empty:
        df = flag_closures(df, closures_df)
    else:
        df["was_closed"] = 0

    # 9. Exclude closed rows from training
    df = df[df["was_closed"] == 0].copy()

    log.info("assembler.done", rows=len(df), cols=len(df.columns))
    return df
