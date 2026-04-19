"""Daily-granularity training pipeline.

Takes the hourly training_features.parquet and aggregates it to daily rows
with daily-level features (min/max/mean weather, holiday flags, lags,
daylight hours, rain streak, temperature shock, tube strikes, etc.).

Input: data/training/training_features.parquet (hourly)
Output: data/training/daily_training_features.parquet

Usage:
    python -m src.data.build_daily_training_set
    python -m src.data.build_daily_training_set --exclude-covid
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.data.daylight import add_daylight_features
from src.data.tube_strikes import add_tube_strike_feature

log = structlog.get_logger(__name__)

DEFAULT_INPUT = Path("data/training/training_features.parquet")
DEFAULT_OUTPUT = Path("data/training/daily_training_features.parquet")

# COVID-era dates to exclude: lockdowns plus re-opening instability.
# We keep everything before 2020-03-16 (first UK lockdown advisory) and
# skip forward to 2022-01-01 (Omicron wave over, Soho activity normalised).
COVID_START = date(2020, 3, 16)
COVID_END = date(2021, 12, 31)


def aggregate_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly features to one row per local date.

    Target: daily total trip count.
    Features: daily weather stats, holiday flags, cultural flags, temporal
    encodings, multi-day lags, rolling means.
    """
    df = hourly.copy()
    df["local_ts"] = pd.to_datetime(df["local_ts"])
    df["date"] = df["local_ts"].dt.date

    agg_map: dict[str, str | tuple] = {
        "cover_count": "sum",  # target: daily total trips
        "restaurant_id": "first",
    }

    # Weather: min/max/mean per day
    for col in ["temperature_c", "apparent_temperature_c", "wind_speed_kmh"]:
        if col in df.columns:
            agg_map[col] = "mean"

    if "precipitation_mm" in df.columns:
        agg_map["precipitation_mm"] = "sum"  # total daily rainfall

    # Binary flags: any-hour-of-day
    for col in ["is_bank_holiday", "is_school_holiday", "is_cultural_period"]:
        if col in df.columns:
            agg_map[col] = "max"

    # Distance-to-event features: use daily value (all hours of a day share)
    for col in ["days_to_next_holiday", "days_from_last_holiday",
                "days_to_next_cultural", "days_from_last_cultural"]:
        if col in df.columns:
            agg_map[col] = "min"

    # Categoricals: first value of day (same for all hours)
    for col in ["city_tier", "footfall_zone_class", "country_code"]:
        if col in df.columns:
            agg_map[col] = "first"

    daily = df.groupby("date").agg(agg_map).reset_index()
    daily = daily.rename(columns={
        "temperature_c": "temp_mean_c",
        "apparent_temperature_c": "apparent_temp_mean_c",
        "wind_speed_kmh": "wind_speed_mean_kmh",
        "precipitation_mm": "precipitation_total_mm",
    })

    # Derived per-day weather details
    temp_extrema = df.groupby("date")["temperature_c"].agg(["min", "max"]).reset_index()
    temp_extrema.columns = ["date", "temp_min_c", "temp_max_c"]
    daily = daily.merge(temp_extrema, on="date", how="left")

    if "temp_anomaly_c" in df.columns:
        anom = df.groupby("date")["temp_anomaly_c"].mean().reset_index()
        anom.columns = ["date", "temp_anomaly_mean_c"]
        daily = daily.merge(anom, on="date", how="left")

    # NOTE: intentionally NOT including peak-hour from same day (data leakage).
    # A future improvement could add lagged peak-hour features (e.g. peak_hour_7d_lag).

    # Calendar features
    daily["date"] = pd.to_datetime(daily["date"])
    daily["dow"] = daily["date"].dt.dayofweek  # 0=Mon
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["month"] = daily["date"].dt.month
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["day_of_year"] = daily["date"].dt.dayofyear

    # Sinusoidal encodings for day-of-week and day-of-year
    daily["dow_sin"] = np.sin(2 * np.pi * daily["dow"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["dow"] / 7)
    daily["doy_sin"] = np.sin(2 * np.pi * daily["day_of_year"] / 365)
    daily["doy_cos"] = np.cos(2 * np.pi * daily["day_of_year"] / 365)

    # Lags (daily) and rolling means
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["covers_1d_lag"] = daily["cover_count"].shift(1)
    daily["covers_7d_lag"] = daily["cover_count"].shift(7)
    daily["covers_14d_lag"] = daily["cover_count"].shift(14)
    daily["covers_28d_lag"] = daily["cover_count"].shift(28)
    daily["covers_365d_lag"] = daily["cover_count"].shift(365)  # year-ago
    daily["covers_7d_mean"] = (
        daily["cover_count"].shift(1).rolling(7, min_periods=1).mean()
    )
    daily["covers_28d_mean"] = (
        daily["cover_count"].shift(1).rolling(28, min_periods=1).mean()
    )
    daily["covers_7d_std"] = (
        daily["cover_count"].shift(1).rolling(7, min_periods=1).std()
    )

    # Same-dow-last-week (fills the gap that lag_7d leaves when DoW shifts)
    daily["covers_same_dow_last_week"] = daily["cover_count"].shift(7)

    # Weather interactions
    if "temp_mean_c" in daily.columns:
        daily["cold_weekend"] = ((daily["temp_mean_c"] < 8) & (daily["is_weekend"] == 1)).astype(int)
        daily["hot_day"] = (daily["temp_max_c"] > 22).astype(int)

    if "precipitation_total_mm" in daily.columns:
        daily["is_wet_day"] = (daily["precipitation_total_mm"] > 2).astype(int)
        # Rain streak: consecutive preceding wet days (saturated at 7).
        is_wet = daily["is_wet_day"].astype(int).values
        streak = np.zeros(len(is_wet), dtype=int)
        for i in range(1, len(is_wet)):
            streak[i] = streak[i-1] + 1 if is_wet[i-1] == 1 else 0
        daily["rain_streak"] = np.minimum(streak, 7)

    # Temperature shock: day-over-day change in mean temp.
    if "temp_mean_c" in daily.columns:
        daily["temp_change_c"] = daily["temp_mean_c"].diff().fillna(0)
        daily["temp_shock"] = (daily["temp_change_c"].abs() > 5).astype(int)

    # Severe weather flag from raw weather code (hourly -> max per day).
    # Weather codes >= 71 indicate snow, >= 95 thunderstorms, 85-87 snow showers.
    if "weather_code" in df.columns:
        wc_max = df.groupby("date")["weather_code"].max().reset_index()
        wc_max.columns = ["date", "weather_code_max"]
        wc_max["date"] = pd.to_datetime(wc_max["date"])
        daily = daily.merge(wc_max, on="date", how="left")
        daily["severe_weather"] = (
            daily["weather_code_max"].fillna(0) >= 71
        ).astype(int)

    # Daylight features (astronomical, deterministic).
    daily = add_daylight_features(daily)

    # Tube strikes (known planned disruptions).
    daily = add_tube_strike_feature(daily)

    return daily


def filter_covid(daily: pd.DataFrame) -> pd.DataFrame:
    """Remove the COVID lockdown window from the daily frame."""
    d = pd.to_datetime(daily["date"]).dt.date
    keep = ~((d >= COVID_START) & (d <= COVID_END))
    dropped = (~keep).sum()
    if dropped:
        log.info("covid.excluded", n_days=int(dropped))
    return daily[keep].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily training set from hourly parquet")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude-covid", action="store_true",
                        help=f"Drop days in [{COVID_START}..{COVID_END}]")
    args = parser.parse_args()

    log.info("daily.build.start", input=str(args.input))
    hourly = pd.read_parquet(args.input)
    log.info("daily.hourly_loaded", rows=len(hourly))

    daily = aggregate_to_daily(hourly)
    if args.exclude_covid:
        daily = filter_covid(daily)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(args.output, index=False)

    log.info(
        "daily.build.done",
        output=str(args.output),
        n_days=len(daily),
        n_features=len(daily.columns),
        mean_daily_trips=float(daily["cover_count"].mean()),
        date_range=f"{daily['date'].min().date()} to {daily['date'].max().date()}",
    )


if __name__ == "__main__":
    main()
