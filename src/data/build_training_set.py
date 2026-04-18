"""Master pipeline: London Soho hourly training dataset.

Single simulated restaurant at Wagamama Soho coordinates, anchored to
MSOA E02000972 (Fitzrovia West & Soho) for demand signal purposes.

Every signal in this pipeline:
    - Has real historical bulk data available
    - Has a live/future data feed for inference time
    - Is free and openly licensed

Data provenance (all REAL):
    Demand: GLA People Counts (O2 Motion) hourly at MSOA E02000972
    Weather: Open-Meteo Archive for Soho (51.5131, -0.1318)
    Holidays: gov.uk bank-holidays.json + curated school calendar
    Cultural: Deterministic London events (Marathon, Wimbledon, Carnival, etc.)
    Temporal: Inherent in timestamps (Fourier harmonics)
    Lags: Derived from the real GLA footfall series

Excluded (either no historical archive, or no comprehensive API):
    events (theatre, concerts, conferences, sports — no single source),
    transport disruptions, social media

Usage:
    # 1. Download GLA CSV to data/gla_busyness/ (see module docstring)
    # 2. Run:
    python -m src.data.build_training_set
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import structlog

from src.data.feature_channels import get_active_channels, get_cold_channels
from src.data.gla_busyness import (
    SOHO_LAT,
    SOHO_LNG,
    SOHO_MSOA_CODE,
    load_gla_busyness,
    restaurant_meta_for_soho,
)
from src.data.historical_weather import fetch_weather
from src.data.uk_cultural_calendar import build_uk_cultural_calendar
from src.features.assembler import assemble_features
from src.signals.holidays import HolidaySource

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path("data/training")


def build(
    gla_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Build complete London Soho training dataset from real data sources."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "training.build.start",
        active_channels=get_active_channels(),
        cold_channels=get_cold_channels(),
    )

    # 1. Load GLA footfall as the demand target
    hourly = load_gla_busyness(data_dir=gla_dir, msoa_code=SOHO_MSOA_CODE)

    ts_min = hourly["timestamp_utc"].min()
    ts_max = hourly["timestamp_utc"].max()
    log.info(
        "training.build.date_range",
        start=str(ts_min),
        end=str(ts_max),
        n_hours=len(hourly),
    )

    # 2. Restaurant metadata (single location)
    meta = restaurant_meta_for_soho()

    # 3. Fetch Open-Meteo historical weather for Soho
    weather = fetch_weather(
        lat=SOHO_LAT,
        lng=SOHO_LNG,
        start_date=ts_min.strftime("%Y-%m-%d"),
        end_date=ts_max.strftime("%Y-%m-%d"),
        restaurant_id=SOHO_MSOA_CODE,
        cache_dir=out_dir,
    )

    # 4. UK bank holidays + school holidays via existing HolidaySource
    holidays = _fetch_uk_holidays(ts_min, ts_max)

    # 5. UK cultural calendar
    cultural = build_uk_cultural_calendar(
        start_date=ts_min.date(),
        end_date=ts_max.date(),
        restaurant_id=SOHO_MSOA_CODE,
    )

    # 6. Merge cultural into holidays (assembler treats them as one daily signal)
    holidays = _merge_cultural_into_holidays(holidays, cultural)

    # 7. Assemble features (no event_df — events channel is cold)
    tz_map = {SOHO_MSOA_CODE: "Europe/London"}
    assembler_meta = meta[
        [
            "restaurant_id",
            "seating_capacity",
            "turnover_rate_per_hour",
            "city_tier",
            "footfall_zone_class",
            "country_code",
        ]
    ].copy()

    feature_df = assemble_features(
        covers_hourly=hourly,
        weather_df=weather,
        holiday_df=holidays,
        event_df=pd.DataFrame(),
        transport_df=pd.DataFrame(),
        closures_df=pd.DataFrame(),
        restaurant_meta=assembler_meta,
        tz_map=tz_map,
    )

    # 8. Save outputs
    hourly.to_parquet(out_dir / "hourly_footfall.parquet", index=False)
    meta.to_parquet(out_dir / "restaurant_meta.parquet", index=False)
    feature_df.to_parquet(out_dir / "training_features.parquet", index=False)

    _log_provenance(hourly, weather, holidays, feature_df)

    log.info(
        "training.build.done",
        output_dir=str(out_dir),
        n_training_rows=len(feature_df),
        n_features=len(feature_df.columns),
    )
    return feature_df


def _fetch_uk_holidays(ts_min: pd.Timestamp, ts_max: pd.Timestamp) -> pd.DataFrame:
    """Fetch UK bank holidays + school holidays via existing HolidaySource."""
    src = HolidaySource()
    df = src._fetch_raw_impl(
        restaurant_id=SOHO_MSOA_CODE,
        start_utc=ts_min.to_pydatetime(),
        end_utc=ts_max.to_pydatetime(),
    )
    log.info("training.holidays.fetched", rows=len(df))
    return df


def _merge_cultural_into_holidays(
    holidays: pd.DataFrame, cultural: pd.DataFrame
) -> pd.DataFrame:
    """Merge cultural calendar into holidays dataframe (per-day join)."""
    if cultural.empty:
        return holidays

    cultural_cols = [
        "restaurant_id",
        "timestamp_utc",
        "is_cultural_period",
        "cultural_period_name",
        "days_to_next_cultural",
        "days_from_last_cultural",
    ]
    available = [c for c in cultural_cols if c in cultural.columns]
    merged = holidays.merge(
        cultural[available],
        on=["restaurant_id", "timestamp_utc"],
        how="left",
    )
    if "is_cultural_period" in merged.columns:
        merged["is_cultural_period"] = merged["is_cultural_period"].fillna(False)
    return merged


def _log_provenance(
    hourly: pd.DataFrame,
    weather: pd.DataFrame,
    holidays: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """Document what's real in the training set."""
    bh_count = (
        int(holidays["is_bank_holiday"].sum())
        if "is_bank_holiday" in holidays.columns
        else 0
    )
    cultural_count = (
        int(holidays["is_cultural_period"].sum())
        if "is_cultural_period" in holidays.columns
        else 0
    )
    log.info(
        "training.provenance",
        location="Wagamama Soho (51.5131, -0.1318), MSOA E02000972",
        demand_target=f"REAL — GLA People Counts ({len(hourly)} hourly rows)",
        weather=f"REAL — Open-Meteo Archive ({len(weather)} hourly rows)",
        holidays=f"REAL — gov.uk bank holidays ({bh_count} flagged days)",
        cultural=f"REAL — deterministic ({cultural_count} cultural days)",
        temporal="REAL — Fourier harmonics from timestamps",
        lags="REAL — computed from real GLA footfall history",
        events="EXCLUDED — no single API covers all event types comprehensively",
        n_training_rows=len(features),
        n_features=len(features.columns),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build London Soho training dataset from real data sources"
    )
    parser.add_argument(
        "--gla-dir",
        type=Path,
        default=None,
        help="Path to GLA People Counts CSV files (default: data/gla_busyness/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for parquet files (default: data/training/)",
    )
    args = parser.parse_args()

    build(gla_dir=args.gla_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
