"""Master pipeline: London Soho hourly training dataset.

Demand target = hourly trip counts (starts + ends) at Soho-area Santander
Cycles docking stations. This is REAL per-date hourly data (TfL Open Data,
OGLv2), directly downloadable and still live.

Feature channels (all real, all live-collectable):
    Weather:  Open-Meteo Archive for Soho (51.5131, -0.1318)
    Holidays: gov.uk bank-holidays.json + England school calendar
    Cultural: Deterministic London events (Marathon, Carnival, etc.)
    Temporal: Fourier harmonics from timestamps
    Lags:     Derived from the Santander target series

Usage:
    # Defaults to 12 months ending yesterday
    python -m src.data.build_training_set
    python -m src.data.build_training_set --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import structlog

from src.data.feature_channels import get_active_channels, get_cold_channels
from src.data.historical_weather import fetch_weather
from src.data.santander import (
    RESTAURANT_ID,
    WAGAMAMA_SOHO_LAT,
    WAGAMAMA_SOHO_LNG,
    load_santander_hourly,
    restaurant_meta,
)
from src.data.uk_cultural_calendar import build_uk_cultural_calendar
from src.features.assembler import assemble_features
from src.signals.holidays import HolidaySource

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path("data/training")


def build(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Build London Soho training dataset from real data sources."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "training.build.start",
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        active_channels=get_active_channels(),
        cold_channels=get_cold_channels(),
    )

    # 1. Santander hourly trip counts at Soho stations — REAL demand target
    hourly = load_santander_hourly(start_date, end_date)
    ts_min = hourly["timestamp_utc"].min()
    ts_max = hourly["timestamp_utc"].max()

    # 2. Restaurant metadata
    meta = restaurant_meta()

    # 3. Open-Meteo historical weather for Soho
    weather = fetch_weather(
        lat=WAGAMAMA_SOHO_LAT,
        lng=WAGAMAMA_SOHO_LNG,
        start_date=ts_min.strftime("%Y-%m-%d"),
        end_date=ts_max.strftime("%Y-%m-%d"),
        restaurant_id=RESTAURANT_ID,
        cache_dir=out_dir,
    )

    # 4. UK bank holidays + school holidays
    holidays = HolidaySource()._fetch_raw_impl(
        restaurant_id=RESTAURANT_ID,
        start_utc=ts_min.to_pydatetime(),
        end_utc=ts_max.to_pydatetime(),
    )

    # 5. UK cultural calendar
    cultural = build_uk_cultural_calendar(
        start_date=ts_min.date(),
        end_date=ts_max.date(),
        restaurant_id=RESTAURANT_ID,
    )

    # 6. Merge cultural into holidays
    holidays = _merge_cultural(holidays, cultural)

    # 7. Assemble features
    tz_map = {RESTAURANT_ID: "Europe/London"}
    assembler_meta = meta[[
        "restaurant_id", "seating_capacity", "turnover_rate_per_hour",
        "city_tier", "footfall_zone_class", "country_code",
    ]].copy()

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
    hourly.to_parquet(out_dir / "hourly_target.parquet", index=False)
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


def _merge_cultural(holidays: pd.DataFrame, cultural: pd.DataFrame) -> pd.DataFrame:
    if cultural.empty:
        return holidays
    cols = [
        "restaurant_id", "timestamp_utc",
        "is_cultural_period", "cultural_period_name",
        "days_to_next_cultural", "days_from_last_cultural",
    ]
    available = [c for c in cols if c in cultural.columns]
    merged = holidays.merge(cultural[available], on=["restaurant_id", "timestamp_utc"], how="left")
    if "is_cultural_period" in merged.columns:
        merged["is_cultural_period"] = merged["is_cultural_period"].fillna(False)
    return merged


def _log_provenance(
    hourly: pd.DataFrame,
    weather: pd.DataFrame,
    holidays: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    log.info(
        "training.provenance",
        location="Soho cycling stations (51.5131, -0.1318)",
        demand_target=(
            f"REAL — Santander Cycles hourly trip counts at Soho stations "
            f"({len(hourly)} hourly rows, "
            f"{int(hourly['cover_count'].sum())} total trips)"
        ),
        weather=f"REAL — Open-Meteo Archive ({len(weather)} hourly rows)",
        holidays="REAL — gov.uk bank holidays",
        cultural="REAL — deterministic London cultural events",
        temporal="REAL — Fourier harmonics from timestamps",
        lags="REAL — computed from Santander hourly target history",
        n_training_rows=len(features),
        n_features=len(features.columns),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build London Soho training dataset")
    default_end = datetime.utcnow().replace(day=1) - timedelta(days=1)
    default_start = default_end - timedelta(days=365)
    parser.add_argument(
        "--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=default_start,
        help="Start date YYYY-MM-DD (default: 12 months before end)",
    )
    parser.add_argument(
        "--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        default=default_end,
        help="End date YYYY-MM-DD (default: last day of previous month)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for parquet files (default: data/training/)",
    )
    args = parser.parse_args()
    build(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
