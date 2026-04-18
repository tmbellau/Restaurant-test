"""Master pipeline: London Soho hourly training dataset (hybrid target).

Demand target is built by combining two real data sources:
    TfL daily entries+exits at Tottenham Court Road station (real daily
    volume — captures weather, holiday, and event effects day-to-day)
    ×
    BestTime typical hourly pattern for Wagamama Soho (real venue-specific
    hourly shape from Google Popular Times observations)

Feature channels (all real, all live-collectable):
    Weather:  Open-Meteo Archive for Soho (51.5131, -0.1318)
    Holidays: gov.uk bank-holidays.json + England school calendar
    Cultural: Deterministic London events (Marathon, Carnival, etc.)
    Temporal: Fourier harmonics from timestamps
    Lags:     Derived from the hybrid hourly target

Usage:
    # 1. Get BestTime API key: besttime.app
    # 2. Download TfL daily station data to data/tfl/
    # 3. Run:
    BESTTIME_API_KEY=<key> python -m src.data.build_training_set
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import structlog

from src.data.besttime import fetch_forecast
from src.data.feature_channels import get_active_channels, get_cold_channels
from src.data.historical_weather import fetch_weather
from src.data.hybrid_target import build_hybrid_hourly
from src.data.tfl_daily import TARGET_STATION, load_tfl_daily
from src.data.uk_cultural_calendar import build_uk_cultural_calendar
from src.features.assembler import assemble_features
from src.signals.holidays import HolidaySource

log = structlog.get_logger(__name__)

OUTPUT_DIR = Path("data/training")

SOHO_LAT = 51.5131
SOHO_LNG = -0.1318
RESTAURANT_ID = "wagamama_soho"


def restaurant_meta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "restaurant_id": RESTAURANT_ID,
                "name": "Wagamama Soho",
                "lat": SOHO_LAT,
                "lng": SOHO_LNG,
                "timezone": "Europe/London",
                "country_code": "GB",
                "seating_capacity": 100,
                "turnover_rate_per_hour": 1.75,
                "city_tier": "tier1",
                "footfall_zone_class": "very_high",
            }
        ]
    )


def build(
    tfl_dir: Path | None = None,
    output_dir: Path | None = None,
    besttime_api_key: str | None = None,
) -> pd.DataFrame:
    """Build London Soho training dataset from real data sources."""
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = besttime_api_key or os.environ.get("BESTTIME_API_KEY", "")

    log.info(
        "training.build.start",
        active_channels=get_active_channels(),
        cold_channels=get_cold_channels(),
    )

    # 1. Load TfL daily station data (real daily volume)
    tfl_daily = load_tfl_daily(data_dir=tfl_dir, station_name=TARGET_STATION)

    # 2. Get BestTime hourly shape for Wagamama Soho (real venue pattern)
    besttime_cache = out_dir / "besttime_wagamama_soho.json"
    if not api_key and not besttime_cache.exists():
        raise RuntimeError(
            "BestTime API key required. Set BESTTIME_API_KEY env var or "
            "run with --besttime-api-key. Register at besttime.app"
        )
    besttime_pattern = fetch_forecast(
        api_key=api_key, cache_path=besttime_cache
    )

    # 3. Build hybrid hourly target (TfL daily × BestTime shape)
    hourly = build_hybrid_hourly(tfl_daily, besttime_pattern, RESTAURANT_ID)

    ts_min = hourly["timestamp_utc"].min()
    ts_max = hourly["timestamp_utc"].max()
    log.info(
        "training.build.hybrid_target",
        n_hours=len(hourly),
        date_range=f"{ts_min} to {ts_max}",
    )

    # 4. Restaurant metadata
    meta = restaurant_meta()

    # 5. Fetch Open-Meteo historical weather for Soho
    weather = fetch_weather(
        lat=SOHO_LAT,
        lng=SOHO_LNG,
        start_date=ts_min.strftime("%Y-%m-%d"),
        end_date=ts_max.strftime("%Y-%m-%d"),
        restaurant_id=RESTAURANT_ID,
        cache_dir=out_dir,
    )

    # 6. UK bank holidays + school holidays
    holidays = HolidaySource()._fetch_raw_impl(
        restaurant_id=RESTAURANT_ID,
        start_utc=ts_min.to_pydatetime(),
        end_utc=ts_max.to_pydatetime(),
    )

    # 7. UK cultural calendar
    cultural = build_uk_cultural_calendar(
        start_date=ts_min.date(),
        end_date=ts_max.date(),
        restaurant_id=RESTAURANT_ID,
    )

    # 8. Merge cultural into holidays
    holidays = _merge_cultural(holidays, cultural)

    # 9. Assemble features
    tz_map = {RESTAURANT_ID: "Europe/London"}
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

    # 10. Save outputs
    hourly.to_parquet(out_dir / "hourly_target.parquet", index=False)
    meta.to_parquet(out_dir / "restaurant_meta.parquet", index=False)
    feature_df.to_parquet(out_dir / "training_features.parquet", index=False)

    _log_provenance(hourly, tfl_daily, weather, holidays, feature_df)

    log.info(
        "training.build.done",
        output_dir=str(out_dir),
        n_training_rows=len(feature_df),
        n_features=len(feature_df.columns),
    )
    return feature_df


def _merge_cultural(
    holidays: pd.DataFrame, cultural: pd.DataFrame
) -> pd.DataFrame:
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
    tfl_daily: pd.DataFrame,
    weather: pd.DataFrame,
    holidays: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    log.info(
        "training.provenance",
        location="Wagamama Soho (51.5131, -0.1318)",
        demand_target=(
            f"HYBRID — TfL daily at {TARGET_STATION} ({len(tfl_daily)} days) "
            f"x BestTime hourly shape → {len(hourly)} hourly rows"
        ),
        weather=f"REAL — Open-Meteo Archive ({len(weather)} hourly rows)",
        holidays=f"REAL — gov.uk bank holidays",
        cultural="REAL — deterministic London cultural events",
        temporal="REAL — Fourier harmonics from timestamps",
        lags="REAL — computed from hybrid hourly target history",
        events="EXCLUDED — no comprehensive source",
        n_training_rows=len(features),
        n_features=len(features.columns),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build London Soho training dataset (TfL + BestTime hybrid)"
    )
    parser.add_argument(
        "--tfl-dir", type=Path, default=None,
        help="Path to TfL daily station CSV files (default: data/tfl/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for parquet files (default: data/training/)",
    )
    parser.add_argument(
        "--besttime-api-key", type=str, default=None,
        help="BestTime API key (or set BESTTIME_API_KEY env var)",
    )
    args = parser.parse_args()
    build(
        tfl_dir=args.tfl_dir,
        output_dir=args.output_dir,
        besttime_api_key=args.besttime_api_key,
    )


if __name__ == "__main__":
    main()
