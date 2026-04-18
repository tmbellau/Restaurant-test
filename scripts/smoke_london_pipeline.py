"""End-to-end smoke test for the London hybrid pipeline.

Generates tiny synthetic TfL daily data + fake BestTime pattern, runs the
full pipeline (load -> weather -> features -> assembler), verifies nothing
crashes. No API keys or real data needed.

Usage:
    python scripts/smoke_london_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def make_synthetic_tfl_csv(tmp_dir: Path, n_days: int = 60) -> Path:
    """Write a synthetic TfL daily station CSV."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "smoke_test_tfl_daily.csv"

    start = datetime(2024, 1, 1)
    rng = np.random.default_rng(42)
    rows = []
    for d in range(n_days):
        dt = start + timedelta(days=d)
        dow = dt.weekday()
        base = 40000 if dow < 5 else 25000
        entries = int(base + rng.normal(0, 3000))
        exits = int(base * 0.95 + rng.normal(0, 2500))
        rows.append({
            "station": "Tottenham Court Road",
            "date": dt.strftime("%Y-%m-%d"),
            "entries": max(0, entries),
            "exits": max(0, exits),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def make_synthetic_besttime_pattern() -> dict[int, list[float]]:
    """Create a realistic-looking hourly pattern (no API needed)."""
    pattern = {}
    for dow in range(7):
        raw = np.zeros(24)
        # Lunch peak
        raw[11:15] = [30, 70, 80, 50]
        # Dinner peak
        raw[17:22] = [40, 70, 90, 85, 50]
        # Some baseline
        raw[8:11] = [10, 15, 20]
        raw[15:17] = [25, 30]
        raw[22:24] = [20, 10]
        if dow >= 5:
            raw = raw * 0.8
            raw[12:16] *= 1.3
        total = raw.sum()
        pattern[dow] = (raw / total).tolist() if total > 0 else [1 / 24] * 24
    return pattern


def run_smoke_test() -> None:
    print("=" * 70)
    print("London hybrid pipeline smoke test")
    print("=" * 70)

    from src.data.feature_channels import get_active_channels, get_cold_channels
    from src.data.hybrid_target import build_hybrid_hourly
    from src.data.tfl_daily import load_tfl_daily
    from src.data.uk_cultural_calendar import build_uk_cultural_calendar
    from src.features.assembler import assemble_features

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tfl_dir = tmp_path / "tfl"

        print(f"\n[1/6] Creating synthetic TfL daily CSV")
        make_synthetic_tfl_csv(tfl_dir, n_days=60)

        print("\n[2/6] Loading TfL daily data")
        tfl_daily = load_tfl_daily(data_dir=tfl_dir, station_name="Tottenham Court Road")
        assert len(tfl_daily) > 0
        print(f"      {len(tfl_daily)} days, avg daily footfall: {int(tfl_daily['total_footfall'].mean())}")

        print("\n[3/6] Building hybrid hourly target (TfL × BestTime shape)")
        besttime_pattern = make_synthetic_besttime_pattern()
        hourly = build_hybrid_hourly(tfl_daily, besttime_pattern, "wagamama_soho")
        assert "cover_count" in hourly.columns
        assert "timestamp_utc" in hourly.columns
        assert len(hourly) == len(tfl_daily) * 24
        print(f"      {len(hourly)} hourly rows, avg hourly: {int(hourly['cover_count'].mean())}")

        print("\n[4/6] Building UK cultural calendar")
        cultural = build_uk_cultural_calendar(
            hourly["timestamp_utc"].min().date(),
            hourly["timestamp_utc"].max().date(),
            "wagamama_soho",
        )
        print(f"      {int(cultural['is_cultural_period'].sum())} cultural days")

        print("\n[5/6] Channel registry check")
        active = get_active_channels()
        cold = get_cold_channels()
        print(f"      Active ({len(active)}): {active}")
        print(f"      Cold   ({len(cold)}): {cold}")

        print("\n[6/6] Assembler with empty signal dataframes")
        meta = pd.DataFrame([{
            "restaurant_id": "wagamama_soho",
            "seating_capacity": 100,
            "turnover_rate_per_hour": 1.75,
            "city_tier": "tier1",
            "footfall_zone_class": "very_high",
            "country_code": "GB",
        }])
        features = assemble_features(
            covers_hourly=hourly,
            weather_df=pd.DataFrame(),
            holiday_df=pd.DataFrame(),
            event_df=pd.DataFrame(),
            transport_df=pd.DataFrame(),
            closures_df=pd.DataFrame(),
            restaurant_meta=meta,
            tz_map={"wagamama_soho": "Europe/London"},
        )
        print(f"      {len(features)} rows, {len(features.columns)} cols")
        assert "cover_count" in features.columns
        assert len(features) > 0

    print("\n" + "=" * 70)
    print("Smoke test PASSED")
    print("=" * 70)
    print("\nTo run with real data:")
    print("  1. Download TfL daily station CSV to data/tfl/")
    print("  2. Get BestTime API key from besttime.app")
    print("  3. Run:")
    print("     BESTTIME_API_KEY=<key> python -m src.data.build_training_set")
    print("     python -m src.models.train_london")


if __name__ == "__main__":
    run_smoke_test()
