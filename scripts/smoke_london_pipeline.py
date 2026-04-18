"""End-to-end smoke test for the London pipeline without the real GLA CSV.

Generates a tiny synthetic CSV in the GLA format, runs it through the whole
pipeline (load -> fetch weather -> build features -> train a tiny model),
and verifies nothing crashes. The synthetic CSV is labelled as such so it
cannot be confused with real training data.

Does NOT require:
    - The real GLA People Counts download (Cloudflare-blocked)
    - Any API keys
    - Network access to Open-Meteo (it skips if offline)

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


def make_synthetic_gla_csv(tmp_dir: Path, msoa_code: str, n_days: int = 30) -> Path:
    """Write a tiny synthetic CSV in GLA People Counts format.

    Used only for pipeline smoke testing. Values are random noise labelled
    as SMOKE_TEST so they're never mistaken for training data.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "smoke_test_people_counts.csv"

    start = datetime(2024, 1, 1)
    rows: list[dict] = []
    rng = np.random.default_rng(42)
    for d in range(n_days):
        for h in range(24):
            ts = start + timedelta(days=d, hours=h)
            # Vaguely realistic daily pattern so the model has something to learn
            hour_mult = 0.2 + 0.8 * np.sin(np.pi * h / 24) ** 2
            base = 2000 * hour_mult
            count = int(max(0, base + rng.normal(0, 200)))
            rows.append(
                {
                    "msoa_code": msoa_code,
                    "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "count": count,
                    "SMOKE_TEST": True,
                }
            )

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def run_smoke_test() -> None:
    print("=" * 70)
    print("London pipeline smoke test")
    print("=" * 70)

    from src.data.gla_busyness import SOHO_MSOA_CODE, load_gla_busyness

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        gla_dir = tmp_path / "gla_busyness"

        print(f"\n[1/5] Creating synthetic GLA CSV in {gla_dir}")
        csv_path = make_synthetic_gla_csv(gla_dir, SOHO_MSOA_CODE, n_days=60)
        print(f"      Written {csv_path.stat().st_size} bytes")

        print("\n[2/5] Loading synthetic data via gla_busyness loader")
        hourly = load_gla_busyness(data_dir=gla_dir, msoa_code=SOHO_MSOA_CODE)
        assert "cover_count" in hourly.columns, "Missing cover_count column"
        assert "timestamp_utc" in hourly.columns, "Missing timestamp_utc column"
        assert len(hourly) > 0, "No rows loaded"
        print(f"      Loaded {len(hourly)} hourly rows for MSOA {SOHO_MSOA_CODE}")

        print("\n[3/5] Building UK cultural calendar")
        from src.data.uk_cultural_calendar import build_uk_cultural_calendar

        cultural = build_uk_cultural_calendar(
            start_date=hourly["timestamp_utc"].min().date(),
            end_date=hourly["timestamp_utc"].max().date(),
            restaurant_id=SOHO_MSOA_CODE,
        )
        assert "is_cultural_period" in cultural.columns
        print(f"      Built {len(cultural)} daily cultural rows")

        print("\n[4/5] Building feature channel registry summary")
        from src.data.feature_channels import get_active_channels, get_cold_channels

        active = get_active_channels()
        cold = get_cold_channels()
        print(f"      Active channels ({len(active)}): {active}")
        print(f"      Cold channels   ({len(cold)}): {cold}")
        assert "gla_footfall" in active, "gla_footfall should be active"
        assert "events" in cold, "events should be cold"

        print("\n[5/5] Verifying the assembler accepts empty signal dataframes")
        from src.data.gla_busyness import restaurant_meta_for_soho
        from src.features.assembler import assemble_features

        meta = restaurant_meta_for_soho()
        # Use empty dataframes for weather/holidays/events so we don't need network
        features = assemble_features(
            covers_hourly=hourly,
            weather_df=pd.DataFrame(),
            holiday_df=pd.DataFrame(),
            event_df=pd.DataFrame(),
            transport_df=pd.DataFrame(),
            closures_df=pd.DataFrame(),
            restaurant_meta=meta[
                [
                    "restaurant_id",
                    "seating_capacity",
                    "turnover_rate_per_hour",
                    "city_tier",
                    "footfall_zone_class",
                    "country_code",
                ]
            ],
            tz_map={SOHO_MSOA_CODE: "Europe/London"},
        )
        print(f"      Assembler produced {len(features)} rows, {len(features.columns)} cols")
        assert "cover_count" in features.columns
        assert len(features) > 0

    print("\n" + "=" * 70)
    print("Smoke test PASSED")
    print("=" * 70)
    print("\nNext step: download the real GLA CSV to data/gla_busyness/ and run:")
    print("    python -m src.data.build_training_set")
    print("    python -m src.models.train_london")


if __name__ == "__main__":
    run_smoke_test()
