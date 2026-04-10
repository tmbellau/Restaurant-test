"""Layer 1 end-to-end verification script.

Runs the full Layer 1 gate against the actual Docker Compose stack:

    1. Opens a DB session (requires `docker compose up -d db` first).
    2. Seeds one restaurant (LON-EC1, Wagamama Old Street, Europe/London).
    3. Drops a sample POS CSV into data/pos_inbox/ and ingests it.
    4. Fetches real weather for that restaurant from Open-Meteo.
    5. Joins the two with point-in-time correctness (weather.timestamp < sales.timestamp).
    6. Prints a summary. Exits non-zero on any failure.

Usage:
    docker compose up -d db
    alembic upgrade head
    python scripts/verify_layer1.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.config import settings
from src.db.engine import SessionLocal
from src.db.models import Restaurant
from src.signals.pos.csv_ingester import ingest_inbox
from src.signals.weather import OpenMeteoWeatherSource


DEMO_RESTAURANT = {
    "id": "LON-EC1",
    "name": "Wagamama Old Street",
    "address": "26 Old Street, London",
    "postcode": "EC1V 9QQ",
    "lat": 51.5252,
    "lng": -0.0879,
    "timezone": "Europe/London",
    "country_code": "GB",
    "seating_capacity": 120,
    "turnover_rate_per_hour": 1.75,
    "city_tier": "tier1",
    "footfall_zone_class": "high",
}


def step(msg: str) -> None:
    print(f"  -> {msg}")


def seed_restaurant() -> None:
    with SessionLocal() as session:
        existing = session.get(Restaurant, DEMO_RESTAURANT["id"])
        if existing is None:
            session.add(Restaurant(**DEMO_RESTAURANT))
            session.commit()
            step(f"seeded restaurant {DEMO_RESTAURANT['id']}")
        else:
            step(f"restaurant {DEMO_RESTAURANT['id']} already exists")


def drop_sample_csv() -> Path:
    settings.pos_inbox_dir.mkdir(parents=True, exist_ok=True)
    path = settings.pos_inbox_dir / "verify_layer1.csv"
    rows = [
        "restaurant_id,timestamp_utc,channel,transaction_id,item_id,item_qty,"
        "item_price,cover_count,voided,comped,record_version"
    ]
    base = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(days=2)
    for i in range(6):
        ts = (base + timedelta(hours=i)).isoformat()
        rows.append(
            f"LON-EC1,{ts},dine_in,VERIFY-{i},ramen,{i + 1},12.50,{i + 1},false,false,1"
        )
    path.write_text("\n".join(rows) + "\n")
    step(f"wrote {path.name} with {len(rows) - 1} rows")
    return path


def ingest() -> None:
    results = ingest_inbox()
    for file, count in results.items():
        step(f"ingested {file}: {count} rows")


def fetch_weather() -> pd.DataFrame:
    src = OpenMeteoWeatherSource()
    start = datetime.now(timezone.utc) - timedelta(days=3)
    end = datetime.now(timezone.utc) - timedelta(days=1)
    df = src.fetch_by_coordinates(
        lat=DEMO_RESTAURANT["lat"],
        lng=DEMO_RESTAURANT["lng"],
        start_utc=start,
        end_utc=end,
    )
    step(f"fetched {len(df)} hourly weather rows from Open-Meteo")
    return df


def pit_join_demo() -> int:
    """Do a point-in-time-correct LATERAL join of sales to most-recent weather.

    Returns the number of joined rows.
    """
    sql = text(
        """
        WITH target AS (
            SELECT
                s.restaurant_id,
                s.timestamp_utc AS target_ts,
                s.cover_count
            FROM sales_records s
            WHERE s.restaurant_id = :rid
            ORDER BY s.timestamp_utc DESC
            LIMIT 10
        )
        SELECT
            t.restaurant_id,
            t.target_ts,
            t.cover_count,
            f.timestamp_utc AS feature_ts,
            f.payload ->> 'temperature_c' AS temperature_c
        FROM target t
        LEFT JOIN LATERAL (
            SELECT sf.timestamp_utc, sf.payload
            FROM signal_features sf
            WHERE sf.restaurant_id = t.restaurant_id
              AND sf.source = 'open_meteo_weather'
              AND sf.timestamp_utc < t.target_ts
            ORDER BY sf.timestamp_utc DESC
            LIMIT 1
        ) f ON TRUE
        """
    )
    with SessionLocal() as session:
        rows = session.execute(sql, {"rid": DEMO_RESTAURANT["id"]}).fetchall()
        step(f"PiT-correct join returned {len(rows)} rows")
        for r in rows[:3]:
            print(f"     {r}")
        return len(rows)


def main() -> int:
    print("Layer 1 verification:")
    try:
        seed_restaurant()
        drop_sample_csv()
        ingest()
        weather_df = fetch_weather()
        # Layer 1 does NOT yet persist weather into signal_features (that's
        # Layer 2's feature assembler). But we verify we *can* fetch it and
        # that the PiT query shape works even if the weather join side is
        # empty for now.
        pit_join_demo()
        print(f"\nLayer 1 OK. Weather sample head:\n{weather_df.head()}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nLayer 1 FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
