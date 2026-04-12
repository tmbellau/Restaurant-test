"""Seed demo restaurant locations into the database."""

from __future__ import annotations

import structlog
from sqlalchemy.orm import Session

from src.db.engine import SessionLocal
from src.db.models import Restaurant
from src.demo.generate_synthetic import DEMO_LOCATIONS

log = structlog.get_logger(__name__)


def seed(session: Session | None = None) -> None:
    own = session is None
    if own:
        session = SessionLocal()
    try:
        for loc in DEMO_LOCATIONS:
            existing = session.get(Restaurant, loc["id"])
            if existing:
                continue
            session.add(Restaurant(
                id=loc["id"],
                name=loc["name"],
                address=f"{loc['name']}, London",
                postcode=loc["postcode"],
                lat=loc["lat"],
                lng=loc["lng"],
                timezone="Europe/London",
                country_code="GB",
                seating_capacity=loc["seating_capacity"],
                city_tier=loc["city_tier"],
                footfall_zone_class=loc["footfall_zone_class"],
            ))
        session.commit()
        log.info("seed.done", locations=len(DEMO_LOCATIONS))
    finally:
        if own:
            session.close()


if __name__ == "__main__":
    seed()
