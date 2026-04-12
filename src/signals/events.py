"""Events signal source (Layer 2).

Pulls from Ticketmaster Discovery API and football-data.org for nearby events
and football matches. Falls back gracefully if API keys are missing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt

import httpx
import pandas as pd
import structlog

from src.config import settings
from src.signals.base import BaseSignalSource

log = structlog.get_logger(__name__)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


class EventSource(BaseSignalSource):
    source_name = "events"
    refresh_interval_seconds = 86400  # daily
    feature_columns = [
        "major_event_within_2km",
        "event_est_attendance",
        "local_football_match",
        "event_name",
    ]

    def _fetch_ticketmaster(
        self, lat: float, lng: float, start_utc: datetime, end_utc: datetime
    ) -> list[dict]:
        if not settings.ticketmaster_api_key:
            log.info("events.ticketmaster.no_key")
            return []
        url = "https://app.ticketmaster.com/discovery/v2/events.json"
        params = {
            "apikey": settings.ticketmaster_api_key,
            "latlong": f"{lat},{lng}",
            "radius": "5",
            "unit": "km",
            "startDateTime": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endDateTime": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "size": "50",
            "sort": "date,asc",
        }
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        events = data.get("_embedded", {}).get("events", [])
        results = []
        for ev in events:
            venue = (ev.get("_embedded", {}).get("venues") or [{}])[0]
            vlat = float(venue.get("location", {}).get("latitude", lat))
            vlng = float(venue.get("location", {}).get("longitude", lng))
            dist = _haversine_km(lat, lng, vlat, vlng)
            results.append({
                "event_name": ev.get("name", "Unknown"),
                "event_date": ev.get("dates", {}).get("start", {}).get("dateTime", ""),
                "distance_km": round(dist, 2),
                "est_attendance": 0,  # Ticketmaster doesn't expose this directly
                "category": ev.get("classifications", [{}])[0].get("segment", {}).get("name", ""),
            })
        return results

    def _fetch_football(self, start_utc: datetime, end_utc: datetime) -> list[dict]:
        """Fetch Premier League matches from football-data.org."""
        key = settings.football_data_api_key
        if not key:
            log.info("events.football.no_key")
            return []
        url = "https://api.football-data.org/v4/competitions/PL/matches"
        headers = {"X-Auth-Token": key}
        params = {
            "dateFrom": start_utc.date().isoformat(),
            "dateTo": end_utc.date().isoformat(),
        }
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
        matches = data.get("matches", [])
        results = []
        for m in matches:
            results.append({
                "event_name": f"{m['homeTeam']['shortName']} vs {m['awayTeam']['shortName']}",
                "event_date": m.get("utcDate", ""),
                "home_team": m["homeTeam"]["shortName"],
                "away_team": m["awayTeam"]["shortName"],
            })
        return results

    def _fetch_raw_impl(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        from src.db.engine import SessionLocal
        from src.db.models import Restaurant

        with SessionLocal() as session:
            loc = session.get(Restaurant, restaurant_id)
            if loc is None:
                raise ValueError(f"Unknown restaurant: {restaurant_id}")
            lat, lng = loc.lat, loc.lng

        tm_events = self._fetch_ticketmaster(lat, lng, start_utc, end_utc)
        football = self._fetch_football(start_utc, end_utc)

        # Build daily summary: for each day, flag if there's a major event
        from datetime import timedelta
        start_d = start_utc.date()
        end_d = end_utc.date()
        days = (end_d - start_d).days + 1
        rows = []
        for i in range(days):
            d = start_d + timedelta(days=i)
            ds = d.isoformat()
            nearby = [e for e in tm_events if e["event_date"][:10] == ds and e["distance_km"] <= 2]
            fb_today = [f for f in football if f["event_date"][:10] == ds]
            rows.append({
                "timestamp_utc": datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
                "restaurant_id": restaurant_id,
                "major_event_within_2km": len(nearby) > 0,
                "event_est_attendance": max((e["est_attendance"] for e in nearby), default=0),
                "local_football_match": len(fb_today) > 0,
                "event_name": nearby[0]["event_name"] if nearby else (
                    fb_today[0]["event_name"] if fb_today else ""
                ),
            })
        return pd.DataFrame(rows)

    def transform_to_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return raw_df

    def get_feature_schema(self) -> dict[str, str]:
        return {
            "major_event_within_2km": "bool",
            "event_est_attendance": "int",
            "local_football_match": "bool",
            "event_name": "str",
        }
