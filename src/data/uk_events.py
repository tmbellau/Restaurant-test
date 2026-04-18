"""Premier League match data via football-data.org.

Historical: free tier covers all PL seasons from 2015/16 onward.
Live: same API, fixtures published months in advance.

For each London match, we compute distance from Soho to the home stadium
and generate feature flags graded by proximity. London-wide matches also
affect central London footfall (fans traveling via the centre).
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

FOOTBALL_DATA_API = "https://api.football-data.org/v4"
PL_COMPETITION = "PL"

# London Premier League clubs and their stadium coordinates (public facts).
LONDON_PL_STADIUMS: dict[str, dict[str, float | int | str]] = {
    "Arsenal FC": {"lat": 51.5549, "lng": -0.1084, "capacity": 60704, "name": "Emirates Stadium"},
    "Chelsea FC": {"lat": 51.4816, "lng": -0.1910, "capacity": 40341, "name": "Stamford Bridge"},
    "Tottenham Hotspur FC": {"lat": 51.6043, "lng": -0.0664, "capacity": 62850, "name": "Tottenham Hotspur Stadium"},
    "West Ham United FC": {"lat": 51.5386, "lng": -0.0166, "capacity": 62500, "name": "London Stadium"},
    "Crystal Palace FC": {"lat": 51.3983, "lng": -0.0855, "capacity": 25486, "name": "Selhurst Park"},
    "Fulham FC": {"lat": 51.4749, "lng": -0.2217, "capacity": 29589, "name": "Craven Cottage"},
    "Brentford FC": {"lat": 51.4906, "lng": -0.2886, "capacity": 17250, "name": "Gtech Community Stadium"},
    "Queens Park Rangers FC": {"lat": 51.5094, "lng": -0.2320, "capacity": 18439, "name": "Loftus Road"},
    "AFC Wimbledon": {"lat": 51.4316, "lng": -0.1867, "capacity": 9300, "name": "Plough Lane"},
}


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_pl_matches(
    seasons: list[int],
    api_key: str,
    cache_path: Path | None = None,
    rate_limit_seconds: float = 6.5,
) -> pd.DataFrame:
    """Fetch all Premier League matches for the given seasons.

    Args:
        seasons: List of season start years, e.g., [2020, 2021, 2022]
        api_key: football-data.org API key (free tier: 10 req/min)
        cache_path: Optional parquet path to cache results
        rate_limit_seconds: Sleep between requests (6.5s = under 10/min)
    """
    if cache_path and cache_path.exists():
        log.info("pl.cache_hit", path=str(cache_path))
        return pd.read_parquet(cache_path)

    if not api_key:
        raise RuntimeError(
            "football-data.org API key missing. Register free at "
            "https://www.football-data.org/client/register and set "
            "FOOTBALL_DATA_API_KEY env var."
        )

    headers = {"X-Auth-Token": api_key}
    all_matches: list[dict] = []

    for i, season in enumerate(seasons):
        url = f"{FOOTBALL_DATA_API}/competitions/{PL_COMPETITION}/matches"
        params = {"season": season}
        log.info("pl.fetch", season=season)

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            log.error("pl.fetch.error", season=season, status=e.response.status_code)
            continue

        matches = data.get("matches", [])
        for m in matches:
            home = m.get("homeTeam", {}).get("name", "")
            away = m.get("awayTeam", {}).get("name", "")
            utc_date = m.get("utcDate")
            all_matches.append(
                {
                    "utc_datetime": utc_date,
                    "home_team": home,
                    "away_team": away,
                    "status": m.get("status", ""),
                    "season": season,
                }
            )

        log.info("pl.fetch.season_done", season=season, matches=len(matches))
        if i < len(seasons) - 1:
            time.sleep(rate_limit_seconds)

    df = pd.DataFrame(all_matches)
    if df.empty:
        raise RuntimeError("No PL matches fetched")

    df["utc_datetime"] = pd.to_datetime(df["utc_datetime"], utc=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    log.info("pl.fetch.all_done", total_matches=len(df))
    return df


def build_pl_event_features(
    match_data: pd.DataFrame,
    restaurant_lat: float,
    restaurant_lng: float,
    restaurant_id: str,
    close_radius_km: float = 5.0,
    london_radius_km: float = 20.0,
) -> pd.DataFrame:
    """Build daily PL match features for a single restaurant.

    Returns columns:
        timestamp_utc (midnight UTC of match date)
        restaurant_id
        pl_match_within_5km (bool)
        pl_match_in_london (bool)
        pl_match_home_team
        pl_match_est_attendance (sum if multiple matches same day)
    """
    rows: list[dict] = []

    # Precompute which London clubs count as close / in-London for this restaurant
    close_clubs: set[str] = set()
    london_clubs: set[str] = set()
    for team, info in LONDON_PL_STADIUMS.items():
        dist = _haversine_km(
            restaurant_lat, restaurant_lng,
            float(info["lat"]), float(info["lng"]),
        )
        if dist <= close_radius_km:
            close_clubs.add(team)
        if dist <= london_radius_km:
            london_clubs.add(team)

    log.info(
        "pl.proximity",
        restaurant=restaurant_id,
        close_clubs=list(close_clubs),
        london_clubs=list(london_clubs),
    )

    # Group matches by date
    if match_data.empty:
        return pd.DataFrame(rows)

    match_data = match_data.copy()
    match_data["date"] = match_data["utc_datetime"].dt.date

    for d, group in match_data.groupby("date"):
        home_teams = group["home_team"].tolist()
        close_match = any(t in close_clubs for t in home_teams)
        london_match = any(t in london_clubs for t in home_teams)

        est_attendance = 0
        home_team_str = ""
        for t in home_teams:
            if t in LONDON_PL_STADIUMS:
                cap = int(LONDON_PL_STADIUMS[t]["capacity"])
                est_attendance += int(cap * 0.90)
                home_team_str = t if not home_team_str else f"{home_team_str}; {t}"

        ts = pd.Timestamp(d.year, d.month, d.day, tz="UTC")
        rows.append(
            {
                "restaurant_id": restaurant_id,
                "timestamp_utc": ts,
                "pl_match_within_5km": close_match,
                "pl_match_in_london": london_match,
                "pl_match_home_team": home_team_str,
                "pl_match_est_attendance": est_attendance,
            }
        )

    df = pd.DataFrame(rows)
    log.info(
        "pl.features.done",
        match_days=len(df),
        close_match_days=int(df["pl_match_within_5km"].sum()),
        london_match_days=int(df["pl_match_in_london"].sum()),
    )
    return df


def seasons_for_date_range(start: datetime, end: datetime) -> list[int]:
    """Figure out which PL seasons cover a date range.

    PL seasons run August to May, labelled by starting year.
    """
    seasons: set[int] = set()
    d = start
    while d <= end:
        if d.month >= 8:
            seasons.add(d.year)
        else:
            seasons.add(d.year - 1)
        d += timedelta(days=30)
    return sorted(seasons)
