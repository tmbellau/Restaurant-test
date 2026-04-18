"""BestTime.app API client — venue-specific hourly busyness patterns.

BestTime Forecast returns the typical weekly hourly pattern for a venue,
derived from Google Popular Times. This is REAL observed data for the
specific venue (not a generic average).

What it gives us: a 7×24 matrix of hourly busyness proportions.
What it does NOT give us: per-day historical time series.

We use it as the hourly SHAPE that gets scaled by TfL daily volume.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import structlog

log = structlog.get_logger(__name__)

FORECAST_URL = "https://besttime.app/api/v1/forecasts"
LIVE_URL = "https://besttime.app/api/v1/forecasts/live"

# Wagamama Soho venue details for the API
WAGAMAMA_SOHO = {
    "venue_name": "Wagamama",
    "venue_address": "10A Lexington St, London W1F 0LD",
}


def fetch_forecast(
    api_key: str,
    venue_name: str | None = None,
    venue_address: str | None = None,
    cache_path: Path | None = None,
) -> dict[int, list[float]]:
    """Fetch typical weekly hourly pattern from BestTime.

    Returns dict mapping day_of_week (0=Mon..6=Sun) to list of 24 hourly
    PROPORTIONS (each day sums to 1.0).
    """
    if cache_path and cache_path.exists():
        log.info("besttime.cache_hit", path=str(cache_path))
        raw = json.loads(cache_path.read_text())
        return _parse_forecast(raw)

    name = venue_name or WAGAMAMA_SOHO["venue_name"]
    addr = venue_address or WAGAMAMA_SOHO["venue_address"]

    log.info("besttime.fetch.start", venue=name, address=addr)

    payload = {
        "api_key_private": api_key,
        "venue_name": name,
        "venue_address": addr,
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(FORECAST_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    if data.get("status") != "OK":
        raise RuntimeError(
            f"BestTime API error: {data.get('message', data.get('status'))}"
        )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2))
        log.info("besttime.cached", path=str(cache_path))

    return _parse_forecast(data)


def _parse_forecast(data: dict) -> dict[int, list[float]]:
    """Extract 7×24 hourly proportions from BestTime response.

    Each day's 24 values are normalized to sum to 1.0 so they can be used
    as proportions of daily total volume.
    """
    analysis = data.get("analysis", [])
    if not analysis:
        raise RuntimeError("BestTime response has no analysis data")

    pattern: dict[int, list[float]] = {}
    for day_data in analysis:
        day_int = day_data["day_info"]["day_int"]
        raw = day_data["day_raw"]
        if len(raw) != 24:
            log.warning("besttime.unexpected_hours", day=day_int, n=len(raw))
            raw = (raw + [0] * 24)[:24]

        arr = np.array(raw, dtype=float)
        arr = np.maximum(arr, 0)
        total = arr.sum()
        if total > 0:
            proportions = (arr / total).tolist()
        else:
            proportions = [1.0 / 24] * 24
        pattern[day_int] = proportions

    log.info(
        "besttime.parsed",
        n_days=len(pattern),
        peak_day=max(pattern, key=lambda d: max(pattern[d])),
    )
    return pattern


def fetch_live(
    api_key: str,
    venue_name: str | None = None,
    venue_address: str | None = None,
) -> dict:
    """Fetch current live busyness from BestTime."""
    name = venue_name or WAGAMAMA_SOHO["venue_name"]
    addr = venue_address or WAGAMAMA_SOHO["venue_address"]

    payload = {
        "api_key_private": api_key,
        "venue_name": name,
        "venue_address": addr,
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(LIVE_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    analysis = data.get("analysis", {})
    return {
        "live_busyness_pct": analysis.get("venue_live_busyness"),
        "forecast_busyness_pct": analysis.get("venue_forecasted_busyness"),
        "delta": analysis.get("venue_live_forecasted_delta"),
        "venue_open": data.get("venue_info", {}).get("venue_open"),
    }
