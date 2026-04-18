"""Feature channel registry.

Every channel listed here must pass BOTH tests:
    1. Real historical bulk data available (for training)
    2. Live / future data available (for real-time prediction)

A channel fails either test -> it's excluded from the model entirely.
No pretending to learn from signals we can't actually feed at inference.

For this London-only pipeline, every active channel has:
    - Verified open-licence historical source
    - Verified live API/feed for ongoing prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChannelStatus(Enum):
    ACTIVE = "active"
    COLD = "cold"


@dataclass(frozen=True)
class FeatureChannel:
    name: str
    status: ChannelStatus
    historical_source: str
    live_source: str
    notes: str


FEATURE_CHANNELS: dict[str, FeatureChannel] = {
    "gla_footfall": FeatureChannel(
        name="gla_footfall",
        status=ChannelStatus.ACTIVE,
        historical_source=(
            "GLA People Counts (O2 Motion) hourly by MSOA — "
            "https://data.london.gov.uk/dataset/busyness-people-counts (OGL)"
        ),
        live_source="Same dataset, updated monthly by GLA",
        notes="The TARGET signal. MSOA E02000972 = Fitzrovia West & Soho.",
    ),
    "weather": FeatureChannel(
        name="weather",
        status=ChannelStatus.ACTIVE,
        historical_source="Open-Meteo Archive API (80+ years, hourly, free)",
        live_source="Open-Meteo Forecast API (free, no key)",
        notes="Hourly temp, rain, wind, apparent temp for Soho coordinates.",
    ),
    "holidays": FeatureChannel(
        name="holidays",
        status=ChannelStatus.ACTIVE,
        historical_source="gov.uk bank-holidays.json (decades, free, no key)",
        live_source="Same API, published years in advance",
        notes="UK bank holidays + England school holiday calendar.",
    ),
    "cultural_calendar": FeatureChannel(
        name="cultural_calendar",
        status=ChannelStatus.ACTIVE,
        historical_source="Deterministic dates (London Marathon, Carnival, etc.)",
        live_source="Calendar rules, published annually",
        notes="Marathon, Carnival, Wimbledon Championships, Pride, NYE, etc.",
    ),
    "temporal": FeatureChannel(
        name="temporal",
        status=ChannelStatus.ACTIVE,
        historical_source="Inherent in timestamps",
        live_source="Inherent in timestamps",
        notes="Fourier harmonics for hour, day-of-week, week-of-year.",
    ),
    "lag_demand": FeatureChannel(
        name="lag_demand",
        status=ChannelStatus.ACTIVE,
        historical_source="Derived from the GLA footfall target signal",
        live_source="Derived from the ongoing GLA series",
        notes="Same-hour-last-week, rolling means from real demand history.",
    ),
    # --- COLD channels: fail the (historical + live + comprehensive) test ---
    "events": FeatureChannel(
        name="events",
        status=ChannelStatus.COLD,
        historical_source=(
            "No single API covers all event types. "
            "football-data.org (PL only), Ticketmaster (partial), "
            "theatre/concerts/conferences/exhibitions all fragmented."
        ),
        live_source="Same fragmentation — no comprehensive live feed",
        notes=(
            "Cherry-picking one event type (e.g., Premier League) introduces "
            "bias without capturing events that actually drive Soho footfall "
            "(West End theatre matinees, conferences, concerts). Excluded "
            "entirely until a comprehensive source is available."
        ),
    ),
    "transport_disruptions": FeatureChannel(
        name="transport_disruptions",
        status=ChannelStatus.COLD,
        historical_source="None — TfL publishes current status only",
        live_source="TfL Unified API (live only, no history)",
        notes="No training data. Excluded until archive becomes available.",
    ),
    "social_media": FeatureChannel(
        name="social_media",
        status=ChannelStatus.COLD,
        historical_source="None — no simple free historical API",
        live_source="Twitter API available but complex + costly",
        notes="Excluded. Could be added later as a plugin channel.",
    ),
}


def get_active_channels() -> list[str]:
    return [k for k, v in FEATURE_CHANNELS.items() if v.status == ChannelStatus.ACTIVE]


def get_cold_channels() -> list[str]:
    return [k for k, v in FEATURE_CHANNELS.items() if v.status == ChannelStatus.COLD]


def is_channel_active(name: str) -> bool:
    ch = FEATURE_CHANNELS.get(name)
    return ch is not None and ch.status == ChannelStatus.ACTIVE
