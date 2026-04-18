"""Feature channel registry.

Every channel listed here must pass BOTH tests:
    1. Real historical bulk data available (for training)
    2. Live / future data available (for real-time prediction)

A channel fails either test -> excluded entirely from the model.
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
    "demand_target": FeatureChannel(
        name="demand_target",
        status=ChannelStatus.ACTIVE,
        historical_source=(
            "TfL Santander Cycles per-trip records (2015+), aggregated to "
            "hourly trip counts at Soho-area docking stations "
            "(Moor Street, Wardour Street, Broadwick Street, Golden Square, "
            "nearby West End stations). https://cycling.data.tfl.gov.uk/"
        ),
        live_source=(
            "Same source — TfL publishes new weekly CSVs continuously. "
            "OGLv2 licence, no API key needed."
        ),
        notes=(
            "Proxy for restaurant demand: cycling activity in Soho captures "
            "the same weather/holiday/event sensitivities that drive walk-in "
            "restaurant demand. When real Wagamama POS arrives, swap in."
        ),
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
        notes="Marathon, Carnival, Wimbledon, Pride, Christmas markets, etc.",
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
        historical_source="Derived from the hybrid target signal",
        live_source="Derived from ongoing target observations",
        notes="Same-hour-last-week, rolling means.",
    ),
    # --- COLD channels ---
    "events": FeatureChannel(
        name="events",
        status=ChannelStatus.COLD,
        historical_source=(
            "No single API covers all event types comprehensively. "
            "Theatre, concerts, conferences, sports all fragmented."
        ),
        live_source="Same fragmentation",
        notes="Excluded until a comprehensive source is available.",
    ),
    "transport_disruptions": FeatureChannel(
        name="transport_disruptions",
        status=ChannelStatus.COLD,
        historical_source="None — TfL publishes current status only",
        live_source="TfL Unified API (live only, no history)",
        notes="No training data. Excluded.",
    ),
    "social_media": FeatureChannel(
        name="social_media",
        status=ChannelStatus.COLD,
        historical_source="None — no simple free historical API",
        live_source="Twitter API available but complex + costly",
        notes="Excluded.",
    ),
}


def get_active_channels() -> list[str]:
    return [k for k, v in FEATURE_CHANNELS.items() if v.status == ChannelStatus.ACTIVE]


def get_cold_channels() -> list[str]:
    return [k for k, v in FEATURE_CHANNELS.items() if v.status == ChannelStatus.COLD]


def is_channel_active(name: str) -> bool:
    ch = FEATURE_CHANNELS.get(name)
    return ch is not None and ch.status == ChannelStatus.ACTIVE
