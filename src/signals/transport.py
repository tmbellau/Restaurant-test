"""TfL transport disruption signal source (Layer 2).

Pulls live tube status from the TfL Unified API. A tube disruption near a
restaurant location reduces dine-in by up to 67% during strikes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd
import structlog

from src.config import settings
from src.signals.base import BaseSignalSource

log = structlog.get_logger(__name__)

# Good service statuses (anything else = disruption)
_GOOD_STATUSES = {"Good Service", "Service Closed"}


class TransportSource(BaseSignalSource):
    source_name = "tfl_transport"
    refresh_interval_seconds = 900  # 15 minutes
    feature_columns = ["tube_disruption", "tube_line_affected_count"]

    def _fetch_raw_impl(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        url = "https://api.tfl.gov.uk/Line/Mode/tube/Status"
        params: dict[str, str] = {}
        if settings.tfl_app_key:
            params["app_key"] = settings.tfl_app_key

        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        disrupted_lines = []
        for line in data:
            statuses = line.get("lineStatuses", [])
            for s in statuses:
                if s.get("statusSeverityDescription") not in _GOOD_STATUSES:
                    disrupted_lines.append(line["name"])
                    break

        now = datetime.now(timezone.utc)
        return pd.DataFrame([{
            "timestamp_utc": now,
            "restaurant_id": restaurant_id,
            "tube_disruption": len(disrupted_lines) > 0,
            "tube_line_affected_count": len(disrupted_lines),
            "disrupted_lines": ",".join(disrupted_lines),
        }])

    def transform_to_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return raw_df[["timestamp_utc", "restaurant_id", "tube_disruption", "tube_line_affected_count"]]

    def get_feature_schema(self) -> dict[str, str]:
        return {
            "tube_disruption": "bool",
            "tube_line_affected_count": "int",
        }
