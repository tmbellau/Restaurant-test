"""Zonal POS adapter (stub).

Zonal is Wagamama's actual POS. Real implementation will call their API
(authentication via OAuth2, paginated reads, incremental cursors). This
stub exists so the ingestion code path is wired up end-to-end in Layer 1;
Layer 2/3 replaces the body with real HTTP calls.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.signals.pos.base_adapter import POSAdapter


class ZonalAdapter(POSAdapter):
    vendor_name = "zonal"

    def fetch(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        # TODO(layer2): implement real Zonal API integration.
        # For now return an empty DataFrame with the correct columns so the
        # ingestion code path is exercised end-to-end.
        return pd.DataFrame(
            columns=[
                "restaurant_id",
                "timestamp_utc",
                "channel",
                "transaction_id",
                "item_id",
                "item_qty",
                "item_price",
                "cover_count",
                "voided",
                "comped",
                "record_version",
            ]
        )
