"""Base class for POS API adapters.

Each real POS system (Zonal, Oracle Micros, Toast, etc.) implements a
concrete subclass that pulls on schedule and normalises into the shared
POS row shape consumed by the CSV ingester's upsert logic.
"""

from __future__ import annotations

import abc
from datetime import datetime

import pandas as pd


class POSAdapter(abc.ABC):
    """Contract for pulling POS data from a vendor API."""

    vendor_name: str = "base"

    @abc.abstractmethod
    def fetch(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        """Fetch rows for a given window, normalised to POS row schema.

        Returns a DataFrame matching src.signals.pos.schemas.POSRowSchema.
        """
