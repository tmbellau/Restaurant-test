"""Pandera schema for POS CSV ingestion.

This is one of the two Pandera boundaries enforced in Layer 1 (the other is
the feature assembler output in Layer 2). Every row must pass these checks
or the whole file is rejected and moved to a quarantine folder.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing.pandas import Series

VALID_CHANNELS = {
    "dine_in",
    "delivery_uber",
    "delivery_deliveroo",
    "delivery_justeat",
    "delivery_own",
}


class POSRowSchema(pa.DataFrameModel):
    """Schema every POS CSV row must satisfy before it enters the database."""

    restaurant_id: Series[str] = pa.Field(nullable=False)
    timestamp_utc: Series[pa.DateTime] = pa.Field(nullable=False)
    channel: Series[str] = pa.Field(isin=list(VALID_CHANNELS), nullable=False)
    transaction_id: Series[str] = pa.Field(nullable=False)
    item_id: Series[str] = pa.Field(nullable=True)
    item_qty: Series[int] = pa.Field(ge=0, nullable=False)
    item_price: Series[float] = pa.Field(ge=0, nullable=False)
    cover_count: Series[int] = pa.Field(ge=0, nullable=False)
    voided: Series[bool] = pa.Field(nullable=False)
    comped: Series[bool] = pa.Field(nullable=False)
    record_version: Series[int] = pa.Field(ge=1, nullable=False)

    class Config:
        strict = False
        coerce = True
