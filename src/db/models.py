"""SQLAlchemy ORM models for Layer 1.

Layer 1 scope: restaurants, sales_records, signal_features, weather_data,
predictions (empty shell). Layer 2-4 additions (delivery_orders, menu_versions,
manager_overrides, recipes, ingredients, audit_log) are stubbed as TODO comments
so Alembic migrations can be extended incrementally.

All timestamps are stored in UTC. Column names end in `_utc` to make the
convention impossible to forget.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Restaurant(Base):
    __tablename__ = "restaurants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    address: Mapped[str] = mapped_column(String(512), nullable=False)
    postcode: Mapped[str] = mapped_column(String(16), nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lng: Mapped[float] = mapped_column(Float, nullable=False)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="Europe/London")
    country_code: Mapped[str] = mapped_column(String(2), nullable=False, default="GB")
    seating_capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    turnover_rate_per_hour: Mapped[float] = mapped_column(Float, nullable=False, default=1.75)
    city_tier: Mapped[str | None] = mapped_column(String(32), nullable=True)
    footfall_zone_class: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.utcnow()
    )

    sales: Mapped[list[SalesRecord]] = relationship(back_populates="restaurant")


class SalesRecord(Base):
    """One row per (location, hour, channel, item) from POS.

    This table is a TimescaleDB hypertable on ``timestamp_utc``. Deduplication
    is by ``transaction_id`` + ``item_id``; later corrections overwrite earlier
    versions via ``record_version``.
    """

    __tablename__ = "sales_records"
    __table_args__ = (
        UniqueConstraint("transaction_id", "item_id", "timestamp_utc", name="uq_sales_txn_item"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Note: composite PK (id, timestamp_utc) in migration for TimescaleDB hypertable
    restaurant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("restaurants.id"), nullable=False, index=True
    )
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    channel: Mapped[str] = mapped_column(String(32), nullable=False)  # dine_in / delivery_*
    transaction_id: Mapped[str] = mapped_column(String(64), nullable=False)
    item_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    item_qty: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    item_price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False, default=0)
    cover_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    voided: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    comped: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    record_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    ingested_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.utcnow()
    )

    restaurant: Mapped[Restaurant] = relationship(back_populates="sales")


class SignalFeature(Base):
    """Generic feature store row -- one row per (location, timestamp, source).

    The ``payload`` JSON column holds source-specific columns. The feature
    assembler extracts them into typed feature tables via continuous aggregates.
    """

    __tablename__ = "signal_features"
    __table_args__ = (
        UniqueConstraint(
            "restaurant_id", "timestamp_utc", "source", name="uq_signal_loc_ts_source"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    restaurant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("restaurants.id"), nullable=False, index=True
    )
    timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    fetched_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.utcnow()
    )


class Prediction(Base):
    """Stored predictions with calibrated intervals. Layer 2 populates this."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    restaurant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("restaurants.id"), nullable=False, index=True
    )
    target_timestamp_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    granularity: Mapped[str] = mapped_column(String(16), nullable=False)  # hourly / daily
    channel: Mapped[str] = mapped_column(String(16), nullable=False)  # dine_in / delivery
    point_forecast: Mapped[float] = mapped_column(Float, nullable=False)
    lower_80: Mapped[float] = mapped_column(Float, nullable=False)
    upper_80: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    capacity_constrained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_contributions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.utcnow()
    )


class LocationClosure(Base):
    """Structural closures (refurb, fire, etc). Excluded from training."""

    __tablename__ = "location_closures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    restaurant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("restaurants.id"), nullable=False, index=True
    )
    start_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
