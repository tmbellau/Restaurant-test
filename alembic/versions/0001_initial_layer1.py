"""Layer 1 initial schema: restaurants, sales_records, signal_features,
predictions, location_closures. TimescaleDB hypertables on sales_records and
signal_features.

Revision ID: 0001_initial_layer1
Revises:
Create Date: 2026-04-10 00:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_layer1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable TimescaleDB extension (no-op if already installed).
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    op.create_table(
        "restaurants",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("address", sa.String(512), nullable=False),
        sa.Column("postcode", sa.String(16), nullable=False),
        sa.Column("lat", sa.Float, nullable=False),
        sa.Column("lng", sa.Float, nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="Europe/London"),
        sa.Column("country_code", sa.String(2), nullable=False, server_default="GB"),
        sa.Column("seating_capacity", sa.Integer, nullable=False),
        sa.Column("turnover_rate_per_hour", sa.Float, nullable=False, server_default="1.75"),
        sa.Column("city_tier", sa.String(32), nullable=True),
        sa.Column("footfall_zone_class", sa.String(32), nullable=True),
        sa.Column(
            "created_at_utc",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "sales_records",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "restaurant_id", sa.String(64), sa.ForeignKey("restaurants.id"), nullable=False
        ),
        sa.Column("timestamp_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("channel", sa.String(32), nullable=False),
        sa.Column("transaction_id", sa.String(64), nullable=False),
        sa.Column("item_id", sa.String(64), nullable=True),
        sa.Column("item_qty", sa.Integer, nullable=False, server_default="1"),
        sa.Column("item_price", sa.Numeric(10, 2), nullable=False, server_default="0"),
        sa.Column("cover_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("voided", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("comped", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("record_version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "ingested_at_utc",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("transaction_id", "item_id", name="uq_sales_txn_item"),
    )
    op.create_index(
        "ix_sales_records_restaurant_ts",
        "sales_records",
        ["restaurant_id", "timestamp_utc"],
    )

    op.create_table(
        "signal_features",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "restaurant_id", sa.String(64), sa.ForeignKey("restaurants.id"), nullable=False
        ),
        sa.Column("timestamp_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column(
            "fetched_at_utc",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "restaurant_id", "timestamp_utc", "source", name="uq_signal_loc_ts_source"
        ),
    )
    op.create_index(
        "ix_signal_features_loc_ts_source",
        "signal_features",
        ["restaurant_id", "timestamp_utc", "source"],
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "restaurant_id", sa.String(64), sa.ForeignKey("restaurants.id"), nullable=False
        ),
        sa.Column("target_timestamp_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("granularity", sa.String(16), nullable=False),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("point_forecast", sa.Float, nullable=False),
        sa.Column("lower_80", sa.Float, nullable=False),
        sa.Column("upper_80", sa.Float, nullable=False),
        sa.Column("confidence_score", sa.Float, nullable=False),
        sa.Column("capacity_constrained", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("feature_contributions", sa.JSON, nullable=True),
        sa.Column(
            "created_at_utc",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_predictions_loc_target",
        "predictions",
        ["restaurant_id", "target_timestamp_utc"],
    )

    op.create_table(
        "location_closures",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "restaurant_id", sa.String(64), sa.ForeignKey("restaurants.id"), nullable=False
        ),
        sa.Column("start_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.String(255), nullable=False),
    )

    # TimescaleDB hypertables on time-series tables.
    op.execute(
        "SELECT create_hypertable('sales_records', 'timestamp_utc', "
        "if_not_exists => TRUE, migrate_data => TRUE)"
    )
    op.execute(
        "SELECT create_hypertable('signal_features', 'timestamp_utc', "
        "if_not_exists => TRUE, migrate_data => TRUE)"
    )


def downgrade() -> None:
    op.drop_table("location_closures")
    op.drop_index("ix_predictions_loc_target", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_signal_features_loc_ts_source", table_name="signal_features")
    op.drop_table("signal_features")
    op.drop_index("ix_sales_records_restaurant_ts", table_name="sales_records")
    op.drop_table("sales_records")
    op.drop_table("restaurants")
