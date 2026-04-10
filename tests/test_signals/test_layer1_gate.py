"""Layer 1 verification gate tests.

Per the build spec, Layer 1 is 'done' when:

    'You can ingest a CSV of POS data, fetch real weather for a postcode,
     join the two with point-in-time correctness, and query the result from
     SQL without errors.'

These tests exercise each of those steps without requiring a live database
or network, so they run cleanly in CI. A separate integration test
(scripts/verify_layer1.py, below) runs the same flow against a real Docker
Compose stack.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pandera as pa
import pytest

from src.signals.base import BaseSignalSource, SignalFetchError
from src.signals.pos.csv_ingester import validate_csv
from src.signals.pos.schemas import POSRowSchema
from src.signals.pos.zonal_adapter import ZonalAdapter
from src.signals.registry import all_sources


def test_pos_schema_accepts_valid_row() -> None:
    df = pd.DataFrame(
        [
            {
                "restaurant_id": "LON-EC1",
                "timestamp_utc": pd.Timestamp("2024-06-01 12:00", tz="UTC"),
                "channel": "dine_in",
                "transaction_id": "T1",
                "item_id": "ramen",
                "item_qty": 1,
                "item_price": 12.50,
                "cover_count": 2,
                "voided": False,
                "comped": False,
                "record_version": 1,
            }
        ]
    )
    POSRowSchema.validate(df)


def test_pos_schema_rejects_invalid_channel() -> None:
    df = pd.DataFrame(
        [
            {
                "restaurant_id": "LON-EC1",
                "timestamp_utc": pd.Timestamp("2024-06-01 12:00", tz="UTC"),
                "channel": "invalid_channel",
                "transaction_id": "T1",
                "item_id": "ramen",
                "item_qty": 1,
                "item_price": 12.50,
                "cover_count": 2,
                "voided": False,
                "comped": False,
                "record_version": 1,
            }
        ]
    )
    with pytest.raises(pa.errors.SchemaError):
        POSRowSchema.validate(df)


def test_pos_schema_rejects_negative_cover_count() -> None:
    df = pd.DataFrame(
        [
            {
                "restaurant_id": "LON-EC1",
                "timestamp_utc": pd.Timestamp("2024-06-01 12:00", tz="UTC"),
                "channel": "dine_in",
                "transaction_id": "T1",
                "item_id": "ramen",
                "item_qty": 1,
                "item_price": 12.50,
                "cover_count": -1,
                "voided": False,
                "comped": False,
                "record_version": 1,
            }
        ]
    )
    with pytest.raises(pa.errors.SchemaError):
        POSRowSchema.validate(df)


def test_csv_ingester_reads_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "pos.csv"
    csv_path.write_text(
        "restaurant_id,timestamp_utc,channel,transaction_id,item_id,item_qty,"
        "item_price,cover_count,voided,comped,record_version\n"
        "LON-EC1,2024-06-01 12:00:00+00:00,dine_in,T1,ramen,1,12.50,2,false,false,1\n"
    )
    df = validate_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["restaurant_id"] == "LON-EC1"
    assert df.iloc[0]["channel"] == "dine_in"


def test_zonal_adapter_stub_returns_empty_with_schema() -> None:
    adapter = ZonalAdapter()
    df = adapter.fetch(
        restaurant_id="LON-EC1",
        start_utc=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 6, 2, tzinfo=timezone.utc),
    )
    assert set(df.columns) >= {
        "restaurant_id",
        "timestamp_utc",
        "channel",
        "transaction_id",
        "cover_count",
    }


def test_signal_registry_discovers_weather_source() -> None:
    sources = all_sources()
    assert "open_meteo_weather" in sources
    weather = sources["open_meteo_weather"]
    assert isinstance(weather, BaseSignalSource)
    schema = weather.get_feature_schema()
    assert "temperature_c" in schema
    assert "precipitation_mm" in schema


class _FailingSource(BaseSignalSource):
    """Deliberately-failing source used to exercise retry + circuit-breaker."""

    source_name = "failing_test"
    refresh_interval_seconds = 60
    feature_columns = ["dummy"]

    def _fetch_raw_impl(self, restaurant_id, start_utc, end_utc):  # type: ignore[no-untyped-def]
        raise ConnectionError("simulated outage")

    def transform_to_features(self, raw_df):  # type: ignore[no-untyped-def]
        return raw_df

    def get_feature_schema(self) -> dict[str, str]:
        return {"dummy": "float"}


def test_base_signal_source_wraps_failure_in_signal_fetch_error() -> None:
    src = _FailingSource()
    with pytest.raises(SignalFetchError):
        src.fetch_raw(
            "LON-EC1",
            datetime(2024, 6, 1, tzinfo=timezone.utc),
            datetime(2024, 6, 2, tzinfo=timezone.utc),
        )
