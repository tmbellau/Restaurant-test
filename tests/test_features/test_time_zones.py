"""Unit tests for TimeZoneManager.

These tests cover DST transitions -- the single most common silent bug in
restaurant forecasting pipelines. A forecast that is "same hour last week"
must resolve to the same *local* wall-clock hour, even across DST boundaries.
"""

from datetime import datetime, timezone

import pytest

from src.util.time_zones import TimeZoneManager


@pytest.fixture()
def london() -> TimeZoneManager:
    return TimeZoneManager("Europe/London")


def test_utc_to_local_gmt(london: TimeZoneManager) -> None:
    # January -- GMT (UTC+0)
    dt_utc = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    local = london.to_local(dt_utc)
    assert local.hour == 12
    assert local.utcoffset().total_seconds() == 0


def test_utc_to_local_bst(london: TimeZoneManager) -> None:
    # July -- BST (UTC+1)
    dt_utc = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
    local = london.to_local(dt_utc)
    assert local.hour == 13
    assert local.utcoffset().total_seconds() == 3600


def test_local_hour_dispatches_in_local_time(london: TimeZoneManager) -> None:
    # 22:00 UTC on a BST day = 23:00 local
    dt_utc = datetime(2024, 6, 1, 22, 0, tzinfo=timezone.utc)
    assert london.local_hour(dt_utc) == 23


def test_same_local_hour_days_ago_across_dst_spring_forward(london: TimeZoneManager) -> None:
    """Spring-forward 2024: 2024-03-31 01:00 -> 02:00 (lost hour).

    Going from 2024-04-07 19:00 BST back 7 days should yield 2024-03-31 19:00 BST,
    which in UTC is 18:00 (both are BST, one hour ahead of UTC).
    """
    dt_utc = datetime(2024, 4, 7, 18, 0, tzinfo=timezone.utc)  # 19:00 BST
    result_utc = london.same_local_hour_days_ago(dt_utc, days=7)
    result_local = london.to_local(result_utc)
    assert result_local.hour == 19
    assert result_local.day == 31
    assert result_local.month == 3


def test_same_local_hour_days_ago_across_dst_fall_back(london: TimeZoneManager) -> None:
    """Fall-back 2024: 2024-10-27 02:00 -> 01:00 (extra hour).

    Going from 2024-11-03 19:00 GMT back 7 days should yield 2024-10-27 19:00 GMT.
    The wall-clock hour must stay 19:00 local, even though the UTC offset changed.
    """
    dt_utc = datetime(2024, 11, 3, 19, 0, tzinfo=timezone.utc)  # 19:00 GMT
    result_utc = london.same_local_hour_days_ago(dt_utc, days=7)
    result_local = london.to_local(result_utc)
    assert result_local.hour == 19
    assert result_local.day == 27
    assert result_local.month == 10


def test_local_to_utc_roundtrip(london: TimeZoneManager) -> None:
    dt_utc = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)
    local = london.to_local(dt_utc)
    back = london.to_utc(local)
    assert back == dt_utc


def test_naive_datetime_treated_as_utc(london: TimeZoneManager) -> None:
    dt_naive = datetime(2024, 1, 15, 12, 0)
    local = london.to_local(dt_naive)
    assert local.hour == 12  # GMT, no offset
