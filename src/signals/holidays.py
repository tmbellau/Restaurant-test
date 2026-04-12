"""UK holidays signal source (Layer 2).

Combines GOV.UK bank holidays with Nager.Date for international support
and a curated school holiday calendar.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone

import httpx
import pandas as pd
import structlog

from src.signals.base import BaseSignalSource

log = structlog.get_logger(__name__)

# UK school holidays 2024-2025 approximate windows (England).
# In production these would be loaded from a DB table or curated JSON.
_SCHOOL_HOLIDAYS_2024_25 = [
    (date(2024, 7, 22), date(2024, 9, 2)),   # Summer
    (date(2024, 10, 28), date(2024, 11, 1)),  # October half-term
    (date(2024, 12, 23), date(2025, 1, 3)),   # Christmas
    (date(2025, 2, 17), date(2025, 2, 21)),   # February half-term
    (date(2025, 4, 7), date(2025, 4, 18)),    # Easter
    (date(2025, 5, 26), date(2025, 5, 30)),   # May half-term
    (date(2025, 7, 21), date(2025, 9, 1)),    # Summer
]


class HolidaySource(BaseSignalSource):
    source_name = "holidays"
    refresh_interval_seconds = 7 * 86400  # weekly
    feature_columns = [
        "is_bank_holiday",
        "bank_holiday_name",
        "is_school_holiday",
        "days_to_next_holiday",
        "days_from_last_holiday",
    ]

    def __init__(self) -> None:
        super().__init__()
        self._bank_holidays: dict[date, str] | None = None

    def _fetch_bank_holidays(self) -> dict[date, str]:
        if self._bank_holidays is not None:
            return self._bank_holidays
        url = "https://www.gov.uk/bank-holidays.json"
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
        hols: dict[date, str] = {}
        for event in data.get("england-and-wales", {}).get("events", []):
            d = date.fromisoformat(event["date"])
            hols[d] = event["title"]
        self._bank_holidays = hols
        return hols

    def _is_school_holiday(self, d: date) -> bool:
        return any(start <= d <= end for start, end in _SCHOOL_HOLIDAYS_2024_25)

    def _fetch_raw_impl(
        self, restaurant_id: str, start_utc: datetime, end_utc: datetime
    ) -> pd.DataFrame:
        hols = self._fetch_bank_holidays()
        start_d = start_utc.date()
        end_d = end_utc.date()
        days = (end_d - start_d).days + 1
        rows = []
        # Build sorted list of holiday dates for proximity calc
        hol_dates = sorted(hols.keys())
        for i in range(days):
            d = start_d + timedelta(days=i)
            is_bh = d in hols
            bh_name = hols.get(d, "")
            is_sh = self._is_school_holiday(d)
            # days to next bank holiday
            days_to = None
            for hd in hol_dates:
                if hd >= d:
                    days_to = (hd - d).days
                    break
            # days from last bank holiday
            days_from = None
            for hd in reversed(hol_dates):
                if hd <= d:
                    days_from = (d - hd).days
                    break
            rows.append({
                "timestamp_utc": datetime(d.year, d.month, d.day, tzinfo=timezone.utc),
                "restaurant_id": restaurant_id,
                "is_bank_holiday": is_bh,
                "bank_holiday_name": bh_name,
                "is_school_holiday": is_sh,
                "days_to_next_holiday": days_to or 999,
                "days_from_last_holiday": days_from or 999,
            })
        return pd.DataFrame(rows)

    def transform_to_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return raw_df

    def get_feature_schema(self) -> dict[str, str]:
        return {
            "is_bank_holiday": "bool",
            "bank_holiday_name": "str",
            "is_school_holiday": "bool",
            "days_to_next_holiday": "int",
            "days_from_last_holiday": "int",
        }
