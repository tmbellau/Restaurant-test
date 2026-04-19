"""UK holidays signal source (Layer 2).

Combines GOV.UK bank holidays with Nager.Date for international support
and a curated school holiday calendar.
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone

import httpx
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# England school holidays (approximate windows).
_SCHOOL_HOLIDAYS = [
    # 2017
    (date(2017, 2, 13), date(2017, 2, 17)),
    (date(2017, 4, 3), date(2017, 4, 14)),
    (date(2017, 5, 29), date(2017, 6, 2)),
    (date(2017, 7, 24), date(2017, 9, 1)),
    (date(2017, 10, 23), date(2017, 10, 27)),
    (date(2017, 12, 18), date(2018, 1, 1)),
    # 2018
    (date(2018, 2, 12), date(2018, 2, 16)),
    (date(2018, 3, 29), date(2018, 4, 13)),
    (date(2018, 5, 28), date(2018, 6, 1)),
    (date(2018, 7, 23), date(2018, 9, 3)),
    (date(2018, 10, 22), date(2018, 10, 26)),
    (date(2018, 12, 21), date(2019, 1, 4)),
    # 2019
    (date(2019, 2, 18), date(2019, 2, 22)),
    (date(2019, 4, 8), date(2019, 4, 22)),
    (date(2019, 5, 27), date(2019, 5, 31)),
    (date(2019, 7, 22), date(2019, 9, 2)),
    (date(2019, 10, 28), date(2019, 11, 1)),
    (date(2019, 12, 23), date(2020, 1, 3)),
    # 2022
    (date(2022, 2, 14), date(2022, 2, 18)),
    (date(2022, 4, 4), date(2022, 4, 18)),
    (date(2022, 5, 30), date(2022, 6, 3)),
    (date(2022, 7, 25), date(2022, 9, 2)),
    (date(2022, 10, 24), date(2022, 10, 28)),
    (date(2022, 12, 19), date(2023, 1, 2)),
    # 2023
    (date(2023, 2, 13), date(2023, 2, 17)),
    (date(2023, 4, 3), date(2023, 4, 14)),
    (date(2023, 5, 29), date(2023, 6, 2)),
    (date(2023, 7, 24), date(2023, 9, 1)),
    (date(2023, 10, 23), date(2023, 10, 27)),
    (date(2023, 12, 18), date(2024, 1, 1)),
    # 2024
    (date(2024, 2, 12), date(2024, 2, 16)),
    (date(2024, 3, 28), date(2024, 4, 12)),
    (date(2024, 5, 27), date(2024, 5, 31)),
    (date(2024, 7, 22), date(2024, 9, 2)),
    (date(2024, 10, 28), date(2024, 11, 1)),
    (date(2024, 12, 23), date(2025, 1, 3)),
    # 2025
    (date(2025, 2, 17), date(2025, 2, 21)),
    (date(2025, 4, 7), date(2025, 4, 18)),
    (date(2025, 5, 26), date(2025, 5, 30)),
    (date(2025, 7, 21), date(2025, 9, 1)),
]


class HolidaySource:
    """UK bank holidays (gov.uk) + England school holidays."""

    def __init__(self) -> None:
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
        return any(start <= d <= end for start, end in _SCHOOL_HOLIDAYS)

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
