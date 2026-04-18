"""UK cultural event calendar — major London events affecting footfall.

Every event here is deterministic: either a fixed calendar date, a fixed
rule (e.g., Notting Hill Carnival = last bank-holiday weekend in August),
or a publicly-announced recurring event (Wimbledon Championships).

Live collection: every one of these is published annually before the event,
so we can always know the upcoming window at inference time.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _last_monday_of_month(year: int, month: int) -> date:
    """Last Monday of a given month (approximates last bank holiday)."""
    d = date(year, month, 28)
    while d.month == month:
        d += timedelta(days=1)
    d -= timedelta(days=1)
    while d.weekday() != 0:
        d -= timedelta(days=1)
    return d


# Historical London Marathon dates (2015-2024) — public record.
# Third Sunday of April for most years, except COVID-shifted 2020-2021.
_LONDON_MARATHON_DATES = {
    2015: date(2015, 4, 26),
    2016: date(2016, 4, 24),
    2017: date(2017, 4, 23),
    2018: date(2018, 4, 22),
    2019: date(2019, 4, 28),
    2020: date(2020, 10, 4),
    2021: date(2021, 10, 3),
    2022: date(2022, 10, 2),
    2023: date(2023, 4, 23),
    2024: date(2024, 4, 21),
    2025: date(2025, 4, 27),
}

# Wimbledon Championships — last week of June / first two weeks of July.
# Precise years known from AELTC press schedules.
_WIMBLEDON_DATES = {
    2015: (date(2015, 6, 29), date(2015, 7, 12)),
    2016: (date(2016, 6, 27), date(2016, 7, 10)),
    2017: (date(2017, 7, 3), date(2017, 7, 16)),
    2018: (date(2018, 7, 2), date(2018, 7, 15)),
    2019: (date(2019, 7, 1), date(2019, 7, 14)),
    2020: None,  # Cancelled (COVID)
    2021: (date(2021, 6, 28), date(2021, 7, 11)),
    2022: (date(2022, 6, 27), date(2022, 7, 10)),
    2023: (date(2023, 7, 3), date(2023, 7, 16)),
    2024: (date(2024, 7, 1), date(2024, 7, 14)),
    2025: (date(2025, 6, 30), date(2025, 7, 13)),
}


def _notting_hill_carnival(year: int) -> tuple[date, date]:
    """Last bank-holiday Sunday + Monday of August."""
    monday = _last_monday_of_month(year, 8)
    sunday = monday - timedelta(days=1)
    return sunday, monday


def _pride_london(year: int) -> date:
    """London Pride parade — late June / early July, first Saturday in July
    for most recent years. Approximate from public records."""
    # Pride is variable; most years it's the first Saturday of July
    d = date(year, 7, 1)
    while d.weekday() != 5:
        d += timedelta(days=1)
    return d


def build_uk_cultural_calendar(
    start_date: date, end_date: date, restaurant_id: str
) -> pd.DataFrame:
    """Build daily UK cultural event flags for the date range.

    Columns: timestamp_utc (midnight UTC), restaurant_id,
             is_cultural_period, cultural_period_name,
             days_to_next_cultural, days_from_last_cultural.
    """
    # Enumerate all cultural windows falling within [start_date - 1y, end_date + 1y]
    event_windows: list[tuple[date, date, str]] = []
    for year in range(start_date.year - 1, end_date.year + 2):
        # New Year period
        event_windows.append((date(year, 12, 31), date(year + 1, 1, 2), "new_year"))
        # Valentine's Day
        event_windows.append((date(year, 2, 14), date(year, 2, 14), "valentines"))
        # St Patrick's Day (relevant in central London bars/restaurants)
        event_windows.append((date(year, 3, 17), date(year, 3, 17), "st_patricks"))
        # London Marathon
        if year in _LONDON_MARATHON_DATES:
            m = _LONDON_MARATHON_DATES[year]
            event_windows.append((m, m, "london_marathon"))
        # May Day bank holiday (first Monday of May)
        d = date(year, 5, 1)
        while d.weekday() != 0:
            d += timedelta(days=1)
        event_windows.append((d, d, "may_day_bh"))
        # Pride London
        pride = _pride_london(year)
        event_windows.append((pride, pride, "pride_london"))
        # Wimbledon Championships
        if year in _WIMBLEDON_DATES and _WIMBLEDON_DATES[year] is not None:
            w = _WIMBLEDON_DATES[year]
            event_windows.append((w[0], w[1], "wimbledon_championships"))
        # Summer school holidays (approximate: last week of July to first week of Sep)
        event_windows.append(
            (date(year, 7, 22), date(year, 9, 3), "summer_school_holidays")
        )
        # Notting Hill Carnival
        nhc = _notting_hill_carnival(year)
        event_windows.append((nhc[0], nhc[1], "notting_hill_carnival"))
        # Bonfire Night
        event_windows.append((date(year, 11, 5), date(year, 11, 5), "bonfire_night"))
        # Christmas markets period (mid-Nov to Christmas Eve)
        event_windows.append(
            (date(year, 11, 20), date(year, 12, 24), "christmas_markets")
        )
        # Christmas
        event_windows.append((date(year, 12, 24), date(year, 12, 26), "christmas"))
        # New Year's Eve
        event_windows.append((date(year, 12, 31), date(year, 12, 31), "new_years_eve"))

    # Expand windows to per-date mapping
    cultural_dates: dict[date, str] = {}
    for start, end, name in event_windows:
        d = start
        while d <= end:
            if start_date <= d <= end_date:
                cultural_dates[d] = name
            d += timedelta(days=1)

    sorted_cultural = sorted(cultural_dates.keys())

    rows: list[dict] = []
    n_days = (end_date - start_date).days + 1
    for i in range(n_days):
        d = start_date + timedelta(days=i)
        is_cultural = d in cultural_dates
        period_name = cultural_dates.get(d, "")

        days_to = 999
        for cd in sorted_cultural:
            if cd >= d:
                days_to = (cd - d).days
                break

        days_from = 999
        for cd in reversed(sorted_cultural):
            if cd <= d:
                days_from = (d - cd).days
                break

        rows.append(
            {
                "restaurant_id": restaurant_id,
                "timestamp_utc": pd.Timestamp(d.year, d.month, d.day, tz="UTC"),
                "is_cultural_period": is_cultural,
                "cultural_period_name": period_name,
                "days_to_next_cultural": days_to,
                "days_from_last_cultural": days_from,
            }
        )

    return pd.DataFrame(rows)
