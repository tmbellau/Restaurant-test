"""Known London Underground strike / major disruption days (2019+).

Sourced from Wikipedia "List of strikes in the United Kingdom" and TfL
announcements. Strikes are rare (0-15/year) but have large effects on
both cycling (spikes) and footfall (drops). A single binary flag per
day is a pragmatic encoding.

Note: this captures PLANNED strikes and named disruptions only. Unplanned
day-to-day line suspensions are not included — there is no free historical
feed for that granularity.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# London Underground strike/major-disruption dates. Multi-day strikes are
# expanded into the full set of affected days.
# Sources: Wikipedia, RMT/ASLEF press releases, TfL announcements.
TUBE_STRIKE_DATES: set[date] = {
    # 2019
    date(2019, 1, 8),   # Bakerloo / Piccadilly strike
    # 2022
    date(2022, 3, 1), date(2022, 3, 3),
    date(2022, 6, 6),
    date(2022, 8, 19),
    date(2022, 11, 10),
    # 2023
    date(2023, 3, 15),
    date(2023, 7, 26), date(2023, 7, 28), date(2023, 7, 31),
    date(2023, 8, 4),
    # 2024
    date(2024, 5, 4), date(2024, 5, 7), date(2024, 5, 8), date(2024, 5, 10),
    date(2024, 11, 1), date(2024, 11, 7), date(2024, 11, 12),
    # 2025
    date(2025, 1, 8), date(2025, 1, 9), date(2025, 1, 12),
    date(2025, 4, 7), date(2025, 4, 10),
}


def add_tube_strike_feature(daily: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Append is_tube_strike binary column."""
    out = daily.copy()
    dates = pd.to_datetime(out[date_col]).dt.date
    out["is_tube_strike"] = [d in TUBE_STRIKE_DATES for d in dates]
    # Also: days since last strike (saturated at 30 for regression stability)
    strike_dates_sorted = sorted(TUBE_STRIKE_DATES)
    days_since = []
    for d in dates:
        past = [s for s in strike_dates_sorted if s < d]
        days_since.append(min((d - past[-1]).days, 30) if past else 30)
    out["days_since_tube_strike"] = days_since
    return out
