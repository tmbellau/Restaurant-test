"""Known London Underground strike / major disruption days.

Sourced from Wikipedia, RMT/ASLEF press releases, TfL announcements, and
contemporary news coverage. Strikes that were announced but subsequently
called off or suspended are EXCLUDED — only days of actual industrial
action with network-wide disruption are included.

Strikes are rare (~10-15/year in peak years) but have large effects on
both cycling (spikes as alternative transport) and footfall (drops).

Note: this captures PLANNED network-wide strikes only. Unplanned
day-to-day line suspensions are not included — there is no free
historical feed for that granularity. For live production use, this
list should be refreshed from the RMT strike calendar.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

# London Underground strike dates with network-wide disruption.
# Multi-day strikes are expanded into the full set of affected days.
TUBE_STRIKE_DATES: set[date] = {
    # 2022 — wave of disputes over pensions and job cuts
    date(2022, 3, 1), date(2022, 3, 3),        # RMT — 2-day strike
    date(2022, 6, 6), date(2022, 6, 7),        # RMT — Platinum Jubilee weekend aftermath
    date(2022, 8, 19), date(2022, 8, 20),      # RMT — alongside national rail
    date(2022, 11, 10), date(2022, 11, 11),    # RMT — station staff
    date(2022, 11, 25),                        # Additional November action

    # 2023 — mostly resolved via agreement; July strikes were cancelled
    date(2023, 3, 15),                         # ASLEF + RMT — 24hr total shutdown

    # 2024 — ASLEF only; RMT strikes cancelled at last minute
    date(2024, 11, 7),                         # ASLEF train operators
    date(2024, 11, 12),                        # ASLEF train operators

    # 2025 — September: the major week-long action over four-day working week
    # Ref: TfL press release, Time Out, Londonist, NationalWorld coverage
    date(2025, 9, 5),                          # Fri — depot operational control
    date(2025, 9, 6),                          # Sat — depot operational control
    date(2025, 9, 7),                          # Sun — depot operational control
    date(2025, 9, 9),                          # Tue — signallers + service control + ERU
    date(2025, 9, 10),                         # Wed — ALL fleet + stations + trains (total shutdown)
    date(2025, 9, 11),                         # Thu — signallers + service control
}


def add_tube_strike_feature(daily: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Append is_tube_strike binary column + days_since_tube_strike."""
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
