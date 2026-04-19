"""London Underground network-wide strike days 2017-2025.

Sourced from the Wikipedia "London Underground strikes" chronology plus
contemporary TfL / RMT / ASLEF press-release coverage for events not
captured there (March 2022 RMT 10,000-member strikes, November 2024
ASLEF driver strikes).

Inclusion criteria: days where industrial action actually took place and
caused network-wide disruption (most or all lines affected). Strikes
announced but subsequently called off are NOT included.

Excluded by choice:
    - Overnight Night Tube strikes (2021-12, 2022-07 onwards): these are
      20:30 Fri/Sat through 04:29 the following morning, so they don't
      disrupt daytime commuting or daytime Santander Cycle usage.
    - 2011-06-19/20: only 6 hours, widely reported as minor.
    - 2016-12-24 strike: called off pre-event; some lines closed for
      other reasons that day.

Strikes are rare (usually 0-12/year) but very high impact when they
happen, since the alternative transport load falls onto bikes, buses,
walking and taxis. LU is an integrated network — when signallers,
drivers, or station staff strike, the whole system goes down; there is
no meaningful "single-line-only" strike.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

TUBE_STRIKE_DATES: set[date] = {
    # 2017 — station staffing dispute
    date(2017, 1, 9),    # 18:00 8 Jan to 18:00 9 Jan; full strike day = 9th

    # 2022 — four-day-week / pay / pensions disputes
    date(2022, 3, 1),    # RMT 10,000 members, full day
    date(2022, 3, 3),    # RMT 10,000 members, full day
    date(2022, 6, 6),    # RMT jobs and pensions
    date(2022, 6, 7),    # continuation
    date(2022, 6, 21),   # RMT, coincident with first national rail day
    date(2022, 6, 22),   # continuation
    date(2022, 8, 19),   # RMT + Unite, coincident with national rail
    date(2022, 8, 20),   # continuation
    date(2022, 11, 10),  # RMT + Unite; 9 of 11 stations closed
    date(2022, 11, 11),  # continuation
    date(2022, 11, 25),  # RMT station staff — partial (some stations)

    # 2023 — four-day total shutdown over pensions
    date(2023, 3, 15),   # RMT + ASLEF, all stations closed
    date(2023, 3, 16),
    date(2023, 3, 17),
    date(2023, 3, 18),

    # 2024 — RMT January pay strike (week-long), ASLEF November driver strikes
    date(2024, 1, 5),    # RMT pay dispute
    date(2024, 1, 6),
    date(2024, 1, 7),
    date(2024, 1, 8),
    date(2024, 1, 9),
    date(2024, 1, 10),
    date(2024, 1, 11),
    date(2024, 11, 7),   # ASLEF drivers — no trains running
    date(2024, 11, 12),  # ASLEF drivers — no trains running

    # 2025 — week-long RMT multi-grade action, most service suspended
    date(2025, 9, 7),    # depot operational control (from 5 Sep weekend)
    date(2025, 9, 8),
    date(2025, 9, 9),    # signallers + service control + ERU
    date(2025, 9, 10),   # all fleet + stations + trains (peak disruption)
    date(2025, 9, 11),   # signallers + service control
    date(2025, 9, 12),
}


def add_tube_strike_feature(daily: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Append is_tube_strike + days_since_tube_strike columns."""
    out = daily.copy()
    dates = pd.to_datetime(out[date_col]).dt.date
    out["is_tube_strike"] = [d in TUBE_STRIKE_DATES for d in dates]
    strike_dates_sorted = sorted(TUBE_STRIKE_DATES)
    days_since = []
    for d in dates:
        past = [s for s in strike_dates_sorted if s < d]
        days_since.append(min((d - past[-1]).days, 30) if past else 30)
    out["days_since_tube_strike"] = days_since
    return out
