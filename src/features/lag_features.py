"""Lag and rolling features computed in local time.

Per the spec: lag features use local time, not UTC. "Same hour last week"
must resolve to the same wall-clock hour, even across DST boundaries.
These are computed by the feature assembler after UTC->local conversion.
"""

from __future__ import annotations

import pandas as pd


def compute_lag_features(
    df: pd.DataFrame,
    target_col: str = "cover_count",
    delivery_col: str | None = None,
) -> pd.DataFrame:
    """Add lag and rolling features to a dataframe sorted by (location, local_ts).

    Expects df sorted by restaurant_id, local_ts with no gaps (filled with NaN).
    """
    out = df.copy()

    # Same hour 1 week ago (168 hours) and 2 weeks ago (336 hours)
    out["covers_same_hour_1w"] = out.groupby("restaurant_id")[target_col].shift(168)
    out["covers_same_hour_2w"] = out.groupby("restaurant_id")[target_col].shift(336)

    # Rolling stats on covers (7d = 168h, 14d = 336h, 30d = 720h)
    grouped = out.groupby("restaurant_id")[target_col]
    out["covers_rolling_7d_mean"] = grouped.transform(
        lambda x: x.rolling(168, min_periods=24).mean()
    )
    out["covers_rolling_7d_std"] = grouped.transform(
        lambda x: x.rolling(168, min_periods=24).std()
    )
    out["covers_rolling_14d_mean"] = grouped.transform(
        lambda x: x.rolling(336, min_periods=48).mean()
    )
    out["covers_rolling_30d_mean"] = grouped.transform(
        lambda x: x.rolling(720, min_periods=168).mean()
    )

    # Delivery lags (if delivery column present)
    if delivery_col and delivery_col in out.columns:
        dgrouped = out.groupby("restaurant_id")[delivery_col]
        out["delivery_rolling_7d_mean"] = dgrouped.transform(
            lambda x: x.rolling(168, min_periods=24).mean()
        )
        out["delivery_rolling_14d_mean"] = dgrouped.transform(
            lambda x: x.rolling(336, min_periods=48).mean()
        )

    return out
