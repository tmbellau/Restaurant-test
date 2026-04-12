"""Temporal feature engineering using Fourier harmonics.

Per the spec: Fourier harmonics encode cyclical patterns better than raw
day-of-week dummies, and the model learns the actual shape rather than
assuming equal-weight categories.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd


def fourier_features(
    timestamps: pd.Series, period: float, n_harmonics: int, prefix: str
) -> pd.DataFrame:
    """Generate sin/cos Fourier harmonics for a time series.

    Args:
        timestamps: Pandas datetime series (should be in local time).
        period: Full period length in the same units as the timestamp component.
        n_harmonics: Number of sin/cos pairs.
        prefix: Column name prefix (e.g., 'hour', 'dow', 'woy').
    """
    cols: dict[str, np.ndarray] = {}
    values = timestamps.values.astype("int64") // 10**9  # seconds
    # Normalise to [0, period)
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * math.pi * k * pd.to_numeric(timestamps.dt.hour if prefix == "hour"
                else timestamps.dt.dayofweek if prefix == "dow"
                else timestamps.dt.isocalendar().week.astype(float) if prefix == "woy"
                else timestamps.dt.hour) / period
        cols[f"{prefix}_sin_k{k}"] = np.sin(angle)
        cols[f"{prefix}_cos_k{k}"] = np.cos(angle)
    return pd.DataFrame(cols, index=timestamps.index)


def compute_temporal_features(df: pd.DataFrame, local_hour_col: str = "local_hour") -> pd.DataFrame:
    """Add all temporal features to a dataframe with a local_hour datetime column.

    Expects columns: local_ts (datetime in local time).
    """
    out = df.copy()
    ts = pd.to_datetime(out["local_ts"])

    # Fourier harmonics
    hour_f = fourier_features(ts, period=24, n_harmonics=6, prefix="hour")
    dow_f = fourier_features(ts, period=7, n_harmonics=3, prefix="dow")
    woy_f = fourier_features(ts, period=52, n_harmonics=6, prefix="woy")

    for col_df in [hour_f, dow_f, woy_f]:
        for c in col_df.columns:
            out[c] = col_df[c].values

    # Simple temporal flags
    out["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    out["local_hour_int"] = ts.dt.hour

    # Payday features (25th of month or last Friday)
    day_of_month = ts.dt.day
    dow = ts.dt.dayofweek  # Mon=0
    # Last Friday: within last 7 days of month and is Friday
    days_in_month = ts.dt.days_in_month
    is_last_friday = (dow == 4) & (day_of_month > (days_in_month - 7))
    is_payday = (day_of_month == 25) | is_last_friday
    out["is_payday_week"] = is_payday.astype(int)

    # Days since nearest payday (25th of current or previous month)
    out["days_since_payday"] = (day_of_month - 25).clip(lower=0)
    out.loc[day_of_month < 25, "days_since_payday"] = (
        day_of_month[day_of_month < 25] + (days_in_month[day_of_month < 25] - 25)
    )

    return out
