"""Naive baseline predictors.

Any real demand model must beat these. If the LightGBM model doesn't
beat them by a meaningful margin, the learned features aren't doing work.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def naive_yesterday(actuals: dict[date, float], target: date) -> float:
    """Predict the target day = yesterday's actual."""
    return float(actuals.get(target - timedelta(days=1), 0.0))


def naive_same_dow_last_week(actuals: dict[date, float], target: date) -> float:
    """Predict target = same day-of-week, last week."""
    return float(actuals.get(target - timedelta(days=7), 0.0))


def dow_month_lookup_fit(daily_train: pd.DataFrame) -> pd.DataFrame:
    """Fit a (day_of_week × month) average lookup table from training data.

    Returns a frame with columns [dow, month, avg_covers] giving the mean
    daily cover_count for each (dow, month) cell.
    """
    df = daily_train.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    table = df.groupby(["dow", "month"])["cover_count"].mean().reset_index()
    table = table.rename(columns={"cover_count": "avg_covers"})
    return table


def dow_month_lookup_predict(table: pd.DataFrame, target: date) -> float:
    """Predict target from the fitted (dow, month) lookup."""
    ts = pd.Timestamp(target)
    match = table[(table["dow"] == ts.dayofweek) & (table["month"] == ts.month)]
    if match.empty:
        return float(table["avg_covers"].mean())
    return float(match["avg_covers"].iloc[0])
