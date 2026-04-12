"""Capacity censoring and closure detection (Layer 2).

Per the spec: a walk-in restaurant at peak frequently hits capacity. POS
records served covers, not demand. Without correction the model learns a
ceiling at capacity and systematically under-predicts peaks.

Detection: flag was_censored=1 if cover_count >= 0.95 * capacity * turnover_rate.
Training: censored rows kept but sample_weight = 0.3.
Prediction: if forecast > capacity, set capacity_constrained = True.
"""

from __future__ import annotations

import pandas as pd


def flag_censored_hours(
    df: pd.DataFrame,
    capacity_col: str = "seating_capacity",
    turnover_col: str = "turnover_rate_per_hour",
    cover_col: str = "cover_count",
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Add was_censored flag to hourly cover data.

    Args:
        df: Must contain cover_col, capacity_col, turnover_col columns.
        threshold: Fraction of effective capacity above which we flag censoring.
    """
    out = df.copy()
    effective_cap = out[capacity_col] * out[turnover_col]
    out["was_censored"] = (out[cover_col] >= threshold * effective_cap).astype(int)
    return out


def flag_closures(
    df: pd.DataFrame,
    closures: pd.DataFrame,
) -> pd.DataFrame:
    """Add was_closed flag using the location_closures table.

    Args:
        df: Hourly data with restaurant_id, timestamp_utc.
        closures: DataFrame with restaurant_id, start_utc, end_utc.
    """
    out = df.copy()
    out["was_closed"] = 0
    for _, closure in closures.iterrows():
        mask = (
            (out["restaurant_id"] == closure["restaurant_id"])
            & (out["timestamp_utc"] >= closure["start_utc"])
            & (out["timestamp_utc"] <= closure["end_utc"])
        )
        out.loc[mask, "was_closed"] = 1
    return out


def compute_sample_weights(
    df: pd.DataFrame,
    censored_weight: float = 0.3,
) -> pd.Series:
    """Compute LightGBM sample_weight: 1.0 for normal rows, reduced for censored.

    Rows with was_closed=1 should already be excluded from training before this.
    """
    weights = pd.Series(1.0, index=df.index)
    if "was_censored" in df.columns:
        weights[df["was_censored"] == 1] = censored_weight
    return weights
