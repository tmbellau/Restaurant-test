"""Build multi-horizon training data from the daily features parquet.

For each historical day t, generates 7 training rows — one per forecast
horizon h ∈ [1..7]. Each row has the feature values as they would appear
at prediction time:

    - origin = t - h days
    - covers_1d_lag = actual[origin]  (becomes progressively more stale with h)
    - covers_Nd_lag = actual[t - N]   (target-anchored, unchanged with h)
    - covers_7d_mean/std = rolling stats ending at origin (shifts backward with h)
    - horizon_days = h (new feature the model can use)
    - target = actual[t]              (the true value, unchanged)

This teaches the quantile ensemble that stale-lag inputs come with
larger prediction errors, so longer-horizon forecasts get wider
intervals — without any hand-tuned horizon multiplier.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

DEFAULT_INPUT = Path("data/training/daily_training_features.parquet")
DEFAULT_OUTPUT = Path("data/training/multi_horizon_training.parquet")
MAX_HORIZON = 7


def build(
    input_path: Path | None = None,
    output_path: Path | None = None,
    max_horizon: int = MAX_HORIZON,
) -> pd.DataFrame:
    input_path = input_path or DEFAULT_INPUT
    output_path = output_path or DEFAULT_OUTPUT

    daily = pd.read_parquet(input_path)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Quick-lookup of actuals for any historical date
    actuals: dict[date, float] = {
        d.date(): float(c) for d, c in zip(daily["date"], daily["cover_count"])
    }

    log.info("multi_horizon.build.start",
             n_days=len(daily),
             date_range=f"{daily['date'].min().date()} to {daily['date'].max().date()}",
             max_horizon=max_horizon)

    rows: list[dict] = []
    for _, base_row in daily.iterrows():
        t: date = base_row["date"].date()
        for h in range(1, max_horizon + 1):
            origin: date = t - timedelta(days=h)
            if origin not in actuals:
                # Skip if we don't have enough history for a stable lag-1 substitute
                continue

            row = base_row.to_dict()
            row["horizon_days"] = h

            # Short-horizon lag feature: substituted with origin's actual
            # (at h=1 this is the true yesterday value, at h=7 it's 7 days stale)
            row["covers_1d_lag"] = actuals[origin]

            # Long-horizon target-anchored lags — unchanged with h so long as
            # target - offset <= origin (true for h <= offset)
            for feat, offset in [("covers_7d_lag", 7), ("covers_14d_lag", 14),
                                 ("covers_28d_lag", 28), ("covers_365d_lag", 365),
                                 ("covers_same_dow_last_week", 7)]:
                look = t - timedelta(days=offset)
                if look <= origin:
                    row[feat] = actuals.get(look, actuals[origin])
                else:
                    # Fallback when the target-anchored lookup is after origin
                    row[feat] = actuals[origin]

            # Rolling stats ending at origin (so they shift backward with h)
            vals7 = [actuals.get(origin - timedelta(days=k)) for k in range(7)]
            vals7 = [v for v in vals7 if v is not None]
            if vals7:
                row["covers_7d_mean"] = float(np.mean(vals7))
                row["covers_7d_std"] = float(np.std(vals7)) if len(vals7) > 1 else 0.0

            vals28 = [actuals.get(origin - timedelta(days=k)) for k in range(28)]
            vals28 = [v for v in vals28 if v is not None]
            if vals28:
                row["covers_28d_mean"] = float(np.mean(vals28))

            rows.append(row)

    df = pd.DataFrame(rows)
    # Drop rows where any essential lag is still missing (edge cases at the very
    # start of our history where we don't have 365 days of lookback).
    df = df.dropna(subset=["covers_7d_lag", "covers_7d_mean", "covers_28d_mean"])
    df = df.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    log.info("multi_horizon.build.done",
             n_rows=len(df),
             n_per_horizon=(df.groupby("horizon_days").size().to_dict()),
             output=str(output_path))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-horizon", type=int, default=MAX_HORIZON)
    args = parser.parse_args()
    build(args.input, args.output, args.max_horizon)


if __name__ == "__main__":
    main()
