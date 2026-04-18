"""Autoregressive multi-day forecast — the honest evaluation.

On Dec 31 2024, predict Jan 1. Then on Jan 1, predict Jan 2 using the
Jan 1 PREDICTION (not the actual) as the lag feature. And so on through
Jan 31. Errors compound — this is how the model actually behaves when
deployed for real forecasting (planning > 1 day ahead).

Compares:
    - True actuals
    - One-step-ahead predictions (the earlier holdout — uses actual lags)
    - Autoregressive predictions (lags replaced with earlier predictions)
    - Naive "yesterday" baseline
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CATEGORICAL = ["city_tier", "footfall_zone_class", "country_code"]
LAG_FEATURES = {
    "covers_1d_lag": 1,
    "covers_7d_lag": 7,
    "covers_14d_lag": 14,
    "covers_28d_lag": 28,
    "covers_365d_lag": 365,
    "covers_same_dow_last_week": 7,
}


def recompute_lags(feature_row: pd.Series, running: dict[date, float], d: date) -> pd.Series:
    row = feature_row.copy()
    for feat, k in LAG_FEATURES.items():
        if feat in row.index:
            row[feat] = running.get(d - timedelta(days=k), np.nan)
    last_7 = [running.get(d - timedelta(days=k), np.nan) for k in range(1, 8)]
    last_7 = [v for v in last_7 if not pd.isna(v)]
    if "covers_7d_mean" in row.index and last_7:
        row["covers_7d_mean"] = float(np.mean(last_7))
    if "covers_7d_std" in row.index:
        row["covers_7d_std"] = float(np.std(last_7)) if len(last_7) > 1 else 0.0
    last_28 = [running.get(d - timedelta(days=k), np.nan) for k in range(1, 29)]
    last_28 = [v for v in last_28 if not pd.isna(v)]
    if "covers_28d_mean" in row.index and last_28:
        row["covers_28d_mean"] = float(np.mean(last_28))
    return row


def predict_one(model, features: list[str], row: pd.Series) -> float:
    X = pd.DataFrame([row[features]])
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if X[col].dtype == object and col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return float(max(0, model.predict(X)[0]))


def main() -> None:
    from src.data.build_daily_training_set import aggregate_to_daily

    hourly = pd.read_parquet("data/holdout/training_features.parquet")
    daily = aggregate_to_daily(hourly)
    daily = daily.dropna(subset=["covers_7d_lag"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])

    with open("data/models/lgbm_daily.pkl", "rb") as fh:
        model = pickle.load(fh)
    features = json.loads(Path("data/models/daily_feature_names.json").read_text())

    origin = pd.Timestamp("2024-12-31")
    end = pd.Timestamp("2025-01-31")
    # Restrict to dates actually present in daily aggregation
    available = set(daily["date"].dt.date)
    forecast_dates = [
        d for d in pd.date_range(origin + pd.Timedelta(days=1), end, freq="D")
        if d.date() in available
    ]
    forecast_dates = pd.DatetimeIndex(forecast_dates)
    print(f"Forecast origin: {origin.date()} (last known actual)")
    print(f"Forecasting: {forecast_dates[0].date()} to {forecast_dates[-1].date()} autoregressively\n")

    # Build running series of actuals up to & including origin
    history = daily[daily["date"] <= origin][["date", "cover_count"]].copy()
    running: dict[date, float] = {
        d.date(): float(c) for d, c in zip(history["date"], history["cover_count"])
    }

    # Autoregressive forecast
    ar_preds = {}
    for fd in forecast_dates:
        d = fd.date()
        row = daily[daily["date"] == fd]
        if row.empty:
            continue
        row = row.iloc[0]
        row = recompute_lags(row, running, d)
        pred = predict_one(model, features, row)
        ar_preds[d] = pred
        running[d] = pred  # feed prediction forward

    # Load the 1-step-ahead predictions (previous holdout — used actual lags)
    onestep = pd.read_parquet("data/holdout/daily_predictions_vs_actuals.parquet")
    onestep["date_dt"] = pd.to_datetime(onestep["date"].str[4:], format="%Y-%m-%d").dt.date
    onestep = onestep.set_index("date_dt")

    # Build comparison frame
    rows = []
    for fd in forecast_dates:
        d = fd.date()
        actual = float(daily[daily["date"] == fd]["cover_count"].iloc[0])
        horizon = (d - origin.date()).days
        rows.append({
            "date": d,
            "horizon_days": horizon,
            "actual": actual,
            "one_step_pred": float(onestep.at[d, "predicted"]) if d in onestep.index else np.nan,
            "autoreg_pred": ar_preds.get(d, np.nan),
            "naive_yesterday": float(daily[daily["date"] == fd - pd.Timedelta(days=1)]["cover_count"].iloc[0]),
        })
    compare = pd.DataFrame(rows)

    compare["one_step_err"] = compare["one_step_pred"] - compare["actual"]
    compare["autoreg_err"]  = compare["autoreg_pred"]  - compare["actual"]
    compare["naive_err"]    = compare["naive_yesterday"] - compare["actual"]

    def wape(err, act): return float(np.abs(err).sum() / max(np.abs(act).sum(), 1e-8))
    def mae(err):       return float(np.abs(err).mean())

    print("=" * 78)
    print("HONEST FORECASTING COMPARISON (Jan 2025, 31 days, starting from Dec 31 2024)")
    print("=" * 78)

    act = compare["actual"].values
    print(f"\n{'Method':<40s} {'MAE':>8s} {'WAPE':>8s}")
    print("-" * 60)
    print(f"{'Naive: predict yesterday':<40s} {mae(compare['naive_err']):>8.1f} {wape(compare['naive_err'], act):>8.1%}")
    print(f"{'One-step (actual lags available)':<40s} {mae(compare['one_step_err']):>8.1f} {wape(compare['one_step_err'], act):>8.1%}")
    print(f"{'Autoregressive (own predictions)':<40s} {mae(compare['autoreg_err']):>8.1f} {wape(compare['autoreg_err'], act):>8.1%}")

    print("\n--- By forecast horizon (autoregressive) ---")
    for h_lo, h_hi, label in [(1, 1, "Day 1 (like 1-step)"), (2, 7, "Days 2-7"),
                               (8, 14, "Days 8-14"), (15, 28, "Days 15-28"),
                               (29, 31, "Days 29-31")]:
        seg = compare[(compare["horizon_days"] >= h_lo) & (compare["horizon_days"] <= h_hi)]
        if len(seg) == 0: continue
        w = wape(seg['autoreg_err'], seg['actual'].values)
        m = mae(seg['autoreg_err'])
        print(f"  {label:<25s} ({len(seg)} days)  MAE {m:6.1f}  WAPE {w:.1%}")

    print("\n--- All 31 days ---")
    pretty = compare.copy()
    pretty["date"] = pretty["date"].astype(str) + " " + pd.to_datetime(pretty["date"]).dt.strftime("%a")
    print(pretty[["date", "horizon_days", "actual", "one_step_pred", "autoreg_pred"]].round(0).to_string(index=False))

    compare.to_parquet("data/holdout/autoregressive_comparison.parquet", index=False)
    print(f"\nSaved: data/holdout/autoregressive_comparison.parquet")


if __name__ == "__main__":
    main()
