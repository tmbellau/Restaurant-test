"""Holdout prediction check — daily model on Jan 2025 unseen data.

Trained on 2022-01-01 to 2024-12-30. Tested on Jan 2025.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.build_daily_training_set import aggregate_to_daily
from src.data.build_training_set import build

HOLDOUT_DIR = Path("data/holdout")
MODEL_DIR = Path("data/models")

HOLDOUT_START = datetime(2024, 1, 1)    # enough to populate 365-day lag
HOLDOUT_END = datetime(2025, 1, 31)
PREDICT_START = pd.Timestamp("2025-01-01")
PREDICT_END = pd.Timestamp("2025-01-31")


def main() -> None:
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DAILY holdout: model trained on 2022-2024, testing on Jan 2025")
    print("=" * 70)

    print(f"\n[1/4] Building hourly features for {HOLDOUT_START.date()} to {HOLDOUT_END.date()}")
    build(start_date=HOLDOUT_START, end_date=HOLDOUT_END, output_dir=HOLDOUT_DIR)
    hourly = pd.read_parquet(HOLDOUT_DIR / "training_features.parquet")

    print("\n[2/4] Aggregating to daily with all lag features")
    daily = aggregate_to_daily(hourly)
    daily = daily.dropna(subset=["covers_7d_lag"]).reset_index(drop=True)
    print(f"      {len(daily)} days after lag-dropna")

    print("\n[3/4] Loading daily model")
    with open(MODEL_DIR / "lgbm_daily.pkl", "rb") as fh:
        model = pickle.load(fh)
    features = json.loads((MODEL_DIR / "daily_feature_names.json").read_text())
    metrics = json.loads((MODEL_DIR / "daily_training_metrics.json").read_text())
    lower_q = metrics["residual_lower_q"]
    upper_q = metrics["residual_upper_q"]
    print(f"      {len(features)} features, train-time CV WAPE {metrics['cv_wape_mean']:.1%}")

    # Filter to holdout month
    daily["date"] = pd.to_datetime(daily["date"])
    hold = daily[(daily["date"] >= PREDICT_START) & (daily["date"] <= PREDICT_END)].copy()
    print(f"      Holdout days: {len(hold)}")

    for col in features:
        if col not in hold.columns:
            hold[col] = 0
    X = hold[features].copy()
    cats = ["city_tier", "footfall_zone_class", "country_code"]
    for c in cats:
        if c in X.columns:
            X[c] = X[c].astype("category")

    y_pred = np.maximum(0, model.predict(X))
    y_true = hold["cover_count"].values
    lower = np.maximum(0, y_pred + lower_q)
    upper = np.maximum(0, y_pred + upper_q)

    mae = np.mean(np.abs(y_true - y_pred))
    wape = np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1e-8)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1)))
    coverage = ((y_true >= lower) & (y_true <= upper)).mean()

    print("\n[4/4] DAILY results on unseen Jan 2025")
    print(f"      Actual mean daily trips:    {np.mean(y_true):.0f}")
    print(f"      Predicted mean daily trips: {np.mean(y_pred):.0f}")
    print(f"      MAE:   {mae:.1f} trips/day")
    print(f"      WAPE:  {wape:.1%}")
    print(f"      MAPE:  {mape:.1%}")
    print(f"      80% interval coverage: {coverage:.1%}")

    compare = pd.DataFrame({
        "date": hold["date"].dt.strftime("%a %Y-%m-%d").values,
        "actual": y_true,
        "predicted": y_pred.round(0).astype(int),
        "error": (y_pred - y_true).round(0).astype(int),
        "lower_80": lower.round(0).astype(int),
        "upper_80": upper.round(0).astype(int),
        "dow": hold["date"].dt.day_name().values,
    })
    pd.concat([compare.drop(columns=["dow"])], axis=0).to_parquet(
        HOLDOUT_DIR / "daily_predictions_vs_actuals.parquet", index=False
    )

    print("\n--- Every day in Jan 2025 ---")
    print(compare.drop(columns=["dow"]).to_string(index=False))

    print("\n--- Worst 5 days ---")
    compare["abs_error"] = compare["error"].abs()
    print(compare.nlargest(5, "abs_error")[["date", "actual", "predicted", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
