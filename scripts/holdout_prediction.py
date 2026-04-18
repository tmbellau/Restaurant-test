"""Holdout prediction check — run the trained model on data it hasn't seen.

Trained model was fit on Jan-Dec 2024. This script:
    1. Builds features for Dec 2024 + Jan 2025 (Dec is needed to populate
       the weekly lag features for early Jan).
    2. Loads the trained model.
    3. Predicts for Jan 2025.
    4. Joins predictions with actual Santander counts.
    5. Reports MAE / WAPE / coverage and prints a few sample hours.
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

from src.data.build_training_set import build
from src.data.feature_channels import FEATURE_CHANNELS
from src.models.trainer import CATEGORICAL_FEATURES

HOLDOUT_DIR = Path("data/holdout")
MODEL_DIR = Path("data/models")
HOLDOUT_START = datetime(2024, 12, 25)  # 1wk pre-buffer to populate lags
HOLDOUT_END = datetime(2025, 1, 31)
PREDICT_START = datetime(2025, 1, 1)    # Only evaluate on truly-unseen dates
PREDICT_END = datetime(2025, 1, 31)


def main() -> None:
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Holdout prediction: model trained on 2024, testing on Jan 2025")
    print("=" * 70)

    print(f"\n[1/4] Building features for {HOLDOUT_START.date()} to {HOLDOUT_END.date()}")
    build(
        start_date=HOLDOUT_START,
        end_date=HOLDOUT_END,
        output_dir=HOLDOUT_DIR,
    )

    df = pd.read_parquet(HOLDOUT_DIR / "training_features.parquet")
    print(f"      Built features: {len(df)} hourly rows")

    print("\n[2/4] Loading trained model")
    with open(MODEL_DIR / "lgbm_quantile.pkl", "rb") as fh:
        model = pickle.load(fh)
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    metrics = json.loads((MODEL_DIR / "training_metrics.json").read_text())
    lower_q = metrics["residual_lower_q"]
    upper_q = metrics["residual_upper_q"]
    print(f"      Loaded model with {len(feature_names)} features")
    print(f"      Training-time CV WAPE: {metrics['cv_wape_mean']:.1%}")

    print(f"\n[3/4] Predicting on {PREDICT_START.date()} to {PREDICT_END.date()}")
    start_utc = pd.Timestamp(PREDICT_START, tz="UTC")
    end_utc = pd.Timestamp(PREDICT_END, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    holdout = df[
        (df["timestamp_utc"] >= start_utc) & (df["timestamp_utc"] <= end_utc)
    ].copy()
    print(f"      Holdout rows: {len(holdout)}")

    missing = [c for c in feature_names if c not in holdout.columns]
    if missing:
        print(f"      WARNING: missing columns: {missing[:5]}")
        for c in missing:
            holdout[c] = 0

    X = holdout[feature_names].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")

    y_pred = np.maximum(0, model.predict(X))
    y_true = holdout["cover_count"].values

    lower = np.maximum(0, y_pred + lower_q)
    upper = np.maximum(0, y_pred + upper_q)

    print("\n[4/4] Results on unseen Jan 2025 data")
    mae = np.mean(np.abs(y_true - y_pred))
    denom = max(np.sum(np.abs(y_true)), 1e-8)
    wape = np.sum(np.abs(y_true - y_pred)) / denom
    coverage = float(((y_true >= lower) & (y_true <= upper)).mean())
    avg_actual = np.mean(y_true)
    avg_pred = np.mean(y_pred)

    print(f"      Actual mean hourly trips:    {avg_actual:.2f}")
    print(f"      Predicted mean hourly trips: {avg_pred:.2f}")
    print(f"      MAE:                         {mae:.2f} trips/hour")
    print(f"      WAPE:                        {wape:.1%}")
    print(f"      80% interval coverage:       {coverage:.1%}")

    compare = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(holdout["timestamp_utc"].values, utc=True),
        "actual": y_true,
        "predicted": y_pred.round(1),
        "lower_80": lower.round(1),
        "upper_80": upper.round(1),
    })
    compare["abs_error"] = (compare["actual"] - compare["predicted"]).abs().round(2)
    compare.to_parquet(HOLDOUT_DIR / "predictions_vs_actuals.parquet", index=False)

    print("\n--- Sample (10 hours around Jan 3, 2025 Friday evening) ---")
    jan3 = compare[
        (compare["timestamp_utc"] >= pd.Timestamp("2025-01-03 14:00", tz="UTC"))
        & (compare["timestamp_utc"] <= pd.Timestamp("2025-01-03 23:00", tz="UTC"))
    ].copy()
    jan3["timestamp_local"] = (
        jan3["timestamp_utc"].dt.tz_convert("Europe/London").dt.strftime("%a %b %d %H:%M")
    )
    print(
        jan3[["timestamp_local", "actual", "predicted", "lower_80", "upper_80", "abs_error"]]
        .to_string(index=False)
    )

    print("\n--- Best 5 predictions (smallest error) ---")
    print(compare.nsmallest(5, "abs_error")[["timestamp_utc", "actual", "predicted", "abs_error"]].to_string(index=False))

    print("\n--- Worst 5 predictions (largest error) ---")
    print(compare.nlargest(5, "abs_error")[["timestamp_utc", "actual", "predicted", "abs_error"]].to_string(index=False))


if __name__ == "__main__":
    main()
