"""Compare autoregressive performance: short-lag model vs long-horizon model.

Runs both models autoregressively from Dec 31 2024 through Jan 30 2025
and reports how each one performs at various forecast horizons.
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

from scripts.autoregressive_forecast import recompute_lags, predict_one, CATEGORICAL


def run_autoregressive(model, features: list[str], daily: pd.DataFrame, origin: pd.Timestamp, end: pd.Timestamp) -> dict:
    available = set(daily["date"].dt.date)
    fdates = [d for d in pd.date_range(origin + pd.Timedelta(days=1), end, freq="D") if d.date() in available]

    history = daily[daily["date"] <= origin][["date", "cover_count"]]
    running: dict[date, float] = {d.date(): float(c) for d, c in zip(history["date"], history["cover_count"])}

    preds = {}
    for fd in fdates:
        d = fd.date()
        row = daily[daily["date"] == fd].iloc[0]
        row = recompute_lags(row, running, d)
        p = predict_one(model, features, row)
        preds[d] = p
        running[d] = p
    return preds


def main() -> None:
    from src.data.build_daily_training_set import aggregate_to_daily

    hourly = pd.read_parquet("data/holdout/training_features.parquet")
    daily = aggregate_to_daily(hourly)
    daily = daily.dropna(subset=["covers_7d_lag"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])

    origin = pd.Timestamp("2024-12-31")
    end = pd.Timestamp("2025-01-31")

    results = {}
    for name, model_pkl, features_json in [
        ("short-lag (original)", "data/models/lgbm_daily.pkl", "data/models/daily_feature_names.json"),
        ("long-horizon",         "data/models/lgbm_daily_longhorizon.pkl", "data/models/daily_feature_names_longhorizon.json"),
    ]:
        with open(model_pkl, "rb") as fh:
            model = pickle.load(fh)
        features = json.loads(Path(features_json).read_text())
        preds = run_autoregressive(model, features, daily, origin, end)
        results[name] = preds

    # Build comparison
    available = sorted(set(results["short-lag (original)"].keys()) & set(results["long-horizon"].keys()))
    rows = []
    for d in available:
        actual = float(daily[daily["date"] == pd.Timestamp(d)]["cover_count"].iloc[0])
        horizon = (d - origin.date()).days
        rows.append({
            "date": d,
            "horizon": horizon,
            "actual": actual,
            "short_lag_pred": results["short-lag (original)"][d],
            "long_horizon_pred": results["long-horizon"][d],
        })
    compare = pd.DataFrame(rows)

    def w(err, act): return float(np.abs(err).sum() / max(np.abs(act).sum(), 1e-8))
    def m(err):      return float(np.abs(err).mean())

    print("=" * 78)
    print("AUTOREGRESSIVE FORECAST COMPARISON (Jan 2025, no future actuals)")
    print("=" * 78)
    print(f"\n{'Model':<35s} {'MAE':>8s} {'WAPE':>8s}")
    print("-" * 55)
    for col, name in [("short_lag_pred", "Short-lag (original)"),
                       ("long_horizon_pred", "Long-horizon (no 1d/7d lags)")]:
        err = compare[col] - compare["actual"]
        print(f"{name:<35s} {m(err):>8.1f} {w(err, compare['actual'].values):>8.1%}")

    print("\n--- By horizon ---")
    print(f"{'Horizon':<18s} {'Short-WAPE':>12s} {'Long-WAPE':>12s}  Δ")
    for lo, hi, lbl in [(1,1,"Day 1"), (2,7,"Days 2-7"), (8,14,"Days 8-14"),
                         (15,28,"Days 15-28"), (29,31,"Days 29-31")]:
        seg = compare[(compare["horizon"] >= lo) & (compare["horizon"] <= hi)]
        if len(seg) == 0: continue
        w_s = w(seg["short_lag_pred"] - seg["actual"], seg["actual"].values)
        w_l = w(seg["long_horizon_pred"] - seg["actual"], seg["actual"].values)
        delta = w_l - w_s
        arrow = "↑ worse" if delta > 0.01 else ("↓ better" if delta < -0.01 else "≈")
        print(f"{lbl:<18s} {w_s:>11.1%} {w_l:>11.1%}   {arrow}")

    print("\n--- All 30 days ---")
    show = compare.copy()
    show["date"] = show["date"].astype(str) + " " + pd.to_datetime(show["date"]).dt.strftime("%a")
    show["short_pred"] = show["short_lag_pred"].round(0).astype(int)
    show["long_pred"] = show["long_horizon_pred"].round(0).astype(int)
    show["actual_i"] = show["actual"].astype(int)
    print(show[["date","horizon","actual_i","short_pred","long_pred"]].to_string(index=False))

    compare.to_parquet("data/holdout/autoregressive_compare.parquet", index=False)
    print("\nSaved: data/holdout/autoregressive_compare.parquet")


if __name__ == "__main__":
    main()
