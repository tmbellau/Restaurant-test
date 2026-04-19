"""Evaluate horizon-aware quantile ensemble on the 2025 holdout.

Compares old (flat-buffer) vs new (per-horizon buffer) per-horizon
coverage, WAPE, MAE, and interval width.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def _predict(booster: lgb.Booster, X: pd.DataFrame, cats: list[str]) -> np.ndarray:
    X_num = X.copy()
    for c in cats:
        if c in X_num.columns:
            X_num[c] = X_num[c].astype("category").cat.codes.astype(float)
    return booster.predict(X_num.values)


def main() -> None:
    # Load multi-horizon test rows for 2025
    df = pd.read_parquet("data/training/multi_horizon_training.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-12-31")].reset_index(drop=True)
    df = df[df["cover_count"] >= 50].reset_index(drop=True)  # drop outages

    # Load models + buffers
    features = json.loads(Path("data/models/horizon_aware_feature_names.json").read_text())
    metrics = json.loads(Path("data/models/horizon_aware_metrics.json").read_text())
    per_horizon_buffer = {int(k): float(v) for k, v in metrics["per_horizon_buffer"].items()}

    boosters = {
        "lower": lgb.Booster(model_file="data/models/lgbm_horizon_q05.txt"),
        "median": lgb.Booster(model_file="data/models/lgbm_horizon_q50.txt"),
        "upper": lgb.Booster(model_file="data/models/lgbm_horizon_q95.txt"),
    }

    cats = ["city_tier", "footfall_zone_class", "country_code"]
    X = df[features]
    for name in boosters:
        df[name] = _predict(boosters[name], X, cats)

    df["buffer"] = df["horizon_days"].map(per_horizon_buffer)
    df["lower_cal"] = np.maximum(0, df["lower"] - df["buffer"])
    df["upper_cal"] = df["upper"] + df["buffer"]
    df["in_interval"] = (df["cover_count"] >= df["lower_cal"]) & (df["cover_count"] <= df["upper_cal"])
    df["abs_err"] = np.abs(df["median"] - df["cover_count"])

    print("=" * 78)
    print("2025 HOLDOUT — horizon-aware model (trained 2017-2023, calibrated on 2024)")
    print("=" * 78)
    print(f"\n{'h':>3s} {'n':>5s} {'Coverage':>9s} {'WAPE':>8s} {'MAE':>8s} {'Avg width':>10s}")
    print("-" * 48)
    for h in range(1, 8):
        seg = df[df["horizon_days"] == h]
        cov = seg["in_interval"].mean()
        wape = seg["abs_err"].sum() / max(seg["cover_count"].abs().sum(), 1e-8)
        mae = seg["abs_err"].mean()
        width = (seg["upper_cal"] - seg["lower_cal"]).mean()
        print(f"{int(h):>3d}d {len(seg):>5d} {cov:>8.1%} {wape:>8.1%} {mae:>8.0f} {width:>10.0f}")

    overall_cov = df["in_interval"].mean()
    overall_wape = df["abs_err"].sum() / max(df["cover_count"].abs().sum(), 1e-8)
    print(f"\nOverall:   coverage {overall_cov:.1%}, WAPE {overall_wape:.1%}")

    # Save for the Streamlit accuracy page
    out = df[[
        "date", "horizon_days", "cover_count",
        "lower_cal", "median", "upper_cal",
        "in_interval", "abs_err", "buffer",
    ]].rename(columns={
        "date": "target",
        "horizon_days": "horizon",
        "cover_count": "actual",
        "lower_cal": "lower",
        "median": "predicted",
        "upper_cal": "upper",
    })
    out["error"] = out["predicted"] - out["actual"]
    out["pct_error"] = out["error"] / out["actual"].clip(lower=1) * 100
    out["dow"] = out["target"].dt.strftime("%a")
    out["month"] = out["target"].dt.month
    out["is_outage"] = False
    out.to_parquet("data/training/eval_2025_horizon_aware.parquet", index=False)
    print(f"\nSaved: data/training/eval_2025_horizon_aware.parquet  ({len(out)} rows)")


if __name__ == "__main__":
    main()
