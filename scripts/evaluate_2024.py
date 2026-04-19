"""Full-year 2024 evaluation — rolling 7-day-ahead forecast + baselines.

Training: everything up to 2023-12-31
Holdout: every day in 2024 (up to last available)

For each target date T in 2024:
  For each horizon h in {1..7}:
    origin O = T - h days
    Predict T using only data available through O:
      - Actual lag features for any lag >= h days
      - Origin-anchored most-recent-known value for covers_1d_lag
      - Rolling means computed over days [O-n..O]

Then compare predictions to baselines:
  - Naive: predict yesterday's actual
  - Seasonal naive: predict same day-of-week last week
  - DoW-Month lookup: mean cover count per (dow, month) from training data

Reports WAPE by horizon for model vs each baseline, plus a feature-group
ablation summary.
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

from src.models.baselines import (
    naive_yesterday,
    naive_same_dow_last_week,
    dow_month_lookup_fit,
    dow_month_lookup_predict,
)

CATEGORICAL = ["city_tier", "footfall_zone_class", "country_code"]


def recompute_features_for_horizon(
    base_row: pd.Series,
    target: date,
    origin: date,
    actuals: dict[date, float],
) -> pd.Series:
    """Recompute all lag / rolling features using only actuals <= origin."""
    row = base_row.copy()

    # Lag features
    if "covers_1d_lag" in row.index:
        row["covers_1d_lag"] = actuals.get(origin, np.nan)
    for feat, offset in [
        ("covers_7d_lag", 7), ("covers_14d_lag", 14),
        ("covers_28d_lag", 28), ("covers_365d_lag", 365),
        ("covers_same_dow_last_week", 7),
    ]:
        if feat in row.index:
            lookup_day = target - timedelta(days=offset)
            if lookup_day <= origin:
                row[feat] = actuals.get(lookup_day, np.nan)
            else:
                row[feat] = actuals.get(origin, np.nan)

    # Rolling stats based on the last N known days ending at origin
    if "covers_7d_mean" in row.index:
        vals = [actuals.get(origin - timedelta(days=k), np.nan) for k in range(0, 7)]
        vals = [v for v in vals if not np.isnan(v)]
        row["covers_7d_mean"] = float(np.mean(vals)) if vals else np.nan

    if "covers_7d_std" in row.index:
        vals = [actuals.get(origin - timedelta(days=k), np.nan) for k in range(0, 7)]
        vals = [v for v in vals if not np.isnan(v)]
        row["covers_7d_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    if "covers_28d_mean" in row.index:
        vals = [actuals.get(origin - timedelta(days=k), np.nan) for k in range(0, 28)]
        vals = [v for v in vals if not np.isnan(v)]
        row["covers_28d_mean"] = float(np.mean(vals)) if vals else np.nan

    return row


def predict_model(model, features: list[str], row: pd.Series) -> float:
    X = pd.DataFrame([row[features]])
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if X[col].dtype == object and col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return float(max(0, model.predict(X)[0]))


def evaluate(
    daily: pd.DataFrame,
    model,
    features: list[str],
    target_dates: pd.DatetimeIndex,
    actuals: dict[date, float],
    daily_idx: dict[date, int],
) -> pd.DataFrame:
    """Run rolling 7-day-ahead forecast for each target date."""
    records = []
    for t_ts in target_dates:
        t = t_ts.date()
        if t not in daily_idx:
            continue
        actual = actuals[t]
        base_row = daily.iloc[daily_idx[t]]
        for h in range(1, 8):
            origin = t - timedelta(days=h)
            row = recompute_features_for_horizon(base_row, t, origin, actuals)
            pred = predict_model(model, features, row)
            records.append({
                "target": t,
                "horizon": h,
                "origin": origin,
                "predicted": pred,
                "actual": actual,
            })
    return pd.DataFrame(records)


def wape(err: np.ndarray, act: np.ndarray) -> float:
    return float(np.abs(err).sum() / max(np.abs(act).sum(), 1e-8))


def mae(err: np.ndarray) -> float:
    return float(np.abs(err).mean())


def main() -> None:
    daily_all = pd.read_parquet("data/training/daily_training_features.parquet")
    daily_all["date"] = pd.to_datetime(daily_all["date"])
    daily_all = daily_all.dropna(subset=["covers_7d_lag", "covers_7d_mean"]).reset_index(drop=True)
    daily_idx = {d.date(): i for d, i in zip(daily_all["date"], daily_all.index)}
    actuals: dict[date, float] = {d.date(): float(c) for d, c in zip(daily_all["date"], daily_all["cover_count"])}

    # Holdout target = all 2025 days
    target_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    target_dates = pd.DatetimeIndex([d for d in target_dates if d.date() in daily_idx])
    print(f"Holdout dates: {target_dates[0].date()} to {target_dates[-1].date()} ({len(target_dates)} days)")

    # === Load models (trained on pre-2024 data) ===
    models = {}
    for name, suffix in [("full", ""), ("weather_only", "_weather"),
                         ("calendar_only", "_calendar"), ("lags_only", "_lags"),
                         ("no_lags", "_nolags")]:
        m_path = Path(f"data/models/lgbm_daily{suffix}.pkl")
        f_path = Path(f"data/models/daily_feature_names{suffix}.json")
        if not m_path.exists():
            print(f"  skipping {name}: {m_path} not found")
            continue
        with open(m_path, "rb") as fh:
            m = pickle.load(fh)
        feats = json.loads(f_path.read_text())
        models[name] = (m, feats)
        print(f"  loaded {name}: {len(feats)} features")

    # === Evaluate each model on the same holdout ===
    results = {}
    for name, (m, feats) in models.items():
        print(f"\nEvaluating {name}...")
        df = evaluate(daily_all, m, feats, target_dates, actuals, daily_idx)
        df["error"] = df["predicted"] - df["actual"]
        results[name] = df

    # === Baselines (train lookup on pre-2024 data only) ===
    pre2024 = daily_all[daily_all["date"] < "2025-01-01"]
    lookup_table = dow_month_lookup_fit(pre2024)

    baseline_records = []
    for t_ts in target_dates:
        t = t_ts.date()
        actual = actuals[t]
        for h in range(1, 8):
            origin = t - timedelta(days=h)
            # Naive "yesterday" at forecast time = origin's actual
            y_pred = float(actuals.get(origin, 0.0))
            # Seasonal naive = actual from (target - 7)
            dow_lw = actuals.get(t - timedelta(days=7), 0.0)
            # DoW×Month lookup doesn't depend on horizon (no lag info)
            lu_pred = dow_month_lookup_predict(lookup_table, t)
            baseline_records.append({
                "target": t, "horizon": h, "actual": actual,
                "naive_yesterday": y_pred,
                "naive_same_dow_lastweek": dow_lw,
                "dow_month_lookup": lu_pred,
            })
    baselines = pd.DataFrame(baseline_records)

    # === Summary table ===
    print("\n" + "=" * 80)
    print("ROLLING 7-DAY-AHEAD WAPE — 2025 HOLDOUT (trained on 2017-2024 ex-COVID)")
    print("=" * 80)

    model_names = list(results.keys())
    hdr = f"{'Horizon':>8s} | "
    hdr += " | ".join(f"{n:>14s}" for n in model_names)
    hdr += " || " + " | ".join(f"{n:>18s}" for n in ["naive_yday", "dow_lastwk", "dow_month_lookup"])
    print(hdr)
    print("-" * len(hdr))

    for h in range(1, 8):
        row = f"{h:>7d}d | "
        for name in model_names:
            seg = results[name][results[name]["horizon"] == h]
            w = wape(seg["error"].values, seg["actual"].values)
            row += f"{w:>13.1%}  | "
        b = baselines[baselines["horizon"] == h]
        for col in ["naive_yesterday", "naive_same_dow_lastweek", "dow_month_lookup"]:
            w = wape((b[col] - b["actual"]).values, b["actual"].values)
            row += f"{w:>17.1%}  | "
        print(row)

    # === Ablation summary ===
    print("\n" + "=" * 80)
    print("FEATURE ABLATION (1-day-ahead WAPE, to show what each group contributes)")
    print("=" * 80)
    for name in model_names:
        seg = results[name][results[name]["horizon"] == 1]
        w = wape(seg["error"].values, seg["actual"].values)
        m = mae(seg["error"].values)
        print(f"  {name:<20s} MAE {m:6.1f}   WAPE {w:.1%}")

    # === Save ===
    out_dir = Path("data/holdout")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_parquet(out_dir / f"eval_2024_{name}.parquet", index=False)
    baselines.to_parquet(out_dir / "eval_2024_baselines.parquet", index=False)
    print(f"\nSaved results under {out_dir}/")


if __name__ == "__main__":
    main()
