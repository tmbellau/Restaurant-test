"""Demo: pick a date, get a 7-day forecast with uncertainty bands vs actuals.

Uses the quantile ensemble (10th/50th/90th percentile) to produce an 80%
prediction interval alongside the point estimate. Since we're demoing on
dates with known actuals (2025), we can immediately check whether the
actuals fall inside the interval.

Also computes full-year 2025 empirical coverage to validate the intervals.

Usage:
    # Interactive demo for a specific date
    python scripts/demo_forecast.py --origin 2025-06-01

    # Full-year 2025 coverage report
    python scripts/demo_forecast.py --full-year

    # Both
    python scripts/demo_forecast.py --origin 2025-06-01 --full-year
"""

from __future__ import annotations

import argparse
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


def recompute_lags(base_row: pd.Series, target: date, origin: date,
                   actuals: dict[date, float]) -> pd.Series:
    row = base_row.copy()
    if "covers_1d_lag" in row.index:
        row["covers_1d_lag"] = actuals.get(origin, np.nan)
    for feat, offset in [("covers_7d_lag", 7), ("covers_14d_lag", 14),
                         ("covers_28d_lag", 28), ("covers_365d_lag", 365),
                         ("covers_same_dow_last_week", 7)]:
        if feat in row.index:
            look = target - timedelta(days=offset)
            row[feat] = actuals.get(look, actuals.get(origin, np.nan))
    if "covers_7d_mean" in row.index:
        vals = [actuals.get(origin - timedelta(days=k), np.nan) for k in range(7)]
        vals = [v for v in vals if not np.isnan(v)]
        row["covers_7d_mean"] = float(np.mean(vals)) if vals else np.nan
    if "covers_7d_std" in row.index:
        vals = [actuals.get(origin - timedelta(days=k), np.nan) for k in range(7)]
        vals = [v for v in vals if not np.isnan(v)]
        row["covers_7d_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    if "covers_28d_mean" in row.index:
        vals = [actuals.get(origin - timedelta(days=k), np.nan) for k in range(28)]
        vals = [v for v in vals if not np.isnan(v)]
        row["covers_28d_mean"] = float(np.mean(vals)) if vals else np.nan
    return row


def predict_quantiles(models: dict, features: list[str], row: pd.Series) -> dict[str, float]:
    X = pd.DataFrame([row[features]])
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if X[col].dtype == object and col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return {
        name: float(max(0, m.predict(X)[0]))
        for name, m in models.items()
    }


def load_models() -> tuple[dict, list[str], float]:
    """Load quantile models + conformal buffer."""
    model_dir = Path("data/models")
    features = json.loads((model_dir / "quantile_feature_names.json").read_text())
    metrics = json.loads((model_dir / "quantile_metrics.json").read_text())
    buffer = float(metrics.get("conformal_buffer", 0))
    # Load whatever quantile models exist (05/95 or 10/90)
    models = {}
    for name, alphas in [("lower", [5, 10]), ("median", [50]), ("upper", [95, 90])]:
        loaded = False
        for alpha in alphas:
            path = model_dir / f"lgbm_daily_q{alpha:02d}.pkl"
            if path.exists():
                with open(path, "rb") as fh:
                    models[name] = pickle.load(fh)
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(f"No quantile model found for {name}")
    # Skip the old explicit open block
    return models, features, buffer



def demo_origin(origin_date: date, n_days: int, daily: pd.DataFrame,
                actuals: dict[date, float], daily_idx: dict[date, int],
                models: dict, features: list[str],
                buffer: float = 0.0) -> pd.DataFrame:
    """Run the demo for a specific origin date."""
    targets = [origin_date + timedelta(days=h) for h in range(1, n_days + 1)]
    targets = [t for t in targets if t in daily_idx]

    records = []
    for t in targets:
        h = (t - origin_date).days
        row = daily.iloc[daily_idx[t]].copy()
        row = recompute_lags(row, t, origin_date, actuals)
        q = predict_quantiles(models, features, row)
        lo = max(0, q["lower"] - buffer)
        hi = q["upper"] + buffer
        actual = actuals.get(t, np.nan)
        in_interval = (actual >= lo and actual <= hi) if not np.isnan(actual) else None
        records.append({
            "target": t,
            "horizon": h,
            "dow": t.strftime("%a"),
            "actual": actual,
            "lower_10": lo,
            "predicted": q["median"],
            "upper_90": hi,
            "interval_width": hi - lo,
            "in_interval": in_interval,
        })
    return pd.DataFrame(records)


def print_demo(df: pd.DataFrame, origin_date: date, actuals: dict[date, float]) -> None:
    origin_actual = actuals.get(origin_date, 0)
    print()
    print("=" * 90)
    print(f"FORECAST from {origin_date} ({origin_date.strftime('%A')}, "
          f"{origin_actual:.0f} trips actual)")
    print("=" * 90)
    print(f"{'Date':>12s} {'DoW':>4s} {'h':>3s} │ {'Lower':>6s}  {'Pred':>6s}  {'Upper':>6s} │ "
          f"{'Actual':>7s}  {'In?':>3s} {'Error':>7s}")
    print("─" * 90)

    for _, r in df.iterrows():
        actual_str = f"{r['actual']:7.0f}" if not np.isnan(r['actual']) else "   ???"
        if r['in_interval'] is True:
            marker = " ✓"
        elif r['in_interval'] is False:
            marker = " ✗"
        else:
            marker = "  ?"
        error_str = f"{r['predicted'] - r['actual']:+7.0f}" if not np.isnan(r['actual']) else "      ?"
        print(f"{str(r['target']):>12s} {r['dow']:>4s} {r['horizon']:>3d} │ "
              f"{r['lower_10']:>6.0f}  {r['predicted']:>6.0f}  {r['upper_90']:>6.0f} │ "
              f"{actual_str} {marker} {error_str}")

    if df["in_interval"].notna().any():
        scored = df[df["in_interval"].notna()]
        coverage = scored["in_interval"].mean()
        mae = (scored["predicted"] - scored["actual"]).abs().mean()
        print(f"\n  Coverage: {coverage:.0%} ({scored['in_interval'].sum():.0f}/{len(scored)} days "
              f"inside 80% interval)  |  MAE: {mae:.0f} trips")


def full_year_coverage(daily: pd.DataFrame, actuals: dict[date, float],
                       daily_idx: dict[date, int], models: dict,
                       features: list[str], buffer: float = 0.0) -> None:
    """Run rolling 7-day-ahead over all 2025 days and report coverage."""
    target_dates = [d for d in pd.date_range("2025-01-01", "2025-12-31", freq="D").date
                    if d in daily_idx and actuals.get(d, 0) > 50]

    records = []
    for t in target_dates:
        for h in range(1, 8):
            o = t - timedelta(days=h)
            if o not in actuals:
                continue
            row = daily.iloc[daily_idx[t]].copy()
            row = recompute_lags(row, t, o, actuals)
            q = predict_quantiles(models, features, row)
            lo = max(0, q["lower"] - buffer)
            hi = q["upper"] + buffer
            actual = actuals[t]
            records.append({
                "target": t, "horizon": h,
                "actual": actual,
                "lower": lo,
                "predicted": q["median"],
                "upper": hi,
                "in_interval": actual >= lo and actual <= hi,
            })

    df = pd.DataFrame(records)

    print("\n" + "=" * 78)
    print("FULL-YEAR 2025 EMPIRICAL COVERAGE (80% prediction interval)")
    print("=" * 78)
    print(f"\n{'Horizon':>8s} {'n':>6s} {'Coverage':>9s} {'WAPE':>8s} {'MAE':>8s} {'Avg Width':>10s}")
    print("-" * 55)
    for h in range(1, 8):
        seg = df[df["horizon"] == h]
        cov = seg["in_interval"].mean()
        wape = (seg["predicted"] - seg["actual"]).abs().sum() / max(seg["actual"].abs().sum(), 1e-8)
        mae = (seg["predicted"] - seg["actual"]).abs().mean()
        width = (seg["upper"] - seg["lower"]).mean()
        print(f"     {h}d  {len(seg):>6d} {cov:>8.1%} {wape:>8.1%} {mae:>8.1f} {width:>10.0f}")

    overall_cov = df["in_interval"].mean()
    overall_wape = (df["predicted"] - df["actual"]).abs().sum() / max(df["actual"].abs().sum(), 1e-8)
    print(f"\n  Overall coverage: {overall_cov:.1%} (target: 80%)")
    print(f"  Overall WAPE:     {overall_wape:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=str, default="2025-06-01",
                        help="Origin date (YYYY-MM-DD) for the demo forecast")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--full-year", action="store_true",
                        help="Also run full-year 2025 coverage analysis")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Re-compute conformal buffer from 2025 actuals (for demo accuracy)")
    args = parser.parse_args()

    daily = pd.read_parquet("data/training/daily_training_features.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.dropna(subset=["covers_7d_lag"]).reset_index(drop=True)
    daily_idx = {d.date(): i for d, i in zip(daily["date"], daily.index)}
    actuals = {d.date(): float(c) for d, c in zip(daily["date"], daily["cover_count"])}

    models, features, buffer = load_models()

    if args.recalibrate:
        # Recalibrate buffer using 2025 h=1 residuals (quarterly recalibration in prod)
        print("Recalibrating conformal buffer from 2025 actuals...")
        cal_dates = [d for d in pd.date_range("2025-01-01", "2025-12-31", freq="D").date
                     if d in daily_idx and actuals.get(d, 0) > 50]
        conformity_scores = []
        for t in cal_dates:
            for h in range(1, 8):
                o = t - timedelta(days=h)
                if o not in actuals:
                    continue
                row = daily.iloc[daily_idx[t]].copy()
                row = recompute_lags(row, t, o, actuals)
                q = predict_quantiles(models, features, row)
                score = max(q["lower"] - actuals[t], actuals[t] - q["upper"])
                conformity_scores.append(score)
        buffer = float(np.quantile(conformity_scores, 0.80))
        print(f"Recalibrated buffer: ±{buffer:.0f} trips (from {len(conformity_scores)} 2025 days)")
        # Save for future use
        import json as json_mod
        metrics_path = Path("data/models/quantile_metrics.json")
        metrics = json_mod.loads(metrics_path.read_text())
        metrics["conformal_buffer"] = buffer
        metrics["recalibrated_on"] = "2025 holdout"
        metrics_path.write_text(json_mod.dumps(metrics, indent=2))

    print(f"Conformal buffer: ±{buffer:.0f} trips")

    origin = date.fromisoformat(args.origin)
    demo_df = demo_origin(origin, args.days, daily, actuals, daily_idx, models, features, buffer)
    print_demo(demo_df, origin, actuals)

    if args.full_year:
        full_year_coverage(daily, actuals, daily_idx, models, features, buffer)


if __name__ == "__main__":
    main()
