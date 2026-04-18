"""Rolling 7-day-ahead forecast with daily refresh.

Each evening at origin O, predict O+1 through O+7. As O advances, lag
features get fresher — covers_1d_lag moves from actual[O] (stale at h=7)
to actual[T-1] (exact at h=1). This is how restaurant ops would use it:
plan 7 days out, refine daily.

Uses the short-lag model (WITH 1d/7d lags) because the whole point is
that recent lags improve as horizon shrinks.
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


def recompute_features_for_horizon(
    base_row: pd.Series,
    target: date,
    origin: date,
    actuals: dict[date, float],
) -> pd.Series:
    """Recompute lag/rolling features using ONLY actuals through origin.

    Key idea: as origin gets closer to target (horizon shrinks),
    covers_1d_lag gets fresher and rolling means get more current.
    """
    row = base_row.copy()
    h = (target - origin).days

    # covers_1d_lag: most recent known actual = actual[origin]
    if "covers_1d_lag" in row.index:
        row["covers_1d_lag"] = actuals.get(origin, np.nan)

    # covers_7d_lag: actual[target - 7]. For h<=7, target-7 <= origin, so available.
    if "covers_7d_lag" in row.index:
        d7 = target - timedelta(days=7)
        row["covers_7d_lag"] = actuals.get(d7, np.nan)

    # covers_same_dow_last_week: same as 7d lag
    if "covers_same_dow_last_week" in row.index:
        d7 = target - timedelta(days=7)
        row["covers_same_dow_last_week"] = actuals.get(d7, np.nan)

    # covers_14d_lag, 28d_lag, 365d_lag: target-anchored, always available
    for feat, offset in [("covers_14d_lag", 14), ("covers_28d_lag", 28), ("covers_365d_lag", 365)]:
        if feat in row.index:
            row[feat] = actuals.get(target - timedelta(days=offset), np.nan)

    # Rolling means/std: computed over the most recent known days (up to origin)
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


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.build_daily_training_set import aggregate_to_daily

    print("Loading data...")
    hourly = pd.read_parquet("data/holdout/training_features.parquet")
    daily = aggregate_to_daily(hourly)
    daily = daily.dropna(subset=["covers_7d_lag"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily_idx = {d.date(): i for d, i in zip(daily["date"], daily.index)}

    with open("data/models/lgbm_daily.pkl", "rb") as fh:
        model = pickle.load(fh)
    features = json.loads(Path("data/models/daily_feature_names.json").read_text())

    actuals_dict: dict[date, float] = {
        d.date(): float(c) for d, c in zip(daily["date"], daily["cover_count"])
    }

    first_target = date(2025, 1, 1)
    last_target = date(2025, 1, 30)

    print(f"Targets: {first_target} to {last_target}")
    print(f"Each target predicted 7 times (h=7d down to h=1d)\n")

    records = []
    for t_ts in pd.date_range(first_target, last_target, freq="D"):
        t = t_ts.date()
        if t not in daily_idx:
            continue
        actual = actuals_dict[t]

        for h in range(1, 8):
            o = t - timedelta(days=h)
            base_row = daily.iloc[daily_idx[t]]
            row = recompute_features_for_horizon(base_row, t, o, actuals_dict)

            X = pd.DataFrame([row[features]])
            for c in CATEGORICAL:
                if c in X.columns:
                    X[c] = X[c].astype("category")
            for col in X.columns:
                if X[col].dtype == object and col not in CATEGORICAL:
                    X[col] = pd.to_numeric(X[col], errors="coerce")

            pred = float(max(0, model.predict(X)[0]))
            records.append({
                "target": t,
                "horizon": h,
                "origin": o,
                "predicted": pred,
                "actual": actual,
                "lag1_value": actuals_dict.get(o, np.nan),
            })

    df = pd.DataFrame(records)
    df["error"] = df["predicted"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = (df["abs_error"] / df["actual"].clip(lower=1) * 100)

    # === Summary by horizon ===
    print("=" * 70)
    print("ACCURACY BY FORECAST HORIZON (averaged across all Jan 2025 days)")
    print("=" * 70)
    print(f"\n{'Horizon':>8s} {'MAE':>8s} {'WAPE':>8s} {'Med%Err':>8s} {'Mean%Err':>9s}")
    print("-" * 50)
    for h in range(1, 8):
        seg = df[df["horizon"] == h]
        mae = seg["abs_error"].mean()
        wape = seg["abs_error"].sum() / max(seg["actual"].sum(), 1e-8)
        med_pct = seg["pct_error"].median()
        mean_pct = seg["pct_error"].mean()
        print(f"     {h}d  {mae:>8.1f} {wape:>8.1%} {med_pct:>7.1f}%  {mean_pct:>8.1f}%")

    # === Each day's predictions across horizons ===
    print("\n" + "=" * 70)
    print("EACH DAY PREDICTED 7 TIMES  (✓ ≤10%  ~ ≤25%  ✗ >25%)")
    print("=" * 70)
    targets_sorted = sorted(df["target"].unique())
    header = f"{'Target':>14s} {'Actual':>6s} |"
    for h in [7, 6, 5, 4, 3, 2, 1]:
        header += f" {h}d-out"
    print(header)
    print("-" * len(header))

    for t in targets_sorted:
        actual = actuals_dict.get(t, 0)
        row_str = f"{str(t) + ' ' + pd.Timestamp(t).strftime('%a'):>14s} {actual:>6.0f} |"
        for h in [7, 6, 5, 4, 3, 2, 1]:
            match = df[(df["target"] == t) & (df["horizon"] == h)]
            if len(match):
                p = match.iloc[0]["predicted"]
                pct = abs(p - actual) / max(actual, 1) * 100
                if pct <= 10:
                    marker = "✓"
                elif pct <= 25:
                    marker = "~"
                else:
                    marker = "✗"
                row_str += f" {p:4.0f}{marker} "
            else:
                row_str += "   -   "
        print(row_str)

    # === Visualization ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1.5]})

    # Top: WAPE by horizon bar chart
    ax = axes[0]
    horizon_stats = []
    for h in range(1, 8):
        seg = df[df["horizon"] == h]
        wape = seg["abs_error"].sum() / max(seg["actual"].sum(), 1e-8)
        horizon_stats.append({"horizon": h, "wape": wape})
    hs = pd.DataFrame(horizon_stats)
    colors = [plt.cm.RdYlGn(1.0 - w) for w in hs["wape"]]
    bars = ax.bar(hs["horizon"], hs["wape"] * 100, color=colors,
                  edgecolor="black", linewidth=0.5, width=0.6)
    for bar, w in zip(bars, hs["wape"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{w:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Forecast Horizon (days ahead)", fontsize=12)
    ax.set_ylabel("WAPE (%)", fontsize=12)
    ax.set_title("Does accuracy improve as forecast horizon shrinks?",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels([f"{h} day" for h in range(1, 8)])
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom: predictions converging on actual for each day
    ax2 = axes[1]
    for t in targets_sorted:
        sub = df[df["target"] == t].sort_values("horizon", ascending=False)
        actual = actuals_dict[t]
        preds_by_h = sub.set_index("horizon")["predicted"]
        xs = list(range(7, 0, -1))
        ys = [preds_by_h.get(h, np.nan) for h in xs]
        ax2.plot(xs, ys, alpha=0.25, color="steelblue", linewidth=0.8)
        ax2.scatter([1], [ys[-1]], s=15, color="steelblue", zorder=3, alpha=0.4)
    # Overlay actuals
    target_dates_num = list(range(len(targets_sorted)))
    ax2.set_xlabel("Days before target (7 = earliest, 1 = final forecast)", fontsize=12)
    ax2.set_ylabel("Predicted value", fontsize=12)
    ax2.set_title("30 target days × 7 forecasts each — how predictions evolve",
                  fontsize=13, fontweight="bold")
    ax2.set_xticks(range(1, 8))
    ax2.set_xticklabels([f"{h}d out" for h in range(1, 8)])
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path("data/holdout/rolling_7day_forecast.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nPlot saved: {out_path}")
    df.to_parquet("data/holdout/rolling_7day_forecast.parquet", index=False)


if __name__ == "__main__":
    main()
