"""Live forward prediction — predict the next N days of Santander trip demand.

Uses:
- Trained LightGBM daily model (data/models/lgbm_daily.pkl)
- Whatever Santander actuals we have on disk as the "known history"
- Open-Meteo forecast API (if prediction dates are future) or archive (if past)

Output:
- predictions saved to data/predictions/{generated_at}.parquet with one row
  per (target, horizon) pair so we can compare to actuals later.

Usage:
    # Predict next 7 days starting tomorrow
    python scripts/predict_future.py

    # Predict from a specific origin (pretend today is 2025-12-31)
    python scripts/predict_future.py --origin 2025-12-31 --days 7

    # Quietly refresh data first
    python scripts/predict_future.py --refresh
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.build_daily_training_set import aggregate_to_daily
from src.data.santander import SOHO_LAT, SOHO_LNG

CATEGORICAL = ["city_tier", "footfall_zone_class", "country_code"]
PREDICTIONS_DIR = Path("data/predictions")


def recompute_features_for_target(
    base_row: pd.Series,
    target: date,
    origin: date,
    actuals: dict[date, float],
) -> pd.Series:
    """Recompute lag + rolling features using only actuals through origin."""
    row = base_row.copy()
    if "covers_1d_lag" in row.index:
        row["covers_1d_lag"] = actuals.get(origin, np.nan)
    for feat, offset in [
        ("covers_7d_lag", 7), ("covers_14d_lag", 14),
        ("covers_28d_lag", 28), ("covers_365d_lag", 365),
        ("covers_same_dow_last_week", 7),
    ]:
        if feat in row.index:
            look = target - timedelta(days=offset)
            row[feat] = actuals.get(look, actuals.get(origin, np.nan))
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


def predict_one(model, features: list[str], row: pd.Series) -> float:
    X = pd.DataFrame([row[features]])
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    for col in X.columns:
        if X[col].dtype == object and col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return float(max(0, model.predict(X)[0]))


def build_forecast_feature_row(target: date, daily_template: pd.Series) -> pd.Series:
    """For a future target date, assemble features that don't depend on the target.

    Calendar / daylight / weather-forecast values are populated. Lag features
    are filled in by recompute_features_for_target. Everything else is taken
    from the nearest-in-time daily template to preserve categoricals.
    """
    from src.data.daylight import daylight_features_for_date
    from src.data.tube_strikes import TUBE_STRIKE_DATES
    from src.signals.holidays import HolidaySource, _SCHOOL_HOLIDAYS

    row = daily_template.copy()

    # Temporal / calendar
    ts = pd.Timestamp(target)
    row["date"] = ts
    row["dow"] = ts.dayofweek
    row["is_weekend"] = int(ts.dayofweek >= 5)
    row["month"] = ts.month
    row["week_of_year"] = int(ts.isocalendar().week)
    row["day_of_year"] = ts.dayofyear
    row["dow_sin"] = np.sin(2 * np.pi * ts.dayofweek / 7)
    row["dow_cos"] = np.cos(2 * np.pi * ts.dayofweek / 7)
    row["doy_sin"] = np.sin(2 * np.pi * ts.dayofyear / 365)
    row["doy_cos"] = np.cos(2 * np.pi * ts.dayofyear / 365)

    # Daylight
    dl = daylight_features_for_date(target)
    for k, v in dl.items():
        row[k] = v

    # Tube strike flag
    row["is_tube_strike"] = target in TUBE_STRIKE_DATES
    past_strikes = [s for s in TUBE_STRIKE_DATES if s < target]
    row["days_since_tube_strike"] = min(
        (target - past_strikes[-1]).days, 30
    ) if past_strikes else 30

    # Bank + school holidays
    hs = HolidaySource()
    hs_df = hs._fetch_raw_impl(
        restaurant_id="soho_cycles",
        start_utc=datetime(target.year - 1, 12, 1),
        end_utc=datetime(target.year + 1, 2, 1),
    )
    row_day = hs_df[hs_df["timestamp_utc"].dt.date == target]
    if not row_day.empty:
        for col in ["is_bank_holiday", "is_school_holiday",
                    "days_to_next_holiday", "days_from_last_holiday"]:
            if col in row_day.columns:
                row[col] = row_day[col].iloc[0]
    row["is_school_holiday"] = any(s <= target <= e for s, e in _SCHOOL_HOLIDAYS)

    # Cultural calendar
    from src.data.uk_cultural_calendar import build_uk_cultural_calendar
    cul = build_uk_cultural_calendar(
        start_date=target - timedelta(days=30),
        end_date=target + timedelta(days=30),
        restaurant_id="soho_cycles",
    )
    cul_today = cul[pd.to_datetime(cul["timestamp_utc"]).dt.date == target]
    if not cul_today.empty:
        for col in ["is_cultural_period", "days_to_next_cultural", "days_from_last_cultural"]:
            if col in cul_today.columns:
                row[col] = cul_today[col].iloc[0]

    return row


def fetch_weather_for_targets(targets: list[date]) -> dict[date, dict]:
    """Fetch weather for a list of dates. Uses forecast API if future, archive if past."""
    from src.data.historical_weather import fetch_weather
    import httpx

    today = datetime.utcnow().date()
    past = [d for d in targets if d <= today]
    future = [d for d in targets if d > today]
    out: dict[date, dict] = {}

    if past:
        start = min(past).isoformat()
        end = max(past).isoformat()
        wdf = fetch_weather(
            lat=SOHO_LAT, lng=SOHO_LNG,
            start_date=start, end_date=end,
            restaurant_id="predict_archive",
            cache_dir=None,
        )
        wdf["local_ts"] = pd.to_datetime(wdf["timestamp_utc"]).dt.tz_convert("Europe/London")
        wdf["d"] = wdf["local_ts"].dt.date
        daily_agg = wdf.groupby("d").agg(
            temp_mean_c=("temperature_c", "mean"),
            temp_min_c=("temperature_c", "min"),
            temp_max_c=("temperature_c", "max"),
            apparent_temp_mean_c=("apparent_temperature_c", "mean"),
            wind_speed_mean_kmh=("wind_speed_kmh", "mean"),
            precipitation_total_mm=("precipitation_mm", "sum"),
            weather_code_max=("weather_code", "max"),
        )
        for d, r in daily_agg.iterrows():
            out[d] = r.to_dict()

    if future:
        from src.data.weather_forecast import fetch_forecast
        max_ahead = (max(future) - today).days + 1
        fdf = fetch_forecast(SOHO_LAT, SOHO_LNG, days=max_ahead)
        fdf["d"] = pd.to_datetime(fdf["date"]).dt.date
        for d, r in fdf.set_index("d").iterrows():
            if d in future:
                out[d] = {
                    "temp_mean_c": r["temp_mean_c"],
                    "temp_min_c": r["temp_min_c"],
                    "temp_max_c": r["temp_max_c"],
                    "apparent_temp_mean_c": r["temp_mean_c"],  # forecast doesn't return apparent
                    "wind_speed_mean_kmh": r["wind_speed_mean_kmh"],
                    "precipitation_total_mm": r["precipitation_total_mm"],
                    "weather_code_max": r["weather_code"],
                }

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=str, default=None,
                        help="Origin date (latest-known actual). Default: use most recent actual in the parquet.")
    parser.add_argument("--days", type=int, default=7,
                        help="How many days ahead to predict")
    parser.add_argument("--refresh", action="store_true",
                        help="Download any new Santander files before predicting")
    args = parser.parse_args()

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if args.refresh:
        from src.data.build_training_set import build as build_training_set
        print("Refreshing training data from TfL...")
        build_training_set(
            start_date=datetime(2017, 1, 1),
            end_date=datetime.utcnow(),
            output_dir=Path("data/training"),
        )
        print("Rebuilding daily aggregation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "src.data.build_daily_training_set", "--exclude-covid"])

    # Load daily history
    daily = pd.read_parquet("data/training/daily_training_features.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    actuals: dict[date, float] = {
        d.date(): float(c) for d, c in zip(daily["date"], daily["cover_count"])
    }

    # Determine origin. Skip backwards past suspected data outages
    # (days with <5% of the trailing-28d mean — usually a station-outage day).
    if args.origin:
        origin = datetime.strptime(args.origin, "%Y-%m-%d").date()
    else:
        sorted_dates = sorted(actuals.keys(), reverse=True)
        origin = sorted_dates[0]
        for d in sorted_dates:
            past_28 = [actuals.get(d - timedelta(days=k)) for k in range(1, 29)]
            past_28 = [v for v in past_28 if v is not None and v > 50]
            if past_28 and actuals[d] > 0.1 * float(np.mean(past_28)):
                origin = d
                break
            print(f"  skipping {d} (only {actuals[d]:.0f} trips — suspected data outage)")
    print(f"Origin (latest valid actual): {origin}  ({actuals[origin]:.0f} trips)")

    # Build target dates
    targets = [origin + timedelta(days=h) for h in range(1, args.days + 1)]
    print(f"Predicting: {targets[0]} to {targets[-1]} ({len(targets)} days)")

    # Fetch weather for each target (forecast if future, archive if past)
    print("\nFetching weather...")
    weather_by_date = fetch_weather_for_targets(targets)

    # Load trained model
    with open("data/models/lgbm_daily.pkl", "rb") as fh:
        model = pickle.load(fh)
    features = json.loads(Path("data/models/daily_feature_names.json").read_text())
    print(f"Loaded model with {len(features)} features")

    # Use the most recent complete daily row as a template for categoricals
    template = daily.iloc[-1].copy()

    # Predict each target day (horizon = target - origin)
    preds = []
    for t in targets:
        horizon = (t - origin).days
        row = build_forecast_feature_row(t, template)

        # Inject weather
        w = weather_by_date.get(t)
        if w:
            for k, v in w.items():
                row[k] = v

        # Derived weather features
        row["is_wet_day"] = int(float(row.get("precipitation_total_mm", 0)) > 2)
        row["severe_weather"] = int(int(row.get("weather_code_max", 0)) >= 71)
        row["cold_weekend"] = int(
            float(row.get("temp_mean_c", 10)) < 8 and int(row.get("is_weekend", 0)) == 1
        )
        row["hot_day"] = int(float(row.get("temp_max_c", 15)) > 22)
        # rain_streak: count preceding wet days from recent history
        streak = 0
        for k in range(1, 8):
            prior = t - timedelta(days=k)
            if prior in actuals:  # we only know weather for past days via archive
                pass
            streak = 0  # simplification: can't easily recompute without pulling more weather
        row["rain_streak"] = streak
        row["temp_change_c"] = 0.0  # placeholder; full reconstruction would need yesterday's forecast
        row["temp_shock"] = 0
        if "temp_anomaly_mean_c" not in row.index or pd.isna(row.get("temp_anomaly_mean_c")):
            row["temp_anomaly_mean_c"] = 0.0

        # Lag features from actuals
        row = recompute_features_for_target(row, t, origin, actuals)

        # Ensure every expected feature exists
        for f in features:
            if f not in row.index or pd.isna(row.get(f)):
                row[f] = 0

        pred = predict_one(model, features, row)
        preds.append({
            "generated_at": datetime.utcnow().isoformat(),
            "origin": origin,
            "target": t,
            "horizon": horizon,
            "dow": t.strftime("%a"),
            "predicted": round(pred, 1),
            "temp_mean_c": w.get("temp_mean_c") if w else None,
            "precipitation_total_mm": w.get("precipitation_total_mm") if w else None,
            "wind_speed_mean_kmh": w.get("wind_speed_mean_kmh") if w else None,
            "is_bank_holiday": bool(row.get("is_bank_holiday", False)),
            "is_tube_strike": bool(row.get("is_tube_strike", False)),
        })

    out_df = pd.DataFrame(preds)
    gen_tag = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = PREDICTIONS_DIR / f"predictions_{gen_tag}.parquet"
    out_df.to_parquet(out_path, index=False)

    print("\n" + "=" * 78)
    print(f"FORECAST  |  origin: {origin} ({actuals[origin]:.0f} trips actual)")
    print("=" * 78)
    print(f"{'Date':>12s} {'DoW':>4s} {'h':>3s} {'Pred':>6s}  {'Temp':>5s} {'Rain':>6s} {'Wind':>5s}  {'Flags':s}")
    print("-" * 70)
    for r in preds:
        flags = []
        if r["is_bank_holiday"]: flags.append("bankhol")
        if r["is_tube_strike"]: flags.append("strike")
        flag_str = ", ".join(flags) if flags else ""
        t_str = f"{r['temp_mean_c']:.1f}°C" if r['temp_mean_c'] is not None else "?"
        rain = f"{r['precipitation_total_mm']:.1f}" if r['precipitation_total_mm'] is not None else "?"
        wind = f"{r['wind_speed_mean_kmh']:.0f}" if r['wind_speed_mean_kmh'] is not None else "?"
        print(f"{str(r['target']):>12s} {r['dow']:>4s} {r['horizon']:>3d} {r['predicted']:>6.0f}  "
              f"{t_str:>5s} {rain:>6s} {wind:>5s}  {flag_str}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
