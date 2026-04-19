"""Train a LightGBM daily-granularity demand model.

A daily model typically beats an hourly model on WAPE by 2-3x because
it smooths away hourly noise — perfect when daily accuracy is what you
care about operationally (staff rostering, food prep volume, etc.).

Usage:
    python -m src.models.train_london_daily
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

DEFAULT_INPUT = Path("data/training/daily_training_features.parquet")
DEFAULT_MODEL_DIR = Path("data/models")

EXCLUDE_COLS = {"cover_count", "date", "restaurant_id"}
CATEGORICAL = ["city_tier", "footfall_zone_class", "country_code"]

# Short-horizon lag features — exclude these to force the model to learn from
# weather/calendar signals rather than just memorizing recent history. Makes
# the model useful for 7+ day ahead forecasting (autoregressively stable).
SHORT_HORIZON_LAGS = {
    "covers_1d_lag",
    "covers_7d_lag",
    "covers_same_dow_last_week",
    "covers_7d_mean",
    "covers_7d_std",
}

# Feature groupings for ablation studies.
FEATURE_GROUPS = {
    "weather": [
        "temp_mean_c", "temp_min_c", "temp_max_c", "temp_anomaly_mean_c",
        "apparent_temp_mean_c", "wind_speed_mean_kmh", "precipitation_total_mm",
        "is_wet_day", "rain_streak", "temp_change_c", "temp_shock",
        "cold_weekend", "hot_day", "weather_code_max", "severe_weather",
    ],
    "calendar": [
        "is_bank_holiday", "is_school_holiday", "is_cultural_period",
        "days_to_next_holiday", "days_from_last_holiday",
        "days_to_next_cultural", "days_from_last_cultural",
        "dow", "is_weekend", "month", "week_of_year", "day_of_year",
        "dow_sin", "dow_cos", "doy_sin", "doy_cos",
        "is_tube_strike", "days_since_tube_strike",
    ],
    "daylight": ["daylight_hours", "sunrise_hour", "sunset_hour"],
    "lags": [
        "covers_1d_lag", "covers_7d_lag", "covers_14d_lag", "covers_28d_lag",
        "covers_365d_lag", "covers_7d_mean", "covers_28d_mean", "covers_7d_std",
        "covers_same_dow_last_week",
    ],
}

DEFAULT_PARAMS = {
    "objective": "quantile",
    "alpha": 0.5,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "verbose": -1,
}


def prepare_xy(
    df: pd.DataFrame,
    exclude_short_lags: bool = False,
    only_groups: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    excluded = set(EXCLUDE_COLS)
    if exclude_short_lags:
        excluded |= SHORT_HORIZON_LAGS

    if only_groups is not None:
        keep: set[str] = set()
        for g in only_groups:
            keep |= set(FEATURE_GROUPS.get(g, []))
        feature_cols = [c for c in df.columns if c in keep and c not in excluded]
    else:
        feature_cols = [c for c in df.columns if c not in excluded]

    X = df[feature_cols].copy()
    y = df["cover_count"].copy()

    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    cats = [c for c in CATEGORICAL if c in X.columns]
    return X, y, cats


def walk_forward(df: pd.DataFrame, n_folds: int = 5, min_train_days: int = 90):
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    fold_size = (n - min_train_days) // n_folds
    splits = []
    for i in range(n_folds):
        train_end = min_train_days + i * fold_size
        val_end = min(train_end + fold_size, n)
        if val_end <= train_end:
            break
        splits.append((df.index[:train_end], df.index[train_end:val_end]))
    return splits


def train(
    input_path: Path | None = None,
    model_dir: Path | None = None,
    coverage: float = 0.80,
    exclude_short_lags: bool = False,
    model_suffix: str = "",
    train_end_date: str | None = None,
    only_groups: list[str] | None = None,
) -> dict:
    input_path = input_path or DEFAULT_INPUT
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])

    # Gate training data by date (e.g. everything strictly before 2024-01-01).
    if train_end_date is not None:
        cutoff = pd.Timestamp(train_end_date)
        before = len(df)
        df = df[df["date"] <= cutoff].reset_index(drop=True)
        log.info("daily_train.train_gated", rows_before=before, rows_after=len(df), cutoff=train_end_date)

    # Drop rows where essential lag features are NaN (first ~365 days)
    df = df.dropna(subset=["covers_7d_lag", "covers_7d_mean"]).reset_index(drop=True)
    log.info(
        "daily_train.loaded",
        rows=len(df),
        date_range=f"{df['date'].min().date()} to {df['date'].max().date()}",
        exclude_short_lags=exclude_short_lags,
        only_groups=only_groups,
    )

    X, y, cats = prepare_xy(df, exclude_short_lags=exclude_short_lags, only_groups=only_groups)
    feature_cols = list(X.columns)

    # Walk-forward CV
    splits = walk_forward(df, n_folds=5, min_train_days=90)
    cv_wapes: list[float] = []
    cv_mapes: list[float] = []
    for i, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(**DEFAULT_PARAMS)
        m.fit(
            X.loc[tr_idx], y.loc[tr_idx],
            eval_set=[(X.loc[va_idx], y.loc[va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            categorical_feature=cats,
        )
        preds = m.predict(X.loc[va_idx])
        actuals = y.loc[va_idx].values
        wape = float(np.sum(np.abs(actuals - preds)) / max(np.sum(np.abs(actuals)), 1e-8))
        mape = float(np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1))))
        cv_wapes.append(wape)
        cv_mapes.append(mape)
        log.info("daily_train.fold", fold=i, wape=wape, mape=mape, n_train=len(tr_idx), n_val=len(va_idx))

    # Final model on full data
    model = lgb.LGBMRegressor(**DEFAULT_PARAMS)
    model.fit(X, y, categorical_feature=cats)

    # Residual-based intervals from last fold
    _, last_va = splits[-1]
    y_pred_calib = model.predict(X.loc[last_va])
    residuals = y.loc[last_va].values - y_pred_calib
    alpha = (1 - coverage) / 2
    lower_q = float(np.quantile(residuals, alpha))
    upper_q = float(np.quantile(residuals, 1 - alpha))
    mae_calib = float(np.mean(np.abs(y.loc[last_va].values - y_pred_calib)))
    wape_calib = float(
        np.sum(np.abs(y.loc[last_va].values - y_pred_calib))
        / max(np.sum(np.abs(y.loc[last_va].values)), 1e-8)
    )

    # Save artefacts (optional suffix for A/B comparison)
    model_path = model_dir / f"lgbm_daily{model_suffix}.pkl"
    features_path = model_dir / f"daily_feature_names{model_suffix}.json"
    metrics_path = model_dir / f"daily_training_metrics{model_suffix}.json"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    features_path.write_text(json.dumps(feature_cols, indent=2))

    metrics = {
        "cv_wape_mean": float(np.mean(cv_wapes)),
        "cv_wape_std": float(np.std(cv_wapes)),
        "cv_wape_per_fold": cv_wapes,
        "cv_mape_mean": float(np.mean(cv_mapes)),
        "mae_calib": mae_calib,
        "wape_calib": wape_calib,
        "coverage_target": coverage,
        "residual_lower_q": lower_q,
        "residual_upper_q": upper_q,
        "n_samples": len(df),
        "n_features": len(feature_cols),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    log.info("daily_train.done", **{k: v for k, v in metrics.items() if not isinstance(v, list)})

    # Feature importance top-10
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    log.info("daily_train.top_features", top10=imp.head(10).to_dict(orient="records"))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--exclude-short-lags", action="store_true",
        help="Drop 1-day and 7-day lag features to force learning "
             "from weather/calendar. Needed for multi-day-ahead forecasts."
    )
    parser.add_argument("--model-suffix", type=str, default="")
    parser.add_argument("--train-end", type=str, default=None,
                        help="Max date (inclusive) to include in training, e.g. 2023-12-31")
    parser.add_argument("--only-groups", type=str, default=None,
                        help="Comma-separated feature groups to include (weather,calendar,daylight,lags)")
    args = parser.parse_args()
    groups = args.only_groups.split(",") if args.only_groups else None
    train(
        input_path=args.input,
        model_dir=args.model_dir,
        exclude_short_lags=args.exclude_short_lags,
        model_suffix=args.model_suffix,
        train_end_date=args.train_end,
        only_groups=groups,
    )


if __name__ == "__main__":
    main()
