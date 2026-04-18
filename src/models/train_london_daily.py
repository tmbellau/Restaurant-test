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


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
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
) -> dict:
    input_path = input_path or DEFAULT_INPUT
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    # Drop rows where essential lag features are NaN (first ~365 days)
    df = df.dropna(subset=["covers_7d_lag", "covers_7d_mean"]).reset_index(drop=True)
    log.info("daily_train.loaded", rows=len(df), date_range=f"{df['date'].min().date()} to {df['date'].max().date()}")

    X, y, cats = prepare_xy(df)
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

    # Save artefacts
    with open(model_dir / "lgbm_daily.pkl", "wb") as fh:
        pickle.dump(model, fh)
    (model_dir / "daily_feature_names.json").write_text(json.dumps(feature_cols, indent=2))

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
    (model_dir / "daily_training_metrics.json").write_text(json.dumps(metrics, indent=2))

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
    args = parser.parse_args()
    train(input_path=args.input, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
