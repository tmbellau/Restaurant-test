"""Train a quantile ensemble: lower (10th), median (50th), upper (90th).

Produces three LightGBM models that together form an 80% prediction
interval. The median model is the point estimate. The lower/upper
models learn the 10th and 90th percentile of the target distribution
given the same features.

Usage:
    python -m src.models.train_quantile_ensemble
    python -m src.models.train_quantile_ensemble --train-end 2024-12-31
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

from src.models.train_london_daily import (
    DEFAULT_PARAMS,
    prepare_xy,
    walk_forward,
)

log = structlog.get_logger(__name__)

DEFAULT_INPUT = Path("data/training/daily_training_features.parquet")
DEFAULT_MODEL_DIR = Path("data/models")

QUANTILES = {
    "lower": 0.05,
    "median": 0.50,
    "upper": 0.95,
}


def train_quantile_ensemble(
    input_path: Path | None = None,
    model_dir: Path | None = None,
    train_end_date: str | None = None,
) -> dict:
    input_path = input_path or DEFAULT_INPUT
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])

    if train_end_date:
        df = df[df["date"] <= pd.Timestamp(train_end_date)].reset_index(drop=True)

    df = df.dropna(subset=["covers_7d_lag", "covers_7d_mean"]).reset_index(drop=True)
    log.info("quantile_train.loaded", rows=len(df),
             date_range=f"{df['date'].min().date()} to {df['date'].max().date()}")

    X, y, cats = prepare_xy(df)
    feature_cols = list(X.columns)

    splits = walk_forward(df, n_folds=5, min_train_days=90)
    results = {}

    for name, alpha in QUANTILES.items():
        params = {**DEFAULT_PARAMS, "alpha": alpha}

        # Walk-forward CV to measure coverage
        all_preds = []
        all_actuals = []
        for tr_idx, va_idx in splits:
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X.loc[tr_idx], y.loc[tr_idx],
                eval_set=[(X.loc[va_idx], y.loc[va_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
                categorical_feature=cats,
            )
            all_preds.extend(m.predict(X.loc[va_idx]).tolist())
            all_actuals.extend(y.loc[va_idx].tolist())

        # Final model on all data
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y, categorical_feature=cats)

        model_path = model_dir / f"lgbm_daily_q{int(alpha*100):02d}.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)

        results[name] = {
            "alpha": alpha,
            "model_path": str(model_path),
            "cv_predictions": all_preds,
            "cv_actuals": all_actuals,
        }
        log.info("quantile_train.model_done", name=name, alpha=alpha,
                 path=str(model_path))

    # Save feature names (same for all three)
    (model_dir / "quantile_feature_names.json").write_text(
        json.dumps(feature_cols, indent=2)
    )

    # Compute empirical coverage from CV, then conformalize.
    # CQR (Conformal Quantile Regression): find the buffer needed to inflate
    # the raw [lower, upper] interval until it achieves the target coverage.
    cv_lower = np.array(results["lower"]["cv_predictions"])
    cv_upper = np.array(results["upper"]["cv_predictions"])
    cv_median = np.array(results["median"]["cv_predictions"])
    cv_actuals = np.array(results["median"]["cv_actuals"])

    # Conformity scores: how far outside the interval each CV sample falls.
    # Positive = outside interval. Negative = inside.
    conformity = np.maximum(cv_lower - cv_actuals, cv_actuals - cv_upper)
    # The buffer at the (1 - target_coverage) quantile of conformity scores
    # guarantees the target coverage on exchangeable future data.
    target_coverage = 0.80
    buffer = float(np.quantile(conformity, target_coverage))

    calibrated_lower = cv_lower - buffer
    calibrated_upper = cv_upper + buffer
    in_interval_raw = (cv_actuals >= cv_lower) & (cv_actuals <= cv_upper)
    in_interval_cal = (cv_actuals >= calibrated_lower) & (cv_actuals <= calibrated_upper)

    raw_coverage = float(in_interval_raw.mean())
    cal_coverage = float(in_interval_cal.mean())
    median_mae = float(np.abs(cv_actuals - cv_median).mean())
    median_wape = float(np.abs(cv_actuals - cv_median).sum() / max(np.abs(cv_actuals).sum(), 1e-8))
    raw_width = float((cv_upper - cv_lower).mean())
    cal_width = float((calibrated_upper - calibrated_lower).mean())

    metrics = {
        "target_coverage": target_coverage,
        "conformal_buffer": buffer,
        "raw_coverage_cv": raw_coverage,
        "calibrated_coverage_cv": cal_coverage,
        "median_mae_cv": median_mae,
        "median_wape_cv": median_wape,
        "raw_interval_width": raw_width,
        "calibrated_interval_width": cal_width,
        "n_cv_samples": len(cv_actuals),
        "n_training_rows": len(df),
        "n_features": len(feature_cols),
    }
    (model_dir / "quantile_metrics.json").write_text(json.dumps(metrics, indent=2))

    log.info(
        "quantile_train.done",
        raw_coverage=f"{raw_coverage:.1%}",
        calibrated_coverage=f"{cal_coverage:.1%}",
        conformal_buffer=f"{buffer:.0f}",
        median_wape=f"{median_wape:.1%}",
        calibrated_width=f"{cal_width:.0f}",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    args = parser.parse_args()
    train_quantile_ensemble(
        input_path=args.input,
        model_dir=args.model_dir,
        train_end_date=args.train_end,
    )


if __name__ == "__main__":
    main()
