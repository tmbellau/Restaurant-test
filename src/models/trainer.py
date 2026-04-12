"""LightGBM trainer with walk-forward cross-validation.

Per the spec:
- One global model per (granularity, channel). location_id is categorical.
- Walk-forward expanding window, 5 folds, minimum 4 weeks per fold.
- Never random splits on time series.
- Quantile objective (alpha=0.5 for median); CQR wraps for intervals.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import structlog

from src.features.censoring import compute_sample_weights

log = structlog.get_logger(__name__)

CATEGORICAL_FEATURES = [
    "restaurant_id",
    "city_tier",
    "footfall_zone_class",
    "precip_intensity_cat",
    "country_code",
]

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "alpha": 0.5,
    "num_leaves": 127,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "cat_smooth": 10,
    "n_estimators": 2000,
    "verbose": -1,
}

# Features that should never be in the model input
_EXCLUDE_COLS = {
    "cover_count", "timestamp_utc", "local_ts", "was_closed",
    "ingested_at_utc", "transaction_id", "item_id", "item_qty",
    "item_price", "voided", "comped", "record_version", "channel",
    "bank_holiday_name", "event_name", "disrupted_lines",
}


def _prepare_xy(
    df: pd.DataFrame,
    target: str = "cover_count",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split features and target, identify categoricals present."""
    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS and c != target]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Convert categoricals
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    return X, y, cat_cols


def walk_forward_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    min_train_hours: int = 672,  # 4 weeks
) -> list[tuple[pd.Index, pd.Index]]:
    """Generate expanding-window walk-forward CV splits.

    Returns list of (train_idx, val_idx) pairs.
    """
    df_sorted = df.sort_values("timestamp_utc")
    n = len(df_sorted)
    fold_size = (n - min_train_hours) // n_folds

    splits = []
    for i in range(n_folds):
        train_end = min_train_hours + i * fold_size
        val_end = min(train_end + fold_size, n)
        if val_end <= train_end:
            break
        train_idx = df_sorted.index[:train_end]
        val_idx = df_sorted.index[train_end:val_end]
        splits.append((train_idx, val_idx))

    return splits


def train_model(
    df: pd.DataFrame,
    target: str = "cover_count",
    params: dict[str, Any] | None = None,
) -> tuple[lgb.LGBMRegressor, list[str], dict[str, float]]:
    """Train a LightGBM model on the full dataset.

    Returns (model, feature_names, metrics_dict).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    X, y, cat_cols = _prepare_xy(df, target)
    weights = compute_sample_weights(df)

    # Walk-forward CV for metrics
    splits = walk_forward_split(df)
    cv_scores: list[float] = []
    for train_idx, val_idx in splits:
        fold_model = lgb.LGBMRegressor(**params)
        fold_model.fit(
            X.loc[train_idx], y.loc[train_idx],
            sample_weight=weights.loc[train_idx],
            eval_set=[(X.loc[val_idx], y.loc[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            categorical_feature=cat_cols,
        )
        preds = fold_model.predict(X.loc[val_idx])
        actuals = y.loc[val_idx].values
        # WAPE
        wape = np.sum(np.abs(actuals - preds)) / max(np.sum(np.abs(actuals)), 1e-8)
        cv_scores.append(wape)

    log.info("trainer.cv_done", n_folds=len(splits), wape_scores=cv_scores)

    # Final model on full data
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(
        X, y,
        sample_weight=weights,
        categorical_feature=cat_cols,
    )

    metrics = {
        "cv_wape_mean": float(np.mean(cv_scores)),
        "cv_wape_std": float(np.std(cv_scores)),
        "n_features": len(X.columns),
        "n_samples": len(X),
    }
    log.info("trainer.done", **metrics)
    return final_model, list(X.columns), metrics
