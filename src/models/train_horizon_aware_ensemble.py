"""Train horizon-aware quantile ensemble.

Trains three LightGBM quantile models (5th/50th/95th percentile) on
multi-horizon training data, with `horizon_days` as an input feature.

Because the training set includes 7 copies of each day (one per horizon)
with progressively stale lag substitutions, the models organically learn
that longer-horizon inputs have more prediction variance — the intervals
widen with horizon because the training data showed larger errors there,
not because we hand-tuned a multiplier.

After training, we also compute a **per-horizon** conformal buffer from
walk-forward CV residuals. This tightens the guarantee so that 80%
coverage holds at EACH horizon, not just in aggregate.

Usage:
    python -m src.models.train_horizon_aware_ensemble --train-end 2024-12-31
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

DEFAULT_INPUT = Path("data/training/multi_horizon_training.parquet")
DEFAULT_MODEL_DIR = Path("data/models")

QUANTILES = {"lower": 0.05, "median": 0.50, "upper": 0.95}
TARGET_COVERAGE = 0.80

EXCLUDE_COLS = {"cover_count", "date", "restaurant_id"}
CATEGORICAL = ["city_tier", "footfall_zone_class", "country_code"]

PARAMS = {
    "objective": "quantile",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 20,  # slightly higher — 7x data means noisier local splits
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


def walk_forward_splits(df: pd.DataFrame, n_folds: int = 5,
                        min_train_rows_per_horizon: int = 90) -> list[tuple]:
    """Walk-forward CV on DATES (not rows). Each fold trains on all horizons
    for dates up to a cutoff and validates on all horizons for the next bucket."""
    df = df.sort_values(["date", "horizon_days"]).reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)
    fold_size = (n_dates - min_train_rows_per_horizon) // n_folds
    splits = []
    for i in range(n_folds):
        train_end = min_train_rows_per_horizon + i * fold_size
        val_end = min(train_end + fold_size, n_dates)
        if val_end <= train_end:
            break
        train_dates = set(unique_dates[:train_end])
        val_dates = set(unique_dates[train_end:val_end])
        tr_idx = df.index[df["date"].isin(train_dates)]
        va_idx = df.index[df["date"].isin(val_dates)]
        splits.append((tr_idx, va_idx))
    return splits


def train(
    input_path: Path | None = None,
    model_dir: Path | None = None,
    train_end_date: str | None = "2024-12-31",
) -> dict:
    """Train on all data through train_end_date. Calibrate per-horizon
    conformal buffer from the LAST walk-forward CV fold's residuals
    (so we don't sacrifice a whole year of training data)."""
    input_path = input_path or DEFAULT_INPUT
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])

    train_cutoff = pd.Timestamp(train_end_date)
    df = df[df["date"] <= train_cutoff].reset_index(drop=True)

    log.info("horizon_aware_train.loaded",
             rows=len(df),
             date_range=f"{df['date'].min().date()} to {df['date'].max().date()}")

    X_all, y_all, cats = prepare_xy(df)
    feature_cols = list(X_all.columns)

    # Walk-forward CV — we need the LAST fold's residuals for per-horizon calibration
    splits = walk_forward_splits(df, n_folds=5)

    trained_models: dict[str, lgb.LGBMRegressor] = {}
    calib_records: list[dict] = []

    for name, alpha in QUANTILES.items():
        params = {**PARAMS, "alpha": alpha}

        # Train on the last fold to get calibration predictions
        _, last_va_idx = splits[-1]
        m_calib = lgb.LGBMRegressor(**params)
        tr_idx_all_but_last = pd.Index([])
        for tr_idx, _ in splits:
            tr_idx_all_but_last = tr_idx_all_but_last.union(tr_idx)
        # Actually just use all indices EXCEPT the last validation fold
        last_va_set = set(last_va_idx.tolist())
        calib_train_idx = df.index[~df.index.isin(last_va_set)]
        m_calib.fit(
            X_all.loc[calib_train_idx], y_all.loc[calib_train_idx],
            eval_set=[(X_all.loc[last_va_idx], y_all.loc[last_va_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
            categorical_feature=cats,
        )
        calib_preds = m_calib.predict(X_all.loc[last_va_idx])
        for idx_val, pred in zip(last_va_idx, calib_preds):
            calib_records.append({
                "horizon": int(df.at[idx_val, "horizon_days"]),
                "actual": float(df.at[idx_val, "cover_count"]),
                "quantile": name,
                "predicted": float(pred),
            })

        # Final model on ALL data (no hold-out — maximise training)
        m = lgb.LGBMRegressor(**params)
        m.fit(X_all, y_all, categorical_feature=cats)
        trained_models[name] = m

        pkl_path = model_dir / f"lgbm_horizon_q{int(alpha*100):02d}.pkl"
        txt_path = model_dir / f"lgbm_horizon_q{int(alpha*100):02d}.txt"
        with open(pkl_path, "wb") as fh:
            pickle.dump(m, fh)
        m.booster_.save_model(str(txt_path))
        log.info("horizon_aware_train.model_done", name=name, alpha=alpha)

    (model_dir / "horizon_aware_feature_names.json").write_text(
        json.dumps(feature_cols, indent=2)
    )

    # Per-horizon conformal buffer from last-fold calibration residuals
    calib_df = pd.DataFrame(calib_records)
    pivoted = calib_df.pivot_table(
        index=["horizon", "actual"],
        columns="quantile", values="predicted",
    ).reset_index()

    per_horizon_buffer: dict[int, float] = {}
    raw_cov: dict[int, float] = {}
    cal_cov: dict[int, float] = {}
    cal_width: dict[int, float] = {}
    per_horizon_wape: dict[int, float] = {}

    for h in sorted(pivoted["horizon"].unique()):
        seg = pivoted[pivoted["horizon"] == h]
        lower_vals = seg["lower"].values
        upper_vals = seg["upper"].values
        actual_vals = seg["actual"].values
        conformity = np.maximum(lower_vals - actual_vals, actual_vals - upper_vals)
        buf = max(0.0, float(np.quantile(conformity, TARGET_COVERAGE)))
        per_horizon_buffer[int(h)] = buf

        raw_cov[int(h)] = float(
            ((actual_vals >= lower_vals) & (actual_vals <= upper_vals)).mean()
        )
        cal_lower = lower_vals - buf
        cal_upper = upper_vals + buf
        cal_cov[int(h)] = float(
            ((actual_vals >= cal_lower) & (actual_vals <= cal_upper)).mean()
        )
        cal_width[int(h)] = float((cal_upper - cal_lower).mean())
        per_horizon_wape[int(h)] = float(
            np.abs(seg["median"].values - actual_vals).sum()
            / max(np.abs(actual_vals).sum(), 1e-8)
        )
        log.info("horizon_aware_train.calibration",
                 h=int(h), raw_cov=f"{raw_cov[int(h)]:.1%}",
                 cal_cov=f"{cal_cov[int(h)]:.1%}",
                 buffer=f"{buf:.0f}", width=f"{cal_width[int(h)]:.0f}",
                 wape=f"{per_horizon_wape[int(h)]:.1%}",
                 n=len(seg))

    metrics = {
        "target_coverage": TARGET_COVERAGE,
        "train_end_date": train_end_date,
        "per_horizon_buffer": per_horizon_buffer,
        "raw_coverage_per_horizon": raw_cov,
        "calibrated_coverage_per_horizon": cal_cov,
        "interval_width_per_horizon": cal_width,
        "median_wape_per_horizon": per_horizon_wape,
        "n_training_rows": len(df),
        "n_features": len(feature_cols),
    }
    (model_dir / "horizon_aware_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    log.info("horizon_aware_train.done",
             mean_cal_cov=f"{np.mean(list(cal_cov.values())):.1%}",
             widths_per_horizon={k: f"{v:.0f}" for k, v in cal_width.items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    args = parser.parse_args()
    train(args.input, args.model_dir, args.train_end)


if __name__ == "__main__":
    main()
