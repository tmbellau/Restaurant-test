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
    train_end_date: str | None = "2023-12-31",
    calibrate_year: int = 2024,
) -> dict:
    """Train on data through train_end_date, calibrate per-horizon conformal
    buffer on a held-out calibration year (default 2024).

    Why this split: CV residuals on training data don't capture the true
    horizon-dependent error distribution because each fold is tiny and the
    model has access to similar patterns at all horizons. Calibrating on a
    full held-out year gives real-world residuals that grow with horizon —
    the buffer comes directly from observation, not a multiplier.
    """
    input_path = input_path or DEFAULT_INPUT
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_parquet(input_path)
    df_all["date"] = pd.to_datetime(df_all["date"])

    train_cutoff = pd.Timestamp(train_end_date)
    calib_start = pd.Timestamp(f"{calibrate_year}-01-01")
    calib_end = pd.Timestamp(f"{calibrate_year}-12-31")

    df_train = df_all[df_all["date"] <= train_cutoff].reset_index(drop=True)
    df_calib = df_all[
        (df_all["date"] >= calib_start) & (df_all["date"] <= calib_end)
    ].reset_index(drop=True)

    log.info("horizon_aware_train.split",
             train_rows=len(df_train),
             train_range=f"{df_train['date'].min().date()} to {df_train['date'].max().date()}",
             calib_rows=len(df_calib),
             calib_range=f"{df_calib['date'].min().date()} to {df_calib['date'].max().date()}")

    X_train, y_train, cats = prepare_xy(df_train)
    X_calib, y_calib, _ = prepare_xy(df_calib)
    feature_cols = list(X_train.columns)

    # Train each quantile model on the full training set (no CV needed —
    # the held-out calibration year provides all the uncertainty info).
    trained_models: dict[str, lgb.LGBMRegressor] = {}
    calib_predictions: dict[str, np.ndarray] = {}

    for name, alpha in QUANTILES.items():
        params = {**PARAMS, "alpha": alpha}
        m = lgb.LGBMRegressor(**params)
        m.fit(X_train, y_train, categorical_feature=cats)
        trained_models[name] = m
        calib_predictions[name] = m.predict(X_calib)

        pkl_path = model_dir / f"lgbm_horizon_q{int(alpha*100):02d}.pkl"
        txt_path = model_dir / f"lgbm_horizon_q{int(alpha*100):02d}.txt"
        with open(pkl_path, "wb") as fh:
            pickle.dump(m, fh)
        m.booster_.save_model(str(txt_path))
        log.info("horizon_aware_train.model_done", name=name, alpha=alpha)

    (model_dir / "horizon_aware_feature_names.json").write_text(
        json.dumps(feature_cols, indent=2)
    )

    # Calibrate per-horizon buffer from held-out residuals
    calib_df = pd.DataFrame({
        "date": df_calib["date"].values,
        "horizon": df_calib["horizon_days"].astype(int).values,
        "actual": y_calib.values,
        "lower": calib_predictions["lower"],
        "median": calib_predictions["median"],
        "upper": calib_predictions["upper"],
    })

    per_horizon_buffer: dict[int, float] = {}
    raw_cov: dict[int, float] = {}
    cal_cov: dict[int, float] = {}
    cal_width: dict[int, float] = {}
    per_horizon_wape: dict[int, float] = {}

    log.info("horizon_aware_train.calibration.header",
             msg="h  raw_cov  cal_cov  buffer  width  WAPE  n")
    for h in sorted(calib_df["horizon"].unique()):
        seg = calib_df[calib_df["horizon"] == h]
        conformity = np.maximum(seg["lower"] - seg["actual"],
                                seg["actual"] - seg["upper"])
        buf = float(np.quantile(conformity, TARGET_COVERAGE))
        per_horizon_buffer[int(h)] = buf

        raw_cov[int(h)] = float(
            ((seg["actual"] >= seg["lower"]) & (seg["actual"] <= seg["upper"])).mean()
        )
        cal_lower = seg["lower"] - buf
        cal_upper = seg["upper"] + buf
        cal_cov[int(h)] = float(
            ((seg["actual"] >= cal_lower) & (seg["actual"] <= cal_upper)).mean()
        )
        cal_width[int(h)] = float((cal_upper - cal_lower).mean())
        per_horizon_wape[int(h)] = float(
            np.abs(seg["median"] - seg["actual"]).sum()
            / max(np.abs(seg["actual"]).sum(), 1e-8)
        )

        log.info("horizon_aware_train.calibration.row",
                 h=int(h), raw_cov=f"{raw_cov[int(h)]:.1%}",
                 cal_cov=f"{cal_cov[int(h)]:.1%}",
                 buffer=f"{buf:.0f}", width=f"{cal_width[int(h)]:.0f}",
                 wape=f"{per_horizon_wape[int(h)]:.1%}",
                 n=len(seg))

    metrics = {
        "target_coverage": TARGET_COVERAGE,
        "train_end_date": train_end_date,
        "calibrate_year": calibrate_year,
        "per_horizon_buffer": per_horizon_buffer,
        "raw_coverage_per_horizon": raw_cov,
        "calibrated_coverage_per_horizon": cal_cov,
        "interval_width_per_horizon": cal_width,
        "median_wape_per_horizon": per_horizon_wape,
        "n_training_rows": len(df_train),
        "n_calibration_rows": len(df_calib),
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
