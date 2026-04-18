"""Train the London Soho demand forecasting model.

Loads the training parquet produced by src.data.build_training_set,
runs walk-forward CV, trains the final LightGBM quantile (median) model,
computes residual-based prediction intervals from the last fold, saves
artefacts and reports metrics.

Usage:
    python -m src.models.train_london
    python -m src.models.train_london --training-path data/training/training_features.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.models.trainer import CATEGORICAL_FEATURES, train_model, walk_forward_split

log = structlog.get_logger(__name__)

DEFAULT_TRAINING_PATH = Path("data/training/training_features.parquet")
DEFAULT_MODEL_DIR = Path("data/models")


def train(
    training_path: Path | None = None,
    model_dir: Path | None = None,
    coverage: float = 0.80,
) -> dict:
    """Full London training workflow."""
    training_path = training_path or DEFAULT_TRAINING_PATH
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    if not training_path.exists():
        raise FileNotFoundError(
            f"Training parquet not found at {training_path}. "
            f"Run `python -m src.data.build_training_set` first."
        )

    log.info("train.start", path=str(training_path))
    df = pd.read_parquet(training_path)
    log.info("train.loaded", rows=len(df), cols=len(df.columns))

    if "cover_count" not in df.columns:
        raise RuntimeError(
            "Training data missing 'cover_count' column (demand target)."
        )

    # 1. Train LightGBM quantile (median) model with walk-forward CV
    model, feature_names, train_metrics = train_model(df, target="cover_count")

    # 2. Residual-based prediction intervals from the last CV fold
    splits = walk_forward_split(df)
    if not splits:
        raise RuntimeError("walk_forward_split produced no folds")
    _, calib_idx = splits[-1]

    X_calib = df.loc[calib_idx, list(feature_names)].copy()
    y_calib = df.loc[calib_idx, "cover_count"].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X_calib.columns:
            X_calib[c] = X_calib[c].astype("category")

    y_pred_calib = model.predict(X_calib)
    residuals = y_calib.values - y_pred_calib
    alpha = (1 - coverage) / 2
    lower_q = float(np.quantile(residuals, alpha))
    upper_q = float(np.quantile(residuals, 1 - alpha))

    # Empirical coverage check on the calib fold
    lower = y_pred_calib + lower_q
    upper = y_pred_calib + upper_q
    hits = ((y_calib.values >= lower) & (y_calib.values <= upper)).mean()
    mae = float(np.mean(np.abs(y_calib.values - y_pred_calib)))
    wape_calib = float(
        np.sum(np.abs(y_calib.values - y_pred_calib))
        / max(np.sum(np.abs(y_calib.values)), 1e-8)
    )

    # 3. Save artefacts
    model_path = model_dir / "lgbm_quantile.pkl"
    features_path = model_dir / "feature_names.json"
    metrics_path = model_dir / "training_metrics.json"

    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    features_path.write_text(json.dumps(feature_names, indent=2))

    metrics = {
        **train_metrics,
        "coverage_target": coverage,
        "coverage_empirical_calib": float(hits),
        "mae_calib": mae,
        "wape_calib": wape_calib,
        "residual_lower_q": lower_q,
        "residual_upper_q": upper_q,
        "n_calib": int(len(X_calib)),
        "model_path": str(model_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    log.info(
        "train.done",
        wape_cv_mean=metrics["cv_wape_mean"],
        wape_cv_std=metrics["cv_wape_std"],
        wape_calib=metrics["wape_calib"],
        mae_calib=metrics["mae_calib"],
        empirical_coverage=metrics["coverage_empirical_calib"],
        n_features=metrics["n_features"],
        n_samples=metrics["n_samples"],
        model_dir=str(model_dir),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train London Soho demand model")
    parser.add_argument("--training-path", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--coverage", type=float, default=0.80)
    args = parser.parse_args()

    train(
        training_path=args.training_path,
        model_dir=args.model_dir,
        coverage=args.coverage,
    )


if __name__ == "__main__":
    main()
