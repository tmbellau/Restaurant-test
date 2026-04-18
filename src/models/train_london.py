"""Train the London Soho demand forecasting model.

Loads the training parquet produced by src.data.build_training_set,
runs walk-forward CV, trains the final LightGBM quantile model, wraps it
with conformal prediction intervals, saves artefacts, and reports metrics.

Usage:
    python -m src.models.train_london
    python -m src.models.train_london --training-path data/training/training_features.parquet
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import structlog

from src.models.conformal import calibrate_cqr
from src.models.trainer import train_model, walk_forward_split

log = structlog.get_logger(__name__)

DEFAULT_TRAINING_PATH = Path("data/training/training_features.parquet")
DEFAULT_MODEL_DIR = Path("data/models")


def train(
    training_path: Path | None = None,
    model_dir: Path | None = None,
    coverage: float = 0.80,
) -> dict:
    """Full London training workflow.

    Returns a metrics dict summarising the run.
    """
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

    # 1. Train the LightGBM quantile model with walk-forward CV
    model, feature_names, train_metrics = train_model(df, target="cover_count")

    # 2. Use the last CV fold as the calibration set for CQR
    splits = walk_forward_split(df)
    if not splits:
        raise RuntimeError("walk_forward_split produced no folds")
    _, calib_idx = splits[-1]

    feature_cols = list(feature_names)
    X_calib = df.loc[calib_idx, feature_cols].copy()
    y_calib = df.loc[calib_idx, "cover_count"].copy()

    # LightGBM needs category dtypes preserved on calibration data
    from src.models.trainer import CATEGORICAL_FEATURES
    for c in CATEGORICAL_FEATURES:
        if c in X_calib.columns:
            X_calib[c] = X_calib[c].astype("category")

    mapie = calibrate_cqr(model, X_calib, y_calib, coverage=coverage)

    # 3. Compute empirical coverage on calibration set
    y_pred, intervals = mapie.predict(X_calib)
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]
    hits = ((y_calib.values >= lower) & (y_calib.values <= upper)).mean()

    # 4. Save artefacts
    model_path = model_dir / "lgbm_quantile.pkl"
    mapie_path = model_dir / "mapie_cqr.pkl"
    features_path = model_dir / "feature_names.json"
    metrics_path = model_dir / "training_metrics.json"

    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    with open(mapie_path, "wb") as fh:
        pickle.dump(mapie, fh)
    features_path.write_text(json.dumps(feature_names, indent=2))

    metrics = {
        **train_metrics,
        "coverage_target": coverage,
        "coverage_empirical": float(hits),
        "n_calib": len(X_calib),
        "model_path": str(model_path),
        "mapie_path": str(mapie_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    log.info(
        "train.done",
        wape_cv_mean=metrics["cv_wape_mean"],
        wape_cv_std=metrics["cv_wape_std"],
        empirical_coverage=metrics["coverage_empirical"],
        n_features=metrics["n_features"],
        model_dir=str(model_dir),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train London Soho demand model")
    parser.add_argument(
        "--training-path",
        type=Path,
        default=None,
        help="Path to training parquet (default: data/training/training_features.parquet)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Output directory for model artefacts (default: data/models/)",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.80,
        help="Prediction interval coverage level (default: 0.80)",
    )
    args = parser.parse_args()

    train(
        training_path=args.training_path,
        model_dir=args.model_dir,
        coverage=args.coverage,
    )


if __name__ == "__main__":
    main()
