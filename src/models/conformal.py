"""Conformalized Quantile Regression (CQR) via MAPIE.

Per the spec: CQR provides finite-sample coverage guarantees that native
quantile regression lacks (Romano, Patterson & Candès, NeurIPS 2019).

Wraps a trained LightGBM median model and produces calibrated prediction
intervals at the desired coverage level (default 80%).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from mapie.regression import MapieQuantileRegressor

log = structlog.get_logger(__name__)


def calibrate_cqr(
    model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    coverage: float = 0.80,
) -> MapieQuantileRegressor:
    """Wrap a trained LightGBM model with CQR calibration.

    Args:
        model: Trained LGBMRegressor (quantile objective, alpha=0.5).
        X_calib: Calibration set features (held out from training).
        y_calib: Calibration set targets.
        coverage: Desired coverage level (0.80 = 80% prediction interval).

    Returns:
        Calibrated MapieQuantileRegressor that produces intervals.
    """
    alpha = 1 - coverage  # MAPIE uses alpha = 1 - coverage

    mapie = MapieQuantileRegressor(
        estimator=model,
        method="quantile",
        cv="prefit",
        alpha=alpha,
    )
    mapie.fit(X_calib, y_calib)
    log.info("cqr.calibrated", coverage=coverage, n_calib=len(X_calib))
    return mapie


def predict_with_intervals(
    mapie_model: MapieQuantileRegressor,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Produce point forecasts + calibrated intervals.

    Returns:
        (point_forecast, lower_bound, upper_bound, confidence_score)
    """
    y_pred, y_intervals = mapie_model.predict(X)

    # y_intervals shape: (n_samples, 2, 1) -> lower and upper
    lower = y_intervals[:, 0, 0]
    upper = y_intervals[:, 1, 0]

    # Ensure non-negative (covers can't be negative)
    lower = np.maximum(lower, 0)
    upper = np.maximum(upper, 0)
    y_pred = np.maximum(y_pred, 0)

    # Confidence score: 1 - (width / (2 * historical_std))
    # Use interval width relative to prediction as proxy
    width = upper - lower
    # Avoid division by zero; use rolling std from training data as denominator
    # For now approximate with prediction-relative width
    denom = np.maximum(y_pred, 1.0) * 2
    confidence = np.clip(1.0 - (width / denom), 0.0, 1.0)

    return y_pred, lower, upper, confidence
