"""Evaluation metrics per the spec: WAPE, pinball loss, CRPS, coverage.

MAPE is NOT used anywhere. WAPE handles zeros and is the industry standard.
"""

from __future__ import annotations

import numpy as np


def wape(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """Weighted Absolute Percentage Error. Target < 12% daily on established locations."""
    denom = np.sum(np.abs(actuals))
    if denom < 1e-8:
        return 0.0
    return float(np.sum(np.abs(actuals - forecasts)) / denom)


def pinball_loss(actuals: np.ndarray, forecasts: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss. Proper scoring rule for quantile forecasts."""
    diff = actuals - forecasts
    return float(np.mean(np.where(diff >= 0, tau * diff, (tau - 1) * diff)))


def crps_quantile(
    actuals: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
) -> float:
    """CRPS approximated as 2x mean pinball loss across quantile levels."""
    losses = [pinball_loss(actuals, forecasts, tau) for tau, forecasts in quantile_forecasts.items()]
    return float(2 * np.mean(losses))


def coverage(
    actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Empirical coverage: fraction of actuals inside the prediction interval."""
    inside = (actuals >= lower) & (actuals <= upper)
    return float(np.mean(inside))


def interval_score(
    actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.2
) -> float:
    """Winkler interval score: penalises wide intervals and missed actuals."""
    width = upper - lower
    below = np.maximum(lower - actuals, 0)
    above = np.maximum(actuals - upper, 0)
    return float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))


def compute_all_metrics(
    actuals: np.ndarray,
    point_forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    """Compute the full evaluation suite."""
    return {
        "wape": wape(actuals, point_forecast),
        "pinball_50": pinball_loss(actuals, point_forecast, 0.5),
        "pinball_10": pinball_loss(actuals, lower, 0.1),
        "pinball_90": pinball_loss(actuals, upper, 0.9),
        "coverage_80": coverage(actuals, lower, upper),
        "interval_score": interval_score(actuals, lower, upper, alpha=0.2),
        "rmse": float(np.sqrt(np.mean((actuals - point_forecast) ** 2))),
    }
