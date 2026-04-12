"""Batch prediction: load champion model, produce forecasts, write to DB.

Per the spec: pre-compute all forecasts hourly, write to Redis/DB. No inference
in the FastAPI request path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.models.conformal import predict_with_intervals
from src.models.explainer import compute_shap_contributions

log = structlog.get_logger(__name__)


def batch_predict(
    model: Any,
    mapie_model: Any,
    X: pd.DataFrame,
    restaurant_ids: pd.Series,
    target_timestamps: pd.Series,
    granularity: str = "hourly",
    channel: str = "dine_in",
    model_version: str = "v0",
    capacity_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Produce predictions with intervals and SHAP explanations.

    Returns a DataFrame matching the predictions table schema.
    """
    point, lower, upper, confidence = predict_with_intervals(mapie_model, X)
    contributions = compute_shap_contributions(model, X)

    # Capacity constraint flag
    cap_constrained = np.zeros(len(X), dtype=bool)
    if capacity_map:
        for i, rid in enumerate(restaurant_ids):
            cap = capacity_map.get(rid, float("inf"))
            if point[i] > cap:
                cap_constrained[i] = True

    now = datetime.now(timezone.utc)
    return pd.DataFrame({
        "restaurant_id": restaurant_ids.values,
        "target_timestamp_utc": target_timestamps.values,
        "granularity": granularity,
        "channel": channel,
        "point_forecast": np.round(point, 1),
        "lower_80": np.round(lower, 1),
        "upper_80": np.round(upper, 1),
        "confidence_score": np.round(confidence, 3),
        "capacity_constrained": cap_constrained,
        "model_version": model_version,
        "feature_contributions": contributions,
        "created_at_utc": now,
    })
