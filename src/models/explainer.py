"""SHAP explainer for per-prediction feature contributions.

Per v2 adjustment: managers need "why this forecast?" SHAP waterfall-style
breakdowns showing the top drivers (e.g., "bank holiday +45, weather +30").
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


def compute_shap_contributions(
    model: Any,
    X: pd.DataFrame,
    top_k: int = 8,
) -> list[dict[str, float]]:
    """Compute per-prediction SHAP feature contributions.

    Returns a list of dicts (one per row in X), each mapping the top-k
    feature names to their SHAP values.

    Uses TreeExplainer for LightGBM (fast, exact for tree models).
    """
    try:
        import shap
    except ImportError:
        log.warning("shap.not_installed")
        return [{} for _ in range(len(X))]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    contributions: list[dict[str, float]] = []
    feature_names = list(X.columns)

    for i in range(len(X)):
        row_shap = shap_values[i]
        # Sort by absolute magnitude, take top-k
        indices = np.argsort(np.abs(row_shap))[::-1][:top_k]
        contrib = {feature_names[j]: round(float(row_shap[j]), 2) for j in indices}
        contributions.append(contrib)

    return contributions
