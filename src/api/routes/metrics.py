"""Model performance metrics endpoints."""

from fastapi import APIRouter

from src.signals.registry import all_sources

router = APIRouter()


@router.get("/metrics/signals")
def signal_health() -> list[dict]:
    """Health status of all registered signal sources."""
    sources = all_sources()
    return [s.health() for s in sources.values()]


@router.get("/metrics/model")
def model_metrics() -> dict:
    """Placeholder for model performance metrics (populated by Layer 2 evaluator)."""
    return {
        "status": "awaiting_training",
        "note": "Run the training pipeline to populate metrics.",
    }
