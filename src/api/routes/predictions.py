"""Prediction endpoints — serve pre-computed forecasts from the DB."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.engine import get_session
from src.db.models import Prediction

router = APIRouter()


@router.get("/locations/{location_id}/predictions")
def get_predictions(
    location_id: str,
    granularity: str = Query(default="hourly", regex="^(hourly|daily)$"),
    channel: str = Query(default="dine_in"),
    days_ahead: int = Query(default=7, ge=1, le=14),
    session: Session = Depends(get_session),
) -> list[dict]:
    now = datetime.now(timezone.utc)
    stmt = (
        select(Prediction)
        .where(Prediction.restaurant_id == location_id)
        .where(Prediction.granularity == granularity)
        .where(Prediction.channel == channel)
        .where(Prediction.target_timestamp_utc >= now)
        .order_by(Prediction.target_timestamp_utc)
        .limit(days_ahead * (24 if granularity == "hourly" else 1))
    )
    preds = session.execute(stmt).scalars().all()
    return [
        {
            "target_timestamp_utc": p.target_timestamp_utc.isoformat(),
            "point_forecast": p.point_forecast,
            "lower_80": p.lower_80,
            "upper_80": p.upper_80,
            "confidence_score": p.confidence_score,
            "capacity_constrained": p.capacity_constrained,
            "model_version": p.model_version,
            "feature_contributions": p.feature_contributions,
        }
        for p in preds
    ]


@router.get("/locations/{location_id}/predictions/today")
def get_today_predictions(
    location_id: str,
    channel: str = Query(default="dine_in"),
    session: Session = Depends(get_session),
) -> list[dict]:
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start.replace(hour=23, minute=59, second=59)
    stmt = (
        select(Prediction)
        .where(Prediction.restaurant_id == location_id)
        .where(Prediction.granularity == "hourly")
        .where(Prediction.channel == channel)
        .where(Prediction.target_timestamp_utc >= today_start)
        .where(Prediction.target_timestamp_utc <= today_end)
        .order_by(Prediction.target_timestamp_utc)
    )
    preds = session.execute(stmt).scalars().all()
    return [
        {
            "hour": p.target_timestamp_utc.hour,
            "point_forecast": p.point_forecast,
            "lower_80": p.lower_80,
            "upper_80": p.upper_80,
            "confidence_score": p.confidence_score,
            "capacity_constrained": p.capacity_constrained,
            "feature_contributions": p.feature_contributions,
        }
        for p in preds
    ]
