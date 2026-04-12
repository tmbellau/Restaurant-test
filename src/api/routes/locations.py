"""Restaurant location endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.db.engine import get_session
from src.db.models import Restaurant

router = APIRouter()


@router.get("/locations")
def list_locations(session: Session = Depends(get_session)) -> list[dict]:
    locations = session.query(Restaurant).all()
    return [
        {
            "id": loc.id,
            "name": loc.name,
            "postcode": loc.postcode,
            "city_tier": loc.city_tier,
            "seating_capacity": loc.seating_capacity,
            "lat": loc.lat,
            "lng": loc.lng,
        }
        for loc in locations
    ]


@router.get("/locations/{location_id}")
def get_location(location_id: str, session: Session = Depends(get_session)) -> dict:
    loc = session.get(Restaurant, location_id)
    if loc is None:
        raise HTTPException(status_code=404, detail="Location not found")
    return {
        "id": loc.id,
        "name": loc.name,
        "address": loc.address,
        "postcode": loc.postcode,
        "lat": loc.lat,
        "lng": loc.lng,
        "timezone": loc.timezone,
        "seating_capacity": loc.seating_capacity,
        "city_tier": loc.city_tier,
        "footfall_zone_class": loc.footfall_zone_class,
    }
