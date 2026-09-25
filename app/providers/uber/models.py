from typing import List, Optional
from pydantic import BaseModel, Field


class UberProduct(BaseModel):
    product_id: str
    display_name: str
    description: Optional[str] = None
    capacity: int = 4
    image: Optional[str] = None


class UberPriceEstimate(BaseModel):
    product_id: str
    currency_code: str = "INR"
    display_name: str
    estimate: Optional[str] = None
    low_estimate: Optional[float] = None
    high_estimate: Optional[float] = None
    surge_multiplier: float = 1.0
    duration: int = 900
    distance: float = 12.5


class UberRideCreatePayload(BaseModel):
    product_id: str
    start_latitude: float
    start_longitude: float
    end_latitude: float
    end_longitude: float
    fare_id: Optional[str] = None


class UberRideResponse(BaseModel):
    request_id: str
    status: str  # processing, accepted, arriving, in_progress, completed, rider_canceled
    product_id: str
    driver: Optional[dict] = None
    vehicle: Optional[dict] = None
    eta: Optional[int] = None
