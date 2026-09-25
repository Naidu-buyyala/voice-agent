from typing import Literal, Optional
from pydantic import BaseModel, Field
from app.domain.location import Location


class RideRequest(BaseModel):
    pickup: Location
    destination: Location
    ride_type: Optional[str] = Field(default="Uber Go", description="Requested ride tier")


class RideOption(BaseModel):
    option_id: str
    name: str
    capacity: int = 4
    eta_minutes: int = 5


class RideEstimate(BaseModel):
    currency: str = "INR"
    amount: float
    surge_multiplier: float = 1.0


class RideBooking(BaseModel):
    booking_id: str
    provider: str
    status: Literal["CONFIRMED", "PROCESSING", "FAILED", "CANCELLED"]
    pickup: Location
    destination: Location
    fare: RideEstimate
    driver_name: Optional[str] = None
    vehicle_plate: Optional[str] = None
    driver_phone: Optional[str] = Field(default="+91 98765 43210", description="Driver contact phone number")
    eta_minutes: Optional[int] = Field(default=4, description="Estimated minutes until arrival")
