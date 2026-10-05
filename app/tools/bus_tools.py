from typing import Any, List, Optional
from pydantic import BaseModel, Field
from app.providers.base import BusProvider
from app.tools.base import Tool


# 1. Search Buses Tool
class SearchBusesInput(BaseModel):
    origin: str
    destination: str
    travel_date: str
    departure_window: Optional[str] = None
    bus_type_preference: Optional[str] = None


class SearchBusesTool(Tool):
    name = "search_buses"
    description = "Searches available intercity buses between origin and destination for a travel date."
    input_schema = SearchBusesInput

    def __init__(self, provider: BusProvider):
        self.provider = provider

    async def execute(self, input_data: SearchBusesInput) -> dict[str, Any]:
        results = await self.provider.search_buses(
            origin=input_data.origin,
            destination=input_data.destination,
            travel_date=input_data.travel_date,
            departure_window=input_data.departure_window,
            bus_type_preference=input_data.bus_type_preference,
        )
        return {"buses": [b.model_dump() for b in results]}


# 2. Book Bus Tool
class BookBusInput(BaseModel):
    service_id: str
    origin: str
    destination: str
    travel_date: str
    passenger_count: int = 1
    selected_seats: List[str] = Field(default_factory=list)
    boarding_point: str = "Ameerpet"
    dropping_point: str = "RTC Complex"
    passenger_names: Optional[List[str]] = None
    fare_amount: Optional[float] = None
    idempotency_key: Optional[str] = None


class BookBusTool(Tool):
    name = "book_bus"
    description = "Executes intercity bus booking with verified provider reservation and SMS delivery."
    input_schema = BookBusInput

    def __init__(self, provider: BusProvider):
        self.provider = provider

    async def execute(self, input_data: BookBusInput) -> dict[str, Any]:
        booking = await self.provider.book_bus(
            service_id=input_data.service_id,
            origin=input_data.origin,
            destination=input_data.destination,
            travel_date=input_data.travel_date,
            passenger_count=input_data.passenger_count,
            selected_seats=input_data.selected_seats,
            boarding_point=input_data.boarding_point,
            dropping_point=input_data.dropping_point,
            passenger_names=input_data.passenger_names,
            fare_amount=input_data.fare_amount,
            idempotency_key=input_data.idempotency_key,
        )
        return booking.model_dump()


# 3. Cancel Bus Tool
class CancelBusInput(BaseModel):
    booking_id: str


class CancelBusTool(Tool):
    name = "cancel_bus_booking"
    description = "Cancels a booked bus ticket and initiates refund."
    input_schema = CancelBusInput

    def __init__(self, provider: BusProvider):
        self.provider = provider

    async def execute(self, input_data: CancelBusInput) -> dict[str, Any]:
        return await self.provider.cancel_bus_booking(input_data.booking_id)
