from typing import Any, List, Optional
from pydantic import BaseModel, Field
from app.providers.base import TrainProvider
from app.tools.base import Tool


# 1. Search Trains Tool
class SearchTrainsInput(BaseModel):
    origin: str
    destination: str
    travel_date: str
    departure_window: Optional[str] = None
    class_preference: Optional[str] = None


class SearchTrainsTool(Tool):
    name = "search_trains"
    description = "Searches available Indian Railway train services between stations for a travel date."
    input_schema = SearchTrainsInput

    def __init__(self, provider: TrainProvider):
        self.provider = provider

    async def execute(self, input_data: SearchTrainsInput) -> dict[str, Any]:
        results = await self.provider.search_trains(
            origin=input_data.origin,
            destination=input_data.destination,
            travel_date=input_data.travel_date,
            departure_window=input_data.departure_window,
            class_preference=input_data.class_preference,
        )
        return {"trains": [t.model_dump() for t in results]}


# 2. Book Train Tool
class BookTrainInput(BaseModel):
    train_number: str
    origin_station: str
    destination_station: str
    travel_date: str
    selected_class: str = "3A"
    passenger_count: int = 1
    passenger_names: Optional[List[str]] = None
    berth_preference: Optional[str] = None
    fare_amount: Optional[float] = None
    idempotency_key: Optional[str] = None


class BookTrainTool(Tool):
    name = "book_train"
    description = "Executes Indian Railways train ticket booking with 10-digit PNR and SMS notification."
    input_schema = BookTrainInput

    def __init__(self, provider: TrainProvider):
        self.provider = provider

    async def execute(self, input_data: BookTrainInput) -> dict[str, Any]:
        booking = await self.provider.book_train(
            train_number=input_data.train_number,
            origin_station=input_data.origin_station,
            destination_station=input_data.destination_station,
            travel_date=input_data.travel_date,
            selected_class=input_data.selected_class,
            passenger_count=input_data.passenger_count,
            passenger_names=input_data.passenger_names,
            berth_preference=input_data.berth_preference,
            fare_amount=input_data.fare_amount,
            idempotency_key=input_data.idempotency_key,
        )
        return booking.model_dump()


# 3. Cancel Train Tool
class CancelTrainInput(BaseModel):
    booking_id: str


class CancelTrainTool(Tool):
    name = "cancel_train_booking"
    description = "Cancels a booked train ticket via PNR/booking ID with standard railway refund rules."
    input_schema = CancelTrainInput

    def __init__(self, provider: TrainProvider):
        self.provider = provider

    async def execute(self, input_data: CancelTrainInput) -> dict[str, Any]:
        return await self.provider.cancel_train_booking(input_data.booking_id)
