from typing import Any
from pydantic import BaseModel, Field
from app.domain.location import Location
from app.domain.ride import RideRequest, RideOption, RideEstimate
from app.providers.base import RideProvider
from app.tools.base import Tool


# 1. GetRideOptionsTool
class GetRideOptionsInput(BaseModel):
    pickup_address: str
    destination_address: str


class GetRideOptionsTool(Tool):
    name = "get_ride_options"
    description = "Retrieves available ride options and tiers between pickup and destination."
    input_schema = GetRideOptionsInput

    def __init__(self, provider: RideProvider):
        self.provider = provider

    async def execute(self, input_data: GetRideOptionsInput) -> dict[str, Any]:
        pickup = Location(address=input_data.pickup_address)
        destination = Location(address=input_data.destination_address)
        options = await self.provider.get_ride_options(pickup, destination)
        return {"options": [opt.model_dump() for opt in options]}


# 2. GetRideEstimateTool
class GetRideEstimateInput(BaseModel):
    pickup_address: str
    destination_address: str
    ride_type: str = Field(default="Uber Go")


class GetRideEstimateTool(Tool):
    name = "get_ride_estimate"
    description = "Calculates price and fare estimate for the requested ride tier."
    input_schema = GetRideEstimateInput

    def __init__(self, provider: RideProvider):
        self.provider = provider

    async def execute(self, input_data: GetRideEstimateInput) -> dict[str, Any]:
        pickup = Location(address=input_data.pickup_address)
        destination = Location(address=input_data.destination_address)
        req = RideRequest(pickup=pickup, destination=destination, ride_type=input_data.ride_type)
        
        # Match option
        option = RideOption(option_id="mock_go", name=input_data.ride_type)
        estimate = await self.provider.get_estimate(req, option)
        return {
            "currency": estimate.currency,
            "amount": estimate.amount,
            "surge_multiplier": estimate.surge_multiplier,
        }


# 3. BookRideTool
class BookRideInput(BaseModel):
    pickup_address: str
    destination_address: str
    ride_type: str = "Uber Go"
    fare_amount: float = 450.0


class BookRideTool(Tool):
    name = "book_ride"
    description = "Performs booking with ride provider after user confirmation."
    input_schema = BookRideInput

    def __init__(self, provider: RideProvider):
        self.provider = provider

    async def execute(self, input_data: BookRideInput) -> dict[str, Any]:
        pickup = Location(address=input_data.pickup_address)
        destination = Location(address=input_data.destination_address)
        req = RideRequest(pickup=pickup, destination=destination, ride_type=input_data.ride_type)
        option = RideOption(option_id="mock_go", name=input_data.ride_type)
        estimate = RideEstimate(currency="INR", amount=input_data.fare_amount)

        booking = await self.provider.book_ride(req, option, estimate)
        return booking.model_dump()
