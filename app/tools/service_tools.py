from typing import Any
from pydantic import BaseModel, Field
from app.providers.base import HomeServiceProvider
from app.tools.base import Tool


# 1. CheckHomeServiceAvailabilityTool
class CheckHomeServiceAvailabilityInput(BaseModel):
    service_name: str = "Deep Cleaning"
    package: str = "Standard"
    service_location: str
    preferred_date: str = "Today"


class CheckHomeServiceAvailabilityTool(Tool):
    name = "check_service_availability"
    description = "Checks available appointment slots and pricing for home services."
    input_schema = CheckHomeServiceAvailabilityInput

    def __init__(self, provider: HomeServiceProvider):
        self.provider = provider

    async def execute(self, input_data: CheckHomeServiceAvailabilityInput) -> dict[str, Any]:
        availability = await self.provider.check_availability(
            service_name=input_data.service_name,
            package=input_data.package,
            service_location=input_data.service_location,
            preferred_date=input_data.preferred_date,
        )
        return availability.model_dump()


# 2. BookHomeServiceTool
class BookHomeServiceInput(BaseModel):
    service_name: str
    package: str
    service_location: str
    preferred_date: str = "Today"
    slot_label: str
    price: float


class BookHomeServiceTool(Tool):
    name = "book_home_service"
    description = "Executes home service booking after user selects slot and confirms."
    input_schema = BookHomeServiceInput

    def __init__(self, provider: HomeServiceProvider):
        self.provider = provider

    async def execute(self, input_data: BookHomeServiceInput) -> dict[str, Any]:
        booking = await self.provider.book_service(
            service_name=input_data.service_name,
            package=input_data.package,
            service_location=input_data.service_location,
            preferred_date=input_data.preferred_date,
            slot_label=input_data.slot_label,
            price=input_data.price,
        )
        return booking.model_dump()
