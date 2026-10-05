from typing import Any, Optional
from pydantic import BaseModel, Field
from app.providers.base import FoodProvider
from app.tools.base import Tool


class PlaceFoodOrderInput(BaseModel):
    food: str
    restaurant: str
    delivery_location: str
    estimated_total: float = 350.0
    idempotency_key: Optional[str] = None


class PlaceFoodOrderTool(Tool):
    name = "place_food_order"
    description = "Places food order with provider after user confirmation."
    input_schema = PlaceFoodOrderInput

    def __init__(self, provider: FoodProvider):
        self.provider = provider

    async def execute(self, input_data: PlaceFoodOrderInput) -> dict[str, Any]:
        result = await self.provider.place_order(
            food=input_data.food,
            restaurant=input_data.restaurant,
            delivery_location=input_data.delivery_location,
            estimated_total=input_data.estimated_total,
            idempotency_key=input_data.idempotency_key,
        )
        return result.model_dump()
