from typing import Any
from app.tools.base import Tool
from app.tools.ride_tools import (
    GetRideOptionsTool,
    GetRideEstimateTool,
    BookRideTool,
)
from app.tools.service_tools import (
    CheckHomeServiceAvailabilityTool,
    BookHomeServiceTool,
)
from app.tools.food_tools import PlaceFoodOrderTool
from app.tools.bus_tools import (
    SearchBusesTool,
    BookBusTool,
    CancelBusTool,
)
from app.tools.train_tools import (
    SearchTrainsTool,
    BookTrainTool,
    CancelTrainTool,
)
from app.providers.factory import (
    get_ride_provider,
    get_home_service_provider,
    get_food_provider,
    get_bus_provider,
    get_train_provider,
)


class ToolExecutor:
    """
    Controlled tool executor.
    Does NOT allow execution of arbitrary tools.
    Validates input using Pydantic input schemas before running.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        ride_provider = get_ride_provider()
        self.register(GetRideOptionsTool(ride_provider))
        self.register(GetRideEstimateTool(ride_provider))
        self.register(BookRideTool(ride_provider))

        home_provider = get_home_service_provider()
        self.register(CheckHomeServiceAvailabilityTool(home_provider))
        self.register(BookHomeServiceTool(home_provider))

        food_provider = get_food_provider()
        self.register(PlaceFoodOrderTool(food_provider))

        bus_provider = get_bus_provider()
        self.register(SearchBusesTool(bus_provider))
        self.register(BookBusTool(bus_provider))
        self.register(CancelBusTool(bus_provider))

        train_provider = get_train_provider()
        self.register(SearchTrainsTool(train_provider))
        self.register(BookTrainTool(train_provider))
        self.register(CancelTrainTool(train_provider))

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    async def execute(self, tool_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not authorized or registered.")

        tool = self._tools[tool_name]
        validated_input = tool.input_schema.model_validate(input_data)
        return await tool.execute(validated_input)


# Singleton executor
tool_executor = ToolExecutor()

