from typing import Any
from app.tools.base import Tool
from app.tools.ride_tools import (
    GetRideOptionsTool,
    GetRideEstimateTool,
    BookRideTool,
)
from app.providers.factory import get_ride_provider


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
        provider = get_ride_provider()
        self.register(GetRideOptionsTool(provider))
        self.register(GetRideEstimateTool(provider))
        self.register(BookRideTool(provider))

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
