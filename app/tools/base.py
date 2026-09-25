from abc import ABC, abstractmethod
from typing import Any, Type
from pydantic import BaseModel


class Tool(ABC):
    """
    Abstract Action Tool base class.
    Each tool has a validated input schema and a single execution method.
    """
    name: str
    description: str
    input_schema: Type[BaseModel]

    @abstractmethod
    async def execute(self, input_data: BaseModel) -> dict[str, Any]:
        pass
