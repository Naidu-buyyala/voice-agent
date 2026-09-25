from typing import Optional
from pydantic import BaseModel, Field


class Location(BaseModel):
    """
    Generic provider-independent location entity.
    Never add Uber-specific fields here.
    """
    address: str = Field(..., description="Human-readable address or landmark name")
    latitude: Optional[float] = Field(default=None, description="GPS latitude coordinate")
    longitude: Optional[float] = Field(default=None, description="GPS longitude coordinate")
