from typing import Literal, Optional
from pydantic import BaseModel, Field


class AgentExtraction(BaseModel):
    """
    Strict schema returned by Gemini for intent and entity understanding.
    LLM = Understand: extracts what user said, without taking actions.
    """
    intent: Literal[
        "BOOK_RIDE",
        "ORDER_FOOD",
        "BOOK_SERVICE",
        "TASK_INQUIRY",
        "GRATITUDE_OR_CLOSING",
        "GREETING",
        "GENERAL_SUPPORT",
        "UNKNOWN",
    ] = Field(
        default="UNKNOWN",
        description="The primary user intent: "
                    "'BOOK_RIDE' (user wants to book/hail a ride, cab, bike, auto, parcel, or gives travel points), "
                    "'ORDER_FOOD' (user wants to order food, meals, groceries), "
                    "'BOOK_SERVICE' (home services like cleaning, urban clean, plumbing, repair), "
                    "'TASK_INQUIRY' (inquiry about driver/vehicle status, phone number, ETA, booking details), "
                    "'GRATITUDE_OR_CLOSING' (thanks, thank you, bye), "
                    "'GREETING' (hi, hello, hey, good morning, greetings), "
                    "'GENERAL_SUPPORT' (asking what apps/services are supported, help), or 'UNKNOWN'.",
    )
    pickup: Optional[str] = Field(
        default=None,
        description="Pickup address, station, landmark or location.",
    )
    destination: Optional[str] = Field(
        default=None,
        description="Drop-off/destination location or landmark.",
    )
    ride_type: Optional[str] = Field(
        default=None,
        description="Desired ride product tier or category (e.g., 'Uber Go', 'Uber Moto', 'Uber Auto', 'Uber Premier', 'Uber XL', 'Uber Connect').",
    )
    service_type: Optional[Literal["cab", "bike", "auto", "parcel"]] = Field(
        default=None,
        description="General vehicle or service category requested: 'cab' (car/taxi), 'bike' (moto/two-wheeler), 'auto' (3-wheeler), 'parcel' (courier/package delivery), or null if not chosen yet.",
    )
    needs_landmark_clarification: bool = Field(
        default=False,
        description="Set to true if user provided only a broad area/locality (e.g., 'Gachibowli', 'Hitech City', 'Madhapur', 'Kondapur') without specifying an exact landmark, building, mall, or street.",
    )
    confirmation: Optional[bool] = Field(
        default=None,
        description="True if user explicitly confirms booking/action (e.g. 'yes', 'confirm', 'book it'). "
                    "False if user rejects/cancels ('no', 'cancel', 'don't'). None if not applicable.",
    )
    conversational_reply: Optional[str] = Field(
        default=None,
        description="A friendly, natural assistant reply to the user (e.g. greeting them back, acknowledging their request, asking for exact building/landmark in broad areas like Gachibowli, or asking if they want a cab, bike, auto, or parcel).",
    )
