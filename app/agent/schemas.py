from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field


class EntityData(BaseModel):
    """Normalized entities across rides, food delivery, home services, bus and train booking."""
    # Ride / Travel entities
    pickup: Optional[str] = Field(default=None, description="Pickup address, station, landmark or locality")
    destination: Optional[str] = Field(default=None, description="Drop-off/destination location or landmark")
    ride_type: Optional[str] = Field(default=None, description="Specific ride tier (e.g., 'Uber Go', 'Uber Moto', 'Uber Auto', 'Uber Premier')")
    service_type: Optional[str] = Field(default=None, description="Service category (e.g. 'cab', 'bike', 'auto', 'parcel', 'food', 'cleaning', 'bus', 'train')")

    # Intercity Bus / Train travel entities
    travel_mode: Optional[str] = Field(default=None, description="'bus' or 'train'")
    origin_city: Optional[str] = Field(default=None, description="Travel origin city or departure station (e.g. 'Hyderabad', 'Secunderabad')")
    destination_city: Optional[str] = Field(default=None, description="Travel destination city or arrival station (e.g. 'Visakhapatnam', 'Tirupati')")
    travel_date: Optional[str] = Field(default=None, description="Resolved departure date (e.g. '2026-10-04' or 'Tomorrow')")
    departure_window: Optional[str] = Field(default=None, description="Preferred departure window (e.g. 'evening', 'after 8 PM', 'morning')")
    passenger_count: Optional[int] = Field(default=None, description="Number of traveling passengers")
    passenger_names: Optional[List[str]] = Field(default=None, description="List of passenger full names")
    bus_type: Optional[str] = Field(default=None, description="Bus type preference (e.g. 'AC Sleeper', 'Volvo Multi-Axle', 'Non-AC')")
    train_class: Optional[str] = Field(default=None, description="Train class code or name (e.g. '3A', '2A', 'SL', 'CC')")
    selected_service_id: Optional[str] = Field(default=None, description="Chosen bus service ID or train number (e.g. '12728')")
    selected_seats: Optional[List[str]] = Field(default=None, description="Selected bus seat labels or berth preference (e.g. ['U1', 'U2'])")
    boarding_point: Optional[str] = Field(default=None, description="Specific bus boarding point (e.g. 'Ameerpet', 'MGBS')")
    dropping_point: Optional[str] = Field(default=None, description="Specific bus dropping point (e.g. 'RTC Complex', 'Gajuwaka')")

    # Food Delivery entities
    food: Optional[str] = Field(default=None, description="Specific dish, cuisine, or meal requested (e.g. 'biryani', 'pizza', 'dosa')")
    restaurant: Optional[str] = Field(default=None, description="Preferred restaurant name, or null if unspecified")
    delivery_location: Optional[str] = Field(default=None, description="Delivery address or neighborhood")

    # Home Services entities (Urban Clean, repairs)
    service_name: Optional[str] = Field(default=None, description="Requested home service (e.g. 'deep cleaning', 'plumbing', 'AC repair')")
    package: Optional[str] = Field(default=None, description="Service package tier (e.g. 'Standard', 'Premium', 'Full Home')")
    service_location: Optional[str] = Field(default=None, description="Address or area for home service")
    preferred_date: Optional[str] = Field(default=None, description="Preferred date for home service (e.g. 'Today', 'Tomorrow')")
    slot: Optional[str] = Field(default=None, description="Selected appointment slot (e.g. '2:00 PM – 4:00 PM')")


class PreferenceData(BaseModel):
    """User preferences, explicit choices, and decision delegation."""
    selection_preference: Optional[Literal["AI_CHOOSE", "USER_SPECIFIED", "NO_PREFERENCE"]] = Field(
        default=None,
        description="Whether user explicitly chooses ('USER_SPECIFIED'), delegates choice to the AI ('AI_CHOOSE' for phrases like 'your wish', 'you decide', 'anything is fine', 'whatever is good'), or has no preference ('NO_PREFERENCE').",
    )
    selection_strategy: Optional[Literal["TOP_RATED", "CHEAPEST", "FASTEST", "NEAREST", "BEST_AVAILABLE", "RECOMMENDED"]] = Field(
        default=None,
        description="Strategy criteria when user expresses a preference or delegation: 'TOP_RATED', 'CHEAPEST' (lowest price/fare), 'FASTEST' (lowest ETA), 'NEAREST', 'BEST_AVAILABLE', or 'RECOMMENDED'.",
    )


class AgentExtraction(BaseModel):
    """
    Rich structured reasoning returned by Gemini for intent, entities,
    contextual preferences, delegation, missing fields, and conversational response.
    Gemini = Understand & Reason.
    Backend = Maintain State & Execute Providers.
    """
    intent: Literal[
        "BOOK_RIDE",
        "ORDER_FOOD",
        "BOOK_SERVICE",
        "BOOK_BUS",
        "BOOK_TRAIN",
        "TASK_INQUIRY",
        "GRATITUDE_OR_CLOSING",
        "GREETING",
        "GENERAL_SUPPORT",
        "UNKNOWN",
    ] = Field(
        default="UNKNOWN",
        description="The primary user intent interpreted in the context of the entire conversation.",
    )
    entities: EntityData = Field(
        default_factory=EntityData,
        description="Entities extracted, refined, or remembered across conversation turns.",
    )
    preferences: PreferenceData = Field(
        default_factory=PreferenceData,
        description="User preferences, criteria, and delegation state.",
    )
    confirmation: Optional[bool] = Field(
        default=None,
        description="True if user affirms/accepts (CONFIRM), False if user rejects/cancels (REJECT), or null.",
    )
    confirmation_type: Optional[Literal["CONFIRM", "REJECT", "CHANGE_SELECTION"]] = Field(
        default=None,
        description="'CONFIRM' if user affirms/accepts, 'REJECT' if user cancels/declines, 'CHANGE_SELECTION' if modifying selection, or null.",
    )
    missing_required_fields: List[str] = Field(
        default_factory=list,
        description="List of fields that are genuinely missing and strictly required before action execution.",
    )
    next_action: Literal[
        "ASK_MISSING_INFORMATION",
        "CONFIRM_ACTION",
        "EXECUTE_ACTION",
        "ANSWER_INQUIRY",
        "GREET_AND_GUIDE",
        "ACKNOWLEDGE_AND_OFFER_HELP",
        "CLARIFY_INPUT",
    ] = Field(
        default="ASK_MISSING_INFORMATION",
        description="The logical next action determined by reasoning over the conversation state.",
    )
    assistant_message: str = Field(
        default="",
        description="Warm, natural, context-aware human response to the user. Never asks for already provided info.",
    )
    needs_landmark_clarification: bool = Field(
        default=False,
        description="Set to true if user provided only a broad area/locality without a specific building, mall, or street.",
    )

    # Convenience properties for backward compatibility
    @property
    def pickup(self) -> Optional[str]:
        return self.entities.pickup

    @pickup.setter
    def pickup(self, val: Optional[str]) -> None:
        self.entities.pickup = val

    @property
    def destination(self) -> Optional[str]:
        return self.entities.destination

    @destination.setter
    def destination(self, val: Optional[str]) -> None:
        self.entities.destination = val

    @property
    def ride_type(self) -> Optional[str]:
        return self.entities.ride_type

    @ride_type.setter
    def ride_type(self, val: Optional[str]) -> None:
        self.entities.ride_type = val

    @property
    def service_type(self) -> Optional[str]:
        return self.entities.service_type

    @service_type.setter
    def service_type(self, val: Optional[str]) -> None:
        self.entities.service_type = val

    @property
    def conversational_reply(self) -> Optional[str]:
        return self.assistant_message or None

    @conversational_reply.setter
    def conversational_reply(self, val: Optional[str]) -> None:
        self.assistant_message = val or ""

    @property
    def is_confirmed(self) -> Optional[bool]:
        """Convenience boolean for existing confirmation checks."""
        if self.confirmation is True or self.confirmation_type == "CONFIRM":
            return True
        elif self.confirmation is False or self.confirmation_type == "REJECT":
            return False
        return None
