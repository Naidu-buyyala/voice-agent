from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.domain.location import Location
from app.domain.ride import RideRequest, RideOption, RideEstimate, RideBooking


class RideProvider(ABC):
    """
    Abstract Interface for Ride Providers (Uber, Rapido, Ola, etc.).
    Workflow, tools, and agent logic depend ONLY on this interface.
    """

    @abstractmethod
    async def get_ride_options(
        self, pickup: Location, destination: Location
    ) -> List[RideOption]:
        pass

    @abstractmethod
    async def get_estimate(
        self, ride_request: RideRequest, selected_option: RideOption
    ) -> RideEstimate:
        pass

    @abstractmethod
    async def book_ride(
        self,
        ride_request: RideRequest,
        selected_option: RideOption,
        estimate: RideEstimate,
    ) -> RideBooking:
        pass


# ==============================================================================
# Home Service (Urban Clean) Provider Interface & Models
# ==============================================================================

class ServiceSlot(BaseModel):
    slot_id: str
    label: str  # e.g. "10:00 AM – 12:00 PM", "2:00 PM – 4:00 PM"
    is_available: bool = True


class ServiceAvailability(BaseModel):
    service_name: str
    package: str
    service_location: str
    date: str
    available_slots: List[ServiceSlot]
    pricing: Dict[str, float] = Field(default_factory=dict)


class ServiceBookingResult(BaseModel):
    booking_id: str
    service_name: str
    package: str
    service_location: str
    date: str
    slot: str
    price: float
    status: str
    assigned_team: Optional[str] = None
    team_lead_phone: Optional[str] = None
    created_at: Optional[str] = None
    simulation_enabled: bool = True


class HomeServiceProvider(ABC):
    """
    Abstract Interface for Home Service Providers (Urban Clean, etc.).
    Decouples real provider APIs from deterministic simulation adapters.
    """

    @abstractmethod
    async def check_availability(
        self,
        service_name: str,
        package: str,
        service_location: str,
        preferred_date: str = "Today",
    ) -> ServiceAvailability:
        pass

    @abstractmethod
    async def book_service(
        self,
        service_name: str,
        package: str,
        service_location: str,
        preferred_date: str,
        slot_label: str,
        price: float,
    ) -> ServiceBookingResult:
        pass


# ==============================================================================
# Food Ordering Provider Interface & Models
# ==============================================================================

class FoodOrderResult(BaseModel):
    order_id: str
    booking_id: str
    restaurant: str
    items: List[str]
    delivery_location: str
    delivery_partner: str = "Suresh V."
    delivery_phone: str = "+91 98765 12345"
    initial_eta_minutes: int = 20
    eta_minutes: int = 20
    status: str = "PREPARING"
    created_at: Optional[str] = None
    estimated_total: float = 350.0
    simulation_enabled: bool = True


class FoodProvider(ABC):
    """
    Abstract Interface for Food Delivery Providers (Swiggy, Zomato, Direct Kitchen).
    """

    @abstractmethod
    async def place_order(
        self,
        food: str,
        restaurant: str,
        delivery_location: str,
        estimated_total: float = 350.0,
        idempotency_key: Optional[str] = None,
    ) -> FoodOrderResult:
        pass


# ==============================================================================
# Bus Booking Provider Interface & Models
# ==============================================================================

class BusServiceOption(BaseModel):
    service_id: str
    operator_name: str
    bus_type: str  # e.g. "AC Sleeper (2+1)", "Non-AC Seater", "Volvo Multi-Axle AC"
    departure_time: str  # e.g. "20:30" or "08:30 PM"
    arrival_time: str  # e.g. "06:00" or "06:00 AM"
    duration: str  # e.g. "9h 30m"
    origin: str  # e.g. "Hyderabad (MGBS)"
    destination: str  # e.g. "Visakhapatnam (RTC Complex)"
    boarding_points: List[str] = Field(default_factory=list)
    dropping_points: List[str] = Field(default_factory=list)
    base_fare: float
    total_fare: float
    available_seats_count: int
    available_seats: List[str] = Field(default_factory=list)  # e.g. ["U1", "U2", "L3", "L4"]
    rating: float = 4.5


class BusBookingResult(BaseModel):
    booking_id: str
    pnr: str
    operator_name: str
    bus_type: str
    origin: str
    destination: str
    travel_date: str
    departure_time: str
    arrival_time: str
    boarding_point: str
    dropping_point: str
    passenger_count: int
    passenger_names: List[str] = Field(default_factory=list)
    selected_seats: List[str] = Field(default_factory=list)
    total_fare: float
    booking_status: str = "CONFIRMED"  # CONFIRMED, PAYMENT_PENDING, FAILED, CANCELLED
    created_at: Optional[str] = None
    sms_notification_sent: bool = True
    sms_message: str = ""
    simulation_enabled: bool = True


class BusProvider(ABC):
    """
    Abstract Interface for Intercity Bus Providers (RedBus, AbhiBus, State RTC).
    """

    @abstractmethod
    async def search_buses(
        self,
        origin: str,
        destination: str,
        travel_date: str,
        departure_window: Optional[str] = None,
        bus_type_preference: Optional[str] = None,
    ) -> List[BusServiceOption]:
        pass

    @abstractmethod
    async def get_seat_layout(
        self,
        service_id: str,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def book_bus(
        self,
        service_id: str,
        origin: str,
        destination: str,
        travel_date: str,
        passenger_count: int,
        selected_seats: List[str],
        boarding_point: str,
        dropping_point: str,
        passenger_names: Optional[List[str]] = None,
        fare_amount: Optional[float] = None,
        idempotency_key: Optional[str] = None,
    ) -> BusBookingResult:
        pass

    @abstractmethod
    async def cancel_bus_booking(
        self,
        booking_id: str,
    ) -> Dict[str, Any]:
        pass


# ==============================================================================
# Train Booking Provider Interface & Models
# ==============================================================================

class TrainClassOption(BaseModel):
    class_code: str  # e.g. "SL", "3A", "2A", "1A", "CC", "2S"
    class_name: str  # e.g. "Sleeper", "AC 3 Tier", "AC 2 Tier", "Chair Car"
    fare: float
    availability_status: str  # e.g. "AVAILABLE-24", "RAC 12", "WL 35"
    is_available: bool = True


class TrainServiceOption(BaseModel):
    train_number: str  # e.g. "12728"
    train_name: str  # e.g. "Godavari Express"
    origin_station: str  # e.g. "Secunderabad Junction (SC)"
    destination_station: str  # e.g. "Visakhapatnam Junction (VSKP)"
    departure_time: str  # e.g. "17:05" or "05:05 PM"
    arrival_time: str  # e.g. "05:45" or "05:45 AM"
    duration: str  # e.g. "12h 40m"
    available_classes: List[TrainClassOption] = Field(default_factory=list)


class TrainBookingResult(BaseModel):
    booking_id: str
    pnr: str
    train_number: str
    train_name: str
    origin_station: str
    destination_station: str
    travel_date: str
    departure_time: str
    arrival_time: str
    selected_class: str
    passenger_count: int
    passenger_names: List[str] = Field(default_factory=list)
    allocated_berths: List[str] = Field(default_factory=list)  # e.g. ["B3-21 (Lower)", "B3-24 (Upper)"]
    total_fare: float
    booking_status: str = "CONFIRMED"  # CONFIRMED, RAC, WAITLISTED, PAYMENT_PENDING, FAILED, CANCELLED
    created_at: Optional[str] = None
    sms_notification_sent: bool = True
    sms_message: str = ""
    simulation_enabled: bool = True


class TrainProvider(ABC):
    """
    Abstract Interface for Indian Railways / Train Providers (IRCTC / National Train Enquiry).
    """

    @abstractmethod
    async def search_trains(
        self,
        origin: str,
        destination: str,
        travel_date: str,
        departure_window: Optional[str] = None,
        class_preference: Optional[str] = None,
    ) -> List[TrainServiceOption]:
        pass

    @abstractmethod
    async def book_train(
        self,
        train_number: str,
        origin_station: str,
        destination_station: str,
        travel_date: str,
        selected_class: str,
        passenger_count: int,
        passenger_names: Optional[List[str]] = None,
        berth_preference: Optional[str] = None,
        fare_amount: Optional[float] = None,
        idempotency_key: Optional[str] = None,
    ) -> TrainBookingResult:
        pass

    @abstractmethod
    async def cancel_train_booking(
        self,
        booking_id: str,
    ) -> Dict[str, Any]:
        pass


