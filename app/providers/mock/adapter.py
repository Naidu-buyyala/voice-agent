import uuid
from typing import List
from app.domain.location import Location
from app.domain.ride import RideRequest, RideOption, RideEstimate, RideBooking
from app.providers.base import RideProvider


class MockRideProvider(RideProvider):
    """
    Deterministic mock ride provider for local verification and testing.
    Allows testing the entire action flow before calling external Uber APIs.
    """

    async def get_ride_options(
        self, pickup: Location, destination: Location
    ) -> List[RideOption]:
        return [
            RideOption(option_id="mock_go", name="Uber Go", capacity=4, eta_minutes=4),
            RideOption(option_id="mock_moto", name="Uber Moto", capacity=1, eta_minutes=2),
            RideOption(option_id="mock_auto", name="Uber Auto", capacity=3, eta_minutes=3),
            RideOption(option_id="mock_premier", name="Uber Premier", capacity=4, eta_minutes=6),
            RideOption(option_id="mock_xl", name="Uber XL", capacity=6, eta_minutes=8),
            RideOption(option_id="mock_connect", name="Uber Connect", capacity=0, eta_minutes=5),
        ]

    async def get_estimate(
        self, ride_request: RideRequest, selected_option: RideOption
    ) -> RideEstimate:
        # Predictable deterministic fares across tiers
        fares = {
            "mock_go": 450.0,
            "mock_moto": 120.0,
            "mock_auto": 220.0,
            "mock_premier": 650.0,
            "mock_xl": 850.0,
            "mock_connect": 180.0,
        }
        name_lower = selected_option.name.lower()
        if "moto" in name_lower or "bike" in name_lower:
            amount = fares["mock_moto"]
        elif "auto" in name_lower:
            amount = fares["mock_auto"]
        elif "premier" in name_lower:
            amount = fares["mock_premier"]
        elif "xl" in name_lower:
            amount = fares["mock_xl"]
        elif "connect" in name_lower or "parcel" in name_lower or "package" in name_lower:
            amount = fares["mock_connect"]
        else:
            amount = fares.get(selected_option.option_id, 450.0)

        return RideEstimate(currency="INR", amount=amount, surge_multiplier=1.0)

    async def book_ride(
        self,
        ride_request: RideRequest,
        selected_option: RideOption,
        estimate: RideEstimate,
    ) -> RideBooking:
        booking_id = f"mock_booking_{uuid.uuid4().hex[:8]}"
        return RideBooking(
            booking_id=booking_id,
            provider="mock",
            status="CONFIRMED",
            pickup=ride_request.pickup,
            destination=ride_request.destination,
            fare=estimate,
            driver_name="Rajesh K.",
            vehicle_plate="TS 09 UB 1234",
            driver_phone="+91 98765 43210",
            eta_minutes=4,
        )
