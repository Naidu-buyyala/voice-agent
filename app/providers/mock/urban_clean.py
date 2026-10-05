import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from app.providers.base import (
    HomeServiceProvider,
    ServiceAvailability,
    ServiceSlot,
    ServiceBookingResult,
)


class UrbanCleanAdapter(HomeServiceProvider):
    """
    Adapter for Urban Clean Home Services.
    Manages package pricing, slot availability lookup, slot reservation validation,
    and verified booking execution.
    """

    PACKAGE_PRICING: Dict[str, float] = {
        "standard": 1499.0,
        "standard cleaning": 1499.0,
        "premium": 2499.0,
        "premium deep cleaning": 2499.0,
        "full home": 3499.0,
        "full home deep cleaning": 3499.0,
        "deep cleaning": 1499.0,
        "home cleaning": 1499.0,
    }

    def __init__(self, unavailable_slots: Optional[List[str]] = None):
        # Allow simulating slot race conditions or unavailability dynamically
        self.unavailable_slots = unavailable_slots or []

    def get_price(self, package: str) -> float:
        pkg_lower = package.lower().strip()
        for k, v in self.PACKAGE_PRICING.items():
            if k in pkg_lower or pkg_lower in k:
                return v
        return 1499.0

    async def check_availability(
        self,
        service_name: str,
        package: str,
        service_location: str,
        preferred_date: str = "Today",
    ) -> ServiceAvailability:
        all_slots = [
            ServiceSlot(slot_id="slot_1", label="10:00 AM – 12:00 PM", is_available=True),
            ServiceSlot(slot_id="slot_2", label="12:00 PM – 2:00 PM", is_available=True),
            ServiceSlot(slot_id="slot_3", label="2:00 PM – 4:00 PM", is_available=True),
            ServiceSlot(slot_id="slot_4", label="4:00 PM – 6:00 PM", is_available=True),
        ]

        # Filter out unavailable slots if configured for dynamic slot changes
        available = []
        for s in all_slots:
            if s.label not in self.unavailable_slots and s.slot_id not in self.unavailable_slots:
                available.append(s)

        price = self.get_price(package)

        return ServiceAvailability(
            service_name=service_name,
            package=package,
            service_location=service_location,
            date=preferred_date,
            available_slots=available,
            pricing={package: price},
        )

    async def book_service(
        self,
        service_name: str,
        package: str,
        service_location: str,
        preferred_date: str,
        slot_label: str,
        price: float,
    ) -> ServiceBookingResult:
        # Check slot availability at booking time (prevent race condition)
        if slot_label in self.unavailable_slots:
            raise ValueError(f"Slot '{slot_label}' is no longer available.")

        booking_id = f"uc_{uuid.uuid4().hex[:8]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        return ServiceBookingResult(
            booking_id=booking_id,
            service_name=service_name,
            package=package,
            service_location=service_location,
            date=preferred_date,
            slot=slot_label,
            price=price,
            status="CONFIRMED",
            assigned_team="Urban Clean Professional Team #4",
            team_lead_phone="+91 98765 88990",
            created_at=now_iso,
            simulation_enabled=True,
        )
