import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from app.providers.base import (
    BusProvider,
    BusServiceOption,
    BusBookingResult,
)


class MockBusProvider(BusProvider):
    """
    Deterministic mock bus provider with alias resolution, dynamic availability,
    seat reservation, idempotency, and cancellation support.
    """

    CITY_ALIASES: Dict[str, str] = {
        "hyd": "Hyderabad",
        "hyderabad": "Hyderabad",
        "vizag": "Visakhapatnam",
        "visakhapatnam": "Visakhapatnam",
        "blr": "Bengaluru",
        "bangalore": "Bengaluru",
        "bengaluru": "Bengaluru",
        "chennai": "Chennai",
        "maa": "Chennai",
        "vijayawada": "Vijayawada",
        "tirupati": "Tirupati",
        "mumbai": "Mumbai",
        "pune": "Pune",
    }

    SAMPLE_OPERATORS = [
        {"operator": "Orange Tours & Travels", "type": "AC Sleeper (2+1)", "dep": "20:30", "arr": "06:00", "dur": "9h 30m", "base_fare": 950.0, "rating": 4.8},
        {"operator": "Morning Star Travels", "type": "Bharat Benz AC Sleeper", "dep": "21:15", "arr": "06:45", "dur": "9h 30m", "base_fare": 880.0, "rating": 4.6},
        {"operator": "APSRTC Garuda Plus", "type": "Multi-Axle Volvo AC", "dep": "18:00", "arr": "04:30", "dur": "10h 30m", "base_fare": 750.0, "rating": 4.4},
        {"operator": "Kaveri Travels", "type": "Non-AC Sleeper / Seater", "dep": "19:00", "arr": "05:30", "dur": "10h 30m", "base_fare": 600.0, "rating": 4.2},
    ]

    def __init__(self):
        self._bookings_by_id: Dict[str, BusBookingResult] = {}
        self._bookings_by_idempotency: Dict[str, BusBookingResult] = {}
        # Dynamic seat availability simulator
        self.unavailable_seats: List[str] = []

    def canonicalize_city(self, city: str) -> str:
        clean = city.strip().lower().rstrip(".,!?")
        return self.CITY_ALIASES.get(clean, city.strip().title())

    async def search_buses(
        self,
        origin: str,
        destination: str,
        travel_date: str,
        departure_window: Optional[str] = None,
        bus_type_preference: Optional[str] = None,
    ) -> List[BusServiceOption]:
        orig = self.canonicalize_city(origin)
        dest = self.canonicalize_city(destination)

        options: List[BusServiceOption] = []
        for idx, op in enumerate(self.SAMPLE_OPERATORS):
            sid = f"bus_srv_{orig[:3].lower()}_{dest[:3].lower()}_{idx+1}"
            
            # Filter by bus type preference if requested
            if bus_type_preference:
                pref_lower = bus_type_preference.lower()
                if "sleeper" in pref_lower and "sleeper" not in op["type"].lower():
                    continue
                if "ac" in pref_lower and "non-ac" in op["type"].lower():
                    continue

            # Available seats
            all_seats = [f"U{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 13)]
            valid_seats = [s for s in all_seats if s not in self.unavailable_seats]

            fare = op["base_fare"]
            tax = round(fare * 0.05, 2)
            total = fare + tax

            option = BusServiceOption(
                service_id=sid,
                operator_name=op["operator"],
                bus_type=op["type"],
                departure_time=op["dep"],
                arrival_time=op["arr"],
                duration=op["dur"],
                origin=f"{orig} (MGBS / Ameerpet)",
                destination=f"{dest} (RTC Complex)",
                boarding_points=["MGBS", "Ameerpet", "KPHB", "Gachibowli"],
                dropping_points=["RTC Complex", "NAD Junction", "Gajuwaka"],
                base_fare=fare,
                total_fare=total,
                available_seats_count=len(valid_seats),
                available_seats=valid_seats[:6],
                rating=op["rating"],
            )
            options.append(option)

        return options

    async def get_seat_layout(self, service_id: str) -> Dict[str, Any]:
        all_seats = [f"U{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 13)]
        available = [s for s in all_seats if s not in self.unavailable_seats]
        return {
            "service_id": service_id,
            "total_seats": len(all_seats),
            "available_seats": available,
        }

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
        if idempotency_key and idempotency_key in self._bookings_by_idempotency:
            return self._bookings_by_idempotency[idempotency_key]

        # Seat availability check
        for seat in selected_seats:
            if seat in self.unavailable_seats:
                raise ValueError(f"Seat {seat} is no longer available. Please select another seat.")

        orig = self.canonicalize_city(origin)
        dest = self.canonicalize_city(destination)

        booking_id = f"bus_{uuid.uuid4().hex[:8]}"
        pnr = f"PNR{uuid.uuid4().hex[:6].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()

        # Find operator
        operator = "Orange Tours & Travels"
        bus_type = "AC Sleeper (2+1)"
        total_price = fare_amount if fare_amount else (997.5 * passenger_count)

        names = passenger_names or [f"Passenger {i+1}" for i in range(passenger_count)]

        sms_text = f"Your bus booking {pnr} from {orig} to {dest} on {travel_date} is confirmed! You will get the SMS with all the booking details."

        result = BusBookingResult(
            booking_id=booking_id,
            pnr=pnr,
            operator_name=operator,
            bus_type=bus_type,
            origin=orig,
            destination=dest,
            travel_date=travel_date,
            departure_time="20:30",
            arrival_time="06:00",
            boarding_point=boarding_point or "Ameerpet",
            dropping_point=dropping_point or "RTC Complex",
            passenger_count=passenger_count,
            passenger_names=names,
            selected_seats=selected_seats or ["U1"],
            total_fare=round(total_price, 2),
            booking_status="CONFIRMED",
            created_at=now_iso,
            sms_notification_sent=True,
            sms_message=sms_text,
            simulation_enabled=True,
        )

        self._bookings_by_id[booking_id] = result
        self._bookings_by_id[pnr] = result
        if idempotency_key:
            self._bookings_by_idempotency[idempotency_key] = result

        return result

    async def cancel_bus_booking(self, booking_id: str) -> Dict[str, Any]:
        target = self._bookings_by_id.get(booking_id)
        if not target:
            return {"status": "FAILED", "message": f"No booking found for reference {booking_id}"}
        target.booking_status = "CANCELLED"
        refund_amount = round(target.total_fare * 0.90, 2)
        return {
            "status": "CANCELLED",
            "booking_id": target.booking_id,
            "pnr": target.pnr,
            "refund_amount": refund_amount,
            "message": f"Bus booking {target.pnr} has been cancelled. Refund of ₹{refund_amount} initiated. You will get the SMS with all the cancellation details.",
        }
