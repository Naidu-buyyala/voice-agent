import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from app.providers.base import (
    TrainProvider,
    TrainServiceOption,
    TrainClassOption,
    TrainBookingResult,
)


class MockTrainProvider(TrainProvider):
    """
    Deterministic mock train provider simulating Indian Railways / IRCTC.
    Handles Secunderabad / Hyderabad station distinction, PNR generation,
    berth preferences, class options (SL, 3A, 2A), waiting list / RAC statuses,
    idempotency, and cancellation.
    """

    STATION_ALIASES: Dict[str, str] = {
        "hyd": "Hyderabad Deccan (HYB)",
        "hyderabad": "Hyderabad Deccan (HYB)",
        "hyb": "Hyderabad Deccan (HYB)",
        "secunderabad": "Secunderabad Junction (SC)",
        "sc": "Secunderabad Junction (SC)",
        "kacheguda": "Kacheguda (KCG)",
        "kcg": "Kacheguda (KCG)",
        "vizag": "Visakhapatnam Junction (VSKP)",
        "visakhapatnam": "Visakhapatnam Junction (VSKP)",
        "vskp": "Visakhapatnam Junction (VSKP)",
        "vijayawada": "Vijayawada Junction (BZA)",
        "bza": "Vijayawada Junction (BZA)",
        "tirupati": "Tirupati (TPTY)",
        "tpty": "Tirupati (TPTY)",
        "chennai": "Chennai Central (MAS)",
        "mas": "Chennai Central (MAS)",
        "bengaluru": "KSR Bengaluru (SBC)",
        "bangalore": "KSR Bengaluru (SBC)",
        "sbc": "KSR Bengaluru (SBC)",
    }

    SAMPLE_TRAINS = [
        {
            "number": "12728",
            "name": "Godavari Express",
            "origin": "Hyderabad Deccan (HYB)",
            "dest": "Visakhapatnam Junction (VSKP)",
            "dep": "17:05",
            "arr": "05:45",
            "dur": "12h 40m",
            "classes": [
                {"code": "SL", "name": "Sleeper", "fare": 435.0, "status": "AVAILABLE-42", "avail": True},
                {"code": "3A", "name": "AC 3 Tier", "fare": 1150.0, "status": "AVAILABLE-18", "avail": True},
                {"code": "2A", "name": "AC 2 Tier", "fare": 1640.0, "status": "AVAILABLE-06", "avail": True},
            ],
        },
        {
            "number": "12760",
            "name": "Charminar Express",
            "origin": "Hyderabad Deccan (HYB)",
            "dest": "Chennai Central (MAS)",
            "dep": "18:00",
            "arr": "07:00",
            "dur": "13h 00m",
            "classes": [
                {"code": "SL", "name": "Sleeper", "fare": 410.0, "status": "AVAILABLE-28", "avail": True},
                {"code": "3A", "name": "AC 3 Tier", "fare": 1090.0, "status": "AVAILABLE-14", "avail": True},
                {"code": "2A", "name": "AC 2 Tier", "fare": 1560.0, "status": "WL 12", "avail": False},
            ],
        },
        {
            "number": "12734",
            "name": "Narayanadri Express",
            "origin": "Secunderabad Junction (SC)",
            "dest": "Tirupati (TPTY)",
            "dep": "18:05",
            "arr": "06:55",
            "dur": "12h 50m",
            "classes": [
                {"code": "SL", "name": "Sleeper", "fare": 395.0, "status": "AVAILABLE-35", "avail": True},
                {"code": "3A", "name": "AC 3 Tier", "fare": 1050.0, "status": "AVAILABLE-22", "avail": True},
                {"code": "2A", "name": "AC 2 Tier", "fare": 1500.0, "status": "AVAILABLE-08", "avail": True},
            ],
        },
        {
            "number": "12710",
            "name": "Simhapuri Express",
            "origin": "Secunderabad Junction (SC)",
            "dest": "Vijayawada Junction (BZA)",
            "dep": "22:05",
            "arr": "04:30",
            "dur": "6h 25m",
            "classes": [
                {"code": "CC", "name": "AC Chair Car", "fare": 580.0, "status": "AVAILABLE-50", "avail": True},
                {"code": "3A", "name": "AC 3 Tier", "fare": 760.0, "status": "AVAILABLE-30", "avail": True},
                {"code": "SL", "name": "Sleeper", "fare": 270.0, "status": "AVAILABLE-60", "avail": True},
            ],
        },
        {
            "number": "12728",
            "name": "Godavari Express",
            "origin": "Secunderabad Junction (SC)",
            "dest": "Visakhapatnam Junction (VSKP)",
            "dep": "17:05",
            "arr": "05:45",
            "dur": "12h 40m",
            "classes": [
                {"code": "SL", "name": "Sleeper", "fare": 435.0, "status": "AVAILABLE-42", "avail": True},
                {"code": "3A", "name": "AC 3 Tier", "fare": 1150.0, "status": "AVAILABLE-18", "avail": True},
                {"code": "2A", "name": "AC 2 Tier", "fare": 1640.0, "status": "AVAILABLE-06", "avail": True},
            ],
        },
        {
            "number": "20834",
            "name": "Vande Bharat Express",
            "origin": "Secunderabad Junction (SC)",
            "dest": "Visakhapatnam Junction (VSKP)",
            "dep": "15:00",
            "arr": "23:30",
            "dur": "8h 30m",
            "classes": [
                {"code": "CC", "name": "AC Chair Car", "fare": 1720.0, "status": "AVAILABLE-24", "avail": True},
                {"code": "EC", "name": "Exec. Chair Car", "fare": 3170.0, "status": "AVAILABLE-08", "avail": True},
            ],
        },
    ]

    def __init__(self):
        self._bookings_by_id: Dict[str, TrainBookingResult] = {}
        self._bookings_by_idempotency: Dict[str, TrainBookingResult] = {}
        # Dynamic class availability override simulator
        self.unavailable_classes: List[str] = []

    def canonicalize_station(self, text: str) -> str:
        clean = text.strip().lower().rstrip(".,!?")
        # Important: check secunderabad vs hyderabad distinction
        if "secunderabad" in clean or clean == "sc":
            return "Secunderabad Junction (SC)"
        if "hyderabad" in clean or clean in ["hyd", "hyb"]:
            return "Hyderabad Deccan (HYB)"
        for k, v in self.STATION_ALIASES.items():
            if k in clean:
                return v
        return text.strip().title()

    async def search_trains(
        self,
        origin: str,
        destination: str,
        travel_date: str,
        departure_window: Optional[str] = None,
        class_preference: Optional[str] = None,
    ) -> List[TrainServiceOption]:
        orig = self.canonicalize_station(origin)
        dest = self.canonicalize_station(destination)

        orig_keyword = origin.strip().lower()
        dest_keyword = destination.strip().lower()

        results: List[TrainServiceOption] = []
        for t in self.SAMPLE_TRAINS:
            # Match origin & destination
            t_orig = t["origin"].lower()
            t_dest = t["dest"].lower()

            # Respect secunderabad vs hyderabad station options
            orig_match = False
            if "secunderabad" in orig_keyword:
                orig_match = "secunderabad" in t_orig
            elif "hyderabad" in orig_keyword or "hyd" in orig_keyword:
                orig_match = "hyderabad" in t_orig or "secunderabad" in t_orig
            else:
                orig_match = any(w in t_orig for w in orig_keyword.split())

            # Match destination
            dest_clean = dest.lower()
            t_dest_clean = t["dest"].lower()
            if "vskp" in dest_clean or "visakhapatnam" in dest_clean or "vizag" in dest_keyword:
                dest_match = "vskp" in t_dest_clean or "visakhapatnam" in t_dest_clean
            elif "bza" in dest_clean or "vijayawada" in dest_clean:
                dest_match = "bza" in t_dest_clean or "vijayawada" in t_dest_clean
            elif "tpty" in dest_clean or "tirupati" in dest_clean:
                dest_match = "tpty" in t_dest_clean or "tirupati" in t_dest_clean
            elif "mas" in dest_clean or "chennai" in dest_clean:
                dest_match = "mas" in t_dest_clean or "chennai" in t_dest_clean
            else:
                dest_match = dest_keyword in t_dest_clean or any(w in t_dest_clean for w in dest_keyword.split())

            if orig_match and dest_match:
                classes: List[TrainClassOption] = []
                for c in t["classes"]:
                    status = c["status"]
                    is_avail = c["avail"]
                    if c["code"] in self.unavailable_classes:
                        status = "REGRET / NOT AVAILABLE"
                        is_avail = False
                    classes.append(
                        TrainClassOption(
                            class_code=c["code"],
                            class_name=c["name"],
                            fare=c["fare"],
                            availability_status=status,
                            is_available=is_avail,
                        )
                    )

                results.append(
                    TrainServiceOption(
                        train_number=t["number"],
                        train_name=t["name"],
                        origin_station=t["origin"],
                        destination_station=t["dest"],
                        departure_time=t["dep"],
                        arrival_time=t["arr"],
                        duration=t["dur"],
                        available_classes=classes,
                    )
                )

        # Fallback if no specific route matched in sample table: return Godavari Express route
        if not results:
            classes = [
                TrainClassOption(class_code="SL", class_name="Sleeper", fare=435.0, availability_status="AVAILABLE-30", is_available=True),
                TrainClassOption(class_code="3A", class_name="AC 3 Tier", fare=1150.0, availability_status="AVAILABLE-15", is_available=True),
                TrainClassOption(class_code="2A", class_name="AC 2 Tier", fare=1640.0, availability_status="AVAILABLE-04", is_available=True),
            ]
            results.append(
                TrainServiceOption(
                    train_number="12728",
                    train_name="Godavari Express",
                    origin_station=orig,
                    destination_station=dest,
                    departure_time="17:05",
                    arrival_time="05:45",
                    duration="12h 40m",
                    available_classes=classes,
                )
            )

        return results

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
        if idempotency_key and idempotency_key in self._bookings_by_idempotency:
            return self._bookings_by_idempotency[idempotency_key]

        if selected_class in self.unavailable_classes:
            raise ValueError(f"Selected class {selected_class} is currently not available / full.")

        booking_id = f"train_{uuid.uuid4().hex[:8]}"
        pnr = f"{uuid.uuid4().int % 9000000000 + 1000000000}"  # 10-digit standard Indian Railway PNR
        now_iso = datetime.now(timezone.utc).isoformat()

        # Find train name
        train_name = "Godavari Express"
        for t in self.SAMPLE_TRAINS:
            if t["number"] == train_number:
                train_name = t["name"]
                break

        unit_fare = fare_amount if fare_amount else 1150.0
        total_price = unit_fare * passenger_count

        names = passenger_names or [f"Passenger {i+1}" for i in range(passenger_count)]
        pref = berth_preference or "Lower"
        berths = [f"B3-{20 + i*3} ({pref})" for i in range(passenger_count)]

        orig = self.canonicalize_station(origin_station)
        dest = self.canonicalize_station(destination_station)

        sms_text = f"Your IRCTC train ticket PNR: {pnr} on {train_name} ({train_number}) from {orig} to {dest} for {travel_date} is confirmed! You will get the SMS with all the booking details."

        result = TrainBookingResult(
            booking_id=booking_id,
            pnr=pnr,
            train_number=train_number,
            train_name=train_name,
            origin_station=orig,
            destination_station=dest,
            travel_date=travel_date,
            departure_time="17:05",
            arrival_time="05:45",
            selected_class=selected_class,
            passenger_count=passenger_count,
            passenger_names=names,
            allocated_berths=berths,
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

    async def cancel_train_booking(self, booking_id: str) -> Dict[str, Any]:
        target = self._bookings_by_id.get(booking_id)
        if not target:
            return {"status": "FAILED", "message": f"No train booking found for reference {booking_id}"}
        target.booking_status = "CANCELLED"
        clerkage_fee = 60.0 * target.passenger_count
        refund_amount = max(0.0, round(target.total_fare - clerkage_fee, 2))
        return {
            "status": "CANCELLED",
            "booking_id": target.booking_id,
            "pnr": target.pnr,
            "refund_amount": refund_amount,
            "message": f"Train ticket PNR {target.pnr} has been cancelled. Refund of ₹{refund_amount} processed after clerkage fee. You will get the SMS with all the cancellation details.",
        }
