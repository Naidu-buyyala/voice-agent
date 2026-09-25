from typing import List, Optional
from app.domain.location import Location
from app.domain.ride import RideOption, RideEstimate, RideBooking
from app.providers.uber.models import UberProduct, UberPriceEstimate, UberRideResponse


class MockLocationResolver:
    """
    Resolves human-readable addresses to GPS coordinates.
    Separated so Google Maps/Mapbox/Uber Places can plug in later.
    """
    CITY_COORDS = {
        "hitech city": (17.4435, 78.3772),
        "hyderabad airport": (17.2403, 78.4294),
        "secunderabad station": (17.4334, 78.5042),
    }

    @classmethod
    def resolve(cls, location: Location) -> tuple[float, float]:
        if location.latitude is not None and location.longitude is not None:
            return location.latitude, location.longitude

        clean = location.address.strip().lower().rstrip(".")
        for key, coords in cls.CITY_COORDS.items():
            if key in clean or clean in key:
                return coords

        # Default Hyderabad metro coordinates if unmapped
        return 17.3850, 78.4867


class UberMapper:
    """
    Two-way transformation between Uber DTOs and Generic Domain Models.
    Ensures zero Uber-specific leakages into domain logic.
    """

    @staticmethod
    def to_ride_options(products_data: List[dict]) -> List[RideOption]:
        options = []
        for p in products_data:
            prod = UberProduct.model_validate(p)
            options.append(
                RideOption(
                    option_id=prod.product_id,
                    name=prod.display_name,
                    capacity=prod.capacity,
                    eta_minutes=5,
                )
            )
        return options

    @staticmethod
    def to_ride_estimate(estimate_data: dict) -> RideEstimate:
        est = UberPriceEstimate.model_validate(estimate_data)
        amount = est.low_estimate or (float(est.estimate.replace("$", "").replace("₹", "")) if est.estimate else 450.0)
        return RideEstimate(
            currency=est.currency_code,
            amount=amount,
            surge_multiplier=est.surge_multiplier,
        )

    @staticmethod
    def to_ride_booking(
        ride_resp: dict,
        pickup: Location,
        destination: Location,
        fare: RideEstimate,
    ) -> RideBooking:
        resp = UberRideResponse.model_validate(ride_resp)
        status_map = {
            "processing": "PROCESSING",
            "accepted": "CONFIRMED",
            "arriving": "CONFIRMED",
            "in_progress": "CONFIRMED",
            "completed": "CONFIRMED",
            "rider_canceled": "CANCELLED",
        }
        domain_status = status_map.get(resp.status.lower(), "CONFIRMED")
        driver_name = resp.driver.get("name") if resp.driver else "Uber Driver"
        vehicle_plate = resp.vehicle.get("license_plate") if resp.vehicle else "TS 09 UB 9999"

        return RideBooking(
            booking_id=resp.request_id,
            provider="uber",
            status=domain_status,
            pickup=pickup,
            destination=destination,
            fare=fare,
            driver_name=driver_name,
            vehicle_plate=vehicle_plate,
        )
