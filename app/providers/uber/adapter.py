from typing import List, Optional
from app.domain.location import Location
from app.domain.ride import RideRequest, RideOption, RideEstimate, RideBooking
from app.providers.base import RideProvider
from app.providers.uber.client import UberClient
from app.providers.uber.mapper import UberMapper, MockLocationResolver


class UberAdapter(RideProvider):
    """
    Implements the RideProvider interface for Uber.
    Converts domain requests -> UberClient HTTP calls -> maps back to Domain entities.
    """

    def __init__(self, client: Optional[UberClient] = None):
        self.client = client or UberClient()
        self.mapper = UberMapper()

    async def get_ride_options(
        self, pickup: Location, destination: Location
    ) -> List[RideOption]:
        lat, lng = MockLocationResolver.resolve(pickup)
        raw_products = await self.client.get_products(lat, lng)
        products_list = raw_products.get("products", raw_products) if isinstance(raw_products, dict) else raw_products
        return self.mapper.to_ride_options(products_list)

    async def get_estimate(
        self, ride_request: RideRequest, selected_option: RideOption
    ) -> RideEstimate:
        start_lat, start_lng = MockLocationResolver.resolve(ride_request.pickup)
        end_lat, end_lng = MockLocationResolver.resolve(ride_request.destination)

        raw_estimates = await self.client.get_price_estimates(
            start_latitude=start_lat,
            start_longitude=start_lng,
            end_latitude=end_lat,
            end_longitude=end_lng,
        )

        estimates_list = (
            raw_estimates.get("prices", raw_estimates)
            if isinstance(raw_estimates, dict)
            else raw_estimates
        )

        # Find matching product estimate or take first
        target_estimate = None
        for item in estimates_list:
            if item.get("display_name", "").lower() == selected_option.name.lower():
                target_estimate = item
                break
        if not target_estimate and estimates_list:
            target_estimate = estimates_list[0]

        if not target_estimate:
            return RideEstimate(currency="INR", amount=450.0, surge_multiplier=1.0)

        return self.mapper.to_ride_estimate(target_estimate)

    async def book_ride(
        self,
        ride_request: RideRequest,
        selected_option: RideOption,
        estimate: RideEstimate,
    ) -> RideBooking:
        start_lat, start_lng = MockLocationResolver.resolve(ride_request.pickup)
        end_lat, end_lng = MockLocationResolver.resolve(ride_request.destination)

        payload = {
            "product_id": selected_option.option_id,
            "start_latitude": start_lat,
            "start_longitude": start_lng,
            "end_latitude": end_lat,
            "end_longitude": end_lng,
        }

        raw_booking = await self.client.create_ride_request(payload)
        return self.mapper.to_ride_booking(
            ride_resp=raw_booking,
            pickup=ride_request.pickup,
            destination=ride_request.destination,
            fare=estimate,
        )
