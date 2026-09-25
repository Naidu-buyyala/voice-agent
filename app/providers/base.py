from abc import ABC, abstractmethod
from typing import List
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
