from functools import lru_cache
from app.config.settings import get_settings
from app.providers.base import (
    RideProvider,
    HomeServiceProvider,
    FoodProvider,
    BusProvider,
    TrainProvider,
)
from app.providers.mock.adapter import MockRideProvider
from app.providers.mock.urban_clean import UrbanCleanAdapter
from app.providers.mock.food import MockFoodProvider
from app.providers.mock.bus import MockBusProvider
from app.providers.mock.train import MockTrainProvider


@lru_cache
def get_ride_provider() -> RideProvider:
    settings = get_settings()
    if settings.DEFAULT_RIDE_PROVIDER == "uber":
        try:
            from app.providers.uber.adapter import UberAdapter
            return UberAdapter()
        except ImportError:
            return MockRideProvider()
    return MockRideProvider()


@lru_cache
def get_home_service_provider() -> HomeServiceProvider:
    return UrbanCleanAdapter()


@lru_cache
def get_food_provider() -> FoodProvider:
    return MockFoodProvider()


@lru_cache
def get_bus_provider() -> BusProvider:
    return MockBusProvider()


@lru_cache
def get_train_provider() -> TrainProvider:
    return MockTrainProvider()

