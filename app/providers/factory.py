from functools import lru_cache
from app.config.settings import get_settings
from app.providers.base import RideProvider
from app.providers.mock.adapter import MockRideProvider


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
