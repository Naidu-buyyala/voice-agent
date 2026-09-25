import pytest
import httpx
from app.domain.location import Location
from app.domain.ride import RideRequest, RideOption, RideEstimate
from app.providers.uber.client import UberClient
from app.providers.uber.adapter import UberAdapter


def mock_transport_handler(request: httpx.Request) -> httpx.Response:
    url_str = str(request.url)

    if "/products" in url_str:
        return httpx.Response(
            200,
            json={
                "products": [
                    {
                        "product_id": "uber_go_123",
                        "display_name": "Uber Go",
                        "capacity": 4,
                    },
                    {
                        "product_id": "uber_premier_456",
                        "display_name": "Uber Premier",
                        "capacity": 4,
                    },
                ]
            },
        )

    if "/estimates/price" in url_str:
        return httpx.Response(
            200,
            json={
                "prices": [
                    {
                        "product_id": "uber_go_123",
                        "currency_code": "INR",
                        "display_name": "Uber Go",
                        "low_estimate": 480.0,
                        "high_estimate": 520.0,
                        "surge_multiplier": 1.0,
                    }
                ]
            },
        )

    if "/requests" in url_str and request.method == "POST":
        return httpx.Response(
            200,
            json={
                "request_id": "uber_req_real_789xyz",
                "status": "accepted",
                "product_id": "uber_go_123",
                "driver": {"name": "Vikram S."},
                "vehicle": {"license_plate": "TS 08 AB 4321"},
            },
        )

    return httpx.Response(404, json={"message": "Not Found"})


@pytest.fixture
def mock_uber_adapter():
    transport = httpx.MockTransport(mock_transport_handler)
    mock_client = httpx.AsyncClient(transport=transport)
    uber_client = UberClient(
        access_token="test_token_123", sandbox=True, client=mock_client
    )
    return UberAdapter(client=uber_client)


@pytest.mark.asyncio
async def test_uber_adapter_get_options(mock_uber_adapter: UberAdapter):
    pickup = Location(address="Hitech City")
    dest = Location(address="Hyderabad Airport")

    options = await mock_uber_adapter.get_ride_options(pickup, dest)
    assert len(options) == 2
    assert options[0].name == "Uber Go"
    assert options[0].option_id == "uber_go_123"


@pytest.mark.asyncio
async def test_uber_adapter_get_estimate(mock_uber_adapter: UberAdapter):
    pickup = Location(address="Hitech City")
    dest = Location(address="Hyderabad Airport")
    req = RideRequest(pickup=pickup, destination=dest, ride_type="Uber Go")
    option = RideOption(option_id="uber_go_123", name="Uber Go")

    estimate = await mock_uber_adapter.get_estimate(req, option)
    assert estimate.currency == "INR"
    assert estimate.amount == 480.0


@pytest.mark.asyncio
async def test_uber_adapter_book_ride(mock_uber_adapter: UberAdapter):
    pickup = Location(address="Hitech City")
    dest = Location(address="Hyderabad Airport")
    req = RideRequest(pickup=pickup, destination=dest, ride_type="Uber Go")
    option = RideOption(option_id="uber_go_123", name="Uber Go")
    estimate = RideEstimate(currency="INR", amount=480.0)

    booking = await mock_uber_adapter.book_ride(req, option, estimate)
    assert booking.booking_id == "uber_req_real_789xyz"
    assert booking.provider == "uber"
    assert booking.status == "CONFIRMED"
    assert booking.driver_name == "Vikram S."
    assert booking.vehicle_plate == "TS 08 AB 4321"
