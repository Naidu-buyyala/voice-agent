import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_onevo_get_state(async_client: AsyncClient):
    res = await async_client.get("/api/v1/onevo/state")
    assert res.status_code == 200
    data = res.json()
    assert data["user"]["name"] == "Rahul Sharma"
    assert data["family"]["name"] == "Sharma Household"
    assert len(data["family"]["members"]) >= 5
    assert data["wallet"]["balance"] == 1420.0
    assert data["wallet"]["max_balance"] == 5000.0
    assert len(data["addresses"]) >= 4
    assert len(data["active_services"]) >= 1


@pytest.mark.asyncio
async def test_onevo_wallet_topup_and_limits(async_client: AsyncClient):
    # Topup 500
    res = await async_client.post(
        "/api/v1/onevo/wallet/topup",
        json={"amount": 500.0, "source": "Voice Top-up"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["new_balance"] == 1920.0

    # Try exceeding max capacity: current is 1920, max is 5000, capacity is 3080. Try adding 4000
    bad_res = await async_client.post(
        "/api/v1/onevo/wallet/topup",
        json={"amount": 4000.0, "source": "Exceed"},
    )
    assert bad_res.status_code == 400
    assert "You can add up to ₹3,080" in bad_res.json()["detail"]


@pytest.mark.asyncio
async def test_onevo_wallet_deduct_and_insufficient_balance(async_client: AsyncClient):
    # Deduct 350 for cab
    res = await async_client.post(
        "/api/v1/onevo/wallet/deduct",
        json={
            "amount": 350.0,
            "service_title": "Cab Booking",
            "service_subtitle": "My Home to Airport",
        },
    )
    assert res.status_code == 200
    assert res.json()["deducted_amount"] == 350.0

    # Deduct huge amount to trigger insufficient balance
    bad_res = await async_client.post(
        "/api/v1/onevo/wallet/deduct",
        json={
            "amount": 10000.0,
            "service_title": "Luxury Flight",
            "service_subtitle": "Del to NYC",
        },
    )
    assert bad_res.status_code == 400
    assert "more" in bad_res.json()["detail"]


@pytest.mark.asyncio
async def test_onevo_frequent_address_saving(async_client: AsyncClient):
    res = await async_client.post(
        "/api/v1/onevo/addresses/save-frequent",
        json={"name": "Brother Home"},
    )
    assert res.status_code == 200
    assert res.json()["address"]["name"] == "Brother Home"

    # Verify in state
    state_res = await async_client.get("/api/v1/onevo/state")
    names = [a["name"] for a in state_res.json()["addresses"]]
    assert "Brother Home" in names


@pytest.mark.asyncio
async def test_onevo_active_service_step(async_client: AsyncClient):
    res = await async_client.post("/api/v1/onevo/active-service/step")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["active_service"]["eta_minutes"] in [6, 4, 0, 8]


@pytest.mark.asyncio
async def test_onevo_reset(async_client: AsyncClient):
    res = await async_client.post("/api/v1/onevo/reset")
    assert res.status_code == 200
    state = res.json()["state"]
    assert state["wallet"]["balance"] == 1420.0
