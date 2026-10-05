import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_e2e_food_ordering_delegation_flow(async_client: AsyncClient):
    """
    Canonical journey:
    User: "Order food" -> "Sure! What food or dish would you like to order?"
    User: "Any biryani" -> "Sure, biryani sounds good. Do you have a restaurant preference, or should I pick a highly rated option for you?"
    User: "your wish" -> "Sure, I'll pick a highly rated biryani restaurant for you. Where should I deliver it?"
    User: "to Cyber Towers, Madhapur" -> Previews order & asks confirmation
    User: "yes" -> Places order
    """
    # 1. Create conversation
    create_res = await async_client.post("/api/v1/conversations")
    assert create_res.status_code == 201
    conv_id = create_res.json()["id"]

    # Turn 1: "Order food"
    res1 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "Order food"},
    )
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["task_state"] == "WAITING_FOR_USER"
    assert "food" in d1["message"].lower() or "eat" in d1["message"].lower()

    # Turn 2: "Any biryani"
    res2 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "Any biryani"},
    )
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["task_state"] == "WAITING_FOR_USER"
    assert "biryani" in d2["message"].lower()
    # Anti-repetition: Must not ask "what food/cuisine do you crave", must ask restaurant or choice
    assert "restaurant" in d2["message"].lower() or "pick" in d2["message"].lower()

    # Turn 3: "your wish"
    res3 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "your wish"},
    )
    assert res3.status_code == 200
    d3 = res3.json()
    assert d3["task_state"] == "WAITING_FOR_USER"
    # Never reset or show welcome message! Must ask for delivery location
    assert "deliver" in d3["message"].lower()

    # Turn 4: "to Cyber Towers, Madhapur"
    res4 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "to Cyber Towers, Madhapur"},
    )
    assert res4.status_code == 200
    d4 = res4.json()
    assert d4["task_state"] == "WAITING_FOR_CONFIRMATION"
    assert "order" in d4["message"].lower()

    # Turn 5: "yes"
    res5 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "yes"},
    )
    assert res5.status_code == 200
    d5 = res5.json()
    assert d5["task_state"] == "COMPLETED"
    assert "placed successfully" in d5["message"].lower()
    assert "Order ID" in d5["message"]


@pytest.mark.asyncio
async def test_e2e_ride_delegation_flow(async_client: AsyncClient):
    """
    Ride delegation journey:
    User: "from Hitech City to Airport"
    User: "your wish"
    Assistant chooses standard tier (Uber Go) and asks confirmation.
    User: "yes" -> Confirmed.
    """
    create_res = await async_client.post("/api/v1/conversations")
    assert create_res.status_code == 201
    conv_id = create_res.json()["id"]

    # Turn 1: Locations
    res1 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "from Hitech City to Airport"},
    )
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["task_state"] == "WAITING_FOR_USER"
    assert any(w in d1["message"] for w in ["Cab", "Bike", "Auto", "Parcel", "What kind of ride", "prefer"])

    # Turn 2: "your wish"
    res2 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "your wish"},
    )
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["task_state"] == "WAITING_FOR_CONFIRMATION"
    assert "Uber Go" in d2["message"]

    # Turn 3: "confirm"
    res3 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "confirm"},
    )
    assert res3.status_code == 200
    d3 = res3.json()
    assert d3["task_state"] == "COMPLETED"
    assert "booked successfully" in d3["message"].lower()


@pytest.mark.asyncio
async def test_e2e_home_cleaning_delegation_flow(async_client: AsyncClient):
    """
    Home cleaning delegation journey:
    User: "I need home cleaning" -> asks service type
    User: "Deep cleaning" -> asks package
    User: "best one" -> selects top-rated package, asks for location
    User: "at Kondapur" -> asks confirmation
    User: "yes" -> appointment booked
    """
    create_res = await async_client.post("/api/v1/conversations")
    assert create_res.status_code == 201
    conv_id = create_res.json()["id"]

    # Turn 1: "I need home cleaning"
    res1 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "I need home cleaning"},
    )
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["task_state"] == "WAITING_FOR_USER"

    # Turn 2: "Deep cleaning"
    res2 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "Deep cleaning"},
    )
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["task_state"] == "WAITING_FOR_USER"
    assert "package" in d2["message"].lower()

    # Turn 3: "best one"
    res3 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "best one"},
    )
    assert res3.status_code == 200
    d3 = res3.json()
    assert d3["task_state"] == "WAITING_FOR_USER"
    assert "address" in d3["message"].lower() or "where" in d3["message"].lower() or "come" in d3["message"].lower()

    # Turn 4: "at Kondapur, Hyderabad"
    res4 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "at Kondapur, Hyderabad"},
    )
    assert res4.status_code == 200
    d4 = res4.json()
    assert d4["task_state"] == "WAITING_FOR_CONFIRMATION"

    # Turn 5: "yes"
    res5 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "yes"},
    )
    assert res5.status_code == 200
    d5 = res5.json()
    assert d5["task_state"] == "COMPLETED"
    assert "confirmed" in d5["message"].lower()

