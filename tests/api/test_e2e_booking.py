import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_end_to_end_conversational_booking(async_client: AsyncClient):
    """
    Acceptance test for the canonical multi-turn booking journey:
    1. User: "I want to book an Uber." -> Agent: asks for pickup
    2. User: "Hitech City." -> Agent: asks for destination
    3. User: "Hyderabad Airport." -> Agent: returns estimate and asks for confirmation
    4. User: "Yes, please book it." -> Agent: executes BookRideTool and returns Booking ID
    """
    # 1. Create conversation
    create_res = await async_client.post("/api/v1/conversations")
    assert create_res.status_code == 201
    conv_id = create_res.json()["id"]

    # Turn 1: Initial Intent
    res1 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "I want to book an Uber."},
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert "pick you up" in data1["message"].lower()
    assert data1["task_state"] == "WAITING_FOR_USER"

    # Turn 2: Provide Pickup
    res2 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "Hitech City."},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert any(w in data2["message"].lower() for w in ["where", "head", "go"])
    assert data2["task_state"] == "WAITING_FOR_USER"

    # Turn 3: Provide Destination -> Agent asks for ride / service type
    res3 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "Hyderabad Airport."},
    )
    assert res3.status_code == 200
    data3 = res3.json()
    assert any(w in data3["message"] for w in ["Cab", "Bike", "Auto", "Parcel", "What kind of ride", "prefer"])
    assert data3["task_state"] == "WAITING_FOR_USER"

    # Turn 3b: Select Service Type -> Agent returns estimate and asks for confirmation
    res3b = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "cab"},
    )
    assert res3b.status_code == 200
    data3b = res3b.json()
    assert "estimated fare is" in data3b["message"]
    assert any(w in data3b["message"].lower() for w in ["book this", "book it", "book"])
    assert data3b["task_state"] == "WAITING_FOR_CONFIRMATION"

    # Turn 4: Confirm Booking
    res4 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "Yes, please book it."},
    )
    assert res4.status_code == 200
    data4 = res4.json()
    assert "booked successfully" in data4["message"].lower()
    assert "mock_booking_" in data4["message"]
    assert data4["task_state"] == "COMPLETED"

    # Verify task state in database
    task_res = await async_client.get(f"/api/v1/tasks/{data4['task_id']}")
    assert task_res.status_code == 200
    task_db = task_res.json()
    assert task_db["status"] == "COMPLETED"
    assert task_db["result"]["booking_id"].startswith("mock_booking_")
    assert task_db["collected_data"]["pickup"] == "Hitech City"
    assert task_db["collected_data"]["destination"] == "Hyderabad Airport"


@pytest.mark.asyncio
async def test_end_to_end_cancellation_flow(async_client: AsyncClient):
    """
    User declines confirmation:
    Agent cancels request and does NOT book ride.
    """
    create_res = await async_client.post("/api/v1/conversations")
    conv_id = create_res.json()["id"]

    # Provide all locations and ride type in one message
    res1 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "from Hitech City to Hyderabad Airport by cab"},
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["task_state"] == "WAITING_FOR_CONFIRMATION"

    # Reject
    res2 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "No, cancel this ride."},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert "cancelled" in data2["message"].lower()
    assert data2["task_state"] == "CANCELLED"


@pytest.mark.asyncio
async def test_end_to_end_broad_locality_clarification_and_bike_booking(async_client: AsyncClient):
    """
    Journey:
    1. User gives broad locality: "Take me from Gachibowli to Airport"
       -> Agent asks for exact building/landmark in Gachibowli.
    2. User specifies building: "DLF Cyber City"
       -> Agent accepts refined location "DLF Cyber City, Gachibowli" and asks for ride type.
    3. User specifies service: "Book a bike"
       -> Agent selects Uber Moto (estimated fare ₹120) and asks for confirmation.
    4. User confirms: "Proceed"
       -> Agent books Uber Moto and returns booking ID.
    """
    create_res = await async_client.post("/api/v1/conversations")
    conv_id = create_res.json()["id"]

    # Turn 1: Broad locality
    res1 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "Take me from Gachibowli to Airport"},
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert "Gachibowli" in data1["message"]

    # Turn 2: Exact landmark specified -> Agent asks for ride type (Cab, Bike, Auto, Parcel)
    res2 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "DLF Cyber City"},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert any(w in data2["message"] for w in ["Cab", "Bike", "Auto", "Parcel", "What kind of ride"])
    assert data2["task_state"] == "WAITING_FOR_USER"

    # Turn 3: Choose Bike / Moto
    res3 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "Book a bike"},
    )
    assert res3.status_code == 200
    data3 = res3.json()
    assert "Uber Moto" in data3["message"]
    assert "₹120" in data3["message"]
    assert data3["task_state"] == "WAITING_FOR_CONFIRMATION"

    # Turn 4: Confirm
    res4 = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "Proceed"},
    )
    assert res4.status_code == 200
    data4 = res4.json()
    assert data4["task_state"] == "COMPLETED"
    assert "Booking ID: mock_booking_" in data4["message"]

