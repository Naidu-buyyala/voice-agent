import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_conversation_and_message_flow(async_client: AsyncClient):
    # 1. Create a new conversation
    conv_res = await async_client.post(
        "/api/v1/conversations", json={"user_id": "test_user_001"}
    )
    assert conv_res.status_code == 201
    conv_data = conv_res.json()
    conv_id = conv_data["id"]
    assert conv_data["status"] == "ACTIVE"

    # 2. Send first message
    msg_res = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "I want to book an Uber to the airport"},
    )
    assert msg_res.status_code == 200
    msg_data = msg_res.json()
    assert msg_data["conversation_id"] == conv_id
    assert "pick you up" in msg_data["message"].lower()
    task_id = msg_data["task_id"]

    # 3. Retrieve conversation history
    detail_res = await async_client.get(f"/api/v1/conversations/{conv_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert len(detail_data["messages"]) == 2  # user + assistant
    assert detail_data["messages"][0]["role"] == "user"
    assert detail_data["messages"][1]["role"] == "assistant"

    # 4. Inspect Task state
    task_res = await async_client.get(f"/api/v1/tasks/{task_id}")
    assert task_res.status_code == 200
    task_data = task_res.json()
    assert task_data["id"] == task_id
    assert task_data["type"] == "BOOK_RIDE"
    assert task_data["status"] == "WAITING_FOR_USER"


@pytest.mark.asyncio
async def test_api_single_word_bus_flow(async_client: AsyncClient):
    """Verify HTTP endpoint handles user sending 'bus' -> returns route question with BOOK_BUS task."""
    conv_res = await async_client.post(
        "/api/v1/conversations", json={"user_id": "test_user_bus"}
    )
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]

    msg_res = await async_client.post(
        f"/api/v1/conversations/{conv_id}/messages",
        json={"message": "bus"},
    )
    assert msg_res.status_code == 200
    msg_data = msg_res.json()
    reply = msg_data["message"].lower()

    assert "where would you like to travel from and to" in reply
    assert "don't support" not in reply

    task_id = msg_data["task_id"]
    task_res = await async_client.get(f"/api/v1/tasks/{task_id}")
    assert task_res.status_code == 200
    task_data = task_res.json()
    assert task_data["type"] == "BOOK_BUS"
