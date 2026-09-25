import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models import ConversationModel, MessageModel, TaskModel, RideRequestModel


@pytest.mark.asyncio
async def test_database_models_persistence(test_db_session: AsyncSession):
    # 1. Create a Conversation
    conversation = ConversationModel(user_id="user_test_42", status="ACTIVE")
    test_db_session.add(conversation)
    await test_db_session.commit()
    await test_db_session.refresh(conversation)

    assert conversation.id is not None
    assert conversation.status == "ACTIVE"

    # 2. Add a Message
    message = MessageModel(
        conversation_id=conversation.id,
        role="user",
        content="I need a ride to the station",
    )
    test_db_session.add(message)
    await test_db_session.commit()

    # 3. Add a Task with JSON payload
    task = TaskModel(
        conversation_id=conversation.id,
        type="BOOK_RIDE",
        status="IN_PROGRESS",
        current_state="COLLECTING_DETAILS",
        collected_data={"pickup": "Hitech City", "destination": "Railway Station"},
    )
    test_db_session.add(task)
    await test_db_session.commit()
    await test_db_session.refresh(task)

    assert task.collected_data["pickup"] == "Hitech City"

    # 4. Add Ride Request
    ride_req = RideRequestModel(
        task_id=task.id,
        provider="mock",
        pickup={"address": "Hitech City", "latitude": 17.44, "longitude": 78.37},
        destination={"address": "Secunderabad", "latitude": 17.43, "longitude": 78.50},
        ride_type="Uber Go",
        status="PENDING",
    )
    test_db_session.add(ride_req)
    await test_db_session.commit()

    # 5. Query and verify relations
    query = select(ConversationModel).where(ConversationModel.id == conversation.id)
    result = await test_db_session.execute(query)
    fetched_conv = result.scalar_one()

    assert fetched_conv.user_id == "user_test_42"
