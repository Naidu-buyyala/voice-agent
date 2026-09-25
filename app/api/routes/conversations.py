from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.conversations import (
    CreateConversationRequest,
    ConversationResponse,
    ConversationDetailResponse,
    SendMessageRequest,
    MessageTurnResponse,
)
from app.db.database import get_db
from app.db.repositories.conversation import ConversationRepository
from app.db.repositories.task import TaskRepository
from app.services.conversation_service import ConversationService
from app.services.task_service import TaskService

router = APIRouter(prefix="/conversations", tags=["Conversations"])


def get_conversation_service(
    session: AsyncSession = Depends(get_db),
) -> ConversationService:
    conv_repo = ConversationRepository(session)
    task_repo = TaskRepository(session)
    task_service = TaskService(task_repo)
    return ConversationService(conv_repo, task_service)


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session",
)
async def create_conversation(
    req: Optional[CreateConversationRequest] = None,
    service: ConversationService = Depends(get_conversation_service),
):
    user_id = req.user_id if req else None
    return await service.create_conversation(user_id=user_id)


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Get conversation details and message history",
)
async def get_conversation(
    conversation_id: str,
    service: ConversationService = Depends(get_conversation_service),
):
    conv = await service.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return conv


@router.post(
    "/{conversation_id}/messages",
    response_model=MessageTurnResponse,
    summary="Send a message to the AI agent",
)
async def send_message(
    conversation_id: str,
    req: SendMessageRequest,
    service: ConversationService = Depends(get_conversation_service),
):
    conv = await service.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    reply_text, task_id, task_state = await service.handle_user_message(
        conversation_id=conversation_id, user_text=req.message
    )

    return MessageTurnResponse(
        conversation_id=conversation_id,
        task_id=task_id,
        message=reply_text,
        task_state=task_state,
    )
