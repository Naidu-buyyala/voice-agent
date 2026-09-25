from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CreateConversationRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional user ID identifier")


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime


class SendMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Text message from user")


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime


class MessageTurnResponse(BaseModel):
    conversation_id: str
    task_id: Optional[str] = None
    message: str
    task_state: str


class ConversationDetailResponse(ConversationResponse):
    messages: List[MessageResponse] = Field(default_factory=list)
