import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional
from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


# JSON type supporting both PostgreSQL JSONB and SQLite/generic JSON for testing
UniversalJSON = JSON().with_variant(JSONB, "postgresql")


class ConversationModel(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    messages: Mapped[List["MessageModel"]] = relationship(
        "MessageModel", back_populates="conversation", cascade="all, delete-orphan", order_by="MessageModel.created_at"
    )
    tasks: Mapped[List["TaskModel"]] = relationship(
        "TaskModel", back_populates="conversation", cascade="all, delete-orphan", order_by="TaskModel.created_at"
    )


class MessageModel(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    conversation: Mapped["ConversationModel"] = relationship(
        "ConversationModel", back_populates="messages"
    )


class TaskModel(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False, default="BOOK_RIDE")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="IN_PROGRESS")
    current_state: Mapped[str] = mapped_column(String(50), nullable=False, default="INITIAL")
    collected_data: Mapped[dict[str, Any]] = mapped_column(UniversalJSON, default=dict)
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(UniversalJSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    conversation: Mapped["ConversationModel"] = relationship(
        "ConversationModel", back_populates="tasks"
    )
    ride_requests: Mapped[List["RideRequestModel"]] = relationship(
        "RideRequestModel", back_populates="task", cascade="all, delete-orphan"
    )


class RideRequestModel(Base):
    __tablename__ = "ride_requests"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(50), nullable=False, default="mock")
    pickup: Mapped[dict[str, Any]] = mapped_column(UniversalJSON, nullable=False)
    destination: Mapped[dict[str, Any]] = mapped_column(UniversalJSON, nullable=False)
    ride_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    provider_request_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    task: Mapped["TaskModel"] = relationship(
        "TaskModel", back_populates="ride_requests"
    )
