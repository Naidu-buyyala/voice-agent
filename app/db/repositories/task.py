from typing import Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import TaskModel


class TaskRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        conversation_id: str,
        task_type: str = "BOOK_RIDE",
        status: str = "IN_PROGRESS",
        current_state: str = "INITIAL",
        collected_data: Optional[dict[str, Any]] = None,
    ) -> TaskModel:
        task = TaskModel(
            conversation_id=conversation_id,
            type=task_type,
            status=status,
            current_state=current_state,
            collected_data=collected_data or {},
        )
        self.session.add(task)
        await self.session.commit()
        await self.session.refresh(task)
        return task

    async def get_by_id(self, task_id: str) -> Optional[TaskModel]:
        stmt = select(TaskModel).where(TaskModel.id == task_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_active_by_conversation(
        self, conversation_id: str
    ) -> Optional[TaskModel]:
        stmt = (
            select(TaskModel)
            .where(
                TaskModel.conversation_id == conversation_id,
                TaskModel.status.in_(["IN_PROGRESS", "WAITING_FOR_USER", "WAITING_FOR_CONFIRMATION"]),
            )
            .order_by(TaskModel.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def update(
        self,
        task: TaskModel,
        status: Optional[str] = None,
        current_state: Optional[str] = None,
        collected_data: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> TaskModel:
        if status is not None:
            task.status = status
        if current_state is not None:
            task.current_state = current_state
        if collected_data is not None:
            task.collected_data = collected_data
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error

        self.session.add(task)
        await self.session.commit()
        await self.session.refresh(task)
        return task
