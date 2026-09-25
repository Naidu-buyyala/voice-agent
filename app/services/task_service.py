from typing import Any, Optional
from app.db.models import TaskModel
from app.db.repositories.task import TaskRepository


class TaskService:
    def __init__(self, repository: TaskRepository):
        self.repository = repository

    async def create_task(
        self,
        conversation_id: str,
        task_type: str = "BOOK_RIDE",
        current_state: str = "INITIAL",
        collected_data: Optional[dict[str, Any]] = None,
    ) -> TaskModel:
        return await self.repository.create(
            conversation_id=conversation_id,
            task_type=task_type,
            status="IN_PROGRESS",
            current_state=current_state,
            collected_data=collected_data or {},
        )

    async def get_active_task(self, conversation_id: str) -> Optional[TaskModel]:
        return await self.repository.get_active_by_conversation(conversation_id)

    async def get_task_by_id(self, task_id: str) -> Optional[TaskModel]:
        return await self.repository.get_by_id(task_id)

    async def update_task_state(
        self,
        task: TaskModel,
        status: Optional[str] = None,
        current_state: Optional[str] = None,
        collected_data: Optional[dict[str, Any]] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> TaskModel:
        return await self.repository.update(
            task=task,
            status=status,
            current_state=current_state,
            collected_data=collected_data,
            result=result,
            error=error,
        )
