from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.tasks import TaskResponse
from app.db.database import get_db
from app.db.repositories.task import TaskRepository
from app.services.task_service import TaskService

router = APIRouter(prefix="/tasks", tags=["Tasks"])


def get_task_service(session: AsyncSession = Depends(get_db)) -> TaskService:
    repo = TaskRepository(session)
    return TaskService(repo)


@router.get(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Get task status, collected data, and execution results",
)
async def get_task(
    task_id: str,
    service: TaskService = Depends(get_task_service),
):
    task = await service.get_task_by_id(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    return task
