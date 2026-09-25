from typing import Optional
from app.db.models import ConversationModel
from app.db.repositories.conversation import ConversationRepository
from app.services.task_service import TaskService
from app.services.agent_service import agent_service


class ConversationService:
    def __init__(
        self,
        conversation_repo: ConversationRepository,
        task_service: TaskService,
    ):
        self.conversation_repo = conversation_repo
        self.task_service = task_service

    async def create_conversation(
        self, user_id: Optional[str] = None
    ) -> ConversationModel:
        return await self.conversation_repo.create(user_id=user_id)

    async def get_conversation(
        self, conversation_id: str
    ) -> Optional[ConversationModel]:
        return await self.conversation_repo.get_by_id(conversation_id)

    async def handle_user_message(
        self, conversation_id: str, user_text: str
    ) -> tuple[str, Optional[str], str]:
        """
        Coordinates user message persistence, active task lookup/creation,
        delegates to AgentService (LangGraph), and updates Task persistence.
        Returns: (agent_response_text, task_id, task_state)
        """
        # 1. Inspect existing conversation messages for context
        conv = await self.get_conversation(conversation_id)
        prev_assistant_msg = None
        if conv and conv.messages:
            for m in reversed(conv.messages):
                if m.role == "assistant":
                    prev_assistant_msg = m.content
                    break

        # 2. Record incoming user message
        await self.conversation_repo.add_message(
            conversation_id=conversation_id, role="user", content=user_text
        )

        # 3. Check for active task or recent completed task
        active_task = await self.task_service.get_active_task(conversation_id)
        latest_task = None
        if not active_task:
            latest_task = await self.task_service.get_latest_task(conversation_id)

        latest_completed_data = None
        if latest_task and latest_task.status == "COMPLETED":
            latest_completed_data = {
                "id": latest_task.id,
                "type": latest_task.type,
                "status": latest_task.status,
                "collected_data": latest_task.collected_data,
                "result": latest_task.result,
            }

        # 4. Invoke LangGraph via AgentService
        reply_text, current_step, updated_data, tool_result, error = (
            await agent_service.process_user_turn(
                conversation_id=conversation_id,
                user_message=user_text,
                task=active_task,
                last_assistant_message=prev_assistant_msg,
                latest_completed_task=latest_completed_data,
            )
        )

        status_map = {
            "WAITING_FOR_USER": "WAITING_FOR_USER",
            "WAITING_FOR_CONFIRMATION": "WAITING_FOR_CONFIRMATION",
            "COMPLETED": "COMPLETED",
            "CANCELLED": "CANCELLED",
            "FAILED": "FAILED",
        }
        new_status = status_map.get(current_step, "IN_PROGRESS")

        # 5. Persist task updates or create new task if action was initiated
        if active_task:
            await self.task_service.update_task_state(
                task=active_task,
                status=new_status,
                current_state=current_step,
                collected_data=updated_data,
                result=tool_result,
                error=error,
            )
            ret_task_id = active_task.id
            ret_step = current_step
        elif current_step in ["WAITING_FOR_USER", "WAITING_FOR_CONFIRMATION", "IN_PROGRESS", "COMPLETED"] and updated_data:
            new_task = await self.task_service.create_task(
                conversation_id=conversation_id,
                task_type="BOOK_RIDE",
                current_state=current_step,
                collected_data=updated_data,
            )
            await self.task_service.update_task_state(
                task=new_task,
                status=new_status,
                current_state=current_step,
                collected_data=updated_data,
                result=tool_result,
                error=error,
            )
            ret_task_id = new_task.id
            ret_step = current_step
        elif latest_task:
            ret_task_id = latest_task.id
            ret_step = latest_task.current_state or latest_task.status
        else:
            ret_task_id = None
            ret_step = "INITIAL"

        # 6. Record assistant response
        await self.conversation_repo.add_message(
            conversation_id=conversation_id, role="assistant", content=reply_text
        )

        return reply_text, ret_task_id, ret_step
