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

        # 3. Get or initialize active task
        task = await self.task_service.get_active_task(conversation_id)
        if not task:
            task = await self.task_service.create_task(
                conversation_id=conversation_id,
                task_type="BOOK_RIDE",
                current_state="INITIAL",
            )

        # 4. Invoke LangGraph via AgentService
        reply_text, current_step, updated_data, tool_result, error = (
            await agent_service.process_user_turn(
                conversation_id=conversation_id,
                user_message=user_text,
                task=task,
                last_assistant_message=prev_assistant_msg,
            )
        )

        # 4. Map current_step to Task status
        status_map = {
            "WAITING_FOR_USER": "WAITING_FOR_USER",
            "WAITING_FOR_CONFIRMATION": "WAITING_FOR_CONFIRMATION",
            "COMPLETED": "COMPLETED",
            "CANCELLED": "CANCELLED",
            "FAILED": "FAILED",
        }
        new_status = status_map.get(current_step, "IN_PROGRESS")

        # 5. Persist task updates
        await self.task_service.update_task_state(
            task=task,
            status=new_status,
            current_state=current_step,
            collected_data=updated_data,
            result=tool_result,
            error=error,
        )

        # 6. Record assistant response
        await self.conversation_repo.add_message(
            conversation_id=conversation_id, role="assistant", content=reply_text
        )

        return reply_text, task.id, current_step
