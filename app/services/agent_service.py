from typing import Any, Optional
from app.agent.graph import agent_graph
from app.agent.state import AgentState
from app.db.models import TaskModel


class AgentService:
    """
    Coordinates LangGraph agent execution for incoming user messages.
    LangGraph = Orchestrate.
    Backend = Control & DB Persistence.
    """

    async def process_user_turn(
        self,
        conversation_id: str,
        user_message: str,
        task: Optional[TaskModel] = None,
        last_assistant_message: Optional[str] = None,
        latest_completed_task: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str, dict[str, Any], Optional[dict[str, Any]], Optional[str]]:
        """
        Executes a turn through LangGraph and returns:
        (response_text, new_task_state, updated_collected_data, tool_result, error)
        """
        # Prepare graph state
        collected = dict(task.collected_data) if task and task.collected_data else {}
        waiting_confirmation = (
            task.status == "WAITING_FOR_CONFIRMATION" if task else False
        )

        initial_state: AgentState = {
            "conversation_id": conversation_id,
            "task_id": task.id if task else None,
            "user_message": user_message,
            "last_assistant_message": last_assistant_message,
            "latest_completed_task": latest_completed_task,
            "intent": None,
            "collected_data": collected,
            "missing_fields": [],
            "_new_fields": None,
            "current_step": task.current_state if task else "INITIAL",
            "waiting_for_confirmation": waiting_confirmation,
            "confirmation_received": None,
            "tool_name": None,
            "tool_input": None,
            "tool_result": None,
            "response": None,
            "error": None,
        }

        final_state = await agent_graph.ainvoke(initial_state)

        response_text = final_state.get(
            "response", "I'm sorry, I couldn't process your request."
        )
        task_step = final_state.get("current_step", "IN_PROGRESS")
        new_collected = final_state.get("collected_data", {})
        tool_result = final_state.get("tool_result")
        error = final_state.get("error")

        return response_text, task_step, new_collected, tool_result, error


agent_service = AgentService()
