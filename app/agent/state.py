from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    """
    Runtime state of the LangGraph agent orchestration.
    Does NOT store secrets or credentials.
    """
    conversation_id: str
    task_id: Optional[str]

    # Current user turn
    user_message: str
    last_assistant_message: Optional[str]
    conversation_history: Optional[list[dict[str, str]]]
    latest_completed_task: Optional[dict[str, Any]]

    # Intent, Entities, Preferences and Extraction
    intent: Optional[str]
    collected_data: dict[str, Any]
    preferences: Optional[dict[str, Any]]
    missing_fields: list[str]
    next_action: Optional[str]
    _new_fields: Optional[dict[str, Any]]
    _llm_reply: Optional[str]
    _needs_landmark_clarification: Optional[bool]

    # Lifecycle State
    current_step: str
    waiting_for_confirmation: bool
    confirmation_received: Optional[bool]

    # Controlled Tool Action
    tool_name: Optional[str]
    tool_input: Optional[dict[str, Any]]
    tool_result: Optional[dict[str, Any]]

    # Response & Errors
    response: Optional[str]
    error: Optional[str]
